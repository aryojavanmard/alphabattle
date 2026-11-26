"""Online Masked-Action PPO engine with GAE, potential-based shaping, and analytical KL anchor.

Designed for 12GB consumer GPUs (Native Windows RTX 3080):
- Uses a single 1.7B base model in VRAM with multiple lightweight PEFT LoRA adapters (~40 MB each):
    * "default"  -> trainable policy
    * "sft_ref"  -> frozen Stage 1 SFT anchor for exact analytical KL penalty
    * "past_opp" -> periodically rotated historical checkpoint for diverse self-play
- Computes exact analytical KL over legal actions using cached 13-float ref_log_probs from
  rollout collection (zero extra reference forward passes during PPO gradient updates).
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.battle import Battle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player

from alphabattle.dataset import NUM_ACTIONS, tokenize_prompt_preserve_tail
from alphabattle.modeling import AlphaBattlePolicyValueModel
from alphabattle.state import (
    ACTION_ID_TO_INDEX,
    ALL_ACTION_IDS,
    TurnHistoryTracker,
    build_action_space,
    format_battle_state,
)


@dataclass
class RolloutTransition:
    """One non-forced decision step captured during online Showdown play."""

    input_ids: list[int]
    legal_mask: list[bool]
    action_idx: int
    old_log_prob: float
    ref_log_probs: list[float]  # 13-element masked log_softmax under frozen sft_ref
    value: float  # V(s_t) in [-1, 1]
    reward: float = 0.0
    done: bool = False


@dataclass
class PPOConfig:
    """Hyperparameters for online Masked-Action PPO."""

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    kl_coef: float = 0.03
    target_kl: float = 0.15
    ppo_epochs: int = 3
    micro_batch_size: int = 8
    grad_accum: int = 8
    max_grad_norm: float = 1.0
    faint_shaping_weight: float = 0.05
    hp_shaping_weight: float = 0.02
    shaping_scale: float = 1.0
    rollout_temperature: float = 1.0


@dataclass
class PPOUpdateStats:
    """Diagnostics returned after one PPO optimization cycle."""

    policy_loss: float
    value_loss: float
    kl_to_ref: float
    approx_step_kl: float
    entropy: float
    clip_fraction: float
    early_stopped: bool
    num_transitions: int
    opt_steps: int


def compute_battle_potential(
    battle: Battle,
    *,
    faint_weight: float = 0.05,
    hp_weight: float = 0.02,
) -> float:
    """Zero-sum potential function Phi(s) bounded well below terminal +/-1.0 win/loss reward.

    Using r_shape = Phi(s') - Phi(s) guarantees switching/healing loops telescope to 0.
    """
    my_team = list(battle.team.values())
    opp_team = list(battle.opponent_team.values())
    max_size = battle.max_team_size or 6

    my_fainted = sum(1 for m in my_team if m.fainted)
    opp_fainted = sum(1 for m in opp_team if m.fainted)

    my_hp = sum(0.0 if m.fainted else float(m.current_hp_fraction) for m in my_team)
    # Unseen opponent mons are assumed at full HP (1.0) so revealing a healthy mon gives 0 delta
    unseen_opp = max(0, max_size - len(opp_team))
    opp_hp = unseen_opp + sum(
        0.0 if m.fainted else float(m.current_hp_fraction) for m in opp_team
    )

    return faint_weight * (opp_fainted - my_fainted) + hp_weight * (my_hp - opp_hp)


def compute_gae(
    transitions: list[RolloutTransition],
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[list[float], list[float]]:
    """Compute Generalized Advantage Estimates and return targets for one or more episodes."""
    n = len(transitions)
    advantages = [0.0] * n
    returns = [0.0] * n
    gae = 0.0

    for t in range(n - 1, -1, -1):
        tr = transitions[t]
        if tr.done or t == n - 1:
            next_value = 0.0
            not_done = 0.0
        else:
            next_value = transitions[t + 1].value
            not_done = 1.0
        delta = tr.reward + gamma * next_value * not_done - tr.value
        gae = delta + gamma * gae_lambda * not_done * gae
        advantages[t] = gae
        returns[t] = gae + tr.value

    return (advantages, returns)


def _masked_log_softmax(masked_logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Compute stable log_softmax strictly over legal entries (illegal entries set to -1e9)."""
    neg_inf = masked_logits.masked_fill(~legal_mask.bool(), float("-inf"))
    log_probs = F.log_softmax(neg_inf, dim=-1)
    return torch.where(legal_mask.bool(), log_probs, torch.zeros_like(log_probs))


def _set_adapter_if_peft(model: AlphaBattlePolicyValueModel, adapter_name: str) -> bool:
    if hasattr(model.lm, "set_adapter") and hasattr(model.lm, "peft_config"):
        if adapter_name in model.lm.peft_config:
            model.lm.set_adapter(adapter_name)
            return True
    return False


def ensure_ref_adapter(
    model: AlphaBattlePolicyValueModel,
    adapter_dir: str | Path,
    *,
    adapter_name: str = "sft_ref",
) -> bool:
    """Load a frozen copy of a LoRA adapter onto the same base weights (~40 MB VRAM)."""
    if not (hasattr(model.lm, "load_adapter") and hasattr(model.lm, "peft_config")):
        return False
    if adapter_name not in model.lm.peft_config:
        model.lm.load_adapter(str(adapter_dir), adapter_name=adapter_name, is_trainable=False)
    for name, param in model.lm.named_parameters():
        if f".{adapter_name}." in name:
            param.requires_grad = False
    model.lm.set_adapter("default")
    return True


def ppo_update(
    model: AlphaBattlePolicyValueModel,
    optimizer: torch.optim.Optimizer,
    transitions: list[RolloutTransition],
    *,
    pad_token_id: int,
    config: PPOConfig,
    seed: int = 42,
) -> PPOUpdateStats:
    """Run K epochs of clipped Masked-Action PPO with analytical KL penalty to frozen SFT ref."""
    if not transitions:
        return PPOUpdateStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, 0, 0)

    _set_adapter_if_peft(model, "default")
    model.train()
    device = next(model.parameters()).device

    advantages_list, returns_list = compute_gae(
        transitions, gamma=config.gamma, gae_lambda=config.gae_lambda
    )
    adv_tensor = torch.tensor(advantages_list, dtype=torch.float32)
    ret_tensor = torch.tensor(returns_list, dtype=torch.float32).clamp(-1.5, 1.5)
    # Normalize advantages across the whole rollout batch
    if adv_tensor.numel() > 1:
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std(unbiased=False) + 1e-8)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n = len(transitions)
    rng = torch.Generator()
    rng.manual_seed(seed)

    pol_loss_acc = 0.0
    val_loss_acc = 0.0
    kl_ref_acc = 0.0
    step_kl_acc = 0.0
    ent_acc = 0.0
    clip_acc = 0.0
    total_seen = 0
    opt_steps = 0
    early_stopped = False

    for _epoch in range(config.ppo_epochs):
        perm = torch.randperm(n, generator=rng).tolist()
        optimizer.zero_grad(set_to_none=True)
        epoch_step_kl_sum = 0.0
        epoch_count = 0
        micro_in_accum = 0

        for start in range(0, n, config.micro_batch_size):
            idxs = perm[start : start + config.micro_batch_size]
            bsz = len(idxs)
            batch_tr = [transitions[i] for i in idxs]

            max_len = max(len(tr.input_ids) for tr in batch_tr)
            rem = max_len % 8
            if rem != 0:
                max_len += 8 - rem

            input_ids = torch.full((bsz, max_len), pad_token_id, dtype=torch.long)
            attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
            last_indices = torch.zeros((bsz,), dtype=torch.long)
            legal_mask = torch.zeros((bsz, NUM_ACTIONS), dtype=torch.bool)
            actions = torch.zeros((bsz,), dtype=torch.long)
            old_lp = torch.zeros((bsz,), dtype=torch.float32)
            ref_lp = torch.zeros((bsz, NUM_ACTIONS), dtype=torch.float32)
            adv_b = torch.zeros((bsz,), dtype=torch.float32)
            ret_b = torch.zeros((bsz,), dtype=torch.float32)

            for i, (idx, tr) in enumerate(zip(idxs, batch_tr)):
                seq_len = len(tr.input_ids)
                input_ids[i, :seq_len] = torch.tensor(tr.input_ids, dtype=torch.long)
                attention_mask[i, :seq_len] = 1
                last_indices[i] = seq_len - 1
                legal_mask[i] = torch.tensor(tr.legal_mask, dtype=torch.bool)
                actions[i] = tr.action_idx
                old_lp[i] = tr.old_log_prob
                ref_lp[i] = torch.tensor(tr.ref_log_probs, dtype=torch.float32)
                adv_b[i] = adv_tensor[idx]
                ret_b[i] = ret_tensor[idx]

            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            last_indices = last_indices.to(device, non_blocking=True)
            legal_mask = legal_mask.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            old_lp = old_lp.to(device, non_blocking=True)
            ref_lp = ref_lp.to(device, non_blocking=True)
            adv_b = adv_b.to(device, non_blocking=True)
            ret_b = ret_b.to(device, non_blocking=True)

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                legal_mask=legal_mask,
                last_token_indices=last_indices,
                return_value=(model.value_head is not None),
            )
            log_probs_all = _masked_log_softmax(out.masked_logits, legal_mask)
            probs_all = torch.where(
                legal_mask, torch.exp(log_probs_all), torch.zeros_like(log_probs_all)
            )
            new_lp = log_probs_all.gather(1, actions.unsqueeze(-1)).squeeze(-1)

            log_ratio = (new_lp - old_lp).clamp(-20.0, 20.0)
            ratio = torch.exp(log_ratio)
            surr1 = ratio * adv_b
            surr2 = torch.clamp(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * adv_b
            policy_loss = -torch.min(surr1, surr2).mean()

            # Exact analytical KL(pi_theta || pi_ref) and entropy over legal actions
            kl_per_step = (probs_all * (log_probs_all - ref_lp)).sum(dim=-1).clamp_min(0.0)
            kl_ref = kl_per_step.mean()
            entropy = -(probs_all * log_probs_all).sum(dim=-1).mean()

            if out.value_logit is not None:
                pred_v = 2.0 * torch.sigmoid(out.value_logit) - 1.0
                value_loss = F.smooth_l1_loss(pred_v, ret_b)
            else:
                value_loss = torch.tensor(0.0, device=device)

            loss = (
                policy_loss
                + config.value_coef * value_loss
                + config.kl_coef * kl_ref
                - config.entropy_coef * entropy
            )

            micro_in_accum += 1
            is_last_micro = (start + config.micro_batch_size) >= n
            denom = config.grad_accum if not is_last_micro else max(1, micro_in_accum)
            (loss / denom).backward()

            if micro_in_accum >= config.grad_accum or is_last_micro:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                opt_steps += 1
                micro_in_accum = 0

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                clip_frac = ((ratio - 1.0).abs() > config.clip_eps).float().mean().item()
                pol_loss_acc += float(policy_loss.item()) * bsz
                val_loss_acc += float(value_loss.item()) * bsz
                kl_ref_acc += float(kl_ref.item()) * bsz
                step_kl_acc += approx_kl * bsz
                ent_acc += float(entropy.item()) * bsz
                clip_acc += clip_frac * bsz
                total_seen += bsz
                epoch_step_kl_sum += approx_kl * bsz
                epoch_count += bsz

        if config.target_kl > 0.0 and (epoch_step_kl_sum / max(1, epoch_count)) > config.target_kl:
            early_stopped = True
            break

    denom_seen = max(1, total_seen)
    return PPOUpdateStats(
        policy_loss=pol_loss_acc / denom_seen,
        value_loss=val_loss_acc / denom_seen,
        kl_to_ref=kl_ref_acc / denom_seen,
        approx_step_kl=step_kl_acc / denom_seen,
        entropy=ent_acc / denom_seen,
        clip_fraction=clip_acc / denom_seen,
        early_stopped=early_stopped,
        num_transitions=n,
        opt_steps=opt_steps,
    )


@dataclass
class _RolloutBatchItem:
    adapter_name: str
    input_ids: list[int]
    legal_mask: list[bool]
    compute_ref: bool
    temperature: float
    future: asyncio.Future[tuple[int, float, list[float], float]]


class SharedModelRolloutBatcher:
    """Micro-batches concurrent rollout requests across multiple PEFT adapters on one GPU."""

    def __init__(
        self,
        model: AlphaBattlePolicyValueModel,
        tokenizer: Any,
        *,
        max_batch_size: int = 16,
        max_wait_ms: float = 8.0,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max(1, max_batch_size)
        self.max_wait_s = max(0.0, max_wait_ms / 1000.0)
        self.device = next(model.parameters()).device
        self.pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else (tokenizer.eos_token_id or 0)
        )
        self._queue: asyncio.Queue[_RolloutBatchItem] | None = None
        self._worker: asyncio.Task[None] | None = None
        self._worker_loop: asyncio.AbstractEventLoop | None = None
        self._closed = False

    def _ensure_worker(self) -> asyncio.Queue[_RolloutBatchItem]:
        loop = asyncio.get_running_loop()
        if (
            self._queue is None
            or self._worker_loop is not loop
            or self._worker is None
            or self._worker.done()
        ):
            self._closed = False
            self._worker_loop = loop
            self._queue = asyncio.Queue()
            self._worker = loop.create_task(self._run_worker())
        return self._queue

    def _eval_group_sync(
        self, group: list[_RolloutBatchItem]
    ) -> list[tuple[int, float, list[float], float]]:
        self.model.eval()
        bsz = len(group)
        max_len = max(len(g.input_ids) for g in group)
        rem = max_len % 8
        if rem != 0:
            max_len += 8 - rem

        input_ids = torch.full((bsz, max_len), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
        last_indices = torch.zeros((bsz,), dtype=torch.long)
        legal_mask = torch.zeros((bsz, NUM_ACTIONS), dtype=torch.bool)

        for i, item in enumerate(group):
            seq_len = len(item.input_ids)
            input_ids[i, :seq_len] = torch.tensor(item.input_ids, dtype=torch.long)
            attention_mask[i, :seq_len] = 1
            last_indices[i] = seq_len - 1
            legal_mask[i] = torch.tensor(item.legal_mask, dtype=torch.bool)

        input_ids = input_ids.to(self.device, non_blocking=True)
        attention_mask = attention_mask.to(self.device, non_blocking=True)
        last_indices = last_indices.to(self.device, non_blocking=True)
        legal_mask = legal_mask.to(self.device, non_blocking=True)

        adapter_name = group[0].adapter_name
        need_ref = any(g.compute_ref for g in group)

        with torch.inference_mode():
            _set_adapter_if_peft(self.model, adapter_name)
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                legal_mask=legal_mask,
                last_token_indices=last_indices,
                return_value=(self.model.value_head is not None),
            )
            log_probs = _masked_log_softmax(out.masked_logits, legal_mask)
            values = (
                (2.0 * torch.sigmoid(out.value_logit) - 1.0)
                if out.value_logit is not None
                else torch.zeros((bsz,), device=self.device)
            )

            if need_ref and _set_adapter_if_peft(self.model, "sft_ref"):
                ref_out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    legal_mask=legal_mask,
                    last_token_indices=last_indices,
                    return_value=False,
                )
                ref_log_probs = _masked_log_softmax(ref_out.masked_logits, legal_mask)
                _set_adapter_if_peft(self.model, "default")
            else:
                ref_log_probs = log_probs

            results: list[tuple[int, float, list[float], float]] = []
            for i, item in enumerate(group):
                lp_i = log_probs[i]
                if item.temperature <= 0.0:
                    chosen_idx = int(torch.argmax(out.masked_logits[i]).item())
                else:
                    scaled = out.masked_logits[i] / item.temperature
                    probs_i = torch.softmax(
                        scaled.masked_fill(~legal_mask[i], float("-inf")), dim=-1
                    )
                    chosen_idx = int(torch.multinomial(probs_i, num_samples=1).item())
                chosen_lp = float(lp_i[chosen_idx].item())
                ref_lp_list = [float(x) for x in ref_log_probs[i].cpu().tolist()]
                val_i = float(values[i].item())
                results.append((chosen_idx, chosen_lp, ref_lp_list, val_i))

        return results

    async def _run_worker(self) -> None:
        assert self._queue is not None
        while not self._closed:
            try:
                first = await self._queue.get()
            except asyncio.CancelledError:
                return
            batch = [first]
            deadline = time.perf_counter() + self.max_wait_s
            while len(batch) < self.max_batch_size:
                rem = deadline - time.perf_counter()
                if rem <= 0.0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self._queue.get(), timeout=rem))
                except asyncio.TimeoutError:
                    break
                except asyncio.CancelledError:
                    return

            # Group by adapter name so each PEFT adapter executes in one batched forward pass
            by_adapter: dict[str, list[_RolloutBatchItem]] = {}
            for item in batch:
                by_adapter.setdefault(item.adapter_name, []).append(item)

            for group in by_adapter.values():
                try:
                    res = await asyncio.to_thread(self._eval_group_sync, group)
                    for item, r in zip(group, res):
                        if not item.future.done():
                            item.future.set_result(r)
                except Exception as exc:
                    for item in group:
                        if not item.future.done():
                            item.future.set_exception(exc)

    async def infer(
        self,
        *,
        adapter_name: str,
        input_ids: list[int],
        legal_mask: list[bool],
        compute_ref: bool,
        temperature: float,
    ) -> tuple[int, float, list[float], float]:
        q = self._ensure_worker()
        fut: asyncio.Future[tuple[int, float, list[float], float]] = (
            asyncio.get_running_loop().create_future()
        )
        await q.put(
            _RolloutBatchItem(
                adapter_name=adapter_name,
                input_ids=input_ids,
                legal_mask=legal_mask,
                compute_ref=compute_ref,
                temperature=temperature,
                future=fut,
            )
        )
        return await fut

    async def aclose(self) -> None:
        self._closed = True
        task = self._worker
        worker_loop = self._worker_loop
        self._worker = None
        self._worker_loop = None
        self._queue = None
        if task is None or task.done():
            return
        try:
            cur_loop = asyncio.get_running_loop()
        except RuntimeError:
            cur_loop = None
        if worker_loop is not None and worker_loop is cur_loop:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        elif worker_loop is not None and worker_loop.is_running():
            worker_loop.call_soon_threadsafe(task.cancel)


class PPORolloutPlayer(Player):
    """Async poke-env Player collecting online PPO transitions with potential-based shaping."""

    def __init__(
        self,
        batcher: SharedModelRolloutBatcher,
        *,
        adapter_name: str = "default",
        collect_transitions: bool = True,
        max_length: int = 1024,
        config: PPOConfig | None = None,
        **player_kwargs: Any,
    ) -> None:
        super().__init__(**player_kwargs)
        self.batcher = batcher
        self.adapter_name = adapter_name
        self.collect_transitions = collect_transitions
        self.max_length = max_length
        self.config = config or PPOConfig()
        self._history = TurnHistoryTracker(window=3)
        self._ep_buffer: dict[str, list[RolloutTransition]] = {}
        self._prev_potential: dict[str, float] = {}
        self.completed_transitions: list[RolloutTransition] = []

    def pop_completed_transitions(self) -> list[RolloutTransition]:
        out = self.completed_transitions
        self.completed_transitions = []
        return out

    async def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if not isinstance(battle, Battle) or battle.wait:
            return self.choose_default_move()

        action_space = build_action_space(battle)
        if action_space.is_empty:
            return self.choose_default_move()

        hist = self._history.update_before_turn(battle)
        if action_space.is_forced_single:
            only = action_space.actions[0]
            self._history.record_action(battle, action_space, only.action_id)
            return only.order

        state_text = format_battle_state(battle, action_space, recent_history=hist)
        input_ids = tokenize_prompt_preserve_tail(
            self.batcher.tokenizer, state_text, max_length=self.max_length
        )
        legal_mask = [False] * NUM_ACTIONS
        for aid in action_space.ids:
            legal_mask[ACTION_ID_TO_INDEX[aid]] = True

        # Potential-based shaping delta assigned to the PREVIOUS decision in this battle
        if self.collect_transitions and self.config.shaping_scale > 0.0:
            pot = compute_battle_potential(
                battle,
                faint_weight=self.config.faint_shaping_weight,
                hp_weight=self.config.hp_shaping_weight,
            )
            tag = battle.battle_tag
            if tag in self._prev_potential and self._ep_buffer.get(tag):
                self._ep_buffer[tag][-1].reward += self.config.shaping_scale * (
                    pot - self._prev_potential[tag]
                )
            self._prev_potential[tag] = pot

        action_idx, log_prob, ref_lps, val = await self.batcher.infer(
            adapter_name=self.adapter_name,
            input_ids=input_ids,
            legal_mask=legal_mask,
            compute_ref=self.collect_transitions,
            temperature=self.config.rollout_temperature,
        )
        chosen_id = ALL_ACTION_IDS[action_idx]
        chosen = action_space.by_id[chosen_id]
        self._history.record_action(battle, action_space, chosen_id)

        if self.collect_transitions:
            self._ep_buffer.setdefault(battle.battle_tag, []).append(
                RolloutTransition(
                    input_ids=input_ids,
                    legal_mask=legal_mask,
                    action_idx=action_idx,
                    old_log_prob=log_prob,
                    ref_log_probs=ref_lps,
                    value=val,
                    reward=0.0,
                    done=False,
                )
            )

        return chosen.order

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        tag = battle.battle_tag
        self._history.pop_battle(tag)
        prev_pot = self._prev_potential.pop(tag, None)
        traj = self._ep_buffer.pop(tag, [])
        if not self.collect_transitions or not traj:
            return

        if isinstance(battle, Battle) and prev_pot is not None and self.config.shaping_scale > 0.0:
            end_pot = compute_battle_potential(
                battle,
                faint_weight=self.config.faint_shaping_weight,
                hp_weight=self.config.hp_shaping_weight,
            )
            traj[-1].reward += self.config.shaping_scale * (end_pot - prev_pot)

        if battle.won is True:
            traj[-1].reward += 1.0
        elif battle.won is False:
            traj[-1].reward -= 1.0
        traj[-1].done = True

        self.completed_transitions.extend(traj)
