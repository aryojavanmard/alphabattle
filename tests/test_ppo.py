"""Unit tests for Stage 2 PPO: GAE math, potential shaping, dual-adapter KL anchor, and updates."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from alphabattle.modeling import (
    ADAPTER_DIRNAME,
    AlphaBattlePolicyValueModel,
    attach_lora,
    load_policy_value_checkpoint,
    resolve_action_token_ids,
)
from alphabattle.ppo import (
    PPOConfig,
    RolloutTransition,
    SharedModelRolloutBatcher,
    compute_battle_potential,
    compute_gae,
    ensure_ref_adapter,
    ppo_update,
)
from alphabattle.state import ALL_ACTION_IDS
from test_modeling_and_training import _build_tiny_qwen3_local


def test_compute_gae_matches_closed_form() -> None:
    # 2-step episode: r=[0.0, 1.0], V=[0.2, 0.5], done=[False, True], gamma=0.9, lam=0.8
    t0 = RolloutTransition([1], [True] * 13, 0, -1.0, [-1.0] * 13, value=0.2, reward=0.0, done=False)
    t1 = RolloutTransition([1], [True] * 13, 0, -1.0, [-1.0] * 13, value=0.5, reward=1.0, done=True)

    adv, ret = compute_gae([t0, t1], gamma=0.9, gae_lambda=0.8)
    # delta1 = 1.0 + 0 - 0.5 = 0.5 -> A1 = 0.5, R1 = 1.0
    # delta0 = 0.0 + 0.9*0.5 - 0.2 = 0.25 -> A0 = 0.25 + 0.9*0.8*0.5 = 0.61, R0 = 0.81
    assert math.isclose(adv[1], 0.5, rel_tol=1e-6)
    assert math.isclose(ret[1], 1.0, rel_tol=1e-6)
    assert math.isclose(adv[0], 0.61, rel_tol=1e-6)
    assert math.isclose(ret[0], 0.81, rel_tol=1e-6)


def test_compute_battle_potential_is_zero_sum_and_bounded() -> None:
    m_full = SimpleNamespace(fainted=False, current_hp_fraction=1.0)
    m_half = SimpleNamespace(fainted=False, current_hp_fraction=0.5)
    m_fnt = SimpleNamespace(fainted=True, current_hp_fraction=0.0)

    # Start of game: 6 healthy vs 1 seen healthy + 5 unseen -> Phi == 0.0
    start_b = SimpleNamespace(
        team={f"m{i}": m_full for i in range(6)},
        opponent_team={"o0": m_full},
        max_team_size=6,
    )
    assert math.isclose(compute_battle_potential(start_b), 0.0, abs_tol=1e-9)  # type: ignore[arg-type]

    # Winning board: 6 healthy vs 5 fainted + 1 half -> positive and well below 1.0
    adv_b = SimpleNamespace(
        team={f"m{i}": m_full for i in range(6)},
        opponent_team={"o0": m_half, **{f"o{i}": m_fnt for i in range(1, 6)}},
        max_team_size=6,
    )
    pot = compute_battle_potential(adv_b, faint_weight=0.05, hp_weight=0.02)  # type: ignore[arg-type]
    assert 0.3 < pot < 0.5


@pytest.mark.asyncio
async def test_dual_adapter_kl_anchor_and_ppo_update_step(tmp_path: Path) -> None:
    model_dir, tokenizer = _build_tiny_qwen3_local(tmp_path / "tiny_qwen3_ppo")
    action_ids = resolve_action_token_ids(tokenizer)
    from transformers import Qwen3ForCausalLM

    base_lm = Qwen3ForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    peft_lm = attach_lora(
        base_lm, r=4, lora_alpha=8, lora_dropout=0.0, gradient_checkpointing=False
    )
    init_model = AlphaBattlePolicyValueModel(peft_lm, action_ids, include_value_head=True)
    ckpt = tmp_path / "init_sft"
    init_model.save_checkpoint(ckpt, base_model_name=str(model_dir), tokenizer=tokenizer, max_length=128)

    model, tok, _ = load_policy_value_checkpoint(
        ckpt, device="cpu", dtype=torch.float32, is_trainable=True
    )
    loaded_ref = ensure_ref_adapter(model, ckpt / ADAPTER_DIRNAME, adapter_name="sft_ref")
    assert loaded_ref is True
    assert "sft_ref" in model.lm.peft_config
    # Verify sft_ref weights are frozen while default adapter weights are trainable
    ref_trainable = [n for n, p in model.lm.named_parameters() if ".sft_ref." in n and p.requires_grad]
    def_trainable = [n for n, p in model.lm.named_parameters() if ".default." in n and p.requires_grad]
    assert len(ref_trainable) == 0
    assert len(def_trainable) > 0

    # Collect transitions via SharedModelRolloutBatcher
    batcher = SharedModelRolloutBatcher(model, tok, max_batch_size=4, max_wait_ms=5.0)
    legal_mask = [False] * len(ALL_ACTION_IDS)
    legal_mask[0] = True
    legal_mask[1] = True
    legal_mask[8] = True

    transitions: list[RolloutTransition] = []
    try:
        for i in range(6):
            inp = tok.encode(f"FORMAT: gen9randombattle | TURN: {i + 1}\nACTION:\n")
            a_idx, lp, ref_lps, val = await batcher.infer(
                adapter_name="default",
                input_ids=inp,
                legal_mask=legal_mask,
                compute_ref=True,
                temperature=1.0,
            )
            assert a_idx in {0, 1, 8}
            transitions.append(
                RolloutTransition(
                    input_ids=inp,
                    legal_mask=legal_mask,
                    action_idx=a_idx,
                    old_log_prob=lp,
                    ref_log_probs=ref_lps,
                    value=val,
                    reward=1.0 if i == 5 else 0.0,
                    done=(i == 5),
                )
            )
    finally:
        await batcher.aclose()

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-3)
    stats = ppo_update(
        model,
        optimizer,
        transitions,
        pad_token_id=tok.pad_token_id,
        config=PPOConfig(ppo_epochs=2, micro_batch_size=3, grad_accum=1, kl_coef=0.05),
    )
    assert stats.num_transitions == 6
    assert stats.opt_steps > 0
    assert stats.entropy > 0.0
    assert stats.kl_to_ref >= 0.0
