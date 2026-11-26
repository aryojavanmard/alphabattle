"""Stage 2 Online Masked-Action PPO trainer on local Pokémon Showdown server(s).

Designed for RTX 3080 12GB + i7-10700K on Native Windows (2-3h weekend runs):
- Single 1.7B base model in VRAM with multiple ~40 MB PEFT LoRA adapters:
    * "default"  -> trainable policy
    * "sft_ref"  -> frozen Stage 1 SFT anchor for analytical KL penalty & anchor sparring
    * "past_opp" -> periodically rotated historical snapshot for diverse self-play
- Mixed opponent pool prevents self-play collapse (Metamon's key finding).
- Multi-port Showdown support (--showdown-ports 8000 8001) saturates 8 CPU cores cleanly.

Example (Native Windows PowerShell):
  # Terminal 1: start 2 local Showdown servers
  .\\scripts\\start_showdown.ps1 -Port 8000 -Instances 2

  # Terminal 2: run 100 PPO updates (~1.5-2h on RTX 3080 12GB)
  python scripts/train_ppo.py ^
      --init-checkpoint checkpoints/m4_sft_with_critic ^
      --output-dir checkpoints/m5_ppo_v1 ^
      --showdown-ports 8000 8001 ^
      --updates 100 --battles-per-update 64 --concurrency 16 ^
      --lr 2e-5 --kl-coef 0.03 --shaping-scale 1.0 --anneal-shaping
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
from poke_env.player import MaxBasePowerPlayer, Player, SimpleHeuristicsPlayer
from poke_env.ps_client import AccountConfiguration, ServerConfiguration

from alphabattle.modeling import (
    ADAPTER_DIRNAME,
    ValueHead,
    load_policy_value_checkpoint,
)
from alphabattle.ppo import (
    PPOConfig,
    PPORolloutPlayer,
    RolloutTransition,
    SharedModelRolloutBatcher,
    ensure_ref_adapter,
    ppo_update,
)

WINDOWS_VRAM_WARN_GIB = 10.2


def _build_optimizer(
    params: list[torch.nn.Parameter], *, lr: float, weight_decay: float, use_8bit: bool
) -> torch.optim.Optimizer:
    if use_8bit and torch.cuda.is_available():
        try:
            import bitsandbytes as bnb  # type: ignore[import-not-found]

            return bnb.optim.PagedAdamW8bit(params, lr=lr, weight_decay=weight_decay)
        except Exception as exc:
            print(f"[warn] bitsandbytes unavailable ({exc!r}); using torch AdamW.")
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def _split_battles_across_pool(
    total_battles: int,
    *,
    has_sft_ref: bool,
    has_past_opp: bool,
    self_frac: float,
    past_frac: float,
    heur_frac: float,
    sft_frac: float,
) -> list[tuple[str, int]]:
    """Allocate `total_battles` across the mixed opponent pool cleanly."""
    weights: list[tuple[str, float]] = [
        ("self", max(0.0, self_frac)),
        ("past_opp", max(0.0, past_frac) if has_past_opp else 0.0),
        ("heuristics", max(0.0, heur_frac)),
        ("sft_ref", max(0.0, sft_frac) if has_sft_ref else 0.0),
    ]
    total_w = sum(w for _, w in weights)
    if total_w <= 0.0:
        return [("heuristics", total_battles)]

    alloc: list[tuple[str, int]] = []
    assigned = 0
    for name, w in weights:
        if w <= 0.0:
            continue
        n = int(round(total_battles * (w / total_w)))
        if n > 0:
            alloc.append((name, n))
            assigned += n

    if not alloc:
        return [("heuristics", total_battles)]
    # Fix rounding delta on the largest bucket
    diff = total_battles - assigned
    if diff != 0:
        max_idx = max(range(len(alloc)), key=lambda i: alloc[i][1])
        name, cnt = alloc[max_idx]
        alloc[max_idx] = (name, max(1, cnt + diff))
    return alloc


async def _collect_rollout_round(
    *,
    batcher: SharedModelRolloutBatcher,
    alloc: list[tuple[str, int]],
    ports: list[int],
    host: str,
    battle_format: str,
    concurrency_per_port: int,
    max_length: int,
    config: PPOConfig,
) -> tuple[list[RolloutTransition], dict[str, tuple[int, int]]]:
    """Play one parallel batch of battles across Showdown ports & opponent types."""
    all_transitions: list[RolloutTransition] = []
    win_loss_by_opp: dict[str, tuple[int, int]] = {}

    async def _run_bucket(opp_kind: str, n_games: int, port_idx: int) -> None:
        port = ports[port_idx % len(ports)]
        server_cfg = ServerConfiguration(
            f"ws://{host}:{port}/showdown/websocket",
            "https://play.pokemonshowdown.com/action.php?",
        )
        common = dict(
            battle_format=battle_format,
            server_configuration=server_cfg,
            max_concurrent_battles=min(concurrency_per_port, max(1, n_games)),
            log_level=logging.WARNING,
        )

        learner = PPORolloutPlayer(
            batcher=batcher,
            adapter_name="default",
            collect_transitions=True,
            max_length=max_length,
            config=config,
            account_configuration=AccountConfiguration.generate("PPOLrn", rand=True),
            **common,
        )

        opp: Player
        if opp_kind == "heuristics":
            opp = SimpleHeuristicsPlayer(
                account_configuration=AccountConfiguration.generate("PPOHeur", rand=True),
                **common,
            )
        elif opp_kind == "max_power":
            opp = MaxBasePowerPlayer(
                account_configuration=AccountConfiguration.generate("PPOMaxP", rand=True),
                **common,
            )
        elif opp_kind in ("self", "sft_ref", "past_opp"):
            opp_adapter = "default" if opp_kind == "self" else opp_kind
            opp = PPORolloutPlayer(
                batcher=batcher,
                adapter_name=opp_adapter,
                collect_transitions=(opp_kind == "self"),  # Double data efficiency in self-play!
                max_length=max_length,
                config=config,
                account_configuration=AccountConfiguration.generate("PPOOpp", rand=True),
                **common,
            )
        else:
            raise ValueError(f"Unknown opponent kind: {opp_kind}")

        await learner.battle_against(opp, n_battles=n_games)
        all_transitions.extend(learner.pop_completed_transitions())
        if isinstance(opp, PPORolloutPlayer) and opp.collect_transitions:
            all_transitions.extend(opp.pop_completed_transitions())

        prev_w, prev_t = win_loss_by_opp.get(opp_kind, (0, 0))
        win_loss_by_opp[opp_kind] = (
            prev_w + learner.n_won_battles,
            prev_t + learner.n_finished_battles,
        )

    tasks = [
        _run_bucket(opp_kind, n_games, idx) for idx, (opp_kind, n_games) in enumerate(alloc)
    ]
    await asyncio.gather(*tasks)
    return (all_transitions, win_loss_by_opp)


async def run_ppo_training(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()

    init_ckpt = Path(args.init_checkpoint)
    model, tokenizer, meta = load_policy_value_checkpoint(
        init_ckpt,
        base_model_override=args.base_model,
        device=args.device,
        qlora=args.qlora,
        is_trainable=True,
    )
    base_model_name = args.base_model or meta["base_model"]
    max_length = int(args.max_length or meta.get("max_length", 1024))
    device = next(model.parameters()).device

    if model.value_head is None:
        model.value_head = ValueHead(model.hidden_size).to(device)

    # Enable gradient checkpointing on base LM for 12GB VRAM headroom during PPO updates
    if not args.no_grad_checkpointing:
        base_lm = model.lm.get_base_model() if hasattr(model.lm, "get_base_model") else model.lm
        if hasattr(base_lm, "gradient_checkpointing_enable"):
            base_lm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if hasattr(base_lm, "enable_input_require_grads"):
                base_lm.enable_input_require_grads()

    # Load frozen SFT anchor adapter onto the exact same base weights (+~40 MB VRAM)
    has_sft_ref = ensure_ref_adapter(model, init_ckpt / ADAPTER_DIRNAME, adapter_name="sft_ref")
    print(f"Loaded PPO policy on {device} (has_sft_ref_adapter={has_sft_ref}).")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = _build_optimizer(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_8bit=(not args.no_8bit_adam),
    )

    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else (tokenizer.eos_token_id or 0)
    )
    batcher = SharedModelRolloutBatcher(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=args.max_batch_size,
        max_wait_ms=args.max_wait_ms,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snap_dir = out_dir / "snapshots"
    snapshots: list[Path] = []
    history: list[dict[str, Any]] = []
    t_start = time.perf_counter()

    try:
        for update_idx in range(1, args.updates + 1):
            # Optionally anneal potential-based shaping linearly down to 0
            shaping_scale = args.shaping_scale
            if args.anneal_shaping and args.updates > 1:
                shaping_scale = args.shaping_scale * max(
                    0.0, 1.0 - (update_idx - 1) / float(args.updates - 1)
                )

            cfg = PPOConfig(
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                clip_eps=args.clip_eps,
                value_coef=args.value_coef,
                entropy_coef=args.entropy_coef,
                kl_coef=args.kl_coef,
                target_kl=args.target_kl,
                ppo_epochs=args.ppo_epochs,
                micro_batch_size=args.micro_batch_size,
                grad_accum=args.grad_accum,
                max_grad_norm=args.max_grad_norm,
                faint_shaping_weight=args.faint_shaping,
                hp_shaping_weight=args.hp_shaping,
                shaping_scale=shaping_scale,
                rollout_temperature=args.rollout_temperature,
            )

            # Rotate a random historical snapshot into the "past_opp" adapter if available
            has_past_opp = False
            if snapshots and hasattr(model.lm, "load_adapter"):
                chosen_snap = random.choice(snapshots)
                if "past_opp" in model.lm.peft_config:
                    model.lm.delete_adapter("past_opp")
                has_past_opp = ensure_ref_adapter(
                    model, chosen_snap / ADAPTER_DIRNAME, adapter_name="past_opp"
                )

            alloc = _split_battles_across_pool(
                args.battles_per_update,
                has_sft_ref=has_sft_ref,
                has_past_opp=has_past_opp,
                self_frac=args.self_frac,
                past_frac=args.past_frac,
                heur_frac=args.heur_frac,
                sft_frac=args.sft_frac,
            )

            t_roll = time.perf_counter()
            transitions, wl_map = await _collect_rollout_round(
                batcher=batcher,
                alloc=alloc,
                ports=args.showdown_ports,
                host=args.showdown_host,
                battle_format=args.format,
                concurrency_per_port=args.concurrency,
                max_length=max_length,
                config=cfg,
            )
            dt_roll = time.perf_counter() - t_roll

            # Pause batcher worker cleanly while running backward pass on the same GPU
            await batcher.aclose()

            t_upd = time.perf_counter()
            stats = ppo_update(
                model=model,
                optimizer=optimizer,
                transitions=transitions,
                pad_token_id=pad_id,
                config=cfg,
                seed=args.seed + update_idx,
            )
            dt_upd = time.perf_counter() - t_upd

            wr_summary = {
                k: round(w / max(1, t), 3) for k, (w, t) in wl_map.items()
            }
            rec: dict[str, Any] = {
                "update": update_idx,
                "transitions": stats.num_transitions,
                "rollout_s": round(dt_roll, 2),
                "update_s": round(dt_upd, 2),
                "shaping_scale": round(shaping_scale, 3),
                "win_rates": wr_summary,
                "policy_loss": round(stats.policy_loss, 4),
                "value_loss": round(stats.value_loss, 4),
                "kl_to_ref": round(stats.kl_to_ref, 4),
                "approx_step_kl": round(stats.approx_step_kl, 4),
                "entropy": round(stats.entropy, 4),
                "clip_frac": round(stats.clip_fraction, 3),
                "early_stopped": stats.early_stopped,
            }
            if torch.cuda.is_available():
                peak_gib = torch.cuda.max_memory_allocated() / (1024**3)
                rec["peak_vram_gib"] = round(peak_gib, 3)

            history.append(rec)
            vram_msg = ""
            if "peak_vram_gib" in rec:
                vram_msg = f" | vram={rec['peak_vram_gib']:.2f}GiB"
                if rec["peak_vram_gib"] > WINDOWS_VRAM_WARN_GIB:
                    vram_msg += " [WARN: spill risk]"

            print(
                f"upd {update_idx}/{args.updates} | N={stats.num_transitions} "
                f"(roll={dt_roll:.1f}s upd={dt_upd:.1f}s) | wr={wr_summary} | "
                f"ploss={stats.policy_loss:.3f} vloss={stats.value_loss:.3f} "
                f"kl_ref={stats.kl_to_ref:.4f} ent={stats.entropy:.3f} "
                f"clip={stats.clip_fraction:.2f}{vram_msg}",
                flush=True,
            )

            if args.snapshot_every > 0 and update_idx % args.snapshot_every == 0:
                sp = snap_dir / f"upd_{update_idx:04d}"
                model.save_checkpoint(
                    sp,
                    base_model_name=base_model_name,
                    tokenizer=tokenizer,
                    max_length=max_length,
                    extra_meta={"ppo_update": update_idx, "metrics": rec},
                )
                snapshots.append(sp)
                if len(snapshots) > args.max_snapshots:
                    snapshots.pop(0)

            if args.save_every > 0 and update_idx % args.save_every == 0:
                model.save_checkpoint(
                    out_dir,
                    base_model_name=base_model_name,
                    tokenizer=tokenizer,
                    max_length=max_length,
                    extra_meta={"ppo_update": update_idx, "metrics": rec},
                )
    finally:
        await batcher.aclose()

    model.save_checkpoint(
        out_dir,
        base_model_name=base_model_name,
        tokenizer=tokenizer,
        max_length=max_length,
        extra_meta={"ppo_updates_completed": args.updates},
    )
    summary = {
        "init_checkpoint": str(init_ckpt),
        "output_dir": str(out_dir),
        "updates": args.updates,
        "elapsed_s": round(time.perf_counter() - t_start, 2),
        "history": history,
    }
    (out_dir / "ppo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"PPO complete in {summary['elapsed_s']:.1f}s. Saved to {out_dir}")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2 Online Masked-Action PPO trainer.")
    parser.add_argument("--init-checkpoint", required=True, help="Stage 1 / 1.5 checkpoint dir.")
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument("--format", default="gen9randombattle")
    parser.add_argument("--showdown-host", default="127.0.0.1")
    parser.add_argument("--showdown-ports", nargs="+", type=int, default=[8000])
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--battles-per-update", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--max-wait-ms", type=float, default=8.0)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--kl-coef", type=float, default=0.03)
    parser.add_argument("--target-kl", type=float, default=0.15)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--faint-shaping", type=float, default=0.05)
    parser.add_argument("--hp-shaping", type=float, default=0.02)
    parser.add_argument("--shaping-scale", type=float, default=1.0)
    parser.add_argument("--anneal-shaping", action="store_true")
    parser.add_argument("--rollout-temperature", type=float, default=1.0)
    parser.add_argument("--self-frac", type=float, default=0.40)
    parser.add_argument("--past-frac", type=float, default=0.30)
    parser.add_argument("--heur-frac", type=float, default=0.20)
    parser.add_argument("--sft-frac", type=float, default=0.10)
    parser.add_argument("--snapshot-every", type=int, default=20)
    parser.add_argument("--max-snapshots", type=int, default=6)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--no-8bit-adam", action="store_true")
    parser.add_argument("--no-grad-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(run_ppo_training(parse_args()))


if __name__ == "__main__":
    main()
