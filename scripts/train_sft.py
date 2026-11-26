"""Train Masked-Action LoRA Policy (Stage 1 SFT) and/or Value Head (Stage 1.5 Critic).

Sized for RTX 3080 12GB on Native Windows (bf16 LoRA + SDPA + last-token 13-way projection):
  ~5.0-6.5 GB peak VRAM for Qwen/Qwen3-1.7B-Base at batch_size=8, max_length=1024.

Examples (Native Windows PowerShell):
  # Milestone 2: Heuristic distillation smoke test (~1-1.5h on RTX 3080 12GB)
  python scripts/train_sft.py ^
      --data data/m2_heuristics_5k.jsonl ^
      --base-model Qwen/Qwen3-1.7B-Base ^
      --output-dir checkpoints/m2_heuristics_sft ^
      --epochs 1 --batch-size 8 --grad-accum 8 --lr 2e-4

  # Milestone 3: High-Elo human replay SFT (~2h on RTX 3080 12GB)
  python scripts/train_sft.py ^
      --data data/gen9randbats_1550plus.jsonl ^
      --base-model Qwen/Qwen3-1.7B-Base ^
      --output-dir checkpoints/m3_human_sft ^
      --min-rating 1550 --winner-only --max-records 250000 ^
      --epochs 1 --batch-size 8 --grad-accum 8 --lr 2e-4

  # Milestone 4: Stage 1.5 Value Head warm-start (30-45 min, policy frozen)
  python scripts/train_sft.py ^
      --data data/gen9randbats_1550plus.jsonl ^
      --resume-checkpoint checkpoints/m3_human_sft ^
      --output-dir checkpoints/m4_sft_with_critic ^
      --freeze-policy --policy-loss-weight 0.0 --value-loss-weight 1.0 ^
      --require-known-outcome --lr 1e-3 --epochs 1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from alphabattle.dataset import (
    BattleSFTDataset,
    LengthBucketBatchSampler,
    MaskedActionCollator,
    load_trajectory_records,
    split_by_battle_id,
)
from alphabattle.metrics import binary_roc_auc, brier_score
from alphabattle.modeling import (
    AlphaBattlePolicyValueModel,
    attach_lora,
    load_base_lm_and_tokenizer,
    load_policy_value_checkpoint,
    resolve_action_token_ids,
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
            print(f"[warn] bitsandbytes PagedAdamW8bit unavailable ({exc!r}); using torch AdamW.")
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def _cosine_lr_multiplier(step: int, total_steps: int, warmup_steps: int) -> float:
    if total_steps <= 0:
        return 1.0
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate_model(
    model: AlphaBattlePolicyValueModel,
    val_loader: DataLoader[Any],
    *,
    device: torch.device,
    policy_loss_weight: float,
    value_loss_weight: float,
) -> dict[str, float]:
    model.eval()
    total_samples = 0
    policy_loss_sum = 0.0
    top1_correct = 0
    top2_correct = 0

    value_loss_sum = 0.0
    value_count = 0
    val_probs: list[float] = []
    val_targets: list[float] = []

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        last_indices = batch["last_token_indices"].to(device, non_blocking=True)
        legal_mask = batch["legal_mask"].to(device, non_blocking=True)
        targets = batch["target_action"].to(device, non_blocking=True)
        outcomes = batch["outcome"].to(device, non_blocking=True)

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            legal_mask=legal_mask,
            last_token_indices=last_indices,
            return_value=(value_loss_weight > 0.0 and model.value_head is not None),
        )
        bsz = targets.size(0)
        total_samples += bsz

        p_loss = F.cross_entropy(out.masked_logits, targets, reduction="sum")
        policy_loss_sum += float(p_loss.item())

        preds_top1 = torch.argmax(out.masked_logits, dim=-1)
        top1_correct += int((preds_top1 == targets).sum().item())

        k = min(2, out.masked_logits.size(-1))
        preds_top2 = torch.topk(out.masked_logits, k=k, dim=-1).indices
        top2_correct += int((preds_top2 == targets.unsqueeze(-1)).any(dim=-1).sum().item())

        if out.value_logit is not None:
            valid = outcomes >= 0.0
            if valid.any():
                v_logits = out.value_logit[valid]
                v_targets = outcomes[valid]
                v_loss = F.binary_cross_entropy_with_logits(v_logits, v_targets, reduction="sum")
                value_loss_sum += float(v_loss.item())
                value_count += int(valid.sum().item())
                val_probs.extend(torch.sigmoid(v_logits).cpu().tolist())
                val_targets.extend(v_targets.cpu().tolist())

    mean_policy_loss = policy_loss_sum / max(1, total_samples)
    mean_value_loss = (value_loss_sum / value_count) if value_count > 0 else 0.0
    combined_loss = policy_loss_weight * mean_policy_loss + value_loss_weight * mean_value_loss

    metrics: dict[str, float] = {
        "val_loss": combined_loss,
        "val_policy_loss": mean_policy_loss,
        "val_top1_acc": top1_correct / max(1, total_samples),
        "val_top2_acc": top2_correct / max(1, total_samples),
    }
    if value_count > 0:
        metrics["val_value_loss"] = mean_value_loss
        metrics["val_brier"] = brier_score(val_probs, val_targets)
        metrics["val_roc_auc"] = binary_roc_auc(val_probs, val_targets)
    return metrics


def train(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()

    print(f"Loading trajectories from {args.data} ...")
    records = load_trajectory_records(
        args.data,
        min_rating=args.min_rating,
        winner_only=args.winner_only,
        max_records=args.max_records,
        require_known_outcome=args.require_known_outcome,
    )
    if not records:
        raise RuntimeError("No valid trajectory records loaded after filtering.")

    train_recs, val_recs = split_by_battle_id(
        records, val_fraction=args.val_fraction, seed=args.seed
    )
    print(
        f"Loaded {len(records):,} turns across "
        f"{len({r.battle_id for r in records}):,} battles -> "
        f"train={len(train_recs):,}, val={len(val_recs):,}."
    )

    if args.resume_checkpoint:
        model, tokenizer, prev_meta = load_policy_value_checkpoint(
            args.resume_checkpoint,
            base_model_override=args.base_model,
            device=args.device,
            qlora=args.qlora,
            is_trainable=True,
        )
        base_model_name = args.base_model or prev_meta["base_model"]
        if args.value_loss_weight > 0.0 and model.value_head is None:
            from alphabattle.modeling import ValueHead

            model.value_head = ValueHead(model.hidden_size).to(next(model.parameters()).device)
    else:
        if not args.base_model:
            raise ValueError("Provide --base-model when not using --resume-checkpoint.")
        base_model_name = args.base_model
        lm, tokenizer = load_base_lm_and_tokenizer(
            base_model_name, device=args.device, qlora=args.qlora
        )
        if not args.no_lora:
            lm = attach_lora(
                lm,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                qlora=args.qlora,
                gradient_checkpointing=not args.no_grad_checkpointing,
            )
        elif not args.no_grad_checkpointing and hasattr(lm, "gradient_checkpointing_enable"):
            lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        action_token_ids = resolve_action_token_ids(tokenizer)
        model = AlphaBattlePolicyValueModel(
            lm=lm,
            action_token_ids=action_token_ids,
            include_value_head=(not args.no_value_head),
        )
        device_lm = next(lm.parameters()).device
        model.action_token_ids = model.action_token_ids.to(device_lm)
        if model.value_head is not None:
            model.value_head = model.value_head.to(device_lm)

    device = next(model.parameters()).device

    if args.freeze_policy:
        for p in model.lm.parameters():
            p.requires_grad = False
        if model.value_head is None:
            raise ValueError("--freeze-policy requires a value head to train.")
        for p in model.value_head.parameters():
            p.requires_grad = True
    elif args.value_loss_weight <= 0.0 and model.value_head is not None:
        for p in model.value_head.parameters():
            p.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in trainable_params)
    print(
        f"Model ready on {device}: {n_trainable:,} trainable / {total_params:,} total params "
        f"({100.0 * n_trainable / max(1, total_params):.2f}%)."
    )

    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else (tokenizer.eos_token_id or 0)
    )
    collator = MaskedActionCollator(pad_token_id=pad_id, pad_to_multiple_of=8)

    train_ds = BattleSFTDataset(train_recs, tokenizer, max_length=args.max_length)
    train_sampler = LengthBucketBatchSampler(
        train_ds.approx_lengths,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        collate_fn=collator,
        num_workers=0,  # Native Windows: num_workers=0 avoids spawn pickling overhead
    )

    val_loader: DataLoader[Any] | None = None
    if val_recs:
        val_ds = BattleSFTDataset(val_recs, tokenizer, max_length=args.max_length)
        val_sampler = LengthBucketBatchSampler(
            val_ds.approx_lengths,
            batch_size=args.eval_batch_size or args.batch_size,
            shuffle=False,
            seed=args.seed,
        )
        val_loader = DataLoader(
            val_ds,
            batch_sampler=val_sampler,
            collate_fn=collator,
            num_workers=0,
        )

    optimizer = _build_optimizer(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_8bit=(not args.no_8bit_adam),
    )

    steps_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accum))
    total_opt_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(round(total_opt_steps * args.warmup_ratio)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: _cosine_lr_multiplier(s, total_opt_steps, warmup_steps),
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    global_opt_step = 0
    t_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_policy:
            model.lm.eval()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_turns = 0
        t_window = time.perf_counter()

        for micro_step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            last_indices = batch["last_token_indices"].to(device, non_blocking=True)
            legal_mask = batch["legal_mask"].to(device, non_blocking=True)
            targets = batch["target_action"].to(device, non_blocking=True)
            outcomes = batch["outcome"].to(device, non_blocking=True)

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                legal_mask=legal_mask,
                last_token_indices=last_indices,
                return_value=(args.value_loss_weight > 0.0 and model.value_head is not None),
            )

            loss = torch.tensor(0.0, device=device)
            if args.policy_loss_weight > 0.0:
                p_loss = F.cross_entropy(out.masked_logits, targets)
                loss = loss + args.policy_loss_weight * p_loss

            if args.value_loss_weight > 0.0 and out.value_logit is not None:
                valid = outcomes >= 0.0
                if valid.any():
                    v_loss = F.binary_cross_entropy_with_logits(
                        out.value_logit[valid], outcomes[valid]
                    )
                    loss = loss + args.value_loss_weight * v_loss

            scaled_loss = loss / args.grad_accum
            scaled_loss.backward()

            bsz = targets.size(0)
            running_loss += float(loss.item()) * bsz
            running_turns += bsz

            is_boundary = (micro_step % args.grad_accum == 0) or (micro_step == len(train_loader))
            if is_boundary:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_opt_step += 1

                if global_opt_step % args.log_every == 0 or global_opt_step == total_opt_steps:
                    dt = max(time.perf_counter() - t_window, 1e-6)
                    avg_loss = running_loss / max(1, running_turns)
                    tps = running_turns / dt
                    vram_str = ""
                    if torch.cuda.is_available():
                        peak_gib = torch.cuda.max_memory_allocated() / (1024**3)
                        vram_str = f" | peak_vram={peak_gib:.2f}GiB"
                        if peak_gib > WINDOWS_VRAM_WARN_GIB:
                            vram_str += " [WARN: >10.2GiB Windows spill risk]"
                    print(
                        f"epoch {epoch}/{args.epochs} | step {global_opt_step}/{total_opt_steps} | "
                        f"loss={avg_loss:.4f} | lr={scheduler.get_last_lr()[0]:.2e} | "
                        f"{tps:.1f} turns/s{vram_str}",
                        flush=True,
                    )
                    running_loss = 0.0
                    running_turns = 0
                    t_window = time.perf_counter()

                if (
                    val_loader is not None
                    and args.eval_every > 0
                    and global_opt_step % args.eval_every == 0
                ):
                    val_metrics = evaluate_model(
                        model,
                        val_loader,
                        device=device,
                        policy_loss_weight=args.policy_loss_weight,
                        value_loss_weight=args.value_loss_weight,
                    )
                    val_metrics["epoch"] = epoch
                    val_metrics["step"] = global_opt_step
                    history.append(val_metrics)
                    print(f"  [VAL step {global_opt_step}] {json.dumps(val_metrics)}", flush=True)
                    if val_metrics["val_loss"] < best_val_loss:
                        best_val_loss = val_metrics["val_loss"]
                        model.save_checkpoint(
                            out_dir,
                            base_model_name=base_model_name,
                            tokenizer=tokenizer,
                            max_length=args.max_length,
                            extra_meta={"best_val_metrics": val_metrics},
                        )
                    model.train()
                    if args.freeze_policy:
                        model.lm.eval()

        # End-of-epoch validation and checkpoint
        if val_loader is not None:
            val_metrics = evaluate_model(
                model,
                val_loader,
                device=device,
                policy_loss_weight=args.policy_loss_weight,
                value_loss_weight=args.value_loss_weight,
            )
            val_metrics["epoch"] = epoch
            val_metrics["step"] = global_opt_step
            history.append(val_metrics)
            print(f"[EPOCH {epoch} VAL] {json.dumps(val_metrics)}")
            if val_metrics["val_loss"] <= best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                model.save_checkpoint(
                    out_dir,
                    base_model_name=base_model_name,
                    tokenizer=tokenizer,
                    max_length=args.max_length,
                    extra_meta={"best_val_metrics": val_metrics},
                )
        else:
            model.save_checkpoint(
                out_dir,
                base_model_name=base_model_name,
                tokenizer=tokenizer,
                max_length=args.max_length,
            )

    elapsed = time.perf_counter() - t_start
    summary: dict[str, Any] = {
        "base_model": base_model_name,
        "output_dir": str(out_dir),
        "train_turns": len(train_recs),
        "val_turns": len(val_recs),
        "total_opt_steps": global_opt_step,
        "elapsed_s": round(elapsed, 2),
        "best_val_loss": best_val_loss if val_loader is not None else None,
        "history": history,
    }
    if torch.cuda.is_available():
        summary["peak_vram_gib"] = round(torch.cuda.max_memory_allocated() / (1024**3), 3)

    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Training complete in {elapsed:.1f}s. Checkpoint saved to {out_dir}")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train AlphaBattle masked-action LoRA policy and/or Value Head."
    )
    parser.add_argument("--data", nargs="+", required=True, help="One or more JSONL dataset files.")
    parser.add_argument("--base-model", default=None, help="HF model ID or local directory.")
    parser.add_argument("--resume-checkpoint", default=None, help="Existing AlphaBattle checkpoint.")
    parser.add_argument("--output-dir", required=True, help="Directory to save checkpoint.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--qlora", action="store_true", help="Load base model in 4-bit NF4.")
    parser.add_argument("--no-lora", action="store_true", help="Full fine-tuning (0.6B/tiny tests).")
    parser.add_argument("--no-value-head", action="store_true")
    parser.add_argument("--freeze-policy", action="store_true", help="Train Value Head only.")
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--policy-loss-weight", type=float, default=1.0)
    parser.add_argument("--value-loss-weight", type=float, default=0.0)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--min-rating", type=int, default=None)
    parser.add_argument("--winner-only", action="store_true")
    parser.add_argument("--require-known-outcome", action="store_true")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--no-8bit-adam", action="store_true")
    parser.add_argument("--no-grad-checkpointing", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
