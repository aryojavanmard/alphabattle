"""Real PyTorch + Transformers + PEFT tests for Milestone 2 modeling, SFT training, and batching."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
import torch
from poke_env.battle.move import Move
from poke_env.player.battle_order import SingleBattleOrder
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from alphabattle.dataset import TrajectoryRecord
from alphabattle.modeling import (
    AlphaBattlePolicyValueModel,
    attach_lora,
    load_policy_value_checkpoint,
    resolve_action_token_ids,
)
from alphabattle.state import ALL_ACTION_IDS, ActionSpace, LegalAction
from alphabattle.torch_backend import PyTorchPolicyBackend


def _build_tiny_qwen3_local(model_dir: Path) -> tuple[Path, PreTrainedTokenizerFast]:
    """Create a tiny random-init Qwen3 model + real HF fast tokenizer on disk for fast CPU tests."""
    chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 \n:|[]()+-/.,%?=_")
    vocab = {"<|pad|>": 0, "<|eos|>": 1, "<|unk|>": 2}
    for ch in chars:
        if ch not in vocab:
            vocab[ch] = len(vocab)

    tok_backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<|unk|>"))
    tok_backend.pre_tokenizer = Split(pattern="", behavior="isolated")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok_backend,
        pad_token="<|pad|>",
        eos_token="<|eos|>",
        unk_token="<|unk|>",
    )
    tokenizer.padding_side = "right"

    cfg = Qwen3Config(
        vocab_size=len(vocab),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
        pad_token_id=0,
        eos_token_id=1,
        tie_word_embeddings=True,
    )
    lm = Qwen3ForCausalLM(cfg)
    model_dir.mkdir(parents=True, exist_ok=True)
    lm.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    return (model_dir, tokenizer)


def test_backbone_slice_matches_full_lm_head_and_lora_gradients_flow(tmp_path: Path) -> None:
    model_dir, tokenizer = _build_tiny_qwen3_local(tmp_path / "tiny_qwen3")
    action_token_ids = resolve_action_token_ids(tokenizer)
    assert len(action_token_ids) == len(ALL_ACTION_IDS)

    base_lm = Qwen3ForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    peft_lm = attach_lora(
        base_lm,
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules="all-linear",
        gradient_checkpointing=False,
    )
    model = AlphaBattlePolicyValueModel(peft_lm, action_token_ids, include_value_head=True)
    model.eval()

    input_ids = torch.tensor([[5, 6, 7, 8, 0, 0], [9, 10, 11, 12, 13, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0]], dtype=torch.long)
    last_idx = torch.tensor([3, 4], dtype=torch.long)
    legal_mask = torch.zeros((2, len(ALL_ACTION_IDS)), dtype=torch.bool)
    legal_mask[0, [0, 4, 8]] = True
    legal_mask[1, [1, 2]] = True

    with torch.no_grad():
        fast_out = model(input_ids, attention_mask, legal_mask, last_token_indices=last_idx)
        full_logits = peft_lm(input_ids=input_ids, attention_mask=attention_mask).logits
        ref_last = full_logits[torch.arange(2), last_idx][:, action_token_ids].float()

    # Sliced 13-token lm_head projection must match full vocab forward pass numerically
    assert torch.allclose(fast_out.raw_action_logits, ref_last, atol=1e-5)
    # Illegal actions must be strongly masked out
    assert fast_out.masked_logits[0, 1].item() <= -1e8
    assert fast_out.value_logit is not None and fast_out.value_logit.shape == (2,)

    # Verify gradients flow to LoRA weights and ValueHead
    model.train()
    out = model(input_ids, attention_mask, legal_mask, last_token_indices=last_idx)
    targets = torch.tensor([0, 1], dtype=torch.long)
    outcomes = torch.tensor([1.0, 0.0], dtype=torch.float32)
    loss = torch.nn.functional.cross_entropy(out.masked_logits, targets) + (
        torch.nn.functional.binary_cross_entropy_with_logits(out.value_logit, outcomes)  # type: ignore[arg-type]
    )
    loss.backward()

    lora_grads = [
        p.grad.abs().sum().item()
        for n, p in model.lm.named_parameters()
        if "lora_" in n and p.grad is not None
    ]
    assert len(lora_grads) > 0 and sum(lora_grads) > 0.0
    assert model.value_head is not None
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.value_head.parameters())

    # Save checkpoint, reload, and verify exact logit parity
    ckpt_dir = tmp_path / "ckpt"
    model.eval()
    with torch.no_grad():
        before_out = model(input_ids, attention_mask, legal_mask, last_token_indices=last_idx)
    model.save_checkpoint(ckpt_dir, base_model_name=str(model_dir), tokenizer=tokenizer, max_length=128)

    reloaded, _, meta = load_policy_value_checkpoint(ckpt_dir, device="cpu", dtype=torch.float32)
    assert meta["has_value_head"] is True
    with torch.no_grad():
        after_out = reloaded(input_ids, attention_mask, legal_mask, last_token_indices=last_idx)
    assert torch.allclose(before_out.masked_logits, after_out.masked_logits, atol=1e-5)
    assert after_out.value_logit is not None and before_out.value_logit is not None
    assert torch.allclose(before_out.value_logit, after_out.value_logit, atol=1e-5)


@pytest.mark.asyncio
async def test_train_sft_end_to_end_and_batched_torch_backend(tmp_path: Path) -> None:
    model_dir, _ = _build_tiny_qwen3_local(tmp_path / "tiny_qwen3_base")
    data_path = tmp_path / "toy.jsonl"

    records = []
    for b in range(10):
        for t in range(1, 5):
            # Simple learnable pattern: "FIRE" -> action "1", "WATER" -> action "5"
            if (b + t) % 2 == 0:
                state = f"FORMAT: gen9randombattle | TURN: {t} | HINT: FIRE"
                aid = "1"
                won = True
            else:
                state = f"FORMAT: gen9randombattle | TURN: {t} | HINT: WATER"
                aid = "5"
                won = False
            records.append(
                TrajectoryRecord(
                    battle_id=f"b{b}",
                    turn=t,
                    state=state,
                    legal_ids=("1", "2", "5"),
                    action_id=aid,
                    won=won,
                    source="toy",
                ).to_json_dict()
            )
    data_path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import train_sft

    ckpt_s1 = tmp_path / "ckpt_s1"
    args_s1 = train_sft.parse_args(
        [
            "--data",
            str(data_path),
            "--base-model",
            str(model_dir),
            "--output-dir",
            str(ckpt_s1),
            "--device",
            "cpu",
            "--lora-r",
            "4",
            "--lora-alpha",
            "8",
            "--batch-size",
            "4",
            "--grad-accum",
            "1",
            "--epochs",
            "2",
            "--lr",
            "5e-3",
            "--val-fraction",
            "0.2",
            "--no-8bit-adam",
            "--no-grad-checkpointing",
            "--log-every",
            "100",
        ]
    )
    summary_s1 = train_sft.train(args_s1)
    assert (ckpt_s1 / "alphabattle_meta.json").exists()
    assert summary_s1["best_val_loss"] is not None

    # Stage 1.5: resume and train frozen-policy Value Head
    ckpt_s15 = tmp_path / "ckpt_s15"
    args_s15 = train_sft.parse_args(
        [
            "--data",
            str(data_path),
            "--resume-checkpoint",
            str(ckpt_s1),
            "--output-dir",
            str(ckpt_s15),
            "--device",
            "cpu",
            "--freeze-policy",
            "--policy-loss-weight",
            "0.0",
            "--value-loss-weight",
            "1.0",
            "--require-known-outcome",
            "--batch-size",
            "4",
            "--grad-accum",
            "1",
            "--epochs",
            "2",
            "--lr",
            "1e-2",
            "--val-fraction",
            "0.2",
            "--no-8bit-adam",
            "--log-every",
            "100",
        ]
    )
    summary_s15 = train_sft.train(args_s15)
    assert "val_roc_auc" in summary_s15["history"][-1]
    assert "val_brier" in summary_s15["history"][-1]

    # Verify concurrent async micro-batching in PyTorchPolicyBackend
    backend = PyTorchPolicyBackend.from_checkpoint(
        ckpt_s15, device="cpu", max_batch_size=4, max_wait_ms=15.0, temperature=0.0
    )
    space = ActionSpace(
        actions=(
            LegalAction("1", "move", "MOVE eq", SingleBattleOrder(Move("earthquake", gen=9))),
            LegalAction("5", "switch", "SWITCH rotom", SingleBattleOrder("/choose switch rotom")),
        ),
        by_id={},
    )
    try:
        decisions = await asyncio.gather(
            *[backend.choose_action(f"TURN: {i} | HINT: FIRE", space) for i in range(6)]
        )
        assert len(decisions) == 6
        assert all(d.action_id in {"1", "5"} and not d.used_fallback for d in decisions)
    finally:
        await backend.aclose()
