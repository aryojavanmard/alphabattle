"""Unit tests for trajectory dataset loading, battle-ID splitting, tokenization, and value metrics."""

from __future__ import annotations

import json
import math
from pathlib import Path

from alphabattle.dataset import (
    BattleSFTDataset,
    LengthBucketBatchSampler,
    MaskedActionCollator,
    TrajectoryRecord,
    load_trajectory_records,
    split_by_battle_id,
    tokenize_prompt_preserve_tail,
)
from alphabattle.metrics import binary_roc_auc, brier_score


class _CharTokenizer:
    """Minimal single-character tokenizer for deterministic unit tests."""

    pad_token_id = 0
    bos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [ord(ch) + 10 for ch in text]
        return ([self.bos_token_id] + ids) if add_special_tokens else ids


def test_load_and_split_by_battle_id_has_zero_leakage(tmp_path: Path) -> None:
    jsonl = tmp_path / "traj.jsonl"
    rows = []
    for b in range(20):
        won = b % 2 == 0
        rating = 1400 + b * 20
        for t in range(1, 6):
            rows.append(
                TrajectoryRecord(
                    battle_id=f"battle-{b}",
                    turn=t,
                    state=f"FORMAT: gen9randombattle | TURN: {t}",
                    legal_ids=("1", "2", "5"),
                    action_id="1",
                    won=won,
                    rating=rating,
                    source="test",
                ).to_json_dict()
            )
    # Add invalid rows that must be filtered out: forced single & illegal action
    rows.append(
        {
            "battle_id": "battle-bad",
            "turn": 1,
            "state": "x",
            "legal_ids": ["1"],
            "action_id": "1",
            "won": True,
            "rating": 1800,
        }
    )
    rows.append(
        {
            "battle_id": "battle-bad2",
            "turn": 1,
            "state": "x",
            "legal_ids": ["1", "2"],
            "action_id": "5",
            "won": True,
            "rating": 1800,
        }
    )
    jsonl.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    all_recs = load_trajectory_records([jsonl])
    assert len(all_recs) == 100

    filtered = load_trajectory_records([jsonl], min_rating=1600, winner_only=True)
    assert all((r.rating or 0) >= 1600 and r.won is True for r in filtered)
    assert len(filtered) > 0

    train, val = split_by_battle_id(all_recs, val_fraction=0.2, seed=7)
    train_ids = {r.battle_id for r in train}
    val_ids = {r.battle_id for r in val}
    assert train_ids.isdisjoint(val_ids)
    assert len(val_ids) == 4
    assert len(train) + len(val) == len(all_recs)

    # Both perspectives (:p1 and :p2) of the same battle must never split across train/val
    paired = []
    for b in range(10):
        for role, won in (("p1", True), ("p2", False)):
            paired.append(
                TrajectoryRecord(f"gen9-{b}:{role}", 1, "s", ("1", "2"), "1", won=won)
            )
    p_train, p_val = split_by_battle_id(paired, val_fraction=0.2, seed=13)
    train_base = {r.battle_id.rsplit(":", 1)[0] for r in p_train}
    val_base = {r.battle_id.rsplit(":", 1)[0] for r in p_val}
    assert train_base.isdisjoint(val_base)
    assert len(val_base) == 2
    assert len(p_val) == 4


def test_tail_preserving_truncation_and_collator() -> None:
    tok = _CharTokenizer()
    long_state = ("X" * 200) + "\nLEGAL_ACTIONS:\n[1] MOVE eq"
    ids = tokenize_prompt_preserve_tail(tok, long_state, max_length=40)
    assert len(ids) == 40
    assert ids[0] == tok.bos_token_id
    # Tail ends with '\nACTION:\n'
    tail_chars = "".join(chr(i - 10) for i in ids[1:])
    assert tail_chars.endswith("\nACTION:\n")

    recs = [
        TrajectoryRecord("b1", 1, "short", ("1", "A", "5"), "A", won=True),
        TrajectoryRecord("b1", 2, "a slightly longer state text", ("2", "6"), "6", won=False),
    ]
    ds = BattleSFTDataset(recs, tok, max_length=128)
    collator = MaskedActionCollator(pad_token_id=0, pad_to_multiple_of=8)
    batch = collator([ds[0], ds[1]])

    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape[1] % 8 == 0
    assert batch["legal_mask"].shape == (2, 13)
    assert batch["legal_mask"][0].sum().item() == 3
    assert batch["legal_mask"][1].sum().item() == 2
    assert batch["outcome"].tolist() == [1.0, 0.0]

    sampler = LengthBucketBatchSampler([10, 50, 12, 48, 15], batch_size=2, shuffle=False)
    batches = list(sampler)
    assert sum(len(b) for b in batches) == 5


def test_brier_and_roc_auc() -> None:
    assert math.isclose(brier_score([1.0, 0.0], [1.0, 0.0]), 0.0)
    assert math.isclose(brier_score([0.5, 0.5], [1.0, 0.0]), 0.25)
    assert math.isclose(binary_roc_auc([0.9, 0.8, 0.2, 0.1], [1.0, 1.0, 0.0, 0.0]), 1.0)
    assert math.isclose(binary_roc_auc([0.5, 0.5, 0.5, 0.5], [1.0, 0.0, 1.0, 0.0]), 0.5)
