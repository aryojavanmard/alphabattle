"""Canonical JSONL trajectory dataset, battle-ID-safe splitting, and PyTorch collators.

Shared across:
- Milestone 2: heuristic distillation (live poke-env recordings)
- Milestone 3: high-Elo human replay SFT
- Milestone 4: critic / value head warm-start
- Stage 2: offline/replay mixing during RL
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from alphabattle.state import ACTION_ID_TO_INDEX, ALL_ACTION_IDS, format_policy_prompt

NUM_ACTIONS = len(ALL_ACTION_IDS)


@dataclass(frozen=True)
class TrajectoryRecord:
    """Single non-forced decision turn with battle outcome metadata."""

    battle_id: str
    turn: int
    state: str
    legal_ids: tuple[str, ...]
    action_id: str
    won: bool | None = None
    rating: int | None = None
    source: str = "unknown"

    def to_json_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["legal_ids"] = list(self.legal_ids)
        return d

    @classmethod
    def from_json_dict(cls, raw: dict[str, Any]) -> TrajectoryRecord:
        legal = tuple(str(x) for x in raw["legal_ids"])
        action_id = str(raw["action_id"])
        won_val = raw.get("won")
        won = bool(won_val) if won_val is not None else None
        rating_val = raw.get("rating")
        rating = int(rating_val) if rating_val is not None else None
        return cls(
            battle_id=str(raw["battle_id"]),
            turn=int(raw["turn"]),
            state=str(raw["state"]),
            legal_ids=legal,
            action_id=action_id,
            won=won,
            rating=rating,
            source=str(raw.get("source", "unknown")),
        )


def load_trajectory_records(
    paths: Sequence[str | Path],
    *,
    min_rating: int | None = None,
    winner_only: bool = False,
    max_records: int | None = None,
    require_known_outcome: bool = False,
) -> list[TrajectoryRecord]:
    """Stream and filter TrajectoryRecords from one or more JSONL files."""
    records: list[TrajectoryRecord] = []
    for path_like in paths:
        path = Path(path_like)
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = TrajectoryRecord.from_json_dict(json.loads(line))
                if rec.action_id not in ACTION_ID_TO_INDEX:
                    continue
                if rec.action_id not in rec.legal_ids:
                    continue
                if len(rec.legal_ids) < 2:
                    continue
                if min_rating is not None and (rec.rating is None or rec.rating < min_rating):
                    continue
                if winner_only and rec.won is not True:
                    continue
                if require_known_outcome and rec.won is None:
                    continue
                records.append(rec)
                if max_records is not None and len(records) >= max_records:
                    return records
    return records


def split_by_battle_id(
    records: Sequence[TrajectoryRecord],
    *,
    val_fraction: float = 0.05,
    seed: int = 42,
) -> tuple[list[TrajectoryRecord], list[TrajectoryRecord]]:
    """Split records into (train, val) strictly grouped by battle_id to prevent leakage."""
    battle_ids = sorted({r.battle_id for r in records})
    if len(battle_ids) <= 1 or val_fraction <= 0.0:
        return (list(records), [])

    rng = random.Random(seed)
    rng.shuffle(battle_ids)
    n_val = max(1, int(round(len(battle_ids) * val_fraction)))
    if n_val >= len(battle_ids):
        n_val = len(battle_ids) - 1
    val_set = set(battle_ids[:n_val])

    train: list[TrajectoryRecord] = []
    val: list[TrajectoryRecord] = []
    for r in records:
        if r.battle_id in val_set:
            val.append(r)
        else:
            train.append(r)
    return (train, val)


def tokenize_prompt_preserve_tail(
    tokenizer: Any,
    state_text: str,
    *,
    max_length: int = 1024,
) -> list[int]:
    """Tokenize state prompt while left-truncating if over max_length.

    Preserving the tail guarantees LEGAL_ACTIONS and the trailing 'ACTION:\\n' suffix
    are never silently truncated away.
    """
    prompt = format_policy_prompt(state_text)
    ids: list[int] = tokenizer.encode(prompt, add_special_tokens=True)
    if len(ids) > max_length:
        bos = getattr(tokenizer, "bos_token_id", None)
        if bos is not None and ids[0] == bos:
            ids = [bos] + ids[-(max_length - 1) :]
        else:
            ids = ids[-max_length:]
    return ids


class BattleSFTDataset:
    """In-memory tokenized dataset for masked-action SFT and Value Head training."""

    def __init__(
        self,
        records: Sequence[TrajectoryRecord],
        tokenizer: Any,
        *,
        max_length: int = 1024,
    ) -> None:
        self.records = list(records)
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Pre-compute approximate lengths for length-bucketed batching without storing all tensors
        self.approx_lengths = [len(r.state) for r in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        input_ids = tokenize_prompt_preserve_tail(
            self.tokenizer, rec.state, max_length=self.max_length
        )
        legal_mask = [False] * NUM_ACTIONS
        for aid in rec.legal_ids:
            aid_idx = ACTION_ID_TO_INDEX.get(aid)
            if aid_idx is not None:
                legal_mask[aid_idx] = True
        target_action = ACTION_ID_TO_INDEX[rec.action_id]
        if rec.won is True:
            outcome = 1.0
        elif rec.won is False:
            outcome = 0.0
        else:
            outcome = -1.0
        return {
            "input_ids": input_ids,
            "legal_mask": legal_mask,
            "target_action": target_action,
            "outcome": outcome,
        }


class MaskedActionCollator:
    """Right-pads tokenized prompts to multiple-of-8 length for Ampere Tensor Core efficiency."""

    def __init__(self, pad_token_id: int, *, pad_to_multiple_of: int = 8) -> None:
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: Sequence[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of > 1:
            rem = max_len % self.pad_to_multiple_of
            if rem != 0:
                max_len += self.pad_to_multiple_of - rem

        batch_size = len(features)
        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        last_token_indices = torch.zeros((batch_size,), dtype=torch.long)
        legal_mask = torch.zeros((batch_size, NUM_ACTIONS), dtype=torch.bool)
        target_action = torch.zeros((batch_size,), dtype=torch.long)
        outcome = torch.full((batch_size,), -1.0, dtype=torch.float32)

        for i, feat in enumerate(features):
            seq = feat["input_ids"]
            seq_len = len(seq)
            input_ids[i, :seq_len] = torch.tensor(seq, dtype=torch.long)
            attention_mask[i, :seq_len] = 1
            last_token_indices[i] = seq_len - 1
            legal_mask[i] = torch.tensor(feat["legal_mask"], dtype=torch.bool)
            target_action[i] = feat["target_action"]
            outcome[i] = feat["outcome"]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "last_token_indices": last_token_indices,
            "legal_mask": legal_mask,
            "target_action": target_action,
            "outcome": outcome,
        }


class LengthBucketBatchSampler:
    """Groups similarly sized turns into mini-batches to minimize padding overhead on 12GB VRAM."""

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        *,
        mega_batch_mult: int = 32,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.mega_batch_size = max(batch_size, batch_size * mega_batch_mult)
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1

        indices = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(indices)

        batches: list[list[int]] = []
        for start in range(0, len(indices), self.mega_batch_size):
            chunk = indices[start : start + self.mega_batch_size]
            chunk.sort(key=lambda idx: self.lengths[idx])
            for b_start in range(0, len(chunk), self.batch_size):
                batches.append(chunk[b_start : b_start + self.batch_size])

        if self.shuffle:
            rng.shuffle(batches)
        yield from batches
