"""Zero-sim2sim-gap trajectory recorders wrapping poke-env Players.

Uses the exact same state formatter and TurnHistoryTracker as AlphaBattlePlayer so
recorded training data and live evaluation prompts are byte-identical.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.battle import Battle
from poke_env.player.baselines import MaxBasePowerPlayer, SimpleHeuristicsPlayer
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player

from alphabattle.dataset import TrajectoryRecord
from alphabattle.state import (
    TurnHistoryTracker,
    build_action_space,
    format_battle_state,
    match_order_to_action_id,
)


@dataclass
class RecordingStats:
    """Counters tracking recorded battles, turns, and any unmapped orders."""

    battles_recorded: int = 0
    turns_written: int = 0
    forced_turns_skipped: int = 0
    unmapped_orders: int = 0


class TrajectoryRecorderMixin:
    """Mixin that records non-forced decisions and flushes with battle outcome on completion."""

    def _init_recorder(
        self,
        *,
        output_path: str | Path,
        source_label: str,
        winner_only: bool = False,
        history_window: int = 3,
    ) -> None:
        self._output_path = Path(output_path)
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._source_label = source_label
        self._winner_only = winner_only
        self._history = TurnHistoryTracker(window=history_window)
        self._pending_by_battle: dict[str, list[dict[str, Any]]] = {}
        self.recording_stats = RecordingStats()

    async def _choose_and_record(
        self,
        battle: AbstractBattle,
        delegate_choose: Any,
    ) -> BattleOrder:
        if not isinstance(battle, Battle) or battle.wait:
            order = delegate_choose(battle)
            return await order if inspect.isawaitable(order) else order

        action_space = build_action_space(battle)
        if action_space.is_empty:
            order = delegate_choose(battle)
            return await order if inspect.isawaitable(order) else order

        hist = self._history.update_before_turn(battle)
        state_text = format_battle_state(battle, action_space, recent_history=hist)

        order = delegate_choose(battle)
        if inspect.isawaitable(order):
            order = await order

        action_id = match_order_to_action_id(order, action_space)
        if action_id is None:
            self.recording_stats.unmapped_orders += 1
            return order

        self._history.record_action(battle, action_space, action_id)

        if action_space.is_forced_single:
            self.recording_stats.forced_turns_skipped += 1
            return order

        self._pending_by_battle.setdefault(battle.battle_tag, []).append(
            {
                "battle_id": f"{battle.battle_tag}:{battle.player_role or 'p'}",
                "turn": battle.turn,
                "state": state_text,
                "legal_ids": tuple(action_space.ids),
                "action_id": action_id,
            }
        )
        return order

    def _flush_battle_records(self, battle: AbstractBattle) -> None:
        self._history.pop_battle(battle.battle_tag)
        turns = self._pending_by_battle.pop(battle.battle_tag, [])
        if not turns:
            return
        won = battle.won
        if self._winner_only and won is not True:
            return

        with self._output_path.open("a", encoding="utf-8") as fh:
            for item in turns:
                rec = TrajectoryRecord(
                    battle_id=item["battle_id"],
                    turn=item["turn"],
                    state=item["state"],
                    legal_ids=item["legal_ids"],
                    action_id=item["action_id"],
                    won=won,
                    rating=None,
                    source=self._source_label,
                )
                fh.write(json.dumps(rec.to_json_dict(), ensure_ascii=False) + "\n")

        self.recording_stats.battles_recorded += 1
        self.recording_stats.turns_written += len(turns)


class RecordingSimpleHeuristicsPlayer(TrajectoryRecorderMixin, SimpleHeuristicsPlayer):
    """SimpleHeuristicsPlayer that logs every non-forced decision + final outcome to JSONL."""

    def __init__(
        self,
        *,
        output_path: str | Path,
        source_label: str = "heuristics",
        winner_only: bool = False,
        history_window: int = 3,
        **player_kwargs: Any,
    ) -> None:
        super().__init__(**player_kwargs)
        self._init_recorder(
            output_path=output_path,
            source_label=source_label,
            winner_only=winner_only,
            history_window=history_window,
        )

    async def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        return await self._choose_and_record(
            battle, lambda b: SimpleHeuristicsPlayer.choose_move(self, b)
        )

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        self._flush_battle_records(battle)


class RecordingMaxBasePowerPlayer(TrajectoryRecorderMixin, MaxBasePowerPlayer):
    """MaxBasePowerPlayer that logs every non-forced decision + final outcome to JSONL."""

    def __init__(
        self,
        *,
        output_path: str | Path,
        source_label: str = "max_power",
        winner_only: bool = False,
        history_window: int = 3,
        **player_kwargs: Any,
    ) -> None:
        super().__init__(**player_kwargs)
        self._init_recorder(
            output_path=output_path,
            source_label=source_label,
            winner_only=winner_only,
            history_window=history_window,
        )

    async def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        return await self._choose_and_record(
            battle, lambda b: MaxBasePowerPlayer.choose_move(self, b)
        )

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        self._flush_battle_records(battle)


class RecordingDelegatePlayer(TrajectoryRecorderMixin, Player):
    """Wraps any Player instance's choose_move to record live zero-gap trajectories."""

    def __init__(
        self,
        delegate: Player,
        *,
        output_path: str | Path,
        source_label: str = "delegate",
        winner_only: bool = False,
        history_window: int = 3,
        **player_kwargs: Any,
    ) -> None:
        super().__init__(**player_kwargs)
        self._delegate = delegate
        self._init_recorder(
            output_path=output_path,
            source_label=source_label,
            winner_only=winner_only,
            history_window=history_window,
        )

    async def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        return await self._choose_and_record(battle, self._delegate.choose_move)

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        self._flush_battle_records(battle)
