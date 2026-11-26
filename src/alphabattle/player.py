"""Async poke-env Player backed by any DecisionBackend with strict legality and telemetry."""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.battle import Battle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player

from alphabattle.backends import DecisionBackend
from alphabattle.metrics import MatchupStats, elo_diff_from_winrate, percentile, wilson_interval
from alphabattle.state import (
    TurnHistoryTracker,
    build_action_space,
    format_battle_state,
)

logger = logging.getLogger(__name__)


@dataclass
class TelemetrySummary:
    """Per-evaluation behavioral diagnostics collected by AlphaBattlePlayer."""

    llm_decisions: int = 0
    forced_decisions: int = 0
    fallbacks: int = 0
    latencies_ms: list[float] | None = None
    action_kinds: Counter[str] | None = None


class AlphaBattlePlayer(Player):
    """poke-env 0.17 async Player that formats state, queries a DecisionBackend, and logs telemetry."""

    def __init__(
        self,
        backend: DecisionBackend,
        *,
        turn_log_path: str | Path | None = None,
        history_window: int = 3,
        **player_kwargs: Any,
    ) -> None:
        super().__init__(**player_kwargs)
        self.backend = backend
        self.history_window = history_window
        self._turn_log_path = Path(turn_log_path) if turn_log_path else None
        if self._turn_log_path is not None:
            self._turn_log_path.parent.mkdir(parents=True, exist_ok=True)

        self._history = TurnHistoryTracker(window=history_window)
        self._reset_telemetry()

    def _reset_telemetry(self) -> None:
        self._llm_decisions = 0
        self._forced_decisions = 0
        self._fallbacks = 0
        self._latencies_ms: list[float] = []
        self._action_kinds: Counter[str] = Counter()
        self._history.clear()

    def reset_battles(self) -> None:
        """Reset both poke-env battle counters and AlphaBattle telemetry."""
        super().reset_battles()
        self._reset_telemetry()

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        self._history.pop_battle(battle.battle_tag)

    def _append_turn_log(self, record: dict[str, Any]) -> None:
        if self._turn_log_path is None:
            return
        with self._turn_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if not isinstance(battle, Battle):
            return self.choose_random_move(battle)
        if battle.wait:
            return self.choose_default_move()

        action_space = build_action_space(battle)
        if action_space.is_empty:
            return self.choose_default_move()

        hist = self._history.update_before_turn(battle)

        # Short-circuit forced single actions (saves LLM calls and avoids zero-entropy RL steps)
        if action_space.is_forced_single:
            only_action = action_space.actions[0]
            self._forced_decisions += 1
            self._action_kinds[only_action.kind] += 1
            self._history.record_action(battle, action_space, only_action.action_id)
            return only_action.order

        state_text = format_battle_state(battle, action_space, recent_history=hist)

        try:
            decision = await self.backend.choose_action(state_text, action_space)
        except Exception as exc:
            logger.warning("Backend error on %s turn %d: %r", battle.battle_tag, battle.turn, exc)
            fallback_action = action_space.actions[0]
            self._llm_decisions += 1
            self._fallbacks += 1
            self._action_kinds[fallback_action.kind] += 1
            self._history.record_action(battle, action_space, fallback_action.action_id)
            return fallback_action.order

        self._llm_decisions += 1
        self._latencies_ms.append(decision.latency_ms)
        if decision.used_fallback or decision.action_id not in action_space.by_id:
            self._fallbacks += 1
            chosen = action_space.by_id.get(decision.action_id, action_space.actions[0])
        else:
            chosen = action_space.by_id[decision.action_id]

        self._action_kinds[chosen.kind] += 1
        self._history.record_action(battle, action_space, chosen.action_id)

        if self._turn_log_path is not None:
            self._append_turn_log(
                {
                    "battle_tag": battle.battle_tag,
                    "turn": battle.turn,
                    "state": state_text,
                    "legal_ids": action_space.ids,
                    "chosen_id": chosen.action_id,
                    "chosen_kind": chosen.kind,
                    "used_fallback": decision.used_fallback,
                    "latency_ms": round(decision.latency_ms, 2),
                    "reasoning": decision.reasoning,
                    "raw_output": decision.raw_output,
                }
            )

        return chosen.order

    def build_matchup_stats(self, opponent_name: str) -> MatchupStats:
        """Aggregate finished battles and telemetry into a MatchupStats snapshot."""
        total = self.n_finished_battles
        wins = self.n_won_battles
        losses = self.n_lost_battles
        ties = self.n_tied_battles
        win_rate = (wins / total) if total > 0 else 0.0
        ci_low, ci_high = wilson_interval(wins, total)
        elo_diff = elo_diff_from_winrate(win_rate)

        turns = [b.turn for b in self.battles.values() if b.finished]
        mean_turns = (sum(turns) / len(turns)) if turns else 0.0

        mean_lat = (
            (sum(self._latencies_ms) / len(self._latencies_ms)) if self._latencies_ms else 0.0
        )
        p95_lat = percentile(self._latencies_ms, 95.0)
        fallback_rate = (
            (self._fallbacks / self._llm_decisions) if self._llm_decisions > 0 else 0.0
        )

        total_actions = sum(self._action_kinds.values())
        move_rate = (
            (self._action_kinds["move"] / total_actions) if total_actions > 0 else 0.0
        )
        tera_rate = (
            (self._action_kinds["tera_move"] / total_actions) if total_actions > 0 else 0.0
        )
        switch_rate = (
            (
                (self._action_kinds["switch"] + self._action_kinds["revive"])
                / total_actions
            )
            if total_actions > 0
            else 0.0
        )

        return MatchupStats(
            opponent=opponent_name,
            wins=wins,
            losses=losses,
            ties=ties,
            total=total,
            win_rate=win_rate,
            ci95_low=ci_low,
            ci95_high=ci_high,
            elo_diff=elo_diff,
            mean_turns=mean_turns,
            llm_decisions=self._llm_decisions,
            forced_decisions=self._forced_decisions,
            fallbacks=self._fallbacks,
            fallback_rate=fallback_rate,
            mean_latency_ms=mean_lat,
            p95_latency_ms=p95_lat,
            move_rate=move_rate,
            tera_rate=tera_rate,
            switch_rate=switch_rate,
        )
