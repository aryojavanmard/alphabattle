"""Unit tests for AlphaBattlePlayer turn handling, short-circuiting, and telemetry."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from poke_env.battle.battle import Battle
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.pokemon_type import PokemonType

from alphabattle.backends import BackendDecision, FirstLegalBackend
from alphabattle.player import AlphaBattlePlayer
from alphabattle.state import TurnHistoryTracker


class _CountingBackend:
    def __init__(self, action_id: str = "1", *, raise_exc: bool = False) -> None:
        self.calls = 0
        self.action_id = action_id
        self.raise_exc = raise_exc

    async def choose_action(self, state_text: str, action_space) -> BackendDecision:
        self.calls += 1
        if self.raise_exc:
            raise RuntimeError("boom")
        return BackendDecision(
            action_id=self.action_id,
            raw_output=self.action_id,
            latency_ms=12.5,
        )

    async def aclose(self) -> None:
        return None


def _make_real_battle_stub(
    *,
    moves: list[Move],
    switches: list[Pokemon],
    can_tera: bool = False,
) -> Battle:
    battle = Battle("battle-gen9randombattle-1", "AlphaBot", None, gen=9)
    active = Pokemon(gen=9, species="garchomp")
    active._active = True
    active._level = 80
    active._current_hp = 200
    active._max_hp = 280
    active._terastallized_type = PokemonType.STEEL
    opp = Pokemon(gen=9, species="corviknight")
    opp._active = True
    opp._level = 80
    opp._current_hp = 80
    opp._max_hp = 100

    battle._team = {"p1: Garchomp": active}
    for idx, sw in enumerate(switches):
        battle._team[f"p1: Bench{idx}"] = sw
    battle._opponent_team = {"p2: Corviknight": opp}
    battle._available_moves = moves
    battle._available_switches = switches
    battle._can_tera = can_tera
    battle._turn = 3
    battle._format = "gen9randombattle"
    return battle


@pytest.mark.asyncio
async def test_player_short_circuits_forced_single_and_logs_turns(tmp_path: Path) -> None:
    log_file = tmp_path / "turns.jsonl"
    backend = _CountingBackend(action_id="2")

    with patch("poke_env.player.player.Player.__init__", return_value=None):
        player = AlphaBattlePlayer.__new__(AlphaBattlePlayer)
        player.backend = backend
        player.history_window = 3
        player._turn_log_path = log_file
        player._history = TurnHistoryTracker(window=3)
        player._reset_telemetry()

    # 1. Forced single action -> should NOT invoke backend
    forced_battle = _make_real_battle_stub(
        moves=[Move("earthquake", gen=9)],
        switches=[],
        can_tera=False,
    )
    order1 = await player.choose_move(forced_battle)
    assert order1.message == "/choose move earthquake"
    assert backend.calls == 0
    assert player._forced_decisions == 1

    # 2. Multiple actions -> invokes backend and logs JSONL
    multi_battle = _make_real_battle_stub(
        moves=[Move("earthquake", gen=9), Move("ironhead", gen=9)],
        switches=[],
        can_tera=False,
    )
    order2 = await player.choose_move(multi_battle)
    assert order2.message == "/choose move ironhead"
    assert backend.calls == 1
    assert player._llm_decisions == 1
    assert player._fallbacks == 0

    records = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["chosen_id"] == "2"
    assert records[0]["legal_ids"] == ["1", "2"]


@pytest.mark.asyncio
async def test_player_handles_backend_exception_gracefully() -> None:
    backend = _CountingBackend(raise_exc=True)
    with patch("poke_env.player.player.Player.__init__", return_value=None):
        player = AlphaBattlePlayer.__new__(AlphaBattlePlayer)
        player.backend = backend
        player.history_window = 3
        player._turn_log_path = None
        player._history = TurnHistoryTracker(window=3)
        player._reset_telemetry()

    battle = _make_real_battle_stub(
        moves=[Move("earthquake", gen=9), Move("ironhead", gen=9)],
        switches=[],
        can_tera=False,
    )
    order = await player.choose_move(battle)
    assert order.message == "/choose move earthquake"
    assert player._fallbacks == 1
    assert player._llm_decisions == 1


@pytest.mark.asyncio
async def test_first_legal_backend_runs() -> None:
    backend = FirstLegalBackend()
    await backend.aclose()
