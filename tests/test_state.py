"""Unit tests for canonical battle state formatting and ActionSpace construction."""

from __future__ import annotations

from types import SimpleNamespace

from poke_env.battle.effect import Effect
from poke_env.battle.field import Field
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status
from poke_env.battle.weather import Weather

from alphabattle.state import build_action_space, format_battle_state


def _make_mon(
    species: str,
    *,
    level: int = 80,
    hp: int = 200,
    max_hp: int = 250,
    active: bool = False,
    status: Status | None = None,
    item: str | None = "leftovers",
    ability: str | None = None,
    tera_type: PokemonType | None = None,
    moves: list[str] | None = None,
    boosts: dict[str, int] | None = None,
    effects: dict[Effect, int] | None = None,
    stats: dict[str, int] | None = None,
) -> Pokemon:
    mon = Pokemon(gen=9, species=species)
    mon._level = level
    mon._current_hp = hp
    mon._max_hp = max_hp
    mon._active = active
    mon._status = status
    mon._item = item
    if ability is not None:
        mon.ability = ability
    if tera_type is not None:
        mon._terastallized_type = tera_type
    if moves:
        for m in moves:
            mon._add_move(m)
    if boosts:
        for k, v in boosts.items():
            mon._boosts[k] = v
    if effects:
        mon._effects.update(effects)
    if stats:
        mon._stats.update(stats)
    return mon


def _make_battle(
    *,
    active: Pokemon,
    opp_active: Pokemon,
    bench: list[Pokemon],
    opp_bench: list[Pokemon] | None = None,
    available_moves: list[Move] | None = None,
    available_switches: list[Pokemon] | None = None,
    can_tera: bool = True,
    force_switch: bool = False,
    trapped: bool = False,
    reviving: bool = False,
    wait: bool = False,
) -> SimpleNamespace:
    team = {"p1: active": active}
    for i, b in enumerate(bench):
        team[f"p1: bench{i}"] = b
    opp_team = {"p2: active": opp_active}
    for i, ob in enumerate(opp_bench or []):
        opp_team[f"p2: bench{i}"] = ob

    return SimpleNamespace(
        format="gen9randombattle",
        turn=7,
        wait=wait,
        force_switch=force_switch,
        trapped=trapped,
        reviving=reviving,
        can_tera=can_tera,
        used_tera=not can_tera,
        opponent_used_tera=False,
        active_pokemon=active,
        opponent_active_pokemon=opp_active,
        available_moves=available_moves if available_moves is not None else [],
        available_switches=available_switches if available_switches is not None else [],
        team=team,
        opponent_team=opp_team,
        max_team_size=6,
        weather={Weather.SUNNYDAY: 3},
        fields={Field.ELECTRIC_TERRAIN: 2},
        side_conditions={SideCondition.STEALTH_ROCK: 1},
        opponent_side_conditions={SideCondition.SPIKES: 2},
    )


def test_action_space_normal_turn_with_tera() -> None:
    garchomp = _make_mon(
        "garchomp",
        level=74,
        hp=214,
        max_hp=282,
        active=True,
        item="lifeorb",
        ability="roughskin",
        tera_type=PokemonType.STEEL,
        moves=["earthquake", "ironhead", "swordsdance"],
        boosts={"atk": 2},
        stats={"spe": 215},
    )
    corviknight = _make_mon(
        "corviknight",
        level=80,
        hp=68,
        max_hp=100,
        active=True,
        item="leftovers",
        ability="pressure",
        moves=["bravebird", "roost"],
    )
    rotom = _make_mon(
        "rotomwash",
        level=82,
        hp=220,
        max_hp=220,
        active=False,
        item="leftovers",
        ability="levitate",
        moves=["hydropump", "voltswitch"],
    )
    moves = [Move("earthquake", gen=9), Move("ironhead", gen=9), Move("swordsdance", gen=9)]
    battle = _make_battle(
        active=garchomp,
        opp_active=corviknight,
        bench=[rotom],
        available_moves=moves,
        available_switches=[rotom],
        can_tera=True,
    )

    space = build_action_space(battle)  # type: ignore[arg-type]
    assert space.ids == ["1", "2", "3", "A", "B", "C", "5"]
    assert space.by_id["1"].order.message == "/choose move earthquake"
    assert space.by_id["A"].order.message == "/choose move earthquake terastallize"
    assert space.by_id["5"].order.message.startswith("/choose switch ")
    # Ground vs Flying/Steel should be immune (x0)
    assert "eff x0" in space.by_id["1"].description
    assert "STAB(1.5x)" in space.by_id["1"].description
    # Steel tera + ironhead gets TERA_STAB(1.5x)
    assert "TERA_STAB(1.5x)" in space.by_id["B"].description

    text = format_battle_state(battle, space, recent_history=["T6: chose [3] MOVE swordsdance"])  # type: ignore[arg-type]
    assert "FORMAT: gen9randombattle | TURN: 7" in text
    assert "MY_ACTIVE: garchomp L74" in text
    assert "OPP_BENCH (unseen=5):" in text
    assert "HISTORY:" in text
    assert "[A] TERA(Steel)+MOVE earthquake" in text


def test_force_switch_and_trapped_and_reviving() -> None:
    active = _make_mon("garchomp", active=True, hp=0, max_hp=280, status=Status.FNT)
    opp = _make_mon("corviknight", active=True)
    b1 = _make_mon("rotomwash", active=False)
    b2 = _make_mon("scizor", active=False, hp=0, max_hp=250, status=Status.FNT)

    # Force switch: only switches (5..9), no moves even if available_moves is non-empty
    fs_battle = _make_battle(
        active=active,
        opp_active=opp,
        bench=[b1, b2],
        available_moves=[Move("earthquake", gen=9)],
        available_switches=[b1],
        force_switch=True,
    )
    fs_space = build_action_space(fs_battle)  # type: ignore[arg-type]
    assert fs_space.ids == ["5"]
    assert fs_space.is_forced_single

    # Trapped: only moves, no switches
    trapped_battle = _make_battle(
        active=_make_mon("garchomp", active=True),
        opp_active=opp,
        bench=[b1],
        available_moves=[Move("earthquake", gen=9)],
        available_switches=[b1],
        can_tera=False,
        trapped=True,
    )
    trapped_space = build_action_space(trapped_battle)  # type: ignore[arg-type]
    assert trapped_space.ids == ["1"]

    # Reviving (Revival Blessing): labeled REVIVE
    rev_battle = _make_battle(
        active=_make_mon("pawmot", active=True),
        opp_active=opp,
        bench=[b2],
        available_moves=[],
        available_switches=[b2],
        force_switch=True,
        reviving=True,
    )
    rev_space = build_action_space(rev_battle)  # type: ignore[arg-type]
    assert rev_space.ids == ["5"]
    assert rev_space.by_id["5"].kind == "revive"
    assert rev_space.by_id["5"].description.startswith("REVIVE scizor")


def test_recharge_and_struggle_moves() -> None:
    slaking = _make_mon("slaking", active=True, level=84)
    opp = _make_mon("corviknight", active=True, level=80)
    for move_id in ("recharge", "struggle"):
        battle = _make_battle(
            active=slaking,
            opp_active=opp,
            bench=[],
            available_moves=[Move(move_id, gen=9)],
            available_switches=[],
            can_tera=False,
            trapped=True,
        )
        space = build_action_space(battle)  # type: ignore[arg-type]
        assert space.ids == ["1"]
        assert space.is_forced_single
        text = format_battle_state(battle, space)  # type: ignore[arg-type]
        assert move_id in text
