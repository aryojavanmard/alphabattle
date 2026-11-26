"""Two-pass spectator-to-POV replay reconstructor for Pokémon Showdown logs (Milestone 3).

Converts public spectator battle logs (e.g. from milkkarten/pokemon-showdown-replays-merged
or poke-env saved replays) into canonical first-person TrajectoryRecords using poke-env 0.17's
own Battle.parse_message engine and AlphaBattle's exact state/action formatters.

Key anti-leakage design:
- Pass 1 collects our own team's revealed species, levels, moves, items, and abilities
  (information a human player privately sees from turn 1 on Showdown).
- Opponent information is NEVER backfilled from Pass 1; it is only revealed strictly causally
  as Pass 2 replays protocol lines into Battle.parse_message.
- Tera availability (can_tera) is always True whenever `not battle.used_tera` so exposing
  Tera action IDs [A]-[D] never leaks whether that Pokémon eventually Terastallized.
- Ambiguous turns (e.g. asleep/paralyzed |cant|, Illusion/Zoroark, Revival Blessing, early
  forfeits, locked multi-turn moves) are conservatively skipped rather than emitting noisy labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

from poke_env.battle.battle import Battle
from poke_env.battle.move import Move
from poke_env.battle.pokemon_type import PokemonType
from poke_env.data import to_id_str
from poke_env.player.player import Player

from alphabattle.dataset import TrajectoryRecord
from alphabattle.state import (
    TurnHistoryTracker,
    build_action_space,
    format_battle_state,
)

_IGNORED_EVENTS = set(Player.MESSAGES_TO_IGNORE) | {
    "j",
    "l",
    "n",
    "c",
    "c:",
    "raw",
    "html",
    "uhtml",
    "t:",
    "upkeep",
    "inactive",
    "inactiveoff",
    "rated",
    "rule",
    "gametype",
    "seed",
    "error",
    "bigerror",
    "request",
    "showteam",
    "init",
    "title",
    "player",
    "teamsize",
    "gen",
    "tier",
    "clearpoke",
    "poke",
    "teampreview",
    "start",
    "win",
    "tie",
    "",
}

_EXTERNAL_MOVE_SKIP_TOKENS = (
    "[from]move:",
    "[from]ability:",
    "[from]lockedmove",
    "[from]Magic Bounce",
    "[from]Magic Coat",
    "[still]",
)
_PIVOT_MOVES = {
    "uturn",
    "voltswitch",
    "flipturn",
    "partingshot",
    "batonpass",
    "teleport",
    "chillyreception",
    "shedtail",
}
_SKIP_MOVE_IDS = {"struggle", "recharge", "revivalblessing"}


@dataclass
class OracleMon:
    """Information observed for one Pokémon across the full replay (used ONLY for our own side)."""

    ident: str  # e.g. "p1: Garchomp"
    details: str  # e.g. "Garchomp, L74, M"
    moves: list[str] = field(default_factory=list)
    item: str | None = None
    ability: str | None = None
    tera_type: str | None = None


@dataclass
class ReplayMetadata:
    """Pass-1 summary of a Showdown spectator replay log."""

    players: dict[str, str]
    ratings: dict[str, int]
    winner_role: str | None
    gen: int
    tier: str
    max_turn: int
    has_illusion: bool
    has_revival: bool
    early_forfeit: bool
    teams: dict[str, dict[str, OracleMon]]

    @property
    def avg_rating(self) -> int | None:
        if not self.ratings:
            return None
        return int(round(sum(self.ratings.values()) / len(self.ratings)))


def _canonical_ident(raw_ident: str) -> tuple[str, str] | None:
    """Convert 'p1a: Garchomp' -> ('p1', 'p1: Garchomp')."""
    if ": " not in raw_ident:
        return None
    prefix, name = raw_ident.split(": ", 1)
    if len(prefix) < 2 or prefix[:2] not in ("p1", "p2"):
        return None
    role = prefix[:2]
    return (role, f"{role}: {name}")


def scan_replay(lines: Sequence[str]) -> ReplayMetadata:
    """Pass 1: scan full replay to collect metadata and each side's revealed roster/moves."""
    players: dict[str, str] = {}
    ratings: dict[str, int] = {}
    winner_name: str | None = None
    gen = 9
    tier = ""
    max_turn = 0
    has_illusion = False
    has_revival = False
    forfeit_seen = False
    teams: dict[str, dict[str, OracleMon]] = {"p1": {}, "p2": {}}

    for raw_line in lines:
        if not raw_line.startswith("|"):
            continue
        parts = raw_line.split("|")
        if len(parts) < 2:
            continue
        ev = parts[1]

        if ev == "player" and len(parts) >= 4:
            role = parts[2]
            name = parts[3]
            if role in ("p1", "p2") and name:
                players[role] = name
                if len(parts) >= 6 and parts[5].isdigit():
                    ratings[role] = int(parts[5])
        elif ev == "gen" and len(parts) >= 3 and parts[2].isdigit():
            gen = int(parts[2])
        elif ev == "tier" and len(parts) >= 3:
            tier = parts[2]
        elif ev == "turn" and len(parts) >= 3 and parts[2].isdigit():
            max_turn = max(max_turn, int(parts[2]))
        elif ev == "win" and len(parts) >= 3:
            winner_name = parts[2]
        elif ev == "replace":
            has_illusion = True
        elif ev == "-message" and len(parts) >= 3:
            msg_lower = parts[2].lower()
            if "forfeited" in msg_lower or "lost due to inactivity" in msg_lower:
                forfeit_seen = True
        elif ev in ("switch", "drag") and len(parts) >= 4:
            parsed = _canonical_ident(parts[2])
            if parsed is None:
                continue
            role, ident = parsed
            details = parts[3]
            if "zoroark" in details.lower() or "zorua" in details.lower():
                has_illusion = True
            if ident not in teams[role]:
                teams[role][ident] = OracleMon(ident=ident, details=details)
        elif ev == "move" and len(parts) >= 4:
            parsed = _canonical_ident(parts[2])
            if parsed is None:
                continue
            role, ident = parsed
            move_id = to_id_str(parts[3])
            if move_id == "revivalblessing":
                has_revival = True
            # Only record moves genuinely belonging to the Pokémon's moveset
            extra = "|".join(parts[4:])
            if (
                move_id
                and move_id not in _SKIP_MOVE_IDS
                and not any(tok in extra for tok in _EXTERNAL_MOVE_SKIP_TOKENS)
                and ident in teams[role]
            ):
                mon = teams[role][ident]
                if move_id not in mon.moves and len(mon.moves) < 4:
                    mon.moves.append(move_id)
        elif ev == "-terastallize" and len(parts) >= 4:
            parsed = _canonical_ident(parts[2])
            if parsed is not None:
                role, ident = parsed
                if ident in teams[role]:
                    teams[role][ident].tera_type = parts[3]
        elif ev in ("-item", "-enditem") and len(parts) >= 4:
            parsed = _canonical_ident(parts[2])
            if parsed is not None:
                role, ident = parsed
                if ident in teams[role] and teams[role][ident].item is None:
                    teams[role][ident].item = to_id_str(parts[3])
        elif ev == "-ability" and len(parts) >= 4:
            parsed = _canonical_ident(parts[2])
            if parsed is not None:
                role, ident = parsed
                if ident in teams[role] and teams[role][ident].ability is None:
                    teams[role][ident].ability = to_id_str(parts[3])

    winner_role: str | None = None
    if winner_name:
        for role, pname in players.items():
            if pname == winner_name:
                winner_role = role
                break

    return ReplayMetadata(
        players=players,
        ratings=ratings,
        winner_role=winner_role,
        gen=gen,
        tier=tier,
        max_turn=max_turn,
        has_illusion=has_illusion,
        has_revival=has_revival,
        early_forfeit=(forfeit_seen and max_turn < 10),
        teams=teams,
    )


def _extract_start_of_turn_choice(
    turn_lines: Sequence[str], role: str
) -> tuple[Literal["move", "tera_move", "switch"], str] | None:
    """Identify what `role` chose at the start of a turn before any faint/pivot replacement."""
    saw_tera = False
    for raw_line in turn_lines:
        if not raw_line.startswith("|"):
            continue
        parts = raw_line.split("|")
        if len(parts) < 3:
            continue
        ev = parts[1]
        parsed = _canonical_ident(parts[2])
        if parsed is None or parsed[0] != role:
            continue
        _, ident = parsed
        extra = "|".join(parts[4:]) if len(parts) > 4 else ""

        if ev == "faint":
            # Fainted before moving -> start-of-turn choice is unobservable
            return None
        if ev == "cant":
            # Paralyzed / asleep / flinched / truant -> choice unobservable
            return None
        if ev == "drag":
            # Phazed out before acting -> choice unobservable
            return None
        if ev == "-terastallize":
            saw_tera = True
            continue
        if ev == "switch":
            if "[from]" in extra:
                return None
            return ("switch", ident)
        if ev == "move" and len(parts) >= 4:
            move_id = to_id_str(parts[3])
            if move_id in _SKIP_MOVE_IDS:
                return None
            if any(tok in extra for tok in _EXTERNAL_MOVE_SKIP_TOKENS):
                return None
            return ("tera_move" if saw_tera else "move", move_id)

    return None


def _sync_pov_action_inputs(
    battle: Battle,
    oracle_team: dict[str, OracleMon],
    *,
    force_switch: bool,
) -> None:
    """Populate poke-env Battle's private request fields for POV ActionSpace construction."""
    battle._force_switch = force_switch
    battle._trapped = False
    battle._reviving = False
    battle._wait = False

    switches = [
        mon for mon in battle.team.values() if (not mon.active and not mon.fainted)
    ]
    battle._available_switches = switches

    active = battle.active_pokemon
    if force_switch or active is None or active.fainted:
        battle._available_moves = []
        battle._can_tera = False
        return

    # Find matching oracle mon to get its full known moveset in consistent order
    active_ident = next((k for k, v in battle.team.items() if v is active), None)
    oracle_moves = oracle_team[active_ident].moves if active_ident in oracle_team else []

    move_objs: list[Move] = []
    seen_ids: set[str] = set()
    for mid in oracle_moves:
        if mid in active.moves:
            move_objs.append(active.moves[mid])
            seen_ids.add(mid)
        else:
            m = active._add_move(mid)
            if m is not None:
                move_objs.append(m)
                seen_ids.add(mid)
    for mid, m in active.moves.items():
        if mid not in seen_ids and len(move_objs) < 4:
            move_objs.append(m)
            seen_ids.add(mid)

    battle._available_moves = move_objs[:4]

    # Always expose Tera options whenever not yet used (prevents future-leakage correlation!)
    if not battle.used_tera:
        if active.tera_type is None and active.type_1 is not None:
            active._terastallized_type = active.type_1
        battle._can_tera = len(battle._available_moves) > 0
    else:
        battle._can_tera = False


def _match_human_choice(
    choice: tuple[Literal["move", "tera_move", "switch"], str],
    battle: Battle,
    action_space_actions: Sequence[object],
) -> str | None:
    kind, target = choice
    if kind == "switch":
        target_mon = battle.team.get(target)
        if target_mon is None:
            return None
        for act in action_space_actions:
            if act.kind == "switch" and act.order.order is target_mon:  # type: ignore[attr-defined]
                return str(act.action_id)  # type: ignore[attr-defined]
        return None

    for act in action_space_actions:
        if act.kind == kind and getattr(act.order.order, "id", None) == target:  # type: ignore[attr-defined]
            return str(act.action_id)  # type: ignore[attr-defined]
    return None


def _safe_parse_message(battle: Battle, parts: list[str]) -> None:
    if len(parts) < 2:
        return
    ev = parts[1]
    if ev in _IGNORED_EVENTS:
        return
    try:
        battle.parse_message(parts)
    except (NotImplementedError, KeyError, ValueError, AssertionError, IndexError):
        return


def reconstruct_replay_pov(
    log_text: str,
    *,
    battle_id: str,
    role: Literal["p1", "p2"],
    rating_override: int | None = None,
    source: str = "human_replay",
    format_name: str = "gen9randombattle",
    min_turns: int = 8,
    meta: ReplayMetadata | None = None,
) -> list[TrajectoryRecord]:
    """Reconstruct first-person TrajectoryRecords for `role` from a raw Showdown spectator log."""
    lines = [ln.strip() for ln in log_text.splitlines() if ln.strip()]
    summary = meta or scan_replay(lines)

    if summary.gen != 9:
        return []
    if summary.max_turn < min_turns:
        return []
    if summary.has_illusion or summary.has_revival or summary.early_forfeit:
        return []
    if role not in summary.players:
        return []

    oracle_team = summary.teams.get(role, {})
    if len(oracle_team) < 2:
        return []

    username = summary.players[role]
    battle = Battle(battle_id, username, None, gen=9)
    battle._player_role = role
    battle._opponent_role = "p2" if role == "p1" else "p1"
    battle._format = format_name

    # Pre-populate ONLY our own team from Pass 1 (opponent stays strictly hidden!)
    for ident, omon in oracle_team.items():
        mon = battle.get_pokemon(ident, details=omon.details)
        for mid in omon.moves:
            mon._add_move(mid)
        if omon.item:
            mon._item = omon.item
        if omon.ability:
            mon.ability = omon.ability
        if omon.tera_type:
            try:
                mon._terastallized_type = PokemonType.from_name(omon.tera_type)
            except Exception:
                pass

    won: bool | None = (
        (summary.winner_role == role) if summary.winner_role is not None else None
    )
    rating = rating_override if rating_override is not None else summary.avg_rating
    history = TurnHistoryTracker(window=3)
    records: list[TrajectoryRecord] = []

    # Partition log into pre-turn-1 setup and per-turn blocks
    pre_turn_lines: list[str] = []
    turn_blocks: list[tuple[int, list[str]]] = []
    cur_turn: int | None = None
    cur_lines: list[str] = []

    for ln in lines:
        parts = ln.split("|")
        if len(parts) >= 3 and parts[1] == "turn" and parts[2].isdigit():
            if cur_turn is not None:
                turn_blocks.append((cur_turn, cur_lines))
            cur_turn = int(parts[2])
            cur_lines = []
        elif cur_turn is None:
            pre_turn_lines.append(ln)
        else:
            cur_lines.append(ln)
    if cur_turn is not None:
        turn_blocks.append((cur_turn, cur_lines))

    # Replay pre-turn-1 setup (lead switches, entry abilities/hazards)
    for ln in pre_turn_lines:
        _safe_parse_message(battle, ln.split("|"))

    # Replay turn-by-turn
    for turn_num, tlines in turn_blocks:
        battle._turn = turn_num

        # 1. Start-of-turn decision (before applying turn_num's events!)
        choice = _extract_start_of_turn_choice(tlines, role)
        if choice is not None and battle.active_pokemon is not None and not battle.active_pokemon.fainted:
            _sync_pov_action_inputs(battle, oracle_team, force_switch=False)
            action_space = build_action_space(battle)
            hist = history.update_before_turn(battle)
            matched_id = _match_human_choice(choice, battle, action_space.actions)
            if matched_id is not None:
                if len(action_space.actions) >= 2:
                    state_text = format_battle_state(battle, action_space, recent_history=hist)
                    records.append(
                        TrajectoryRecord(
                            battle_id=f"{battle_id}:{role}",
                            turn=turn_num,
                            state=state_text,
                            legal_ids=tuple(action_space.ids),
                            action_id=matched_id,
                            won=won,
                            rating=rating,
                            source=source,
                        )
                    )
                history.record_action(battle, action_space, matched_id)

        # 2. Replay turn events line by line, also capturing post-faint / pivot forced switches
        awaiting_forced_switch = False
        for ln in tlines:
            parts = ln.split("|")
            if len(parts) < 2:
                continue
            ev = parts[1]
            parsed = _canonical_ident(parts[2]) if len(parts) >= 3 else None

            if awaiting_forced_switch and ev == "switch" and parsed is not None and parsed[0] == role:
                _, target_ident = parsed
                _sync_pov_action_inputs(battle, oracle_team, force_switch=True)
                fs_space = build_action_space(battle)
                fs_matched = _match_human_choice(("switch", target_ident), battle, fs_space.actions)
                if fs_matched is not None:
                    if len(fs_space.actions) >= 2:
                        hist = history.update_before_turn(battle)
                        state_text = format_battle_state(battle, fs_space, recent_history=hist)
                        records.append(
                            TrajectoryRecord(
                                battle_id=f"{battle_id}:{role}",
                                turn=turn_num,
                                state=state_text,
                                legal_ids=tuple(fs_space.ids),
                                action_id=fs_matched,
                                won=won,
                                rating=rating,
                                source=source,
                            )
                        )
                    history.record_action(battle, fs_space, fs_matched)
                awaiting_forced_switch = False

            _safe_parse_message(battle, parts)

            if parsed is not None and parsed[0] == role:
                if ev == "faint":
                    awaiting_forced_switch = True
                elif ev == "move" and len(parts) >= 4 and to_id_str(parts[3]) in _PIVOT_MOVES:
                    awaiting_forced_switch = True

    return records


def reconstruct_replay_both_sides(
    log_text: str,
    *,
    battle_id: str,
    rating_override: int | None = None,
    winner_only: bool = False,
    source: str = "human_replay",
    format_name: str = "gen9randombattle",
    min_turns: int = 8,
) -> list[TrajectoryRecord]:
    """Reconstruct POV trajectories for both players (or winner only) from one replay."""
    lines = [ln.strip() for ln in log_text.splitlines() if ln.strip()]
    meta = scan_replay(lines)

    roles: list[Literal["p1", "p2"]] = ["p1", "p2"]
    if winner_only:
        if meta.winner_role in ("p1", "p2"):
            roles = [meta.winner_role]  # type: ignore[list-item]
        else:
            return []

    out: list[TrajectoryRecord] = []
    for role in roles:
        out.extend(
            reconstruct_replay_pov(
                log_text,
                battle_id=battle_id,
                role=role,
                rating_override=rating_override,
                source=source,
                format_name=format_name,
                min_turns=min_turns,
                meta=meta,
            )
        )
    return out
