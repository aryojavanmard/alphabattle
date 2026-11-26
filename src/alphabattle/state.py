"""Compact, deterministic battle state formatting and canonical single-token action mapping.

Action ID convention (13 single-character IDs, guaranteed 1 token in Qwen3/Llama/Gemma BPE):
- '1'..'4': Use available move slot 1..4
- 'A'..'D': Terastallize + use available move slot 1..4
- '5'..'9': Switch to (or revive, under Revival Blessing) available switch slot 1..5
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal, Sequence

from poke_env.battle.battle import Battle
from poke_env.battle.move import Move
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.pokemon_type import PokemonType
from poke_env.data import GenData
from poke_env.player.battle_order import BattleOrder, SingleBattleOrder

MOVE_IDS: tuple[str, ...] = ("1", "2", "3", "4")
TERA_MOVE_IDS: tuple[str, ...] = ("A", "B", "C", "D")
SWITCH_IDS: tuple[str, ...] = ("5", "6", "7", "8", "9")
ALL_ACTION_IDS: tuple[str, ...] = MOVE_IDS + TERA_MOVE_IDS + SWITCH_IDS
ACTION_ID_TO_INDEX: dict[str, int] = {aid: i for i, aid in enumerate(ALL_ACTION_IDS)}
POLICY_PROMPT_SUFFIX: str = "\nACTION:\n"

ActionKind = Literal["move", "tera_move", "switch", "revive"]


@dataclass(frozen=True)
class LegalAction:
    """One legal battle order paired with its canonical single-token action ID."""

    action_id: str
    kind: ActionKind
    description: str
    order: SingleBattleOrder


@dataclass(frozen=True)
class ActionSpace:
    """All legal actions available on the current turn."""

    actions: tuple[LegalAction, ...]
    by_id: dict[str, LegalAction]

    @property
    def ids(self) -> list[str]:
        return [a.action_id for a in self.actions]

    @property
    def is_empty(self) -> bool:
        return len(self.actions) == 0

    @property
    def is_forced_single(self) -> bool:
        return len(self.actions) == 1


def _fmt_type(ptype: PokemonType | None) -> str:
    if ptype is None:
        return "?"
    return ptype.name.capitalize()


def _fmt_types(mon: Pokemon) -> str:
    types = [t for t in mon.types if t is not None]
    if not types:
        return "?"
    label = "/".join(_fmt_type(t) for t in types)
    if mon.is_terastallized:
        return f"TERA:{label}"
    return label


def _fmt_item(item: str | None) -> str:
    if not item or item == GenData.UNKNOWN_ITEM:
        return "?"
    return item


def _fmt_ability(mon: Pokemon, is_opponent: bool) -> str:
    if mon.ability:
        return mon.ability
    if is_opponent and mon.possible_abilities:
        if len(mon.possible_abilities) <= 2:
            return "or".join(mon.possible_abilities)
    return "?"


def _fmt_boosts(boosts: dict[str, int]) -> str:
    parts = [f"{k}{v:+d}" for k, v in boosts.items() if v != 0]
    return ",".join(parts) if parts else "none"


def _fmt_effects(mon: Pokemon) -> str:
    if not mon.effects:
        return ""
    names = sorted(e.name.lower() for e in mon.effects.keys())
    return ",".join(names[:5])


def _safe_damage_multiplier(defender: Pokemon | None, attack: Move | PokemonType) -> float | None:
    if defender is None:
        return None
    try:
        return float(defender.damage_multiplier(attack))
    except Exception:
        return None


def _fmt_mult(mult: float | None) -> str:
    if mult is None:
        return "?"
    if mult == int(mult):
        return f"x{int(mult)}"
    return f"x{mult:g}"


def _has_stab(attacker: Pokemon, move: Move, terastallizing_now: bool = False) -> str:
    if move.category == MoveCategory.STATUS:
        return ""
    move_type = move.type
    if move_type is None:
        return ""

    base_types = {t for t in attacker.base_types if t is not None}
    cur_types = {t for t in attacker.types if t is not None}

    if terastallizing_now and attacker.tera_type is not None:
        if move_type == attacker.tera_type:
            return "ADAPT_STAB(2x)" if move_type in base_types else "TERA_STAB(1.5x)"
        if move_type in base_types:
            return "STAB(1.5x)"
        return ""

    if attacker.is_terastallized and attacker.tera_type is not None:
        if move_type == attacker.tera_type and move_type in base_types:
            return "ADAPT_STAB(2x)"
        if move_type == attacker.tera_type or move_type in base_types:
            return "STAB(1.5x)"
        return ""

    if move_type in cur_types:
        return "STAB(1.5x)"
    return ""


def _safe_move_attr(move: Move, attr: str, default: object = None) -> object:
    try:
        return getattr(move, attr)
    except (KeyError, AttributeError, TypeError):
        return default


def _move_tags(move: Move) -> list[str]:
    tags: list[str] = []
    priority = _safe_move_attr(move, "priority", 0)
    if isinstance(priority, int) and priority != 0:
        tags.append(f"pri{priority:+d}")
    heal = _safe_move_attr(move, "heal", 0)
    if isinstance(heal, (int, float)) and heal > 0:
        tags.append("heal")
    if _safe_move_attr(move, "self_switch", False):
        tags.append("pivot")
    if _safe_move_attr(move, "force_switch", False):
        tags.append("phaze")
    status = _safe_move_attr(move, "status", None)
    if status is not None:
        tags.append(f"inflict:{status.name.lower()}")
    self_boost = _safe_move_attr(move, "self_boost", None)
    boosts = _safe_move_attr(move, "boosts", None)
    if isinstance(self_boost, dict) and self_boost:
        b = ",".join(f"{k}{v:+d}" for k, v in self_boost.items() if v != 0)
        if b:
            tags.append(f"self:{b}")
    elif isinstance(boosts, dict) and boosts:
        b = ",".join(f"{k}{v:+d}" for k, v in boosts.items() if v != 0)
        if b:
            tags.append(f"boost:{b}")
    side_cond = _safe_move_attr(move, "side_condition", None)
    if side_cond is not None:
        tags.append(f"side:{side_cond.name.lower()}")
    return tags


def _describe_move(
    move: Move,
    active: Pokemon,
    opp_active: Pokemon | None,
    *,
    terastallize: bool,
) -> str:
    cat_map = {
        MoveCategory.PHYSICAL: "Phys",
        MoveCategory.SPECIAL: "Spec",
        MoveCategory.STATUS: "Stat",
    }
    category = _safe_move_attr(move, "category", None)
    cat = cat_map.get(category, "?")
    mtype = _fmt_type(_safe_move_attr(move, "type", None))  # type: ignore[arg-type]
    raw_acc = _safe_move_attr(move, "accuracy", 1.0)
    acc = int(round(float(raw_acc) * 100)) if isinstance(raw_acc, (int, float)) else 100
    cur_pp = _safe_move_attr(move, "current_pp", 1)
    max_pp = _safe_move_attr(move, "max_pp", 1)
    pp = f"{cur_pp}/{max_pp}"

    parts = [f"{mtype} {cat}"]
    if category != MoveCategory.STATUS:
        bp = _safe_move_attr(move, "base_power", 0)
        parts.append(f"BP {bp}")
        mult = _safe_damage_multiplier(opp_active, move)
        if mult is not None:
            parts.append(f"eff {_fmt_mult(mult)}")
        stab = _has_stab(active, move, terastallizing_now=terastallize)
        if stab:
            parts.append(stab)
    parts.append(f"Acc {acc}%")
    parts.append(f"PP {pp}")
    parts.extend(_move_tags(move))

    if terastallize:
        tera_lbl = _fmt_type(active.tera_type)
        return f"TERA({tera_lbl})+MOVE {move.id} ({', '.join(parts)})"
    return f"MOVE {move.id} ({', '.join(parts)})"


def _describe_switch(mon: Pokemon, opp_active: Pokemon | None, *, reviving: bool) -> str:
    hp_pct = int(round(mon.current_hp_fraction * 100))
    status = mon.status.name.lower() if mon.status else "ok"
    verb = "REVIVE" if reviving else "SWITCH"
    details = [f"L{mon.level}", _fmt_types(mon), f"HP {hp_pct}%", f"stat={status}"]

    if not reviving and opp_active is not None:
        opp_types = [t for t in opp_active.types if t is not None]
        if opp_types:
            matchups = []
            for ot in opp_types:
                m = _safe_damage_multiplier(mon, ot)
                if m is not None:
                    matchups.append(f"{_fmt_type(ot)} {_fmt_mult(m)}")
            if matchups:
                details.append("takes:" + "/".join(matchups))

    return f"{verb} {mon.species} ({', '.join(details)})"


def build_action_space(battle: Battle) -> ActionSpace:
    """Construct the deterministic legal ActionSpace for a singles Battle turn."""
    if battle.wait:
        return ActionSpace(actions=(), by_id={})

    actions: list[LegalAction] = []
    active = battle.active_pokemon
    opp_active = battle.opponent_active_pokemon

    if not battle.force_switch and active is not None:
        moves = list(battle.available_moves)[: len(MOVE_IDS)]
        for idx, move in enumerate(moves):
            aid = MOVE_IDS[idx]
            actions.append(
                LegalAction(
                    action_id=aid,
                    kind="move",
                    description=_describe_move(move, active, opp_active, terastallize=False),
                    order=SingleBattleOrder(move),
                )
            )
        if battle.can_tera:
            for idx, move in enumerate(moves):
                aid = TERA_MOVE_IDS[idx]
                actions.append(
                    LegalAction(
                        action_id=aid,
                        kind="tera_move",
                        description=_describe_move(move, active, opp_active, terastallize=True),
                        order=SingleBattleOrder(move, terastallize=True),
                    )
                )

    if not battle.trapped:
        switches = list(battle.available_switches)[: len(SWITCH_IDS)]
        reviving = bool(battle.reviving)
        kind: ActionKind = "revive" if reviving else "switch"
        for idx, mon in enumerate(switches):
            aid = SWITCH_IDS[idx]
            actions.append(
                LegalAction(
                    action_id=aid,
                    kind=kind,
                    description=_describe_switch(mon, opp_active, reviving=reviving),
                    order=SingleBattleOrder(mon),
                )
            )

    by_id = {a.action_id: a for a in actions}
    return ActionSpace(actions=tuple(actions), by_id=by_id)


def _fmt_active_mon(mon: Pokemon | None, *, is_opponent: bool) -> str:
    if mon is None:
        return "NONE"
    hp_pct = int(round(mon.current_hp_fraction * 100))
    if not is_opponent and mon.max_hp:
        hp_str = f"{mon.current_hp}/{mon.max_hp} ({hp_pct}%)"
    else:
        hp_str = f"{hp_pct}%"
    status = mon.status.name.lower() if mon.status else "none"
    tera_hint = ""
    if not mon.is_terastallized and mon.tera_type is not None:
        tera_hint = f" (tera:{_fmt_type(mon.tera_type)})"

    parts = [
        f"{mon.species} L{mon.level} {_fmt_types(mon)}{tera_hint}",
        f"HP {hp_str}",
        f"stat={status}",
        f"boosts={_fmt_boosts(mon.boosts)}",
        f"item={_fmt_item(mon.item)}",
        f"abil={_fmt_ability(mon, is_opponent=is_opponent)}",
    ]

    effects = _fmt_effects(mon)
    if effects:
        parts.append(f"vol={effects}")

    if not is_opponent and mon.stats and mon.stats.get("spe") is not None:
        parts.append(f"spe={mon.stats['spe']}")
    elif mon.base_stats and "spe" in mon.base_stats:
        parts.append(f"base_spe={mon.base_stats['spe']}")

    if is_opponent:
        seen_moves = sorted(mon.moves.keys())
        parts.append(f"seen_moves={','.join(seen_moves) if seen_moves else 'none'}")

    return " | ".join(parts)


def _fmt_bench_mon(mon: Pokemon, *, is_opponent: bool) -> str:
    if mon.fainted:
        return f"- {mon.species} L{mon.level} | FNT"
    hp_pct = int(round(mon.current_hp_fraction * 100))
    status = mon.status.name.lower() if mon.status else "ok"
    moves_list = sorted(mon.moves.keys())
    move_key = "seen" if is_opponent else "moves"
    move_str = ",".join(moves_list) if moves_list else "?"
    return (
        f"- {mon.species} L{mon.level} {_fmt_types(mon)} | {hp_pct}% {status} | "
        f"item={_fmt_item(mon.item)} abil={_fmt_ability(mon, is_opponent=is_opponent)} | "
        f"{move_key}={move_str}"
    )


def _fmt_side_conditions(conds: dict) -> str:
    if not conds:
        return "none"
    parts = []
    for cond, val in sorted(conds.items(), key=lambda kv: kv[0].name):
        name = cond.name.lower()
        parts.append(f"{name}({val})" if val else name)
    return ",".join(parts)


def format_battle_state(
    battle: Battle,
    action_space: ActionSpace,
    recent_history: Sequence[str] = (),
) -> str:
    """Serialize a poke-env Battle + ActionSpace into a compact (~350-550 token) text state."""
    fmt = battle.format or "gen9randombattle"
    my_tera = "USED" if battle.used_tera else "AVAIL"
    opp_tera = "USED" if battle.opponent_used_tera else "AVAIL"
    flags = []
    if battle.force_switch:
        flags.append("FORCE_SWITCH")
    if battle.reviving:
        flags.append("REVIVING")
    if battle.trapped:
        flags.append("TRAPPED")
    flag_str = f" | FLAGS: {','.join(flags)}" if flags else ""

    lines = [
        f"FORMAT: {fmt} | TURN: {battle.turn} | TERA: my={my_tera} opp={opp_tera}{flag_str}"
    ]

    # Field conditions (only emit non-empty entries to save tokens)
    field_parts: list[str] = []
    if battle.weather:
        w = ",".join(f"{k.name.lower()}(t{v})" for k, v in battle.weather.items())
        field_parts.append(f"weather={w}")
    if battle.fields:
        f = ",".join(f"{k.name.lower()}(t{v})" for k, v in battle.fields.items())
        field_parts.append(f"terrain={f}")
    if battle.side_conditions:
        field_parts.append(f"my_side={_fmt_side_conditions(battle.side_conditions)}")
    if battle.opponent_side_conditions:
        field_parts.append(f"opp_side={_fmt_side_conditions(battle.opponent_side_conditions)}")
    if field_parts:
        lines.append("FIELD: " + " | ".join(field_parts))

    # Active Pokémon
    lines.append(f"MY_ACTIVE: {_fmt_active_mon(battle.active_pokemon, is_opponent=False)}")
    lines.append(f"OPP_ACTIVE: {_fmt_active_mon(battle.opponent_active_pokemon, is_opponent=True)}")

    # Our bench (non-active)
    my_bench = [m for m in battle.team.values() if not m.active]
    if my_bench:
        lines.append("MY_BENCH:")
        for mon in my_bench:
            lines.append(_fmt_bench_mon(mon, is_opponent=False))

    # Opponent team (revealed non-active + count of unseen)
    opp_all = list(battle.opponent_team.values())
    opp_bench = [m for m in opp_all if not m.active]
    max_team = battle.max_team_size or 6
    unseen_count = max(0, max_team - len(opp_all))
    lines.append(f"OPP_BENCH (unseen={unseen_count}):")
    if opp_bench:
        for mon in opp_bench:
            lines.append(_fmt_bench_mon(mon, is_opponent=True))
    else:
        lines.append("- none revealed yet")

    # Short recent history (last 3 turns)
    if recent_history:
        lines.append("HISTORY:")
        for item in recent_history[-3:]:
            lines.append(f"- {item}")

    # Legal actions
    lines.append("LEGAL_ACTIONS:")
    for act in action_space.actions:
        lines.append(f"[{act.action_id}] {act.description}")

    return "\n".join(lines)


def format_policy_prompt(state_text: str) -> str:
    """Append the canonical base-LLM decision suffix.

    Ending with a newline prevents BPE tokenizers (Qwen3, Llama 3, Gemma) from merging a
    trailing space with the single-character action token ('1'..'9', 'A'..'D').
    """
    return f"{state_text}{POLICY_PROMPT_SUFFIX}"


def match_order_to_action_id(order: BattleOrder, action_space: ActionSpace) -> str | None:
    """Map a poke-env BattleOrder back to its canonical single-token action ID."""
    msg = order.message
    for act in action_space.actions:
        if act.order.message == msg:
            return act.action_id
    return None


class TurnHistoryTracker:
    """Tracks compact per-battle recent turn summaries shared across eval, recording, and RL."""

    def __init__(self, window: int = 3) -> None:
        self.window = window
        self._by_battle: dict[str, deque[str]] = {}

    def clear(self) -> None:
        self._by_battle.clear()

    def pop_battle(self, battle_tag: str) -> None:
        self._by_battle.pop(battle_tag, None)

    def update_before_turn(self, battle: Battle) -> tuple[str, ...]:
        hist = self._by_battle.setdefault(battle.battle_tag, deque(maxlen=self.window))
        if battle.turn > 1 and hist and hist[-1].startswith(f"T{battle.turn - 1}: "):
            opp = battle.opponent_active_pokemon
            opp_last = opp.last_move.id if (opp and opp.last_move) else "?"
            opp_name = opp.species if opp else "?"
            opp_hp = int(round(opp.current_hp_fraction * 100)) if opp else 0
            my = battle.active_pokemon
            my_name = my.species if my else "?"
            my_hp = int(round(my.current_hp_fraction * 100)) if my else 0
            hist[-1] = (
                f"{hist[-1]} -> opp({opp_name} {opp_hp}%) last={opp_last}; "
                f"us({my_name} {my_hp}%)"
            )
        return tuple(hist)

    def record_action(
        self, battle: Battle, action_space: ActionSpace, action_id: str
    ) -> None:
        hist = self._by_battle.setdefault(battle.battle_tag, deque(maxlen=self.window))
        chosen = action_space.by_id[action_id]
        short_desc = chosen.description.split(" (")[0].strip()
        hist.append(f"T{battle.turn}: chose [{action_id}] {short_desc}")
