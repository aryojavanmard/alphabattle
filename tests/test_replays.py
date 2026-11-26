"""Unit tests for two-pass spectator-to-POV replay reconstruction and anti-leakage guarantees."""

from __future__ import annotations

from alphabattle.replays import reconstruct_replay_both_sides, reconstruct_replay_pov, scan_replay

SAMPLE_REPLAY = """
|player|p1|Alice|101|1650
|player|p2|Bob|102|1680
|teamsize|p1|6
|teamsize|p2|6
|gen|9
|tier|[Gen 9] Random Battle
|start
|switch|p1a: Garchomp|Garchomp, L74, M|100/100
|switch|p2a: Corviknight|Corviknight, L80, F|100/100
|turn|1
|-terastallize|p1a: Garchomp|Steel
|move|p1a: Garchomp|Swords Dance|p1a: Garchomp
|-boost|p1a: Garchomp|atk|2
|move|p2a: Corviknight|Brave Bird|p1a: Garchomp
|-resisted|p1a: Garchomp
|-damage|p1a: Garchomp|78/100
|turn|2
|switch|p1a: Rotom|Rotom-Wash, L82|100/100
|move|p2a: Corviknight|Body Press|p1a: Rotom
|-damage|p1a: Rotom|70/100
|turn|3
|move|p1a: Rotom|Hydro Pump|p2a: Corviknight
|-damage|p2a: Corviknight|40/100
|move|p2a: Corviknight|Roost|p2a: Corviknight
|-heal|p2a: Corviknight|90/100
|turn|4
|move|p1a: Rotom|Volt Switch|p2a: Corviknight
|-supereffective|p2a: Corviknight
|-damage|p2a: Corviknight|20/100
|switch|p1a: Garchomp|Garchomp, L74, M|78/100|[from] Volt Switch
|move|p2a: Corviknight|Roost|p2a: Corviknight
|-heal|p2a: Corviknight|70/100
|turn|5
|move|p1a: Garchomp|Earthquake|p2a: Corviknight
|-immune|p2a: Corviknight
|move|p2a: Corviknight|Body Press|p1a: Garchomp
|-supereffective|p1a: Garchomp
|-damage|p1a: Garchomp|0 fnt
|faint|p1a: Garchomp
|switch|p1a: Scizor|Scizor, L79, M|100/100
|turn|6
|switch|p2a: Chien-Pao|Chien-Pao, L72|100/100
|-ability|p2a: Chien-Pao|Sword of Ruin
|move|p1a: Scizor|Bullet Punch|p2a: Chien-Pao
|-supereffective|p2a: Chien-Pao
|-damage|p2a: Chien-Pao|0 fnt
|faint|p2a: Chien-Pao
|switch|p2a: Corviknight|Corviknight, L80, F|70/100
|turn|7
|cant|p1a: Scizor|par
|move|p2a: Corviknight|Brave Bird|p1a: Scizor
|-damage|p1a: Scizor|55/100
|turn|8
|move|p1a: Scizor|Close Combat|p2a: Corviknight
|-damage|p2a: Corviknight|0 fnt
|faint|p2a: Corviknight
|win|Alice
""".strip()


def test_two_pass_replay_reconstructs_moves_tera_switches_and_blocks_opponent_leakage() -> None:
    meta = scan_replay(SAMPLE_REPLAY.splitlines())
    assert meta.winner_role == "p1"
    assert meta.avg_rating == 1665
    assert meta.max_turn == 8
    assert not meta.has_illusion

    p1_recs = reconstruct_replay_pov(
        SAMPLE_REPLAY, battle_id="r1", role="p1", min_turns=8
    )
    assert len(p1_recs) >= 6
    assert all(r.won is True for r in p1_recs)
    assert all(r.action_id in r.legal_ids for r in p1_recs)

    # Turn 1 must record Tera + Swords Dance (action ID in A..D)
    t1 = next(r for r in p1_recs if r.turn == 1)
    assert t1.action_id in {"A", "B", "C", "D"}
    # Opponent's future Chien-Pao (not seen until turn 6) MUST NOT appear in turn 1 state!
    assert "chienpao" not in t1.state.lower()
    # Opponent's future moves (bravebird, bodypress, roost) MUST NOT appear on turn 1!
    assert "seen_moves=none" in t1.state
    # Our own bench (rotomwash, scizor) IS visible from turn 1 (private POV)
    assert "rotomwash" in t1.state.lower()
    assert "scizor" in t1.state.lower()

    # Turn 2: p1 switched to Rotom -> action ID in 5..9
    t2 = next(r for r in p1_recs if r.turn == 2)
    assert t2.action_id in {"5", "6", "7", "8", "9"}

    # Turn 5: Garchomp attacked AND then fainted -> must emit BOTH normal turn + FORCE_SWITCH!
    t5_recs = [r for r in p1_recs if r.turn == 5]
    assert len(t5_recs) == 2
    assert "FORCE_SWITCH" not in t5_recs[0].state
    assert "FORCE_SWITCH" in t5_recs[1].state
    assert t5_recs[1].action_id in {"5", "6", "7", "8", "9"}

    # Turn 7 had |cant|p1a: Scizor|par -> MUST NOT emit a guessed action label!
    assert not any(r.turn == 7 for r in p1_recs)


def test_winner_only_and_illusion_filtering() -> None:
    both = reconstruct_replay_both_sides(SAMPLE_REPLAY, battle_id="r2", winner_only=False, min_turns=8)
    win_only = reconstruct_replay_both_sides(SAMPLE_REPLAY, battle_id="r2", winner_only=True, min_turns=8)
    assert len(both) > len(win_only) > 0
    assert all(r.won is True for r in win_only)

    illusion_replay = SAMPLE_REPLAY + "\n|replace|p2a: Zoroark|Zoroark, L80, M"
    assert reconstruct_replay_both_sides(illusion_replay, battle_id="r3", min_turns=8) == []


def test_hf_data_files_bracket_escaping() -> None:
    import importlib.util
    from pathlib import Path

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_replay_dataset.py"
    spec = importlib.util.spec_from_file_location("build_replay_dataset", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    out = mod._normalize_hf_data_files(
        ["[Gen 9] RANDOMBATTLE_part11.parquet", "data/part-*.parquet"]
    )
    assert out == ["[[]Gen 9] RANDOMBATTLE_part11.parquet", "data/part-*.parquet"]
