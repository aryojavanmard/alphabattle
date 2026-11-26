"""Unit tests for Wilson confidence intervals, two-proportion tests, and Elo deltas."""

from __future__ import annotations

import math

from alphabattle.metrics import (
    MatchupStats,
    elo_diff_from_winrate,
    format_eval_table,
    percentile,
    two_proportion_z_test,
    wilson_interval,
)


def test_wilson_interval_edges_and_midpoint() -> None:
    lo, hi = wilson_interval(50, 100)
    assert 0.39 < lo < 0.41
    assert 0.59 < hi < 0.61

    lo0, hi0 = wilson_interval(0, 100)
    assert lo0 == 0.0
    assert 0.03 < hi0 < 0.05

    lo1, hi1 = wilson_interval(100, 100)
    assert 0.95 < lo1 < 0.97
    assert hi1 == 1.0


def test_two_proportion_z_test() -> None:
    z_same, p_same = two_proportion_z_test(50, 100, 50, 100)
    assert z_same == 0.0
    assert p_same == 1.0

    z_diff, p_diff = two_proportion_z_test(80, 100, 50, 100)
    assert z_diff > 4.0
    assert p_diff < 1e-4


def test_elo_diff_and_percentile() -> None:
    assert abs(elo_diff_from_winrate(0.5)) < 1e-9
    assert math.isclose(elo_diff_from_winrate(0.75), 190.85, rel_tol=1e-2)
    assert elo_diff_from_winrate(1.0) > 1000.0
    assert elo_diff_from_winrate(0.0) < -1000.0

    assert percentile([10.0, 20.0, 30.0, 40.0], 50.0) == 25.0
    assert percentile([], 95.0) == 0.0


def test_format_eval_table() -> None:
    row = MatchupStats(
        opponent="SimpleHeuristicsPlayer",
        wins=65,
        losses=35,
        ties=0,
        total=100,
        win_rate=0.65,
        ci95_low=0.553,
        ci95_high=0.736,
        elo_diff=107.5,
        mean_turns=29.4,
        llm_decisions=2400,
        forced_decisions=310,
        fallbacks=0,
        fallback_rate=0.0,
        mean_latency_ms=42.3,
        p95_latency_ms=78.1,
        move_rate=0.72,
        tera_rate=0.03,
        switch_rate=0.25,
    )
    table = format_eval_table([row])
    assert "SimpleHeuristicsPlayer" in table
    assert "65-35-0" in table
    assert "0.00%" in table
