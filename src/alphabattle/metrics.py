"""Evaluation statistics: Wilson confidence intervals, two-proportion tests, and Elo deltas."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


def wilson_interval(wins: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return the two-sided Wilson score confidence interval (default 95%) for a binomial rate."""
    if total <= 0:
        return (0.0, 1.0)
    p = wins / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    margin = (z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total)) / denom
    lo = 0.0 if wins == 0 else max(0.0, center - margin)
    hi = 1.0 if wins == total else min(1.0, center + margin)
    return (lo, hi)


def two_proportion_z_test(wins_a: int, total_a: int, wins_b: int, total_b: int) -> tuple[float, float]:
    """Two-sided pooled z-test comparing win rates A vs B. Returns (z_stat, p_value)."""
    if total_a <= 0 or total_b <= 0:
        return (0.0, 1.0)
    p_a = wins_a / total_a
    p_b = wins_b / total_b
    p_pool = (wins_a + wins_b) / (total_a + total_b)
    se = math.sqrt(max(p_pool * (1.0 - p_pool) * (1.0 / total_a + 1.0 / total_b), 0.0))
    if se == 0.0:
        return (0.0, 1.0)
    z = (p_a - p_b) / se
    p_val = math.erfc(abs(z) / math.sqrt(2.0))
    return (z, p_val)


def elo_diff_from_winrate(win_rate: float) -> float:
    """Convert win rate p in [0, 1] to approximate logistic Elo difference vs opponent.

    Clamps p to [1e-4, 1 - 1e-4] so 0% and 100% produce finite (+/-1600) bounds.
    """
    p = min(max(win_rate, 1e-4), 1.0 - 1e-4)
    return -400.0 * math.log10((1.0 / p) - 1.0)


def percentile(values: list[float], pct: float) -> float:
    """Compute linear-interpolated percentile in [0, 100] without external dependencies."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * (pct / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def brier_score(probs: list[float], targets: list[float]) -> float:
    """Mean squared error between predicted win probabilities in [0, 1] and binary targets."""
    if not probs:
        return 0.0
    return sum((p - y) ** 2 for p, y in zip(probs, targets)) / len(probs)


def binary_roc_auc(probs: list[float], targets: list[float]) -> float:
    """Exact ROC-AUC via Mann-Whitney U rank statistic (handles ties, zero extra deps)."""
    if len(probs) != len(targets) or not probs:
        return 0.5
    n_pos = sum(1 for y in targets if y >= 0.5)
    n_neg = len(targets) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    paired = sorted(zip(probs, targets), key=lambda x: x[0])
    rank_sum_pos = 0.0
    i = 0
    n = len(paired)
    while i < n:
        j = i + 1
        while j < n and paired[j][0] == paired[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            if paired[k][1] >= 0.5:
                rank_sum_pos += avg_rank
        i = j

    u_stat = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    return u_stat / (n_pos * n_neg)


@dataclass
class MatchupStats:
    """Summary metrics for evaluating an agent against one fixed opponent."""

    opponent: str
    wins: int
    losses: int
    ties: int
    total: int
    win_rate: float
    ci95_low: float
    ci95_high: float
    elo_diff: float
    mean_turns: float
    llm_decisions: int
    forced_decisions: int
    fallbacks: int
    fallback_rate: float
    mean_latency_ms: float
    p95_latency_ms: float
    move_rate: float
    tera_rate: float
    switch_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def format_eval_table(results: list[MatchupStats]) -> str:
    """Render a clean markdown-compatible ASCII summary table for terminal and log output."""
    header = (
        f"{'Opponent':<22} | {'W-L-T':<11} | {'WinRate (95% CI)':<23} | "
        f"{'dElo':>6} | {'Turns':>5} | {'Lat ms':>7} | {'Fallback':>8} | {'Sw%':>5}"
    )
    sep = "-" * len(header)
    rows = [header, sep]
    for r in results:
        wlt = f"{r.wins}-{r.losses}-{r.ties}"
        wr = f"{r.win_rate * 100:5.1f}% [{r.ci95_low * 100:4.1f}%, {r.ci95_high * 100:4.1f}%]"
        rows.append(
            f"{r.opponent:<22} | {wlt:<11} | {wr:<23} | "
            f"{r.elo_diff:+6.0f} | {r.mean_turns:5.1f} | {r.mean_latency_ms:7.1f} | "
            f"{r.fallback_rate * 100:7.2f}% | {r.switch_rate * 100:4.1f}%"
        )
    return "\n".join(rows)
