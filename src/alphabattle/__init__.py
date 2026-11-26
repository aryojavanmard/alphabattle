"""AlphaBattle: Post-training a small LLM as a competitive Pokémon Showdown agent."""

from alphabattle.backends import (
    BackendDecision,
    DecisionBackend,
    FirstLegalBackend,
    OpenAICompatBackend,
    RandomLegalBackend,
)
from alphabattle.dataset import TrajectoryRecord, load_trajectory_records, split_by_battle_id
from alphabattle.metrics import (
    MatchupStats,
    binary_roc_auc,
    brier_score,
    elo_diff_from_winrate,
    wilson_interval,
)
from alphabattle.player import AlphaBattlePlayer, TelemetrySummary
from alphabattle.replays import reconstruct_replay_both_sides, reconstruct_replay_pov
from alphabattle.state import (
    ALL_ACTION_IDS,
    ActionSpace,
    LegalAction,
    TurnHistoryTracker,
    build_action_space,
    format_battle_state,
    format_policy_prompt,
)

__all__ = [
    "ALL_ACTION_IDS",
    "ActionSpace",
    "AlphaBattlePlayer",
    "BackendDecision",
    "DecisionBackend",
    "FirstLegalBackend",
    "LegalAction",
    "MatchupStats",
    "OpenAICompatBackend",
    "RandomLegalBackend",
    "TelemetrySummary",
    "TrajectoryRecord",
    "TurnHistoryTracker",
    "binary_roc_auc",
    "brier_score",
    "build_action_space",
    "elo_diff_from_winrate",
    "format_battle_state",
    "format_policy_prompt",
    "load_trajectory_records",
    "reconstruct_replay_both_sides",
    "reconstruct_replay_pov",
    "split_by_battle_id",
    "wilson_interval",
]
