"""Record zero-sim2sim-gap trajectories from local Pokémon Showdown battles (Milestone 2).

Examples (Native Windows PowerShell):
  # Record 5,000 SimpleHeuristicsPlayer mirror matches (~250K turns, both sides recorded):
  python scripts/record_trajectories.py --games 5000 --concurrency 16 ^
      --teacher heuristics --opponent heuristics --record-both ^
      --output data/m2_heuristics_5k.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer, SimpleHeuristicsPlayer
from poke_env.ps_client import AccountConfiguration, ServerConfiguration

from alphabattle.recorder import (
    RecordingMaxBasePowerPlayer,
    RecordingSimpleHeuristicsPlayer,
)


def _make_player(
    role: str,
    *,
    record: bool,
    output_path: Path,
    winner_only: bool,
    battle_format: str,
    server_config: ServerConfiguration,
    concurrency: int,
    prefix: str,
) -> Player:
    common = dict(
        battle_format=battle_format,
        server_configuration=server_config,
        max_concurrent_battles=concurrency,
        log_level=logging.WARNING,
        account_configuration=AccountConfiguration.generate(prefix, rand=True),
    )
    if record:
        if role == "heuristics":
            return RecordingSimpleHeuristicsPlayer(
                output_path=output_path,
                source_label="heuristics",
                winner_only=winner_only,
                **common,
            )
        if role == "max_power":
            return RecordingMaxBasePowerPlayer(
                output_path=output_path,
                source_label="max_power",
                winner_only=winner_only,
                **common,
            )
        raise ValueError(f"Recording not supported for role '{role}'. Use heuristics or max_power.")

    if role == "heuristics":
        return SimpleHeuristicsPlayer(**common)
    if role == "max_power":
        return MaxBasePowerPlayer(**common)
    if role == "random":
        return RandomPlayer(**common)
    raise ValueError(f"Unknown opponent role '{role}'.")


async def run_recording(args: argparse.Namespace) -> None:
    out_path = Path(args.output)
    if args.overwrite and out_path.exists():
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    server_config = ServerConfiguration(
        f"ws://{args.showdown_host}:{args.showdown_port}/showdown/websocket",
        "https://play.pokemonshowdown.com/action.php?",
    )

    p1 = _make_player(
        args.teacher,
        record=True,
        output_path=out_path,
        winner_only=args.winner_only,
        battle_format=args.format,
        server_config=server_config,
        concurrency=args.concurrency,
        prefix="RecP1",
    )
    p2 = _make_player(
        args.opponent,
        record=args.record_both,
        output_path=out_path,
        winner_only=args.winner_only,
        battle_format=args.format,
        server_config=server_config,
        concurrency=args.concurrency,
        prefix="RecP2",
    )

    print(
        f"Recording {args.games} {args.format} games "
        f"({args.teacher} vs {args.opponent}, record_both={args.record_both}, "
        f"winner_only={args.winner_only}, concurrency={args.concurrency}) -> {out_path}"
    )
    t0 = time.perf_counter()
    await p1.battle_against(p2, n_battles=args.games)
    elapsed = time.perf_counter() - t0

    s1 = p1.recording_stats  # type: ignore[attr-defined]
    total_turns = s1.turns_written
    total_battles = s1.battles_recorded
    total_unmapped = s1.unmapped_orders
    if args.record_both:
        s2 = p2.recording_stats  # type: ignore[attr-defined]
        total_turns += s2.turns_written
        total_battles += s2.battles_recorded
        total_unmapped += s2.unmapped_orders

    print(
        f"Done in {elapsed:.1f}s ({args.games / max(elapsed, 1e-6):.1f} games/s): "
        f"wrote {total_turns:,} turns from {total_battles:,} player-trajectories "
        f"(unmapped_orders={total_unmapped})."
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record local Showdown trajectories to JSONL.")
    parser.add_argument("--teacher", choices=["heuristics", "max_power"], default="heuristics")
    parser.add_argument(
        "--opponent", choices=["heuristics", "max_power", "random"], default="heuristics"
    )
    parser.add_argument(
        "--record-both",
        action="store_true",
        help="Record trajectories from both players (recommended for mirror matches).",
    )
    parser.add_argument(
        "--winner-only",
        action="store_true",
        help="Only write trajectories from the winning side of each game.",
    )
    parser.add_argument("--games", type=int, default=5000)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--format", default="gen9randombattle")
    parser.add_argument("--showdown-host", default="127.0.0.1")
    parser.add_argument("--showdown-port", type=int, default=8000)
    parser.add_argument("--output", required=True, help="Output JSONL dataset path.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output file.")
    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(run_recording(parse_args()))


if __name__ == "__main__":
    main()
