"""Pre-registered Pokémon Showdown ladder & human-challenge runner (Milestone 6).

Safety & scientific rigor guardrails:
- Default mode is `accept` (only accepts direct challenges, optionally restricted by
  `--allow-opponents`), so you never accidentally queue an untested bot on the public ladder.
- Ranked `ladder` mode requires `--ladder-ack` and a fixed finite `--games N` (pre-registered
  sample size, preventing cherry-picking/optional-stopping bias).
- Respects Showdown bot etiquette: zero chat spam, clean pacing, saves full battle records
  and official ratings to a JSON report.

Examples (Native Windows PowerShell):
  # Safe mode: accept 5 challenges from yourself on Showdown
  python scripts/run_ladder.py --mode accept --games 5 --allow-opponents YourMainAccount ^
      --username AlphaBattleBot --password $env:PS_PASSWORD ^
      --backend torch --checkpoint checkpoints/m5_ppo_v1

  # Pre-registered 100-game rated ladder evaluation (only after beating SimpleHeuristics!)
  python scripts/run_ladder.py --mode ladder --ladder-ack --games 100 ^
      --username AlphaBattleBot --password $env:PS_PASSWORD ^
      --backend torch --checkpoint checkpoints/m5_ppo_v1 ^
      --output results/ladder_m5_ppo_v1.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poke_env.ps_client import (
    AccountConfiguration,
    LocalhostServerConfiguration,
    ShowdownServerConfiguration,
)

from alphabattle.backends import DecisionBackend, OpenAICompatBackend
from alphabattle.metrics import format_eval_table
from alphabattle.player import AlphaBattlePlayer


def _build_backend(args: argparse.Namespace) -> DecisionBackend:
    if args.backend == "torch":
        if not args.checkpoint:
            raise ValueError("--backend torch requires --checkpoint <dir>.")
        from alphabattle.torch_backend import PyTorchPolicyBackend

        return PyTorchPolicyBackend.from_checkpoint(
            args.checkpoint,
            base_model_override=args.base_model,
            device=args.device,
            qlora=args.qlora,
            max_batch_size=1,
            max_wait_ms=2.0,
            temperature=args.temperature,
        )
    if args.backend == "openai":
        return OpenAICompatBackend(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            mode=args.mode_prompt,
            constrain=args.constrain,
            temperature=args.temperature,
            timeout_s=args.timeout,
        )
    raise ValueError(f"Unsupported backend: {args.backend}")


async def run_live(args: argparse.Namespace) -> dict[str, Any]:
    if args.games <= 0:
        raise ValueError("--games must be > 0 (pre-register a fixed number of battles).")
    if args.mode == "ladder" and not args.ladder_ack:
        raise SystemExit(
            "Refusing to queue on the public Showdown ladder without --ladder-ack. "
            "Verify your checkpoint beats SimpleHeuristicsPlayer locally first, then pass "
            "--ladder-ack with a pre-registered --games N."
        )

    server_cfg = (
        LocalhostServerConfiguration if args.local_server else ShowdownServerConfiguration
    )
    account = AccountConfiguration(args.username, args.password)
    backend = _build_backend(args)
    save_replays: bool | str = str(Path(args.save_replays_dir)) if args.save_replays_dir else False

    player = AlphaBattlePlayer(
        backend=backend,
        turn_log_path=args.log_turns,
        account_configuration=account,
        battle_format=args.format,
        server_configuration=server_cfg,
        max_concurrent_battles=1,  # Respectful 1-at-a-time pacing on public Showdown
        start_timer_on_battle_start=(args.mode == "ladder"),
        save_replays=save_replays,
        log_level=logging.INFO,
    )

    t0 = time.perf_counter()
    try:
        if args.mode == "ladder":
            print(
                f"Starting pre-registered {args.games}-game {args.format} ladder run "
                f"as '{args.username}'..."
            )
            await player.ladder(args.games)
        else:
            opponent_filter = args.allow_opponents if args.allow_opponents else None
            print(
                f"Waiting to accept {args.games} {args.format} challenge(s) "
                f"as '{args.username}' (allow={opponent_filter or 'ANY'})..."
            )
            await player.accept_challenges(opponent_filter, args.games)
    finally:
        await backend.aclose()

    elapsed = time.perf_counter() - t0
    stats = player.build_matchup_stats(f"{args.mode}:{args.format}")

    battle_records: list[dict[str, Any]] = []
    ratings_seen: list[int] = []
    for tag, b in player.battles.items():
        if b.rating is not None:
            ratings_seen.append(int(b.rating))
        battle_records.append(
            {
                "battle_tag": tag,
                "opponent": b.opponent_username,
                "won": b.won,
                "turns": b.turn,
                "rating": b.rating,
                "opponent_rating": b.opponent_rating,
            }
        )

    print("\n=== LIVE SESSION SUMMARY ===")
    print(format_eval_table([stats]))
    if ratings_seen:
        print(
            f"Recorded Elo ratings: start={ratings_seen[0]}, "
            f"peak={max(ratings_seen)}, final={ratings_seen[-1]}"
        )

    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "format": args.format,
        "username": args.username,
        "local_server": args.local_server,
        "preregistered_games": args.games,
        "backend": args.backend,
        "checkpoint": args.checkpoint,
        "elapsed_s": round(elapsed, 2),
        "summary": stats.to_dict(),
        "final_rating": ratings_seen[-1] if ratings_seen else None,
        "peak_rating": max(ratings_seen) if ratings_seen else None,
        "battles": battle_records,
    }
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved report to {out_path}")

    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-registered Pokémon Showdown live runner.")
    parser.add_argument("--mode", choices=["accept", "ladder"], default="accept")
    parser.add_argument(
        "--ladder-ack",
        action="store_true",
        help="Required explicit confirmation before queuing on the public Showdown ladder.",
    )
    parser.add_argument(
        "--local-server",
        action="store_true",
        help="Connect to localhost:8000 instead of public Pokémon Showdown.",
    )
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", default=None)
    parser.add_argument("--allow-opponents", nargs="*", default=None)
    parser.add_argument("--format", default="gen9randombattle")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--backend", choices=["torch", "openai"], default="torch")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="qwen3-4b")
    parser.add_argument("--api-key", default="local-no-key-needed")
    parser.add_argument("--mode-prompt", choices=["direct", "cot"], default="direct")
    parser.add_argument("--constrain", choices=["json_schema", "gbnf", "none"], default="json_schema")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--save-replays-dir", default="replays/live")
    parser.add_argument("--log-turns", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_live(parse_args()))


if __name__ == "__main__":
    main()
