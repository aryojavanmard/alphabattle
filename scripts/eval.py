"""Fixed evaluation harness for AlphaBattle against standard poke-env baselines.

Examples (Native Windows PowerShell):
  # 1. Pipeline smoke test (no LLM needed):
  python scripts/eval.py --backend random --games 20 --concurrency 8

  # 2. Stage 0 baseline against local llama-server (GBNF 1-token constrained):
  python scripts/eval.py --backend openai --base-url http://127.0.0.1:8080/v1 ^
      --model qwen3-4b --mode direct --constrain gbnf --games 100 --concurrency 4

  # 3. Stage 0 baseline against local Ollama (JSON Schema constrained):
  python scripts/eval.py --backend openai --base-url http://127.0.0.1:11434/v1 ^
      --model qwen3:4b --mode direct --constrain json_schema --games 100 --concurrency 4
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

# Allow running directly via `python scripts/eval.py` without editable install
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer, SimpleHeuristicsPlayer
from poke_env.ps_client import AccountConfiguration, ServerConfiguration

from alphabattle.backends import (
    DecisionBackend,
    FirstLegalBackend,
    OpenAICompatBackend,
    RandomLegalBackend,
)
from alphabattle.metrics import MatchupStats, format_eval_table
from alphabattle.player import AlphaBattlePlayer


def build_backend(args: argparse.Namespace) -> DecisionBackend:
    if args.backend == "random":
        return RandomLegalBackend(seed=args.seed)
    if args.backend == "first":
        return FirstLegalBackend()
    if args.backend == "openai":
        return OpenAICompatBackend(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            mode=args.mode,
            constrain=args.constrain,
            temperature=args.temperature,
            timeout_s=args.timeout,
            qwen3_no_think=not args.allow_qwen3_think,
        )
    if args.backend == "torch":
        if not args.checkpoint:
            raise ValueError("--backend torch requires --checkpoint <dir>.")
        from alphabattle.torch_backend import PyTorchPolicyBackend

        return PyTorchPolicyBackend.from_checkpoint(
            args.checkpoint,
            base_model_override=args.base_model,
            device=args.device,
            qlora=args.qlora,
            max_batch_size=args.max_batch_size,
            max_wait_ms=args.max_wait_ms,
            temperature=args.temperature,
        )
    raise ValueError(f"Unsupported backend: {args.backend}")


def build_opponent(
    name: str,
    *,
    battle_format: str,
    server_config: ServerConfiguration,
    max_concurrent_battles: int,
) -> tuple[str, Player]:
    common = dict(
        battle_format=battle_format,
        server_configuration=server_config,
        max_concurrent_battles=max_concurrent_battles,
        log_level=logging.WARNING,
    )
    if name == "random":
        return (
            "RandomPlayer",
            RandomPlayer(
                account_configuration=AccountConfiguration.generate("EvalRand", rand=True),
                **common,
            ),
        )
    if name == "max_power":
        return (
            "MaxBasePowerPlayer",
            MaxBasePowerPlayer(
                account_configuration=AccountConfiguration.generate("EvalMaxPwr", rand=True),
                **common,
            ),
        )
    if name == "heuristics":
        return (
            "SimpleHeuristicsPlayer",
            SimpleHeuristicsPlayer(
                account_configuration=AccountConfiguration.generate("EvalHeur", rand=True),
                **common,
            ),
        )
    raise ValueError(f"Unknown opponent '{name}'. Expected one of: random, max_power, heuristics.")


async def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    server_config = ServerConfiguration(
        f"ws://{args.showdown_host}:{args.showdown_port}/showdown/websocket",
        "https://play.pokemonshowdown.com/action.php?",
    )

    backend = build_backend(args)
    extra_backends: list[DecisionBackend] = []
    save_replays: bool | str = str(Path(args.save_replays_dir)) if args.save_replays_dir else False

    agent = AlphaBattlePlayer(
        backend=backend,
        turn_log_path=args.log_turns,
        account_configuration=AccountConfiguration.generate("AlphaBot", rand=True),
        battle_format=args.format,
        server_configuration=server_config,
        max_concurrent_battles=args.concurrency,
        save_replays=save_replays,
        log_level=logging.WARNING,
    )

    all_stats: list[MatchupStats] = []
    t_start = time.perf_counter()

    try:
        opponent_runs: list[tuple[str, Player]] = []
        for opp_key in args.opponents:
            opponent_runs.append(
                build_opponent(
                    opp_key,
                    battle_format=args.format,
                    server_config=server_config,
                    max_concurrent_battles=args.concurrency,
                )
            )

        for ckpt_path in args.opponent_checkpoint or []:
            from alphabattle.torch_backend import PyTorchPolicyBackend

            opp_backend = PyTorchPolicyBackend.from_checkpoint(
                ckpt_path,
                base_model_override=args.base_model,
                device=args.device,
                qlora=args.qlora,
                max_batch_size=args.max_batch_size,
                max_wait_ms=args.max_wait_ms,
                temperature=args.temperature,
            )
            extra_backends.append(opp_backend)
            opp_label = f"Ckpt:{Path(ckpt_path).name}"
            opp_player = AlphaBattlePlayer(
                backend=opp_backend,
                account_configuration=AccountConfiguration.generate("EvalCkpt", rand=True),
                battle_format=args.format,
                server_configuration=server_config,
                max_concurrent_battles=args.concurrency,
                log_level=logging.WARNING,
            )
            opponent_runs.append((opp_label, opp_player))

        for opp_label, opponent in opponent_runs:
            agent.reset_battles()
            print(
                f"\nRunning {args.games} games ({args.format}, concurrency={args.concurrency}) "
                f"vs {opp_label}..."
            )
            t0 = time.perf_counter()
            await agent.battle_against(opponent, n_battles=args.games)
            elapsed = time.perf_counter() - t0

            stats = agent.build_matchup_stats(opp_label)
            all_stats.append(stats)
            print(
                f"Finished vs {opp_label} in {elapsed:.1f}s "
                f"({args.games / max(elapsed, 1e-6):.2f} battles/s): "
                f"{stats.wins}-{stats.losses}-{stats.ties} "
                f"({stats.win_rate * 100:.1f}%, 95% CI "
                f"[{stats.ci95_low * 100:.1f}%, {stats.ci95_high * 100:.1f}%])"
            )
    finally:
        await backend.aclose()
        for eb in extra_backends:
            await eb.aclose()

    total_elapsed = time.perf_counter() - t_start
    print("\n=== EVALUATION SUMMARY ===")
    print(format_eval_table(all_stats))
    print(f"Total elapsed: {total_elapsed:.1f}s")

    model_label = (
        args.model
        if args.backend == "openai"
        else (str(args.checkpoint) if args.backend == "torch" else args.backend)
    )
    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "format": args.format,
        "games_per_opponent": args.games,
        "concurrency": args.concurrency,
        "backend": args.backend,
        "model": model_label,
        "mode": args.mode if args.backend == "openai" else "masked_logits",
        "constrain": args.constrain if args.backend == "openai" else "logit_mask",
        "temperature": args.temperature,
        "total_elapsed_s": round(total_elapsed, 2),
        "results": [s.to_dict() for s in all_stats],
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved JSON report to {out_path}")

    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AlphaBattle fixed evaluation harness.")
    parser.add_argument(
        "--backend",
        choices=["openai", "torch", "random", "first"],
        default="openai",
        help="Model backend to evaluate.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="qwen3-4b")
    parser.add_argument("--api-key", default="local-no-key-needed")
    parser.add_argument("--mode", choices=["direct", "cot"], default="direct")
    parser.add_argument("--constrain", choices=["json_schema", "gbnf", "none"], default="json_schema")
    parser.add_argument("--checkpoint", default=None, help="Path to saved PyTorch LoRA checkpoint.")
    parser.add_argument("--base-model", default=None, help="Optional override for base HF model.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--max-wait-ms", type=float, default=8.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--allow-qwen3-think",
        action="store_true",
        help="Do not append '/no_think' for Qwen3 models in direct mode.",
    )
    parser.add_argument("--format", default="gen9randombattle")
    parser.add_argument(
        "--opponents",
        nargs="*",
        choices=["random", "max_power", "heuristics"],
        default=["random", "max_power", "heuristics"],
    )
    parser.add_argument(
        "--opponent-checkpoint",
        nargs="*",
        default=None,
        help="Optional prior AlphaBattle checkpoint dir(s) to evaluate head-to-head against.",
    )
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--showdown-host", default="127.0.0.1")
    parser.add_argument("--showdown-port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-replays-dir", default=None)
    parser.add_argument("--log-turns", default=None, help="Optional JSONL path to log every turn.")
    parser.add_argument("--output", default=None, help="Optional JSON report path.")
    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    args = parse_args()
    asyncio.run(run_evaluation(args))


if __name__ == "__main__":
    main()
