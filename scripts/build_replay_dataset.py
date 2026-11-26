"""Build a Stage 1 high-Elo SFT dataset from Hugging Face or local Pokémon Showdown replays.

Examples (Native Windows PowerShell):
  # Stream high-Elo Gen 9 Random Battles directly from Hugging Face (requires `pip install datasets`):
  python scripts/build_replay_dataset.py ^
      --hf-dataset milkkarten/pokemon-showdown-replays-merged ^
      --format-filter "randombattle" ^
      --gen-filter 9 ^
      --min-rating 1550 ^
      --winner-only ^
      --max-turns 250000 ^
      --output data/gen9randbats_1550plus.jsonl

  # Or parse local Showdown .log / .json / .jsonl replay files:
  python scripts/build_replay_dataset.py ^
      --input-dir replays/ ^
      --min-rating 0 ^
      --output data/local_replays.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alphabattle.replays import reconstruct_replay_both_sides

_HTML_LOG_RE = re.compile(r"<script[^>]*battle-log-data[^>]*>(.*?)</script>", flags=re.S)


def _matches_format(fmt: str, *, format_filter: str, gen_filter: int | None) -> bool:
    norm = fmt.lower().replace(" ", "").replace("[", "").replace("]", "")
    if format_filter and format_filter.lower() not in norm:
        return False
    if gen_filter is not None and f"gen{gen_filter}" not in norm:
        return False
    return True


def _iter_local_replays(input_dir: Path) -> Iterator[dict[str, Any]]:
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix == ".log":
            yield {
                "id": path.stem,
                "format": "gen9randombattle",
                "rating": None,
                "log": path.read_text(encoding="utf-8", errors="replace"),
            }
        elif path.suffix == ".html":
            html = path.read_text(encoding="utf-8", errors="replace")
            match = _HTML_LOG_RE.search(html)
            if match:
                yield {
                    "id": path.stem,
                    "format": "gen9randombattle",
                    "rating": None,
                    "log": match.group(1),
                }
        elif path.suffix == ".json":
            raw = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            if isinstance(raw, dict) and "log" in raw:
                yield raw
        elif path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                for idx, line in enumerate(fh):
                    line = line.strip()
                    if not line:
                        continue
                    raw = json.loads(line)
                    if isinstance(raw, dict) and "log" in raw:
                        raw.setdefault("id", f"{path.stem}-{idx}")
                        yield raw


def _iter_hf_replays(dataset_name: str, split: str) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset  # type: ignore[import-not-found]

    ds = load_dataset(dataset_name, split=split, streaming=True)
    yield from ds


def build_dataset(args: argparse.Namespace) -> dict[str, int]:
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and out_path.exists():
        out_path.unlink()

    if args.hf_dataset:
        stream = _iter_hf_replays(args.hf_dataset, args.hf_split)
    elif args.input_dir:
        stream = _iter_local_replays(Path(args.input_dir))
    else:
        raise ValueError("Specify either --hf-dataset or --input-dir.")

    scanned = 0
    matched_replays = 0
    used_replays = 0
    turns_written = 0
    t0 = time.perf_counter()

    with out_path.open("a", encoding="utf-8") as fh:
        for item in stream:
            scanned += 1
            fmt = str(item.get("format") or item.get("gamemode") or "")
            if fmt and not _matches_format(
                fmt, format_filter=args.format_filter, gen_filter=args.gen_filter
            ):
                continue

            raw_rating = item.get("rating")
            rating: int | None = None
            if raw_rating is not None:
                try:
                    rating = int(float(raw_rating))
                except (TypeError, ValueError):
                    rating = None

            if args.min_rating > 0 and (rating is None or rating < args.min_rating):
                continue

            log_text = item.get("log") or item.get("text")
            if not isinstance(log_text, str) or "|turn|" not in log_text:
                continue

            matched_replays += 1
            battle_id = str(item.get("id") or f"replay-{scanned}")
            recs = reconstruct_replay_both_sides(
                log_text,
                battle_id=battle_id,
                rating_override=rating,
                winner_only=args.winner_only,
                source=args.source_label,
                format_name=args.target_format,
                min_turns=args.min_turns,
            )
            if not recs:
                continue

            used_replays += 1
            for rec in recs:
                fh.write(json.dumps(rec.to_json_dict(), ensure_ascii=False) + "\n")
                turns_written += 1
                if args.max_turns and turns_written >= args.max_turns:
                    break

            if used_replays % 500 == 0:
                dt = max(time.perf_counter() - t0, 1e-6)
                print(
                    f"scanned={scanned:,} | matched={matched_replays:,} | used={used_replays:,} | "
                    f"turns={turns_written:,} ({turns_written / dt:.0f} turns/s)"
                )

            if args.max_turns and turns_written >= args.max_turns:
                break
            if args.max_replays and used_replays >= args.max_replays:
                break

    elapsed = time.perf_counter() - t0
    print(
        f"Done in {elapsed:.1f}s: scanned={scanned:,}, matched={matched_replays:,}, "
        f"used={used_replays:,}, wrote {turns_written:,} turns -> {out_path}"
    )
    return {
        "scanned": scanned,
        "matched_replays": matched_replays,
        "used_replays": used_replays,
        "turns_written": turns_written,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AlphaBattle SFT JSONL from Showdown replays.")
    parser.add_argument("--hf-dataset", default=None, help="Hugging Face dataset repo ID.")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--input-dir", default=None, help="Local directory of .log/.json/.jsonl replays.")
    parser.add_argument("--format-filter", default="randombattle")
    parser.add_argument("--gen-filter", type=int, default=9)
    parser.add_argument("--target-format", default="gen9randombattle")
    parser.add_argument("--min-rating", type=int, default=1550)
    parser.add_argument("--min-turns", type=int, default=8)
    parser.add_argument("--winner-only", action="store_true")
    parser.add_argument("--max-turns", type=int, default=250000)
    parser.add_argument("--max-replays", type=int, default=0)
    parser.add_argument("--source-label", default="human_replay")
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    build_dataset(parse_args())


if __name__ == "__main__":
    main()
