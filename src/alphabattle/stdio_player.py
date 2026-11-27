"""Persistent, battle-ID-multiplexed JSONL stdio player for the sealed BattleStream judge.

Protocol (newline-delimited JSON on stdin -> stdout; all logs/warnings go strictly to stderr):
  Startup handshake (player -> judge):
    {"type": "ready", "protocol_version": 1, "backend": "torch", "checkpoint": "...", "device": "..."}

  Incoming messages (judge -> player):
    {"type": "init", "battle_id": "b1", "role": "p1", "format": "gen9randombattle", "username": "P1"}
    {"type": "protocol", "battle_id": "b1", "lines": ["|switch|...", "|turn|1", ...]}
    {"type": "request", "battle_id": "b1", "request": {...}}
    {"type": "eval_states", "request_id": "e1", "states": [{"state": "...", "legal_ids": ["1","2"]}]}
    {"type": "end", "battle_id": "b1"}
    {"type": "quit"}

  Outgoing responses (player -> judge):
    {"type": "choice", "battle_id": "b1", "rqid": 2, "choice": "move 1", "action_id": "1",
     "latency_ms": 12.3, "used_fallback": false}
    {"type": "noop", "battle_id": "b1", "rqid": 2}
    {"type": "eval_states_result", "request_id": "e1", "results": [{"probs": {...}, "top1": "1"}]}
    {"type": "error", "battle_id": "b1", "message": "..."}

Why persistent & battle-ID-multiplexed?
Loading Qwen3-1.7B-Base once per SPRT and multiplexing 8-16 concurrent battles over one
stdin/stdout pipe avoids per-game model reload overhead and enables GPU micro-batching (5-10x faster).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, TextIO

from poke_env.battle.battle import Battle
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.data import to_id_str
from poke_env.player.baselines import MaxBasePowerPlayer, SimpleHeuristicsPlayer
from poke_env.player.battle_order import BattleOrder, SingleBattleOrder
from poke_env.player.player import Player

from alphabattle.backends import (
    BackendDecision,
    DecisionBackend,
    FirstLegalBackend,
    RandomLegalBackend,
)
from alphabattle.state import (
    ALL_ACTION_IDS,
    TurnHistoryTracker,
    build_action_space,
    format_battle_state,
    match_order_to_action_id,
)

PROTOCOL_VERSION = 1

_IGNORED_PROTOCOL_EVENTS = set(Player.MESSAGES_TO_IGNORE) | {
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


@dataclass
class _BattleSlot:
    battle: Battle
    role: str
    history: TurnHistoryTracker = field(default_factory=lambda: TurnHistoryTracker(window=3))
    last_request: dict[str, Any] | None = None


def _parse_gen_from_format(battle_format: str) -> int:
    fmt = battle_format.lower()
    if fmt.startswith("gen") and len(fmt) >= 4 and fmt[3].isdigit():
        return int(fmt[3])
    return 9


def _safe_parse_protocol_line(battle: Battle, raw_line: str) -> None:
    line = raw_line.strip()
    if not line.startswith("|"):
        return
    parts = line.split("|")
    if len(parts) < 2:
        return
    ev = parts[1]
    if ev == "turn" and len(parts) >= 3 and parts[2].isdigit():
        battle._turn = int(parts[2])
        return
    if ev in _IGNORED_PROTOCOL_EVENTS:
        return
    try:
        battle.parse_message(parts)
    except (NotImplementedError, KeyError, ValueError, AssertionError, IndexError):
        return


def order_to_stream_choice(
    order: BattleOrder,
    battle: Battle,
    request: dict[str, Any],
) -> str:
    """Convert a poke-env BattleOrder into an unambiguous Showdown BattleStream choice string."""
    if not isinstance(order, SingleBattleOrder):
        msg = getattr(order, "message", "default")
        if isinstance(msg, str) and msg.startswith("/choose "):
            return msg[len("/choose ") :]
        return "default"

    inner = order.order
    if isinstance(inner, Move):
        if inner.id in ("recharge", "struggle"):
            return "move 1"
        active_list = request.get("active") or []
        moves_req = active_list[0].get("moves", []) if active_list else []
        chosen_slot: int | None = None
        for idx, mdict in enumerate(moves_req, start=1):
            if mdict.get("disabled"):
                continue
            mid = to_id_str(str(mdict.get("id") or mdict.get("move") or ""))
            if mid == inner.id:
                chosen_slot = idx
                break
        if chosen_slot is None:
            # Fallback to first non-disabled move if request used an alias (e.g. Hidden Power)
            for idx, mdict in enumerate(moves_req, start=1):
                if not mdict.get("disabled"):
                    chosen_slot = idx
                    break
        if chosen_slot is None:
            return "default"
        cmd = f"move {chosen_slot}"
        if order.terastallize and active_list and active_list[0].get("canTerastallize"):
            cmd += " terastallize"
        elif order.mega and active_list and active_list[0].get("canMegaEvo"):
            cmd += " mega"
        elif order.z_move:
            cmd += " zmove"
        elif order.dynamax:
            cmd += " dynamax"
        return cmd

    if isinstance(inner, Pokemon):
        side_mons = request.get("side", {}).get("pokemon", [])
        role = battle.player_role or ""
        target_ident = f"{role}: {inner.name}" if role else inner.name
        reviving = bool(battle.reviving)

        def _eligible(pdict: dict[str, Any]) -> bool:
            cond = str(pdict.get("condition", ""))
            is_fnt = cond.endswith("fnt") or cond == "0 fnt"
            if reviving:
                return is_fnt
            return (not bool(pdict.get("active"))) and (not is_fnt)

        # Pass 1: exact ident match
        for idx, pdict in enumerate(side_mons, start=1):
            if _eligible(pdict) and str(pdict.get("ident", "")) == target_ident:
                return f"switch {idx}"
        # Pass 2: species match from details ("Species, L80, M")
        target_species = to_id_str(inner.species)
        for idx, pdict in enumerate(side_mons, start=1):
            if not _eligible(pdict):
                continue
            details = str(pdict.get("details", "")).split(",", 1)[0]
            if to_id_str(details) == target_species:
                return f"switch {idx}"
        # Pass 3: first legal switch slot
        for idx, pdict in enumerate(side_mons, start=1):
            if _eligible(pdict):
                return f"switch {idx}"
        return "default"

    msg = order.message
    if isinstance(msg, str) and msg.startswith("/choose "):
        return msg[len("/choose ") :]
    return str(msg) if msg else "default"


class StdioBattleServer:
    """Async JSONL battle-ID-multiplexed handler sharing one loaded backend/model."""

    def __init__(
        self,
        *,
        backend_kind: str,
        decision_backend: DecisionBackend | None = None,
        offline_player: Player | None = None,
        checkpoint: str | None = None,
        device_label: str = "cpu",
    ) -> None:
        self.backend_kind = backend_kind
        self.decision_backend = decision_backend
        self.offline_player = offline_player
        self.checkpoint = checkpoint
        self.device_label = device_label
        self._slots: dict[str, _BattleSlot] = {}
        self._battle_locks: dict[str, asyncio.Lock] = {}

    def _get_lock(self, battle_id: str) -> asyncio.Lock:
        lock = self._battle_locks.get(battle_id)
        if lock is None:
            lock = asyncio.Lock()
            self._battle_locks[battle_id] = lock
        return lock

    def _ensure_slot(
        self,
        battle_id: str,
        *,
        role: str = "p1",
        battle_format: str = "gen9randombattle",
        username: str = "AlphaBattle",
    ) -> _BattleSlot:
        slot = self._slots.get(battle_id)
        if slot is None:
            gen = _parse_gen_from_format(battle_format)
            battle = Battle(battle_id, username, None, gen=gen)
            battle._player_role = role
            battle._opponent_role = "p2" if role == "p1" else "p1"
            battle._format = battle_format
            slot = _BattleSlot(battle=battle, role=role)
            self._slots[battle_id] = slot
        return slot

    async def handle_message(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        mtype = msg.get("type")
        if mtype == "ping":
            return {"type": "pong"}

        if mtype == "init":
            bid = str(msg["battle_id"])
            role = str(msg.get("role", "p1"))
            fmt = str(msg.get("format", "gen9randombattle"))
            uname = str(msg.get("username", f"Player_{role}"))
            async with self._get_lock(bid):
                # Reset any prior slot with the same ID
                self._slots.pop(bid, None)
                self._ensure_slot(bid, role=role, battle_format=fmt, username=uname)
            return {"type": "ack", "battle_id": bid}

        if mtype == "protocol":
            bid = str(msg["battle_id"])
            lines = msg.get("lines") or []
            async with self._get_lock(bid):
                slot = self._ensure_slot(bid)
                for ln in lines:
                    if isinstance(ln, str):
                        _safe_parse_protocol_line(slot.battle, ln)
            return None

        if mtype == "end":
            bid = str(msg["battle_id"])
            async with self._get_lock(bid):
                self._slots.pop(bid, None)
            self._battle_locks.pop(bid, None)
            return None

        if mtype == "eval_states":
            return await self._handle_eval_states(msg)

        if mtype == "request":
            bid = str(msg["battle_id"])
            req = msg.get("request")
            if not isinstance(req, dict):
                return {"type": "error", "battle_id": bid, "message": "Missing request dict"}
            async with self._get_lock(bid):
                return await self._handle_request(bid, req)

        return {"type": "error", "message": f"Unknown message type: {mtype!r}"}

    async def _handle_request(self, battle_id: str, req: dict[str, Any]) -> dict[str, Any]:
        slot = self._ensure_slot(battle_id)
        battle = slot.battle
        rqid = req.get("rqid")
        slot.last_request = req

        battle.parse_request(req)
        if battle.wait or req.get("wait"):
            return {"type": "noop", "battle_id": battle_id, "rqid": rqid}

        if battle.teampreview or req.get("teamPreview"):
            return {
                "type": "choice",
                "battle_id": battle_id,
                "rqid": rqid,
                "choice": "default",
                "action_id": "teampreview",
                "latency_ms": 0.0,
                "used_fallback": False,
            }

        t0 = time.perf_counter()
        action_space = build_action_space(battle)
        if action_space.is_empty:
            return {
                "type": "choice",
                "battle_id": battle_id,
                "rqid": rqid,
                "choice": "default",
                "action_id": "default",
                "latency_ms": 0.0,
                "used_fallback": True,
            }

        hist = slot.history.update_before_turn(battle)
        if action_space.is_forced_single:
            only = action_space.actions[0]
            slot.history.record_action(battle, action_space, only.action_id)
            choice_str = order_to_stream_choice(only.order, battle, req)
            return {
                "type": "choice",
                "battle_id": battle_id,
                "rqid": rqid,
                "choice": choice_str,
                "action_id": only.action_id,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                "used_fallback": False,
            }

        if self.offline_player is not None:
            order = self.offline_player.choose_move(battle)
            if asyncio.iscoroutine(order):
                order = await order
            aid = match_order_to_action_id(order, action_space) or action_space.ids[0]
            slot.history.record_action(battle, action_space, aid)
            choice_str = order_to_stream_choice(order, battle, req)
            return {
                "type": "choice",
                "battle_id": battle_id,
                "rqid": rqid,
                "choice": choice_str,
                "action_id": aid,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                "used_fallback": False,
            }

        assert self.decision_backend is not None
        state_text = format_battle_state(battle, action_space, recent_history=hist)
        try:
            decision: BackendDecision = await self.decision_backend.choose_action(
                state_text, action_space
            )
            chosen_id = decision.action_id
            used_fallback = bool(decision.used_fallback)
            if chosen_id not in action_space.by_id:
                chosen_id = action_space.ids[0]
                used_fallback = True
        except Exception:
            chosen_id = action_space.ids[0]
            used_fallback = True

        chosen_act = action_space.by_id[chosen_id]
        slot.history.record_action(battle, action_space, chosen_id)
        choice_str = order_to_stream_choice(chosen_act.order, battle, req)
        return {
            "type": "choice",
            "battle_id": battle_id,
            "rqid": rqid,
            "choice": choice_str,
            "action_id": chosen_id,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
            "used_fallback": used_fallback,
        }

    async def _handle_eval_states(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Score fixed battle states with exact legal action probability distributions.

        Used by the sealed judge's Speedup-Pay probability parity gate (mean KL < 1e-5,
        top-1 agreement >= 99.8% across 2,000 fixed states).
        """
        req_id = msg.get("request_id", "eval")
        states = msg.get("states") or []
        if self.backend_kind != "torch" or self.decision_backend is None:
            return {
                "type": "error",
                "request_id": req_id,
                "message": "eval_states requires --backend torch",
            }

        from alphabattle.dataset import NUM_ACTIONS, tokenize_prompt_preserve_tail
        from alphabattle.state import ACTION_ID_TO_INDEX
        from alphabattle.torch_backend import PyTorchPolicyBackend

        assert isinstance(self.decision_backend, PyTorchPolicyBackend)
        backend = self.decision_backend

        def _score_batch() -> list[dict[str, Any]]:
            import torch

            results: list[dict[str, Any]] = []
            chunk_size = max(1, backend.max_batch_size)
            pad_id = (
                backend.tokenizer.pad_token_id
                if backend.tokenizer.pad_token_id is not None
                else (backend.tokenizer.eos_token_id or 0)
            )
            for start in range(0, len(states), chunk_size):
                chunk = states[start : start + chunk_size]
                encoded = [
                    tokenize_prompt_preserve_tail(
                        backend.tokenizer, str(item["state"]), max_length=backend.max_length
                    )
                    for item in chunk
                ]
                max_len = max(len(x) for x in encoded)
                rem = max_len % 8
                if rem != 0:
                    max_len += 8 - rem

                bsz = len(chunk)
                input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
                attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
                last_indices = torch.zeros((bsz,), dtype=torch.long)
                legal_mask = torch.zeros((bsz, NUM_ACTIONS), dtype=torch.bool)

                for i, (seq, item) in enumerate(zip(encoded, chunk)):
                    seq_len = len(seq)
                    input_ids[i, :seq_len] = torch.tensor(seq, dtype=torch.long)
                    attention_mask[i, :seq_len] = 1
                    last_indices[i] = seq_len - 1
                    for aid in item["legal_ids"]:
                        idx = ACTION_ID_TO_INDEX.get(str(aid))
                        if idx is not None:
                            legal_mask[i, idx] = True

                with torch.inference_mode():
                    out = backend.model(
                        input_ids=input_ids.to(backend.device),
                        attention_mask=attention_mask.to(backend.device),
                        legal_mask=legal_mask.to(backend.device),
                        last_token_indices=last_indices.to(backend.device),
                        return_value=False,
                    )
                    logits = out.masked_logits.masked_fill(~legal_mask.to(backend.device), float("-inf"))
                    probs = torch.softmax(logits, dim=-1).cpu().tolist()
                    top1_idxs = torch.argmax(out.masked_logits, dim=-1).cpu().tolist()

                for i, item in enumerate(chunk):
                    prob_map = {
                        str(aid): float(probs[i][ACTION_ID_TO_INDEX[str(aid)]])
                        for aid in item["legal_ids"]
                        if str(aid) in ACTION_ID_TO_INDEX
                    }
                    results.append(
                        {
                            "top1": ALL_ACTION_IDS[top1_idxs[i]],
                            "probs": prob_map,
                        }
                    )
            return results

        scored = await asyncio.to_thread(_score_batch)
        return {
            "type": "eval_states_result",
            "request_id": req_id,
            "results": scored,
        }

    async def aclose(self) -> None:
        if self.decision_backend is not None:
            await self.decision_backend.aclose()


def build_stdio_server(args: argparse.Namespace) -> StdioBattleServer:
    if args.backend == "random":
        return StdioBattleServer(
            backend_kind="random",
            decision_backend=RandomLegalBackend(seed=args.seed),
            device_label="cpu",
        )
    if args.backend == "first":
        return StdioBattleServer(
            backend_kind="first",
            decision_backend=FirstLegalBackend(),
            device_label="cpu",
        )
    if args.backend == "heuristics":
        player = SimpleHeuristicsPlayer(
            battle_format=args.format,
            start_listening=False,
        )
        return StdioBattleServer(
            backend_kind="heuristics",
            offline_player=player,
            device_label="cpu",
        )
    if args.backend == "max_power":
        player = MaxBasePowerPlayer(
            battle_format=args.format,
            start_listening=False,
        )
        return StdioBattleServer(
            backend_kind="max_power",
            offline_player=player,
            device_label="cpu",
        )
    if args.backend == "torch":
        if not args.checkpoint:
            raise ValueError("--backend torch requires --checkpoint <dir>.")
        from alphabattle.torch_backend import PyTorchPolicyBackend

        backend = PyTorchPolicyBackend.from_checkpoint(
            args.checkpoint,
            base_model_override=args.base_model,
            device=args.device,
            qlora=args.qlora,
            max_batch_size=args.max_batch_size,
            max_wait_ms=args.max_wait_ms,
            temperature=args.temperature,
        )
        return StdioBattleServer(
            backend_kind="torch",
            decision_backend=backend,
            checkpoint=str(args.checkpoint),
            device_label=str(backend.device),
        )
    raise ValueError(f"Unsupported backend: {args.backend}")


async def serve_stdio(
    server: StdioBattleServer,
    in_stream: TextIO,
    out_stream: TextIO,
) -> None:
    """Read JSONL lines from `in_stream`, multiplex concurrently, and write replies to `out_stream`."""
    loop = asyncio.get_running_loop()
    write_lock = asyncio.Lock()

    async def _emit(payload: dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        async with write_lock:
            out_stream.write(line)
            out_stream.flush()

    await _emit(
        {
            "type": "ready",
            "protocol_version": PROTOCOL_VERSION,
            "backend": server.backend_kind,
            "checkpoint": server.checkpoint,
            "device": server.device_label,
        }
    )

    in_queue: asyncio.Queue[str | None] = asyncio.Queue()

    def _stdin_reader() -> None:
        try:
            for raw in in_stream:
                loop.call_soon_threadsafe(in_queue.put_nowait, raw)
        finally:
            loop.call_soon_threadsafe(in_queue.put_nowait, None)

    reader_thread = threading.Thread(target=_stdin_reader, daemon=True)
    reader_thread.start()

    pending_tasks: set[asyncio.Task[None]] = set()
    last_task_by_battle: dict[str, asyncio.Task[None]] = {}

    async def _dispatch(raw_msg: dict[str, Any]) -> None:
        try:
            reply = await server.handle_message(raw_msg)
            if reply is not None:
                await _emit(reply)
        except Exception as exc:
            await _emit(
                {
                    "type": "error",
                    "battle_id": raw_msg.get("battle_id"),
                    "message": repr(exc),
                }
            )

    async def _run_chained(
        prev: asyncio.Task[None] | None,
        raw_msg: dict[str, Any],
        bid: str | None,
    ) -> None:
        if prev is not None and not prev.done():
            try:
                await prev
            except Exception:
                pass
        await _dispatch(raw_msg)
        if bid is not None and raw_msg.get("type") == "end":
            cur = asyncio.current_task()
            if last_task_by_battle.get(bid) is cur:
                last_task_by_battle.pop(bid, None)

    try:
        while True:
            raw_line = await in_queue.get()
            if raw_line is None:
                break
            line = raw_line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError as exc:
                await _emit({"type": "error", "message": f"Invalid JSON: {exc}"})
                continue

            if not isinstance(msg, dict):
                await _emit({"type": "error", "message": "Expected JSON object"})
                continue

            mtype = msg.get("type")
            if mtype == "quit":
                break

            bid_val = msg.get("battle_id")
            bid = str(bid_val) if bid_val is not None else None
            prev = last_task_by_battle.get(bid) if bid is not None else None
            task = asyncio.create_task(_run_chained(prev, msg, bid))
            if bid is not None:
                last_task_by_battle[bid] = task
            pending_tasks.add(task)
            task.add_done_callback(pending_tasks.discard)

        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)
    finally:
        await server.aclose()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Persistent battle-ID-multiplexed JSONL stdio player for BattleStream judging."
    )
    parser.add_argument(
        "--backend",
        choices=["torch", "heuristics", "max_power", "random", "first"],
        default="torch",
    )
    parser.add_argument("--checkpoint", default=None, help="Path to AlphaBattle checkpoint dir.")
    parser.add_argument("--base-model", default=None, help="Optional base HF model override.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument("--format", default="gen9randombattle")
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--max-wait-ms", type=float, default=4.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    # Reserve real stdout exclusively for JSONL protocol frames so library logs/warnings never corrupt it
    protocol_out = sys.stdout
    sys.stdout = sys.stderr
    args = parse_args(argv)
    server = build_stdio_server(args)
    asyncio.run(serve_stdio(server, sys.stdin, protocol_out))


if __name__ == "__main__":
    main()
