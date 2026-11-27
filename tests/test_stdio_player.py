"""Unit & integration tests for the persistent battle-ID-multiplexed JSONL stdio player."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from alphabattle.modeling import (
    AlphaBattlePolicyValueModel,
    attach_lora,
    resolve_action_token_ids,
)
from alphabattle.stdio_player import StdioBattleServer, build_stdio_server, parse_args, serve_stdio
from test_modeling_and_training import _build_tiny_qwen3_local


def _sample_request(rqid: int = 2, *, can_tera: bool = True, force_switch: bool = False) -> dict:
    side_pokemon = [
        {
            "ident": "p1: Garchomp",
            "details": "Garchomp, L74, M",
            "condition": "282/282" if not force_switch else "0 fnt",
            "active": True,
            "stats": {"atk": 225, "def": 165, "spa": 140, "spd": 150, "spe": 175},
            "moves": ["earthquake", "swordsdance"],
            "baseAbility": "roughskin",
            "item": "leftovers",
            "pokeball": "pokeball",
            "ability": "roughskin",
            "teraType": "Steel",
            "terastallized": "",
        },
        {
            "ident": "p1: Rotom",
            "details": "Rotom-Wash, L82",
            "condition": "216/216",
            "active": False,
            "stats": {"atk": 110, "def": 200, "spa": 195, "spd": 200, "spe": 165},
            "moves": ["hydropump", "voltswitch"],
            "baseAbility": "levitate",
            "item": "leftovers",
            "pokeball": "pokeball",
            "ability": "levitate",
            "teraType": "Electric",
            "terastallized": "",
        },
        {
            "ident": "p1: Scizor",
            "details": "Scizor, L79, M",
            "condition": "240/240",
            "active": False,
            "stats": {"atk": 230, "def": 180, "spa": 100, "spd": 150, "spe": 125},
            "moves": ["bulletpunch", "closecombat"],
            "baseAbility": "technician",
            "item": "choiceband",
            "pokeball": "pokeball",
            "ability": "technician",
            "teraType": "Steel",
            "terastallized": "",
        },
    ]
    req: dict = {
        "rqid": rqid,
        "side": {"name": "Cand", "id": "p1", "pokemon": side_pokemon},
    }
    if force_switch:
        req["forceSwitch"] = [True]
    else:
        active_entry: dict = {
            "moves": [
                {"move": "Earthquake", "id": "earthquake", "pp": 16, "maxpp": 16, "target": "allAdjacent", "disabled": False},
                {"move": "Swords Dance", "id": "swordsdance", "pp": 32, "maxpp": 32, "target": "self", "disabled": False},
            ]
        }
        if can_tera:
            active_entry["canTerastallize"] = "Steel"
        req["active"] = [active_entry]
    return req


@pytest.mark.asyncio
async def test_stdio_player_multiplexes_two_battles_and_handles_force_switch() -> None:
    server = build_stdio_server(parse_args(["--backend", "heuristics"]))

    frames = [
        {"type": "init", "battle_id": "b1", "role": "p1", "format": "gen9randombattle", "username": "Cand"},
        {"type": "init", "battle_id": "b2", "role": "p1", "format": "gen9randombattle", "username": "Cand"},
        {
            "type": "protocol",
            "battle_id": "b1",
            "lines": [
                "|switch|p1a: Garchomp|Garchomp, L74, M|282/282",
                "|switch|p2a: Corviknight|Corviknight, L80, F|100/100",
                "|turn|1",
            ],
        },
        {
            "type": "protocol",
            "battle_id": "b2",
            "lines": [
                "|switch|p1a: Garchomp|Garchomp, L74, M|282/282",
                "|switch|p2a: Heatran|Heatran, L79, M|100/100",
                "|turn|1",
            ],
        },
        {"type": "request", "battle_id": "b1", "request": _sample_request(rqid=2)},
        {"type": "request", "battle_id": "b2", "request": _sample_request(rqid=2, force_switch=True)},
        {"type": "end", "battle_id": "b1"},
        {"type": "end", "battle_id": "b2"},
        {"type": "quit"},
    ]
    in_buf = io.StringIO("\n".join(json.dumps(f) for f in frames) + "\n")
    out_buf = io.StringIO()

    await serve_stdio(server, in_buf, out_buf)

    replies = [json.loads(line) for line in out_buf.getvalue().splitlines() if line.strip()]
    assert replies[0]["type"] == "ready"
    choices = {r["battle_id"]: r for r in replies if r["type"] == "choice"}
    assert set(choices.keys()) == {"b1", "b2"}
    assert choices["b1"]["choice"].startswith(("move ", "switch "))
    assert choices["b2"]["choice"] in ("switch 2", "switch 3")
    assert server._slots == {}


@pytest.mark.asyncio
async def test_stdio_player_torch_and_eval_states_parity(tmp_path: Path) -> None:
    base_dir, _ = _build_tiny_qwen3_local(tmp_path)
    ckpt_dir = tmp_path / "ckpt"

    from alphabattle.modeling import load_base_lm_and_tokenizer

    lm, tok2 = load_base_lm_and_tokenizer(str(base_dir), device="cpu")
    lm = attach_lora(lm, r=4, lora_alpha=8, gradient_checkpointing=False)
    model = AlphaBattlePolicyValueModel(lm, resolve_action_token_ids(tok2), include_value_head=False)
    model.save_checkpoint(ckpt_dir, base_model_name=str(base_dir), tokenizer=tok2, max_length=256)

    server = build_stdio_server(
        parse_args(["--backend", "torch", "--checkpoint", str(ckpt_dir), "--device", "cpu"])
    )
    assert isinstance(server, StdioBattleServer)
    try:
        res = await server.handle_message(
            {
                "type": "eval_states",
                "request_id": "parity1",
                "states": [
                    {"state": "FORMAT: gen9randombattle | TURN: 1", "legal_ids": ["1", "2", "5"]},
                    {"state": "FORMAT: gen9randombattle | TURN: 2", "legal_ids": ["A", "5"]},
                ],
            }
        )
        assert res is not None and res["type"] == "eval_states_result"
        assert len(res["results"]) == 2
        assert abs(sum(res["results"][0]["probs"].values()) - 1.0) < 1e-5
        assert res["results"][0]["top1"] in {"1", "2", "5"}
        assert res["results"][1]["top1"] in {"A", "5"}
    finally:
        await server.aclose()


def test_stdio_player_subprocess_roundtrip() -> None:
    proc = subprocess.Popen(
        [sys.executable, "-m", "alphabattle.stdio_player", "--backend", "random", "--seed", "7"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdin is not None and proc.stdout is not None
    ready = json.loads(proc.stdout.readline())
    assert ready["type"] == "ready"
    assert ready["backend"] == "random"

    proc.stdin.write(json.dumps({"type": "ping"}) + "\n")
    proc.stdin.flush()
    pong = json.loads(proc.stdout.readline())
    assert pong["type"] == "pong"

    proc.stdin.write(json.dumps({"type": "quit"}) + "\n")
    proc.stdin.flush()
    proc.wait(timeout=10)
    assert proc.returncode == 0
