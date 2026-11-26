"""Unit tests for constrained decoding payloads, action parsing, and OpenAICompatBackend."""

from __future__ import annotations

import json

import httpx
import pytest
from poke_env.battle.move import Move
from poke_env.player.battle_order import SingleBattleOrder

from alphabattle.backends import (
    OpenAICompatBackend,
    build_gbnf_grammar,
    build_response_schema,
    parse_action_from_text,
)
from alphabattle.state import ActionSpace, LegalAction


def _sample_action_space() -> ActionSpace:
    actions = (
        LegalAction("1", "move", "MOVE earthquake", SingleBattleOrder(Move("earthquake", gen=9))),
        LegalAction("A", "tera_move", "TERA+MOVE earthquake", SingleBattleOrder(Move("earthquake", gen=9), terastallize=True)),
        LegalAction("5", "switch", "SWITCH rotomwash", SingleBattleOrder("/choose switch rotomwash")),
    )
    return ActionSpace(actions=actions, by_id={a.action_id: a for a in actions})


def test_parse_action_from_text_variants() -> None:
    legal = ["1", "A", "5"]

    assert parse_action_from_text("A", legal) == ("A", None)
    assert parse_action_from_text("[5]", legal) == ("5", None)
    assert parse_action_from_text('{"action": "1"}', legal) == ("1", None)
    assert parse_action_from_text(
        '<think>Let me check matchups...</think>\n{"reasoning": "SE hit", "action": "A"}',
        legal,
    ) == ("A", "SE hit")
    assert parse_action_from_text("Best play here.\nAction: [5]", legal) == ("5", None)
    # Illegal action ID must be rejected
    assert parse_action_from_text('{"action": "2"}', legal) == (None, None)
    assert parse_action_from_text("I choose thunderbolt", legal) == (None, None)


def test_gbnf_and_schema_builders() -> None:
    g = build_gbnf_grammar(["1", "A", "5"])
    assert g == 'root ::= ("1" | "A" | "5")\n'

    schema = build_response_schema(["1", "A", "5"], mode="cot")
    props = schema["json_schema"]["schema"]["properties"]
    assert list(props.keys()) == ["reasoning", "action"]
    assert props["action"]["enum"] == ["1", "A", "5"]


@pytest.mark.asyncio
async def test_openai_compat_backend_json_and_fallback() -> None:
    space = _sample_action_space()

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["response_format"]["json_schema"]["schema"]["properties"]["action"]["enum"] == [
            "1",
            "A",
            "5",
        ]
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"action": "A"}'}}]},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(base_url="http://mock/v1", transport=transport) as client:
        backend = OpenAICompatBackend(
            base_url="http://mock/v1",
            model="qwen3-4b",
            mode="direct",
            constrain="json_schema",
            client=client,
        )
        decision = await backend.choose_action("STATE", space)
        assert decision.action_id == "A"
        assert not decision.used_fallback

    # Simulate server error -> safe legal fallback flagged explicitly
    def error_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server error")

    err_transport = httpx.MockTransport(error_handler)
    async with httpx.AsyncClient(base_url="http://mock/v1", transport=err_transport) as client:
        backend = OpenAICompatBackend(
            base_url="http://mock/v1",
            model="qwen3-4b",
            client=client,
        )
        decision = await backend.choose_action("STATE", space)
        assert decision.action_id == "1"
        assert decision.used_fallback
