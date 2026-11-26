"""Model backends for AlphaBattle.

Supports:
- OpenAICompatBackend: native Windows local inference via llama-server (llama.cpp),
  Ollama, LM Studio, or any OpenAI-compatible API, with strict legal-action
  constraining (JSON Schema enum or llama.cpp GBNF single-token grammar).
- RandomLegalBackend / FirstLegalBackend: deterministic/fast zero-GPU backends for
  testing the Showdown harness without a GPU.
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Literal, Protocol, Sequence

import httpx

from alphabattle.state import ActionSpace

PromptMode = Literal["direct", "cot"]
ConstrainMode = Literal["json_schema", "gbnf", "none"]

SYSTEM_PROMPT_DIRECT = (
    "You are an expert competitive Pokémon Showdown player (Gen 9 Singles). "
    "Analyze the battle state and LEGAL_ACTIONS, then choose the single best legal action ID. "
    "Consider type matchups, STAB, speed order, boosts, status, hazards, HP preservation, "
    "and Terastallization timing. Do not panic-switch unnecessarily."
)

SYSTEM_PROMPT_COT = (
    "You are an expert competitive Pokémon Showdown player (Gen 9 Singles). "
    "Briefly reason (under 60 words) about speed order, damage/KO thresholds, switches, "
    "and Tera timing, then select the single best legal action ID from LEGAL_ACTIONS."
)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_JSON_OBJ_RE = re.compile(r"\{.*\}", flags=re.DOTALL)
_ACTION_LABEL_RE = re.compile(
    r"(?:action|choice|move)\s*[:=]\s*[\"'`\[]*([1-9A-D])[\"'`\]]*",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class BackendDecision:
    """Result of one backend decision call."""

    action_id: str
    raw_output: str
    latency_ms: float
    used_fallback: bool = False
    reasoning: str | None = None


class DecisionBackend(Protocol):
    """Protocol implemented by every local/remote inference backend (Stage 0-2)."""

    async def choose_action(
        self, state_text: str, action_space: ActionSpace
    ) -> BackendDecision: ...

    async def aclose(self) -> None: ...


def build_gbnf_grammar(legal_ids: Sequence[str]) -> str:
    """Build a llama.cpp GBNF grammar constraining output to exactly one legal action ID."""
    alts = " | ".join(f'"{aid}"' for aid in legal_ids)
    return f"root ::= ({alts})\n"


def build_response_schema(legal_ids: Sequence[str], mode: PromptMode) -> dict[str, Any]:
    """Build an OpenAI-compatible strict JSON Schema constraining 'action' to legal_ids."""
    if mode == "cot":
        properties: dict[str, Any] = {
            "reasoning": {
                "type": "string",
                "description": "Concise tactical reasoning (<= 60 words).",
            },
            "action": {
                "type": "string",
                "enum": list(legal_ids),
                "description": "Exactly one legal action ID from LEGAL_ACTIONS.",
            },
        }
        required = ["reasoning", "action"]
    else:
        properties = {
            "action": {
                "type": "string",
                "enum": list(legal_ids),
                "description": "Exactly one legal action ID from LEGAL_ACTIONS.",
            }
        }
        required = ["action"]

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "pokemon_battle_action",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def parse_action_from_text(
    raw_text: str, legal_ids: Sequence[str]
) -> tuple[str | None, str | None]:
    """Extract (action_id, optional_reasoning) strictly validated against legal_ids.

    Returns (None, None) if no unambiguous legal action ID can be recovered.
    """
    legal_set = set(legal_ids)
    cleaned = _THINK_BLOCK_RE.sub("", raw_text).strip()
    if not cleaned:
        return (None, None)

    # 1. Exact single-token match (e.g. GBNF grammar output)
    stripped_brackets = cleaned.strip("[]\"'` \n\r\t.")
    if stripped_brackets in legal_set:
        return (stripped_brackets, None)

    # 2. Strict JSON object parse
    json_candidate = cleaned
    if not json_candidate.startswith("{"):
        match = _JSON_OBJ_RE.search(cleaned)
        if match:
            json_candidate = match.group(0)
    try:
        parsed = json.loads(json_candidate)
        if isinstance(parsed, dict):
            aid = str(parsed.get("action", "")).strip().strip("[]")
            reasoning = parsed.get("reasoning")
            reasoning_str = str(reasoning) if reasoning is not None else None
            if aid in legal_set:
                return (aid, reasoning_str)
    except json.JSONDecodeError:
        pass

    # 3. Labeled regex fallback (e.g., "Action: [A]" or "action=1")
    matches = _ACTION_LABEL_RE.findall(cleaned)
    if matches:
        candidate = matches[-1].upper()
        if candidate in legal_set:
            return (candidate, None)

    return (None, None)


class OpenAICompatBackend:
    """Async client for local llama-server, Ollama, LM Studio, or OpenAI-compatible endpoints."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8080/v1",
        model: str = "qwen3-4b",
        api_key: str = "local-no-key-needed",
        mode: PromptMode = "direct",
        constrain: ConstrainMode = "json_schema",
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout_s: float = 30.0,
        qwen3_no_think: bool = True,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.mode = mode
        self.constrain = constrain
        self.temperature = temperature
        self.max_tokens = max_tokens if max_tokens is not None else (160 if mode == "cot" else 24)
        self.qwen3_no_think = qwen3_no_think
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_s),
            headers={"Authorization": f"Bearer {api_key}"},
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
        )

    def _build_payload(self, state_text: str, legal_ids: Sequence[str]) -> dict[str, Any]:
        sys_prompt = SYSTEM_PROMPT_COT if self.mode == "cot" else SYSTEM_PROMPT_DIRECT
        if self.mode == "direct" and self.qwen3_no_think:
            # Qwen3 hybrid-thinking models respect '/no_think' cleanly across Ollama & llama.cpp
            # without requiring non-standard top-level JSON parameters.
            sys_prompt = f"{sys_prompt} /no_think"

        if self.constrain == "gbnf":
            user_prompt = (
                f"{state_text}\n\n"
                f"Respond with ONLY the single legal action ID from {{{', '.join(legal_ids)}}}."
            )
        elif self.mode == "cot":
            user_prompt = (
                f"{state_text}\n\n"
                f'Respond with JSON: {{"reasoning": "<brief analysis>", '
                f'"action": "<one of: {", ".join(legal_ids)}>"}}.'
            )
        else:
            user_prompt = (
                f"{state_text}\n\n"
                f'Respond with JSON: {{"action": "<one of: {", ".join(legal_ids)}>"}}.'
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": 4 if self.constrain == "gbnf" else self.max_tokens,
        }

        if self.constrain == "json_schema":
            payload["response_format"] = build_response_schema(legal_ids, self.mode)
        elif self.constrain == "gbnf":
            payload["grammar"] = build_gbnf_grammar(legal_ids)

        return payload

    async def choose_action(
        self, state_text: str, action_space: ActionSpace
    ) -> BackendDecision:
        legal_ids = action_space.ids
        payload = self._build_payload(state_text, legal_ids)
        t0 = time.perf_counter()
        raw_content = ""
        try:
            resp = await self._client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
            raw_content = str(data["choices"][0]["message"].get("content") or "")
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return BackendDecision(
                action_id=legal_ids[0],
                raw_output=f"<HTTP_ERROR: {exc!r}>",
                latency_ms=latency_ms,
                used_fallback=True,
            )

        latency_ms = (time.perf_counter() - t0) * 1000.0
        action_id, reasoning = parse_action_from_text(raw_content, legal_ids)
        if action_id is None:
            return BackendDecision(
                action_id=legal_ids[0],
                raw_output=raw_content,
                latency_ms=latency_ms,
                used_fallback=True,
            )
        return BackendDecision(
            action_id=action_id,
            raw_output=raw_content,
            latency_ms=latency_ms,
            used_fallback=False,
            reasoning=reasoning,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class RandomLegalBackend:
    """Zero-GPU random baseline over the canonical ActionSpace (useful for pipeline smoke tests)."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    async def choose_action(
        self, state_text: str, action_space: ActionSpace
    ) -> BackendDecision:
        choice = self._rng.choice(action_space.ids)
        return BackendDecision(action_id=choice, raw_output=choice, latency_ms=0.0)

    async def aclose(self) -> None:
        return None


class FirstLegalBackend:
    """Deterministic zero-GPU backend picking the first legal action (for unit/smoke tests)."""

    async def choose_action(
        self, state_text: str, action_space: ActionSpace
    ) -> BackendDecision:
        choice = action_space.ids[0]
        return BackendDecision(action_id=choice, raw_output=choice, latency_ms=0.0)

    async def aclose(self) -> None:
        return None
