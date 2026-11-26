"""Async GPU micro-batched PyTorch inference backend implementing DecisionBackend.

Allows 16+ concurrent poke-env battles to share one GPU forward pass per ~8-15 ms window
with 100% legal single-token action masking (zero parsing ambiguity).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from alphabattle.backends import BackendDecision
from alphabattle.dataset import NUM_ACTIONS, tokenize_prompt_preserve_tail
from alphabattle.modeling import AlphaBattlePolicyValueModel, load_policy_value_checkpoint
from alphabattle.state import ACTION_ID_TO_INDEX, ALL_ACTION_IDS, ActionSpace


@dataclass
class _PendingRequest:
    state_text: str
    legal_ids: tuple[str, ...]
    future: asyncio.Future[BackendDecision]
    enqueued_at: float


class PyTorchPolicyBackend:
    """In-process batched PyTorch backend for Stage 1 SFT and Stage 2 RL checkpoints."""

    def __init__(
        self,
        model: AlphaBattlePolicyValueModel,
        tokenizer: Any,
        *,
        max_length: int = 1024,
        max_batch_size: int = 16,
        max_wait_ms: float = 8.0,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_batch_size = max(1, max_batch_size)
        self.max_wait_s = max(0.0, max_wait_ms / 1000.0)
        self.temperature = temperature
        self.device = next(self.model.parameters()).device

        self._queue: asyncio.Queue[_PendingRequest] | None = None
        self._worker_task: asyncio.Task[None] | None = None
        self._worker_loop: asyncio.AbstractEventLoop | None = None
        self._closed = False

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        *,
        base_model_override: str | None = None,
        device: str = "auto",
        qlora: bool = False,
        max_batch_size: int = 16,
        max_wait_ms: float = 8.0,
        temperature: float = 0.0,
    ) -> PyTorchPolicyBackend:
        model, tokenizer, meta = load_policy_value_checkpoint(
            checkpoint_dir,
            base_model_override=base_model_override,
            device=device,
            qlora=qlora,
            is_trainable=False,
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            max_length=int(meta.get("max_length", 1024)),
            max_batch_size=max_batch_size,
            max_wait_ms=max_wait_ms,
            temperature=temperature,
        )

    def _ensure_worker(self) -> asyncio.Queue[_PendingRequest]:
        loop = asyncio.get_running_loop()
        if (
            self._queue is None
            or self._worker_loop is not loop
            or self._worker_task is None
            or self._worker_task.done()
        ):
            self._closed = False
            self._worker_loop = loop
            self._queue = asyncio.Queue()
            self._worker_task = loop.create_task(self._batch_worker())
        return self._queue

    def _run_batch_sync(self, batch: list[_PendingRequest]) -> list[tuple[str, float]]:
        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else (self.tokenizer.eos_token_id or 0)
        )
        encoded = [
            tokenize_prompt_preserve_tail(
                self.tokenizer, req.state_text, max_length=self.max_length
            )
            for req in batch
        ]
        max_len = max(len(x) for x in encoded)
        rem = max_len % 8
        if rem != 0:
            max_len += 8 - rem

        bsz = len(batch)
        input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
        last_indices = torch.zeros((bsz,), dtype=torch.long)
        legal_mask = torch.zeros((bsz, NUM_ACTIONS), dtype=torch.bool)

        for i, (seq, req) in enumerate(zip(encoded, batch)):
            seq_len = len(seq)
            input_ids[i, :seq_len] = torch.tensor(seq, dtype=torch.long)
            attention_mask[i, :seq_len] = 1
            last_indices[i] = seq_len - 1
            for aid in req.legal_ids:
                idx = ACTION_ID_TO_INDEX.get(aid)
                if idx is not None:
                    legal_mask[i, idx] = True

        input_ids = input_ids.to(self.device, non_blocking=True)
        attention_mask = attention_mask.to(self.device, non_blocking=True)
        last_indices = last_indices.to(self.device, non_blocking=True)
        legal_mask = legal_mask.to(self.device, non_blocking=True)

        with torch.inference_mode():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                legal_mask=legal_mask,
                last_token_indices=last_indices,
                return_value=False,
            )
            logits = out.masked_logits
            if self.temperature <= 0.0:
                chosen_indices = torch.argmax(logits, dim=-1)
                probs = torch.softmax(logits, dim=-1)
            else:
                scaled = logits / self.temperature
                probs = torch.softmax(scaled, dim=-1)
                chosen_indices = torch.multinomial(probs, num_samples=1).squeeze(-1)

            chosen_probs = probs.gather(1, chosen_indices.unsqueeze(-1)).squeeze(-1)
            chosen_list = chosen_indices.cpu().tolist()
            prob_list = chosen_probs.cpu().tolist()

        return [(ALL_ACTION_IDS[idx], float(p)) for idx, p in zip(chosen_list, prob_list)]

    async def _batch_worker(self) -> None:
        assert self._queue is not None
        while not self._closed:
            try:
                first = await self._queue.get()
            except asyncio.CancelledError:
                return

            batch = [first]
            if self.max_batch_size > 1:
                deadline = time.perf_counter() + self.max_wait_s
                while len(batch) < self.max_batch_size:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0.0:
                        break
                    try:
                        nxt = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                        batch.append(nxt)
                    except asyncio.TimeoutError:
                        break
                    except asyncio.CancelledError:
                        return

            try:
                results = await asyncio.to_thread(self._run_batch_sync, batch)
                now = time.perf_counter()
                for req, (aid, prob) in zip(batch, results):
                    if not req.future.done():
                        req.future.set_result(
                            BackendDecision(
                                action_id=aid,
                                raw_output=f"{aid} (p={prob:.3f})",
                                latency_ms=(now - req.enqueued_at) * 1000.0,
                                used_fallback=False,
                            )
                        )
            except Exception as exc:
                for req in batch:
                    if not req.future.done():
                        req.future.set_exception(exc)

    async def choose_action(
        self, state_text: str, action_space: ActionSpace
    ) -> BackendDecision:
        queue = self._ensure_worker()
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[BackendDecision] = loop.create_future()
        await queue.put(
            _PendingRequest(
                state_text=state_text,
                legal_ids=tuple(action_space.ids),
                future=fut,
                enqueued_at=time.perf_counter(),
            )
        )
        return await fut

    async def aclose(self) -> None:
        self._closed = True
        task = self._worker_task
        worker_loop = self._worker_loop
        self._worker_task = None
        self._worker_loop = None
        self._queue = None
        if task is None or task.done():
            return
        try:
            cur_loop = asyncio.get_running_loop()
        except RuntimeError:
            cur_loop = None
        if worker_loop is not None and worker_loop is cur_loop:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        elif worker_loop is not None and worker_loop.is_running():
            worker_loop.call_soon_threadsafe(task.cancel)
