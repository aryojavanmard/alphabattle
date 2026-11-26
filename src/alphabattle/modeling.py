"""PyTorch Masked-Action Policy + Value Head wrapper over pretrained causal LLMs.

Key 12GB VRAM optimization:
Instead of projecting all (B, L) hidden states through the 152K-token lm_head (which wastes
2-4 GB VRAM on intermediate logits/gradients for tokens we discard), we run the transformer
backbone to get hidden states, slice ONLY the final real token h_last (B, H), and multiply
ONLY against the 13 canonical action rows of lm_head.weight: (B, H) @ (13, H)^T -> (B, 13).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from alphabattle.state import ALL_ACTION_IDS, POLICY_PROMPT_SUFFIX

META_FILENAME = "alphabattle_meta.json"
VALUE_HEAD_FILENAME = "value_head.pt"
ADAPTER_DIRNAME = "adapter"


@dataclass(frozen=True)
class PolicyValueOutput:
    """Outputs from one batched forward pass of AlphaBattlePolicyValueModel."""

    raw_action_logits: torch.Tensor  # (B, 13), float32
    masked_logits: torch.Tensor  # (B, 13), float32 with illegal entries set to -1e9
    value_logit: torch.Tensor | None  # (B,), float32 win-probability logit (sigmoid -> p_win)
    last_hidden: torch.Tensor  # (B, H), model dtype


def resolve_action_token_ids(tokenizer: Any) -> list[int]:
    """Verify every canonical action ID maps cleanly to 1 token without merging with suffix."""
    suffix_ids = tokenizer.encode(POLICY_PROMPT_SUFFIX, add_special_tokens=False)
    token_ids: list[int] = []
    seen: set[int] = set()

    for aid in ALL_ACTION_IDS:
        ids = tokenizer.encode(aid, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"Action ID '{aid}' tokenized into {len(ids)} tokens ({ids}) instead of 1."
            )
        tid = int(ids[0])
        if tid in seen:
            raise ValueError(f"Duplicate token ID {tid} for action '{aid}'.")
        # Sanity check: appending aid to POLICY_PROMPT_SUFFIX must keep tid as the exact last token
        combined = tokenizer.encode(f"{POLICY_PROMPT_SUFFIX}{aid}", add_special_tokens=False)
        if not combined or combined[-1] != tid or len(combined) != len(suffix_ids) + 1:
            raise ValueError(
                f"Action ID '{aid}' merged with suffix '{POLICY_PROMPT_SUFFIX!r}': "
                f"suffix={suffix_ids}, combined={combined}"
            )
        seen.add(tid)
        token_ids.append(tid)

    return token_ids


class ValueHead(nn.Module):
    """Lightweight 2-layer MLP predicting battle win-probability logit from last hidden state."""

    def __init__(self, hidden_size: int, intermediate_size: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.SiLU(),
            nn.Linear(intermediate_size, 1),
        )
        # Near-zero init on output layer so initial V(s) starts calibrated near 0.5 win prob
        nn.init.zeros_(self.net[-1].bias)
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.01)

    def forward(self, h_last: torch.Tensor) -> torch.Tensor:
        return self.net(h_last.float()).squeeze(-1)


def _extract_backbone_and_lm_head(lm: nn.Module) -> tuple[nn.Module, nn.Module]:
    """Unwrap PEFT (if present) to get (transformer_backbone, output_lm_head) preserving LoRA."""
    base = lm.get_base_model() if hasattr(lm, "get_base_model") else lm
    lm_head = base.get_output_embeddings()
    if lm_head is None:
        raise ValueError(f"Model {type(base).__name__} does not expose get_output_embeddings().")

    for attr in ("model", "transformer", "base_model"):
        candidate = getattr(base, attr, None)
        if isinstance(candidate, nn.Module) and candidate is not base:
            return (candidate, lm_head)
    raise ValueError(f"Could not locate transformer backbone attribute on {type(base).__name__}.")


def _infer_hidden_size(lm: nn.Module) -> int:
    cfg = getattr(lm, "config", None)
    if cfg is None and hasattr(lm, "get_base_model"):
        cfg = lm.get_base_model().config
    for attr in ("hidden_size", "n_embd", "d_model"):
        val = getattr(cfg, attr, None)
        if isinstance(val, int) and val > 0:
            return val
    raise ValueError("Could not determine hidden_size from model config.")


def _autocast_ctx(device: torch.device) -> Any:
    """Use native bf16 autocast on Ampere+ CUDA so PEFT fp32 LoRA layers run on Tensor Cores."""
    from contextlib import nullcontext

    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.amp.autocast("cuda", dtype=dtype)
    return nullcontext()


class AlphaBattlePolicyValueModel(nn.Module):
    """Pretrained LLM backbone (+ optional PEFT LoRA) with 13-way masked action policy + ValueHead."""

    def __init__(
        self,
        lm: nn.Module,
        action_token_ids: list[int],
        *,
        include_value_head: bool = True,
        value_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if len(action_token_ids) != len(ALL_ACTION_IDS):
            raise ValueError(
                f"Expected {len(ALL_ACTION_IDS)} action token IDs, got {len(action_token_ids)}."
            )
        self.lm = lm
        self.register_buffer(
            "action_token_ids", torch.tensor(action_token_ids, dtype=torch.long), persistent=False
        )
        self.hidden_size = _infer_hidden_size(lm)
        self.value_head: ValueHead | None = (
            ValueHead(self.hidden_size, value_hidden_dim) if include_value_head else None
        )

    def _forward_last_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        last_token_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, nn.Module]:
        from contextlib import nullcontext

        backbone, lm_head = _extract_backbone_and_lm_head(self.lm)
        backbone_trainable = any(p.requires_grad for p in backbone.parameters())
        grad_ctx = (
            nullcontext() if (torch.is_grad_enabled() and backbone_trainable) else torch.no_grad()
        )
        with grad_ctx, _autocast_ctx(input_ids.device):
            outputs = backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        hidden = outputs.last_hidden_state  # (B, L, H)
        if last_token_indices is None:
            last_token_indices = attention_mask.long().sum(dim=1).clamp_min(1) - 1
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        h_last = hidden[batch_idx, last_token_indices]  # (B, H)
        return (h_last, lm_head)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        legal_mask: torch.Tensor,
        *,
        last_token_indices: torch.Tensor | None = None,
        return_value: bool = True,
    ) -> PolicyValueOutput:
        h_last, lm_head = self._forward_last_hidden(
            input_ids, attention_mask, last_token_indices=last_token_indices
        )

        # Project ONLY against the 13 action token embeddings (avoids 152K-vocab logits allocation!)
        with _autocast_ctx(input_ids.device):
            action_weight = lm_head.weight.index_select(0, self.action_token_ids)
            action_bias = (
                lm_head.bias.index_select(0, self.action_token_ids)
                if getattr(lm_head, "bias", None) is not None
                else None
            )
            raw_logits = F.linear(h_last, action_weight, action_bias).float()  # (B, 13)
        masked_logits = raw_logits.masked_fill(~legal_mask.bool(), -1e9)

        value_logit: torch.Tensor | None = None
        if return_value and self.value_head is not None:
            value_logit = self.value_head(h_last)

        return PolicyValueOutput(
            raw_action_logits=raw_logits,
            masked_logits=masked_logits,
            value_logit=value_logit,
            last_hidden=h_last,
        )

    def save_checkpoint(
        self,
        output_dir: str | Path,
        *,
        base_model_name: str,
        tokenizer: Any,
        max_length: int = 1024,
        extra_meta: dict[str, Any] | None = None,
    ) -> Path:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        adapter_dir = out / ADAPTER_DIRNAME
        is_peft = hasattr(self.lm, "peft_config")
        if is_peft:
            self.lm.save_pretrained(adapter_dir)
        else:
            self.lm.save_pretrained(adapter_dir)

        tokenizer.save_pretrained(out / "tokenizer")

        if self.value_head is not None:
            torch.save(self.value_head.state_dict(), out / VALUE_HEAD_FILENAME)

        meta: dict[str, Any] = {
            "base_model": base_model_name,
            "is_peft": is_peft,
            "action_ids": list(ALL_ACTION_IDS),
            "action_token_ids": [int(x) for x in self.action_token_ids.tolist()],
            "max_length": max_length,
            "has_value_head": self.value_head is not None,
            "hidden_size": self.hidden_size,
        }
        if extra_meta:
            meta.update(extra_meta)
        (out / META_FILENAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return out


def load_base_lm_and_tokenizer(
    base_model_name: str,
    *,
    device: str | torch.device = "auto",
    dtype: torch.dtype | None = None,
    qlora: bool = False,
    attn_implementation: str | None = None,
) -> tuple[nn.Module, Any]:
    """Load HF CausalLM + tokenizer with Native Windows RTX 3080 defaults (bf16 + SDPA)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.padding_side = "right"

    if device == "auto":
        use_cuda = torch.cuda.is_available()
        target_device = torch.device("cuda" if use_cuda else "cpu")
    else:
        target_device = torch.device(device)
        use_cuda = target_device.type == "cuda"

    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if dtype is None:
        if use_cuda and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif use_cuda:
            dtype = torch.float16
        else:
            dtype = torch.float32

    model_kwargs: dict[str, Any] = {"dtype": dtype}
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation
    elif use_cuda:
        model_kwargs["attn_implementation"] = "sdpa"

    if qlora:
        if not use_cuda:
            raise RuntimeError("QLoRA NF4 quantization requires a CUDA GPU.")
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        model_kwargs["device_map"] = {"": target_device.index or 0}

    lm = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
    if not qlora:
        lm = lm.to(target_device)

    return (lm, tokenizer)


def attach_lora(
    lm: nn.Module,
    *,
    r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    target_modules: str | list[str] = "all-linear",
    qlora: bool = False,
    gradient_checkpointing: bool = True,
) -> nn.Module:
    """Attach PEFT LoRA adapter cleanly (supports both bf16 LoRA and QLoRA NF4)."""
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    if qlora:
        lm = prepare_model_for_kbit_training(
            lm, use_gradient_checkpointing=gradient_checkpointing
        )
    elif gradient_checkpointing:
        if hasattr(lm, "activation_checkpointing_enable"):
            lm.activation_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        elif hasattr(lm, "gradient_checkpointing_enable"):
            lm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        if hasattr(lm, "enable_input_require_grads"):
            lm.enable_input_require_grads()

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    return get_peft_model(lm, lora_cfg)


def load_policy_value_checkpoint(
    checkpoint_dir: str | Path,
    *,
    base_model_override: str | None = None,
    device: str | torch.device = "auto",
    dtype: torch.dtype | None = None,
    qlora: bool = False,
    is_trainable: bool = False,
    merge_for_inference: bool = True,
) -> tuple[AlphaBattlePolicyValueModel, Any, dict[str, Any]]:
    """Reload a saved AlphaBattlePolicyValueModel (PEFT LoRA + ValueHead + tokenizer)."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ckpt = Path(checkpoint_dir)
    meta = json.loads((ckpt / META_FILENAME).read_text(encoding="utf-8"))
    base_model_name = base_model_override or meta["base_model"]
    adapter_dir = ckpt / ADAPTER_DIRNAME

    if meta.get("is_peft", True):
        base_lm, tokenizer = load_base_lm_and_tokenizer(
            base_model_name, device=device, dtype=dtype, qlora=qlora
        )
        tok_dir = ckpt / "tokenizer"
        if tok_dir.exists():
            tokenizer = AutoTokenizer.from_pretrained(tok_dir)
            if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
        lm = PeftModel.from_pretrained(base_lm, adapter_dir, is_trainable=is_trainable)
        if (not is_trainable) and merge_for_inference and (not qlora) and hasattr(lm, "merge_and_unload"):
            lm = lm.merge_and_unload()
    else:
        target_device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        use_dtype = dtype or (
            torch.bfloat16
            if (target_device.type == "cuda" and torch.cuda.is_bf16_supported())
            else torch.float32
        )
        lm = AutoModelForCausalLM.from_pretrained(adapter_dir, dtype=use_dtype).to(target_device)
        tokenizer = AutoTokenizer.from_pretrained(ckpt / "tokenizer")
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    action_token_ids = resolve_action_token_ids(tokenizer)
    has_value_head = bool(meta.get("has_value_head", False))
    model = AlphaBattlePolicyValueModel(
        lm=lm,
        action_token_ids=action_token_ids,
        include_value_head=has_value_head,
    )

    vh_path = ckpt / VALUE_HEAD_FILENAME
    if has_value_head and vh_path.exists() and model.value_head is not None:
        state_dict = torch.load(vh_path, map_location="cpu", weights_only=True)
        model.value_head.load_state_dict(state_dict)

    # Align buffers and ValueHead with the LM's device
    model_device = next(lm.parameters()).device
    model.action_token_ids = model.action_token_ids.to(model_device)
    if model.value_head is not None:
        model.value_head = model.value_head.to(model_device)

    if not is_trainable:
        model.eval()
    return (model, tokenizer, meta)
