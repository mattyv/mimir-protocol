"""Qwen 2.5 0.5B base model wrapper, with optional LoRA installation.

Used by the sentinel-LoRA POC (`docs/sentinel-lora-poc-spec.md`). The base
model is frozen; the LoRA adapter is the trainable surface that teaches
the sentinel-consumption protocol.

This module deliberately keeps the wrapper thin — most heavy lifting lives
in `transformers` and `peft`. Behavior we own here:

  - load Qwen on MPS in fp16, set `eval()` and freeze base params
  - generate next-token text deterministically (greedy)
  - wrap with a LoRA adapter on the standard attn + FFN target set
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

# Qwen 2.5 transformer blocks expose these projection module names.
DEFAULT_LORA_TARGETS: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass
class WrappedModel:
    """A SentinelModel after LoRA installation. Carries the peft model so
    callers can train it; exposes the same generate() API as SentinelModel."""

    peft_model: PeftModel
    tokenizer: PreTrainedTokenizer
    device: str

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 16) -> str:
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        out = self.peft_model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)


class SentinelModel:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        device: str = "cpu",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.base: PreTrainedModel = (
            AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()
        )
        for p in self.base.parameters():
            p.requires_grad_(False)

    @property
    def config(self):  # noqa: ANN201 — passthrough; types come from HF
        return self.base.config

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 16) -> str:
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        out = self.base.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def with_lora(
        self,
        rank: int = 16,
        alpha: int = 32,
        target_modules: tuple[str, ...] = DEFAULT_LORA_TARGETS,
        dropout: float = 0.0,
    ) -> WrappedModel:
        """Install a LoRA adapter and return a WrappedModel. The base remains
        frozen; only the LoRA matrices are trainable."""
        cfg = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=list(target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        peft_model = get_peft_model(self.base, cfg)
        return WrappedModel(peft_model=peft_model, tokenizer=self.tokenizer, device=self.device)
