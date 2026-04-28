"""Multi-layer decode-time residual injection.

Prior residual-injection scripts hooked layers during prefill only.
The KV cache from boilerplate tokens then propagates unmodified during
decode, anchoring the continuation to the lexical reading.

This script registers hooks at multiple layers AND keeps them active
during the decode loop, so each generated token gets its residual
nudged at every chosen layer before producing logits. Optionally
stacks with decode-time logit biasing.

Layers to hit are based on the patching probe finding for Balance
Publisher: L26 last-position (+3.55) is primary; L12-14 was secondary
on the term-token but doesn't matter at decode-time since the term
isn't in the generated stream. So we layer at 20 + 26 by default
(both upper-band, complementary).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marker.run_logit_bias_decode import (
    BP_CONTINUATIONS,
    BP_INTENDED_PARAPHRASES_PATH,
    BP_LEXICAL_PARAPHRASES_PATH,
    BP_PROMPTS,
    ST_CONTINUATIONS,
    ST_INTENDED_PARAPHRASES_PATH,
    ST_LEXICAL_PARAPHRASES_PATH,
    ST_PROMPTS,
    _load_paraphrases,
    capture_concept_completion_residual,
    compute_logit_bias,
)

AXIOM_CFGS = {
    "bp": (BP_INTENDED_PARAPHRASES_PATH, BP_LEXICAL_PARAPHRASES_PATH, BP_CONTINUATIONS, BP_PROMPTS),
    "shoe": (
        ST_INTENDED_PARAPHRASES_PATH,
        ST_LEXICAL_PARAPHRASES_PATH,
        ST_CONTINUATIONS,
        ST_PROMPTS,
    ),
}

ROOT = Path(__file__).resolve().parents[2]


class MultiLayerInjector:
    """Maintains hooks at multiple layers; injects α·v at the LAST
    position of every forward pass (prefill and each decode step)."""

    def __init__(self, model, layer_to_vector: dict[int, np.ndarray], alpha: float) -> None:
        self.model = model
        self.alpha = alpha
        self.vectors = {L: torch.tensor(v, dtype=torch.float32) for L, v in layer_to_vector.items()}
        base = model
        if hasattr(base, "base_model") and hasattr(base.base_model, "model"):
            base = base.base_model.model
        self.layers_module = base.model.layers
        self.handles: list = []

    def __enter__(self):
        for L, vec in self.vectors.items():
            handle = self.layers_module[L].register_forward_hook(self._make_hook(vec))
            self.handles.append(handle)
        return self

    def __exit__(self, *args):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def _make_hook(self, vec: torch.Tensor):
        alpha = self.alpha

        def hook(module, inputs, output):  # noqa: ANN001, ARG001
            h = output[0] if isinstance(output, tuple) else output
            v_dev = vec.to(dtype=h.dtype, device=h.device)
            h_new = h.clone()
            h_new[:, -1, :] = h_new[:, -1, :] + alpha * v_dev
            if isinstance(output, tuple):
                return (h_new, *output[1:])
            return h_new

        return hook


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    layer_vectors: dict[int, np.ndarray] | None,
    alpha: float,
    logit_bias: torch.Tensor | None,
    max_new: int = 80,
) -> str:
    device = next(model.parameters()).device
    bias_dev = logit_bias.to(device=device, dtype=torch.float32) if logit_bias is not None else None

    def _step_logits(logits_row: torch.Tensor) -> torch.Tensor:
        x = logits_row.float()
        if bias_dev is not None:
            x = x + bias_dev
        return x

    ctx = MultiLayerInjector(model, layer_vectors, alpha) if layer_vectors else None
    if ctx is not None:
        ctx.__enter__()
    try:
        ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        out = model(ids, use_cache=True)
        past = out.past_key_values
        nxt = _step_logits(out.logits[0, -1]).argmax().unsqueeze(0).unsqueeze(0)
        ids = torch.cat([ids, nxt], dim=1)
        if int(nxt.item()) == tokenizer.eos_token_id:
            return ""
        for _ in range(max_new - 1):
            out = model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = _step_logits(out.logits[0, -1]).argmax().unsqueeze(0).unsqueeze(0)
            ids = torch.cat([ids, nxt], dim=1)
            if int(nxt.item()) == tokenizer.eos_token_id:
                break
        full = tokenizer.decode(ids[0], skip_special_tokens=True)
        return full[len(prompt) :]
    finally:
        if ctx is not None:
            ctx.__exit__()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--layers", type=int, nargs="+", default=[20, 26])
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    parser.add_argument("--logit-alpha", type=float, default=0.0, help="0 disables logit bias")
    parser.add_argument("--max-new", type=int, default=80)
    parser.add_argument("--axiom", choices=["bp", "shoe"], default="bp")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}  layers: {args.layers}  logit_alpha: {args.logit_alpha}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16)
        .to(device)
        .eval()
    )
    base = model
    if hasattr(base, "base_model") and hasattr(base.base_model, "model"):
        base = base.base_model.model
    lm_head = base.lm_head if hasattr(base, "lm_head") else base.get_output_embeddings()

    intended_path, lexical_path, continuations, prompts = AXIOM_CFGS[args.axiom]
    intended = _load_paraphrases(intended_path)
    lexical = _load_paraphrases(lexical_path)

    print("=== building per-layer vectors ===")
    layer_vectors: dict[int, np.ndarray] = {}
    for L in args.layers:
        v_int = capture_concept_completion_residual(model, tokenizer, intended, continuations, L)
        v_lex = capture_concept_completion_residual(model, tokenizer, lexical, continuations, L)
        v = v_int - v_lex
        layer_vectors[L] = v
        print(
            f"  L{L}: norm intended {np.linalg.norm(v_int):.1f}, contrastive {np.linalg.norm(v):.2f}"
        )

    logit_bias = None
    if args.logit_alpha != 0.0:
        # Use the L26 vector for the logit bias (matches the prior winning result)
        v_for_bias = layer_vectors.get(26)
        if v_for_bias is None:
            v_for_bias = next(iter(layer_vectors.values()))
        logit_bias = compute_logit_bias(lm_head.weight, v_for_bias, args.logit_alpha)
        print(f"  logit bias built from L26 v at α={args.logit_alpha}")
    print()

    for prompt in prompts:
        print("=" * 78)
        print(f"USER: {prompt}")
        baseline = generate(model, tokenizer, prompt, None, 0.0, None, args.max_new)
        print(f"  [baseline                      ]: {baseline.replace(chr(10), ' ').strip()[:280]}")
        for alpha in args.alphas:
            out = generate(model, tokenizer, prompt, layer_vectors, alpha, None, args.max_new)
            print(
                f"  [decode-inject α={alpha:>4.1f}            ]: {out.replace(chr(10), ' ').strip()[:280]}"
            )
            if logit_bias is not None:
                out2 = generate(
                    model, tokenizer, prompt, layer_vectors, alpha, logit_bias, args.max_new
                )
                print(
                    f"  [decode-inject α={alpha:>4.1f} + logit α={args.logit_alpha:.2f}]: "
                    f"{out2.replace(chr(10), ' ').strip()[:280]}"
                )
        print()


if __name__ == "__main__":
    main()
