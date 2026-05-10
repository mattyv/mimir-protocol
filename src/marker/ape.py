"""APE (Adaptive Parallel Encoding) — historical / negative result.

**Status (2026-05-10):** Superseded by the composed-axiom (H) approach
in `axiom_registry.composed_description` + `Prefix.from_axiom`. APE
helps for direct fact lookup at 3-5 stacked prefixes but fails on
counterfactual / DAG-traversal queries. Kept here for reproducibility
of the negative result and for the 2-prefix RoPE-fix gauntlet.

Original docstring follows.

----

APE — three dials to fix attention entropy collapse at 3+ stacked prefixes.

Reference: Yang, Chen, Chen, "Adaptive Parallel Encoding for Efficient
LLM Serving" (ICLR 2025, arxiv 2502.05431).

The paper's three corrections:
  1. Shared prefix prepended once before all axioms (one consistent
     attention sink instead of N competing ones).
  2. Temperature T<1 to sharpen post-concat attention.
  3. Scale factor S<1 to compensate for inflated LogSumExp.

This v1 collapses (2) and (3) into one knob `q_scale` (multiplied into
Q before attention computation). Mathematically: scaling Q by c gives
softmax(c * QK^T / sqrt(d)), which is the SAME family of effects T and
S target — sharpening when c>1, softening when c<1. Future v2 can split
them back into separate T and S if needed.

Implementation: register a forward_hook on every layer's `q_proj` that
multiplies the projection output by `q_scale`. Rotation (RoPE) and
attention are then computed with the scaled Q. No model weights change.
"""

from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache

from marker.prefix_tuning import (
    Prefix,
    _get_layers,
    _get_rope_theta,
    _model_dtype,
    combined_cache,
)


def install_q_scale_hook(model, q_scale: float) -> list:  # noqa: ANN001
    """Install a forward_hook on every transformer layer's q_proj that
    multiplies its output by `q_scale`. Returns a list of hook handles
    — caller must call `.remove()` on each when done.

    Scaling Q by c is equivalent to softmax(c * QK^T / sqrt(d)) — i.e.
    a temperature/scale knob on attention without modifying weights.
    """
    layers = _get_layers(model)
    handles = []

    def make_hook(scale: float):
        def hook(_module, _input, output):
            if isinstance(output, tuple):
                # Some attention impls return tuples — scale the first elem.
                return (output[0] * scale, *output[1:])
            return output * scale

        return hook

    for layer in layers:
        attn = layer.self_attn
        q_proj = getattr(attn, "q_proj", None)
        if q_proj is None:
            raise RuntimeError(
                f"layer {type(attn).__name__} has no q_proj; APE not yet supported here"
            )
        handles.append(q_proj.register_forward_hook(make_hook(q_scale)))
    return handles


@torch.no_grad()
def generate_with_ape(
    model,  # noqa: ANN001
    tokenizer,
    prompt: str,
    prefixes: list[Prefix],
    shared_prefix_text: str = "",
    q_scale: float = 1.0,
    max_new: int = 60,
    rope_correct: bool = True,
) -> str:
    """Greedy decode with APE: optional shared prefix prepended to all
    axioms + q_scale on attention.

    `shared_prefix_text=""` and `q_scale=1.0` reduce to the existing
    `generate_with_prefixes` behavior exactly (used by the no-op test).
    """
    device = next(model.parameters()).device
    dtype = _model_dtype(model)
    rope_theta = _get_rope_theta(model)
    layers = list(range(model.config.num_hidden_layers))

    prefixes_with_shared: list[Prefix] = []
    if shared_prefix_text:
        shared = Prefix.from_description(model, tokenizer, shared_prefix_text, target_layers=layers)
        prefixes_with_shared.append(shared)
    prefixes_with_shared.extend(prefixes)

    if prefixes_with_shared:
        cache = combined_cache(
            prefixes_with_shared,
            dtype=dtype,
            device=device,
            rope_theta=rope_theta,
            rope_correct=rope_correct,
        )
    else:
        cache = DynamicCache()

    handles: list = []
    if q_scale != 1.0:
        handles = install_q_scale_hook(model, q_scale=q_scale)
    try:
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        out = model(ids, past_key_values=cache, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        full_ids = torch.cat([ids, nxt], dim=1)
        if int(nxt.item()) == tokenizer.eos_token_id:
            return ""
        for _ in range(max_new - 1):
            out = model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            full_ids = torch.cat([full_ids, nxt], dim=1)
            if int(nxt.item()) == tokenizer.eos_token_id:
                break
        new_ids = full_ids[0, ids.shape[1] :]
        return tokenizer.decode(new_ids, skip_special_tokens=True)
    finally:
        for h in handles:
            h.remove()
