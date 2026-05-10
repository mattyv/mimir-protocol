"""Per-block attention — historical / negative result.

**Status (2026-05-10):** Confirmed dead end. Reproduces the ICLR 2025
Block-Attention frozen-model ablation (67.9 → 48.0% accuracy). On
Qwen 2.5-32B, both `uniform` and `cosine` combiners produce token-level
garbage on 5/5 hierarchy prompts. Superseded by composed-axiom (H) in
`axiom_registry.composed_description`. Kept for the negative-result
record and for the per-block SDPA tests (which exercise the kernel in
isolation).

Original docstring follows.

----

Per-block attention (custom SDPA) for axiom-aware KV cache reuse.

Why this exists: APE (q_scale + shared prefix) helps for direct fact
recall on 5+ axioms but fails on counterfactual reasoning ("if X breaks,
what cascades?"). The diagnosis is attention-entropy collapse — one big
softmax over many cached blocks goes flat.

Per-block attention runs a separate softmax INSIDE each axiom's slot
range and combines per-block outputs. Two combiners:

  * uniform — each block gets equal weight 1/N. Forces budget per axiom.
  * lse     — log-sum-exp weighted (mathematically recovers vanilla flat
              attention; included as a sanity check).
  * cosine  — Q · mean(K_block) similarity weighted, then softmax across
              blocks. Routes attention to the relevant axiom.

Implementation: monkey-patch `torch.nn.functional.scaled_dot_product_attention`
during decode. Boundaries set via thread-local `_BOUNDARIES`. Stays
frozen-model — no weights touched.

References (training-free family):
- Zhang et al, "Attention Entropy is a Key Factor", ACL 2025.
- CacheClip (arxiv 2510.10129): closest in spirit, uses an auxiliary
  model for token routing instead of per-block softmax.
"""

from __future__ import annotations

import math
import threading
from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812
from transformers.cache_utils import DynamicCache

from marker.prefix_tuning import (
    Prefix,
    _get_rope_theta,
    _model_dtype,
    combined_cache,
)

# ---------------------------------------------------------------------------
# Thread-local boundaries + combiner mode
# ---------------------------------------------------------------------------

_STATE = threading.local()


def set_block_boundaries(
    boundaries: list[tuple[int, int]] | None,
    combiner: Literal["uniform", "lse", "cosine"] | None = None,
) -> None:
    """Set the cached-K block boundaries for the next forward call.

    `boundaries` is a list of (start, end) half-open intervals over the
    K sequence dimension; together they must span the cached portion.
    Set to None to disable (fall back to vanilla SDPA).
    """
    _STATE.boundaries = boundaries
    if combiner is not None:
        _STATE.combiner = combiner


def _get_state() -> tuple[list[tuple[int, int]] | None, str]:
    return getattr(_STATE, "boundaries", None), getattr(_STATE, "combiner", "uniform")


# ---------------------------------------------------------------------------
# Per-block SDPA kernel
# ---------------------------------------------------------------------------


def per_block_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    boundaries: list[tuple[int, int]],
    combiner: Literal["uniform", "lse", "cosine"] = "uniform",
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute attention with separate softmax per block, combined by `combiner`.

    Shapes: q (B, H, Lq, D); k, v (B, H, Lk, D). Output (B, H, Lq, D).

    `boundaries` must cover the cached prefix portion of K. Any positions
    in K beyond the last boundary are treated as a single trailing block
    (handles new query tokens attending to themselves during decode).
    """
    Lk = k.shape[-2]
    # Handle Grouped-Query Attention: q has H_q heads, k/v have H_kv heads
    # where H_q is a multiple of H_kv. Standard sdpa broadcasts; we have to.
    h_q = q.shape[1]
    h_kv = k.shape[1]
    if h_q != h_kv:
        if h_q % h_kv != 0:
            raise RuntimeError(f"GQA mismatch: q_heads={h_q}, kv_heads={h_kv}")
        repeats = h_q // h_kv
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
    # Append a trailing block for any K positions past the last boundary
    # (these are the new query positions appended during decode).
    eff_bounds = list(boundaries)
    if not eff_bounds or eff_bounds[-1][1] < Lk:
        last = eff_bounds[-1][1] if eff_bounds else 0
        eff_bounds.append((last, Lk))

    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)

    block_outputs: list[torch.Tensor] = []
    block_lse: list[torch.Tensor] = []  # (B, H, Lq)
    block_centroid_sims: list[torch.Tensor] = []  # (B, H, Lq) — cosine combiner only

    for s, e in eff_bounds:
        if s >= e:
            continue
        k_b = k[..., s:e, :]
        v_b = v[..., s:e, :]
        logits = torch.matmul(q, k_b.transpose(-1, -2)) * scale  # (B, H, Lq, e-s)
        if attn_mask is not None:
            mask_b = attn_mask[..., s:e]
            logits = logits + mask_b
        m = logits.max(dim=-1, keepdim=True).values
        e_b = (logits - m).exp()
        w_b = e_b.sum(dim=-1, keepdim=True)
        out_b = torch.matmul(e_b, v_b) / w_b
        block_outputs.append(out_b)
        block_lse.append(m.squeeze(-1) + w_b.squeeze(-1).log())

        if combiner == "cosine":
            # Centroid of K in the block, normalized
            centroid = k_b.mean(dim=-2, keepdim=True)  # (B, H, 1, D)
            q_n = q / (q.norm(dim=-1, keepdim=True).clamp_min(1e-6))
            c_n = centroid / (centroid.norm(dim=-1, keepdim=True).clamp_min(1e-6))
            sim = (q_n * c_n).sum(dim=-1)  # (B, H, Lq)
            block_centroid_sims.append(sim)

    n = len(block_outputs)
    stacked = torch.stack(block_outputs, dim=0)  # (N, B, H, Lq, D)

    if combiner == "uniform":
        out = stacked.mean(dim=0)
    elif combiner == "lse":
        lse_stack = torch.stack(block_lse, dim=0)  # (N, B, H, Lq)
        weights = F.softmax(lse_stack, dim=0).unsqueeze(-1)  # (N, B, H, Lq, 1)
        out = (stacked * weights).sum(dim=0)
    elif combiner == "cosine":
        sim_stack = torch.stack(block_centroid_sims, dim=0)
        weights = F.softmax(sim_stack, dim=0).unsqueeze(-1)
        out = (stacked * weights).sum(dim=0)
    else:
        raise ValueError(f"unknown combiner {combiner!r}")
    if n == 1:
        # Single-block uniform/lse/cosine all collapse to vanilla.
        return out
    return out


# ---------------------------------------------------------------------------
# SDPA monkey-patch
# ---------------------------------------------------------------------------


_ORIG_SDPA = F.scaled_dot_product_attention


def _patched_sdpa(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw
):  # noqa: ANN001
    boundaries, combiner = _get_state()
    if boundaries is None:
        return _ORIG_SDPA(
            query, key, value, attn_mask, dropout_p, is_causal=is_causal, scale=scale, **kw
        )
    # Build the additive mask we need for per-block attention. We only need
    # the mask elements over K positions; for `is_causal=True` during prefill
    # this is the lower-triangular mask, which sdpa builds internally — we
    # have to build it ourselves here.
    mask = attn_mask
    if is_causal and mask is None:
        Lq, Lk = query.shape[-2], key.shape[-2]
        causal = torch.full((Lq, Lk), float("-inf"), dtype=query.dtype, device=query.device)
        causal = torch.triu(causal, diagonal=1 + Lk - Lq)
        mask = causal
    return per_block_sdpa(
        query, key, value, boundaries=boundaries, combiner=combiner, attn_mask=mask
    )


class _PatchHandle:
    def __init__(self) -> None:
        self._installed = True

    def remove(self) -> None:
        if self._installed:
            F.scaled_dot_product_attention = _ORIG_SDPA
            self._installed = False


def install_per_block_attention(
    model,  # noqa: ANN001 (unused — kept for API symmetry with hooks)
    combiner: Literal["uniform", "lse", "cosine"] = "uniform",
) -> _PatchHandle:
    """Install the per-block SDPA patch globally on torch.nn.functional.
    Caller must subsequently call `set_block_boundaries(...)` before each
    forward, and `set_block_boundaries(None)` after to disable.
    """
    set_block_boundaries(None, combiner=combiner)
    F.scaled_dot_product_attention = _patched_sdpa
    return _PatchHandle()


# ---------------------------------------------------------------------------
# Top-level decode
# ---------------------------------------------------------------------------


def _boundaries_from_prefixes(prefixes: list[Prefix]) -> list[tuple[int, int]]:
    """Compute (start, end) per axiom in the joint cache."""
    out: list[tuple[int, int]] = []
    cursor = 0
    for p in prefixes:
        out.append((cursor, cursor + p.n_tokens))
        cursor += p.n_tokens
    return out


@torch.no_grad()
def generate_with_per_block(
    model,  # noqa: ANN001
    tokenizer,
    prompt: str,
    prefixes: list[Prefix],
    combiner: Literal["uniform", "lse", "cosine"] = "uniform",
    max_new: int = 60,
    rope_correct: bool = True,
) -> str:
    """Greedy decode with per-block attention over the cached axiom blocks.

    The trailing block (positions past the cache) is implicit — it
    captures the prompt + decoded tokens attending to themselves.
    """
    device = next(model.parameters()).device
    dtype = _model_dtype(model)
    rope_theta = _get_rope_theta(model)

    if prefixes:
        cache = combined_cache(
            prefixes,
            dtype=dtype,
            device=device,
            rope_theta=rope_theta,
            rope_correct=rope_correct,
        )
        boundaries = _boundaries_from_prefixes(prefixes)
    else:
        cache = DynamicCache()
        boundaries = []

    handle = install_per_block_attention(model, combiner=combiner)
    try:
        set_block_boundaries(boundaries, combiner=combiner)
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
        set_block_boundaries(None)
        handle.remove()
