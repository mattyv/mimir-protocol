"""CacheBlend-style selective recompute — historical / negative result.

**Status (2026-05-10):** v1 vanilla-copy variant tested on Modal vs
rope-fix concat at 3 prefixes; identical failure mode (model loops on
one fact). Superseded by composed-axiom (H) in
`axiom_registry.composed_description`. Kept for the negative-result
record.

Original docstring follows.

----

CacheBlend-style selective recompute for 3+ prefix chains.

Background: 2-prefix concat works with RoPE re-rotation
(`prefix_tuning.combined_cache(rope_correct=True)`). 3+ prefixes
sometimes regress because the model never saw "three independent
documents stacked back-to-back" during pretrain. CacheBlend's fix:
identify the ~10–15% of cached positions whose K vectors deviate most
from a joint-context K, and "recompute" only those positions with full
cross-prefix attention awareness.

Reference: Yao et al 2024, "CacheBlend: Fast LLM Serving for RAG with
Cached Knowledge Fusion" (arxiv 2405.16444), section 3.

This module is v1 — correctness-first. It uses a "vanilla-copy"
approximation rather than the true sparse-query custom forward loop:
1. Run a full vanilla joint forward to produce a complete reference cache.
2. For the flagged positions, copy K/V from the vanilla cache into the
   base (cached) cache; leave non-flagged positions untouched.

This costs the same as Path 2 (full prefill) per query but produces a
hybrid cache where most positions retain the RoPE-corrected cached K/V
and only the deviant positions are replaced. Whether this hybrid beats
pure Path 2 on the chain prompts is the demo's question to answer.

TODO (v2, future): swap the vanilla-copy step for a true layer-by-layer
custom forward that recomputes K/V at flagged positions only, using the
partially-patched cache for cross-attention. That gives the CacheBlend
efficiency win (~6× cheaper than full prefill at top_k_pct=0.15).
"""

from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache

from marker.prefix_tuning import Prefix, _get_rope_theta, _model_dtype, combined_cache

# ----------------------------------------------------------------------
# Deviation detection (pure tensor function — easy to unit-test)
# ----------------------------------------------------------------------


def _deviation_indices(
    cached_k: torch.Tensor,
    joint_k: torch.Tensor,
    top_k_pct: float,
) -> torch.Tensor:
    """L2-norm of (cached_k - joint_k) per position; return indices of
    the top `top_k_pct` fraction of positions, sorted by deviation
    descending.

    Shapes: both inputs are (1, n_kv_heads, n_positions, head_dim).
    Output: 1-D LongTensor of indices in [0, n_positions).
    """
    if cached_k.shape != joint_k.shape:
        raise ValueError(f"shape mismatch: {cached_k.shape} vs {joint_k.shape}")
    diff = (cached_k.float() - joint_k.float()).pow(2).sum(dim=(0, 1, 3)).sqrt()
    n_positions = diff.shape[0]
    k = max(1, round(n_positions * top_k_pct))
    k = min(k, n_positions)
    _, idx = torch.topk(diff, k=k, largest=True)
    return idx.sort().values.to(torch.long).cpu()


# ----------------------------------------------------------------------
# Joint forward + deviation lookup
# ----------------------------------------------------------------------


@torch.no_grad()
def find_high_deviation_positions(
    model,  # noqa: ANN001
    cached_k_layer1: torch.Tensor,
    joint_input_ids: torch.Tensor,
    top_k_pct: float = 0.15,
    layer_for_deviation: int = 1,
) -> torch.Tensor:
    """Run a fresh joint forward and compare K at `layer_for_deviation`
    against the cached K at the same layer. Return top-deviation indices.

    Per CacheBlend, layer 1 is cheap and predictive of downstream
    divergence — but on deeper models it can be too generic. The plan
    suggests sweeping layer ∈ {1, 4, 8, n_layers/4}; expose as a kwarg.
    """
    out = model(joint_input_ids, past_key_values=DynamicCache(), use_cache=True)
    joint_cache: DynamicCache = out.past_key_values
    joint_k = joint_cache.layers[layer_for_deviation].keys
    if joint_k.shape != cached_k_layer1.shape:
        raise ValueError(f"joint K shape {joint_k.shape} != cached K shape {cached_k_layer1.shape}")
    return _deviation_indices(cached_k_layer1, joint_k, top_k_pct=top_k_pct)


# ----------------------------------------------------------------------
# Selective recompute (v1: vanilla-copy)
# ----------------------------------------------------------------------


@torch.no_grad()
def selective_recompute_prefix_cache(
    model,  # noqa: ANN001
    base_cache: DynamicCache,
    joint_input_ids: torch.Tensor,
    high_deviation_positions: torch.Tensor,
    target_layers: list[int] | None = None,
) -> DynamicCache:
    """Replace K/V at `high_deviation_positions` (across all layers in
    `target_layers`, or all layers if None) with K/V from a fresh vanilla
    forward. Other positions retain the K/V already in `base_cache`.

    v1 implementation: full prefill + selective copy. v2 (TODO) would
    skip the prefill cost at non-flagged positions.

    Returns a NEW DynamicCache; does not mutate `base_cache` in place.
    """
    if high_deviation_positions.numel() == 0:
        # Nothing to do — return a fresh DynamicCache copy of base_cache.
        return _copy_cache(base_cache)

    out = model(joint_input_ids, past_key_values=DynamicCache(), use_cache=True)
    vanilla_cache: DynamicCache = out.past_key_values

    n_layers = len(base_cache)
    pos = high_deviation_positions.to(torch.long).cpu()

    new_cache = DynamicCache()
    layers_to_patch = set(target_layers) if target_layers is not None else set(range(n_layers))

    for layer_idx in range(n_layers):
        base_k = base_cache.layers[layer_idx].keys
        base_v = base_cache.layers[layer_idx].values
        new_k = base_k.clone()
        new_v = base_v.clone()
        if layer_idx in layers_to_patch:
            v_k = vanilla_cache.layers[layer_idx].keys
            v_v = vanilla_cache.layers[layer_idx].values
            new_k[..., pos, :] = v_k[..., pos, :].to(dtype=new_k.dtype, device=new_k.device)
            new_v[..., pos, :] = v_v[..., pos, :].to(dtype=new_v.dtype, device=new_v.device)
        new_cache.update(new_k, new_v, layer_idx)
    return new_cache


def _copy_cache(cache: DynamicCache) -> DynamicCache:
    out = DynamicCache()
    for i in range(len(cache)):
        out.update(cache.layers[i].keys.clone(), cache.layers[i].values.clone(), i)
    return out


# ----------------------------------------------------------------------
# blend_prefixes — top-level entry
# ----------------------------------------------------------------------


@torch.no_grad()
def blend_prefixes(
    model,  # noqa: ANN001
    prefixes: list[Prefix],
    rope_corrected: bool = True,
    selective_recompute: bool = True,
    top_k_pct: float = 0.15,
    layer_for_deviation: int = 1,
) -> DynamicCache:
    """Build a DynamicCache for `prefixes` with optional CacheBlend
    selective recompute on top of the existing RoPE-corrected concat.

    1. Build naive concat with RoPE correction (existing path).
    2. If `selective_recompute` and len(prefixes) >= 3:
       a. Concat each prefix's source token-ids to form `joint_input_ids`.
       b. find_high_deviation_positions(...) at `layer_for_deviation`.
       c. selective_recompute_prefix_cache(...) to patch flagged positions.
    3. Return.

    Note: step 2a requires each Prefix to know its source token ids.
    Prefix as currently implemented does NOT store ids. The simplest
    workaround is to pass `joint_input_ids` directly when calling. For
    convenience, this function also accepts `prefixes` whose source
    descriptions can be re-tokenized — but to keep the API minimal here,
    callers wanting selective recompute should use the lower-level
    `selective_recompute_prefix_cache` directly with explicit
    `joint_input_ids`. This `blend_prefixes` falls back to plain
    RoPE-corrected concat when selective recompute would need ids it
    doesn't have.
    """
    device = next(model.parameters()).device
    dtype = _model_dtype(model)
    rope_theta = _get_rope_theta(model)
    base = combined_cache(
        prefixes,
        dtype=dtype,
        device=device,
        rope_theta=rope_theta,
        rope_correct=rope_corrected,
    )
    if not selective_recompute or len(prefixes) < 3:
        return base
    joint_input_ids = _joint_ids_from_prefixes(prefixes, device)
    if joint_input_ids is None:
        # No way to reconstruct joint ids from these prefixes; return base.
        return base
    cached_k_l1 = base.layers[layer_for_deviation].keys
    flagged = find_high_deviation_positions(
        model,
        cached_k_layer1=cached_k_l1,
        joint_input_ids=joint_input_ids,
        top_k_pct=top_k_pct,
        layer_for_deviation=layer_for_deviation,
    )
    return selective_recompute_prefix_cache(
        model=model,
        base_cache=base,
        joint_input_ids=joint_input_ids,
        high_deviation_positions=flagged,
    )


def _joint_ids_from_prefixes(
    prefixes: list[Prefix],
    device: torch.device,
) -> torch.Tensor | None:
    """If every prefix has a `source_ids` attribute (set by the
    description-based constructor in a future patch), concat them. Else
    return None and the caller will skip selective recompute.
    """
    ids_list: list[torch.Tensor] = []
    for p in prefixes:
        sids = getattr(p, "source_ids", None)
        if sids is None:
            return None
        ids_list.append(sids.to(device).view(-1))
    return torch.cat(ids_list, dim=0).unsqueeze(0)
