"""Per-axiom signature injection (Path 3) — break the binding-ID collision.

Hypothesis: when multiple per-axiom prefixes are stacked into a joint
cache, the model's binding heads can't tell which fact belongs to which
entity because every axiom's K vectors carry the same "I'm the focal
entity of my own little document" content marker. Add a tiny unique
fingerprint to each axiom's K vectors at load time so the binding heads
have something distinguishing to latch onto.

The signature is:
  - **Deterministic** — same name always gives the same fingerprint.
  - **Axiom-specific** — different names give different fingerprints.
  - **Unit per (layer, head)** — magnitude is applied externally.
  - **K-only** — V vectors are not modified.

Mechanistic motivation (Feng & Steinhardt 2023; Dai et al 2025): the
model represents (entity, attribute) pairs by attaching matching
binding-ID vectors in a learned subspace. Independent captures share
that subspace → IDs collide. Adding a small per-axiom offset to K
gives each cached entity a distinguishable "ID color" while leaving V
(the actual fact content) intact.
"""

from __future__ import annotations

import hashlib

import torch

from marker.prefix_tuning import Prefix


def signature_vector(
    name: str,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Deterministic per-name fingerprint of shape (n_layers, n_kv_heads, head_dim).

    Each (layer, head) slice is a unit vector — the caller scales by a
    magnitude at injection time. Two different names give two
    near-orthogonal-ish unit vectors (with high probability in
    high-dimensional spaces).

    Construction: SHA-256 hash of `name` seeds a generator; we draw
    (n_layers * n_kv_heads) Gaussian head_dim-vectors, normalize each.
    """
    h = hashlib.sha256(name.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big", signed=False) % (2**31 - 1)
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(n_layers, n_kv_heads, head_dim, generator=g, dtype=dtype)
    norms = raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return raw / norms


@torch.no_grad()
def apply_signatures(
    prefixes: list[Prefix],
    names: list[str],
    magnitude: float = 0.1,
) -> list[Prefix]:
    """Return a new list of Prefix objects with per-axiom K signatures
    added. Original prefixes are NOT mutated.

    `magnitude` scales each unit-length per-(layer, head) signature
    before it's broadcast-added to every token position of that axiom's
    K. magnitude=0.0 is a strict no-op.
    """
    if len(prefixes) != len(names):
        raise ValueError(f"prefixes ({len(prefixes)}) and names ({len(names)}) length mismatch")
    out: list[Prefix] = []
    for p, name in zip(prefixes, names, strict=True):
        new_keys: list[torch.nn.Parameter] = []
        new_values: list[torch.nn.Parameter] = []
        if magnitude == 0.0:
            for k_t in p.keys:
                new_keys.append(torch.nn.Parameter(k_t.detach().clone()))
            for v_t in p.values:
                new_values.append(torch.nn.Parameter(v_t.detach().clone()))
        else:
            n_layers_target = len(p.target_layers)
            ref_k = p.keys[0]
            n_kv_heads, head_dim = ref_k.shape[1], ref_k.shape[3]
            sig = signature_vector(
                name=name,
                n_layers=n_layers_target,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                dtype=ref_k.dtype,
            ).to(ref_k.device)  # (n_layers_target, n_kv_heads, head_dim)
            for i, k_t in enumerate(p.keys):
                # Broadcast (n_kv_heads, head_dim) over the seq dim:
                # k_t shape (1, n_kv_heads, n_tokens, head_dim).
                offset = (sig[i] * magnitude).unsqueeze(0).unsqueeze(-2)
                new_keys.append(torch.nn.Parameter((k_t + offset).detach().clone()))
            for v_t in p.values:
                new_values.append(torch.nn.Parameter(v_t.detach().clone()))
        out.append(
            Prefix(
                n_tokens=p.n_tokens,
                n_total_layers=p.n_total_layers,
                n_kv_heads=p.n_kv_heads,
                head_dim=p.head_dim,
                target_layers=list(p.target_layers),
                keys=new_keys,
                values=new_values,
                per_layer_shapes=list(p.per_layer_shapes),
                source_ids=p.source_ids,
            )
        )
    return out
