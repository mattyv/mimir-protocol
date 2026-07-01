"""Hypernetwork axiom store: per-axiom latent + shared decoder → KV.

Contrast with kv_compression.KVCompressor, which stores the compressed KV
*output* per layer. Here we store, per axiom, only:

    AxiomCode = (z: small latent, fact_text: verbatim facts)

and a single SHARED, frozen-after-training KVHypernet expands z into the
"scaffold" KV (the predictable prose structure). The facts — max-entropy
values like "250ms", "balances.raw" — are NOT produced by the net; they're
encoded verbatim via compute_axiom_kv and concatenated, so they're lossless.

Why this shape:
  - Realtime add: register an axiom = encode(full_kv) → z  (one forward pass,
    NO training) + extract fact_text. The heavy net is trained once offline.
  - Facts are solved by not compressing them: z carries structure, fact_text
    carries information.
  - Reuses merge_axiom_kvs (RoPE-corrected) to stitch scaffold + facts.

This module is a scaffold: shapes run end-to-end, but train_hypernet() is a
stub — the decoder must be trained once on existing axioms (same recipe as
train_compressor: inject decoded KV, cross-entropy on answers).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from marker.prefix_tuning import _get_rope_theta
from marker.run_axiom_mlp_demo import (
    AxiomKV,
    compute_axiom_kv,
    merge_axiom_kvs,
)


@dataclass
class AxiomCode:
    """What we store per axiom instead of a full/compressed KV."""

    term: str
    z: torch.Tensor  # (d_latent,) — compressible structure
    fact_text: str  # verbatim irreducible facts, e.g. "poll_interval=250ms; topic=balances.raw"


class KVHypernet(nn.Module):
    """Shared, frozen-after-training. encode: full KV → z. decode: z → scaffold KV."""

    def __init__(
        self,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        d_latent: int = 512,
        n_scaffold: int = 4,
        embed_dim: int = 32,
        bottleneck: int = 512,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.d_latent = d_latent
        self.n_scaffold = n_scaffold
        self.layer_embed = nn.Embedding(n_layers, embed_dim)

        # encode: per-layer pooled (K,V) → partial latent; averaged over layers → z
        self.enc = nn.Sequential(
            nn.Linear(2 * head_dim + embed_dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, d_latent),
        )
        # decode: (z, layer_embed) → n_scaffold × (K,V) for that layer
        self.dec = nn.Sequential(
            nn.Linear(d_latent + embed_dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, n_scaffold * 2 * n_kv_heads * head_dim),
        )

    @torch.no_grad()
    def encode(self, full_kv: AxiomKV) -> torch.Tensor:
        """Realtime add: full KV → z. One forward pass, no gradients."""
        device = full_kv.keys[0].device
        per_layer = []
        for layer_idx in range(full_kv.n_layers):
            k_mean = full_kv.keys[layer_idx].to(device).mean(dim=(1, 2))  # (1, head_dim)
            v_mean = full_kv.values[layer_idx].to(device).mean(dim=(1, 2))
            l_emb = self.layer_embed(torch.tensor(layer_idx, device=device)).view(1, -1)
            per_layer.append(self.enc(torch.cat([k_mean, v_mean, l_emb], dim=-1).float()))
        return torch.stack(per_layer).mean(0).squeeze(0)  # (d_latent,)

    def decode_scaffold(self, z: torch.Tensor, device: torch.device) -> AxiomKV:
        """z → n_scaffold virtual KV tokens per layer (the predictable structure)."""
        z = z.to(device).view(1, -1)
        keys, values = [], []
        for layer_idx in range(self.n_layers):
            l_emb = self.layer_embed(torch.tensor(layer_idx, device=device)).view(1, -1)
            out = self.dec(torch.cat([z, l_emb], dim=-1).float())
            out = out.view(1, self.n_kv_heads, self.n_scaffold, 2, self.head_dim)
            keys.append(out[:, :, :, 0, :])
            values.append(out[:, :, :, 1, :])
        return AxiomKV(n_layers=self.n_layers, keys=keys, values=values)


def build_axiom_kv(hypernet: KVHypernet, code: AxiomCode, model, tokenizer) -> AxiomKV:  # noqa: ANN001
    """Load-time reconstruction: scaffold(z) ++ verbatim facts → injectable KV.

    Scaffold comes from the tiny latent; facts are encoded losslessly and
    concatenated (RoPE-corrected via merge_axiom_kvs).
    """
    device = next(model.parameters()).device
    scaffold = hypernet.decode_scaffold(code.z, device)
    facts_kv = compute_axiom_kv(model, tokenizer, code.fact_text, term=code.term)
    return merge_axiom_kvs([scaffold, facts_kv], _get_rope_theta(model))


def make_axiom_code(hypernet: KVHypernet, full_kv: AxiomKV, fact_text: str, term: str) -> AxiomCode:
    """Realtime registration: full KV → tiny stored code. No training."""
    return AxiomCode(term=term, z=hypernet.encode(full_kv).cpu(), fact_text=fact_text)


def train_hypernet(*args, **kwargs):  # noqa: ANN002, ANN003, ANN201
    """STUB. Train the decoder once, offline, on existing axioms.

    Recipe (mirror kv_compression.train_compressor):
      for step: sample axiom → build_axiom_kv(decoded) → inject + MLP hooks →
                cross-entropy on the answer → backprop into hypernet only.
    Weight the loss toward high-surprisal (fact) tokens so decoder capacity
    goes to structure, not to memorising the numbers (facts ride fact_text).
    Freeze after training → realtime add stays training-free.
    """
    raise NotImplementedError("train the shared decoder once on existing axioms")
