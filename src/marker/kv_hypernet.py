"""Hypernetwork axiom store: per-axiom latent + shared decoder → KV.

Contrast with kv_compression.KVCompressor, which stores the compressed KV
*output* per layer. Here we store, per axiom, only:

    AxiomCode = (z: small latent, fact_text: verbatim facts)

and a single SHARED KVHypernet (encoder + decoder), trained once offline and
frozen. The decoder expands z into the "scaffold" KV (the predictable prose
structure). The facts — max-entropy values like "250ms", "balances.raw" — are
NOT produced by the net; they're encoded verbatim via compute_axiom_kv and
concatenated, so they're lossless.

Why this shape:
  - Realtime add: register an axiom = encode(full_kv) → z  (one forward pass,
    NO training) + keep fact_text. The heavy net is trained once offline, so
    axiom #1000 is as fast to add as axiom #1.
  - Facts are solved by not compressing them: z carries structure, fact_text
    carries information.
  - Reuses merge_axiom_kvs (RoPE-corrected) to stitch scaffold + facts.

Training (train_hypernet) optimises encode→decode end-to-end against answer
cross-entropy, so the frozen encoder generalises to new axioms.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from marker.prefix_tuning import _get_rope_theta
from marker.run_axiom_mlp_demo import (
    TEMPLATE,
    AxiomKV,
    AxiomMLP,
    _build_dynamic_cache,
    _find_term_positions,
    compute_axiom_kv,
    install_hooks,
    merge_axiom_kvs,
)


@dataclass
class AxiomCode:
    """What we store per axiom instead of a full/compressed KV."""

    term: str
    z: torch.Tensor  # (d_latent,) — compressible structure
    fact_text: str  # verbatim irreducible facts, e.g. "poll_interval=250ms; topic=balances.raw"


class KVHypernet(nn.Module):
    """Shared encoder+decoder. encode: full KV → z. decode: z → scaffold KV."""

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

        # encode: per-layer pooled (K,V) → partial latent; mean over layers → z
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

    def encode(self, full_kv: AxiomKV) -> torch.Tensor:
        """Full KV → z. Differentiable (train jointly); wrap in no_grad to add."""
        device = self.layer_embed.weight.device
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


def assemble_kv(scaffold: AxiomKV, facts_kv: AxiomKV, rope_theta: float) -> AxiomKV:
    """[scaffold ++ verbatim facts] → one injectable KV (RoPE-corrected).

    Scaffold is cast to the facts' dtype/device so the concat is clean; facts'
    keys are re-rotated by the scaffold length inside merge_axiom_kvs.
    """
    dtype = facts_kv.keys[0].dtype
    device = facts_kv.keys[0].device
    scaffold = AxiomKV(
        n_layers=scaffold.n_layers,
        keys=[k.to(device=device, dtype=dtype) for k in scaffold.keys],
        values=[v.to(device=device, dtype=dtype) for v in scaffold.values],
    )
    return merge_axiom_kvs([scaffold, facts_kv], rope_theta)


def build_axiom_kv(hypernet: KVHypernet, code: AxiomCode, model, tokenizer) -> AxiomKV:  # noqa: ANN001
    """Load-time reconstruction: scaffold(z) ++ verbatim facts → injectable KV."""
    device = next(model.parameters()).device
    scaffold = hypernet.decode_scaffold(code.z, device)
    facts_kv = compute_axiom_kv(model, tokenizer, code.fact_text, term=code.term)
    return assemble_kv(scaffold, facts_kv, _get_rope_theta(model))


def make_axiom_code(hypernet: KVHypernet, full_kv: AxiomKV, fact_text: str, term: str) -> AxiomCode:
    """Realtime registration: full KV → tiny stored code. No training."""
    with torch.no_grad():
        z = hypernet.encode(full_kv).detach().cpu()
    return AxiomCode(term=term, z=z, fact_text=fact_text)


def train_hypernet(
    model,  # noqa: ANN001
    tokenizer,  # noqa: ANN001
    hypernet: KVHypernet,
    axiom_mlps: list[AxiomMLP],
    fact_texts: dict[str, str],  # term → verbatim fact string
    qa_map: dict[str, list[tuple[str, str]]],  # term → list of (q, a)
    n_steps: int = 1500,
    lr: float = 1e-3,
) -> KVHypernet:
    """Train encoder+decoder once, offline, on existing axioms.

    Each step: sample an axiom → z = encode(full_kv) → decode → concat verbatim
    facts → inject + MLP hooks → cross-entropy on the answer → backprop into the
    hypernet only. Facts' KV is precomputed once per axiom (fixed content).

    Model and axiom-MLP weights are frozen. After training, freeze the hypernet;
    adding a new axiom is then encode()-only (no training).
    """
    for p in model.parameters():
        p.requires_grad_(False)
    for a in axiom_mlps:
        for p in a.mlps.parameters():
            p.requires_grad_(False)

    device = next(model.parameters()).device
    hypernet = hypernet.to(device=device, dtype=torch.float32)
    rope_theta = _get_rope_theta(model)

    # Precompute the verbatim-fact KV per axiom (content is fixed).
    facts_kv_by_term: dict[str, AxiomKV] = {}
    trainable = []
    for a in axiom_mlps:
        if a.kv is None or a.term not in fact_texts or not qa_map.get(a.term):
            continue
        facts_kv_by_term[a.term] = compute_axiom_kv(
            model, tokenizer, fact_texts[a.term], term=a.term
        )
        trainable.append(a)
    if not trainable:
        raise ValueError("no axioms with kv + fact_text + qa to train on")

    optimizer = torch.optim.AdamW(hypernet.parameters(), lr=lr)
    eos_id = tokenizer.eos_token_id
    rng = random.Random(42)
    losses: list[float] = []
    skipped = 0

    for _ in range(n_steps):
        a = rng.choice(trainable)
        q, ans = rng.choice(qa_map[a.term])
        q_text = TEMPLATE.format(q=q)
        full_text = q_text + " " + ans

        q_ids = tokenizer(q_text, add_special_tokens=False, return_tensors="pt").input_ids.to(
            device
        )
        full_ids = tokenizer(full_text, add_special_tokens=False, return_tensors="pt").input_ids.to(
            device
        )
        if eos_id is not None:
            full_ids = torch.cat([full_ids, torch.tensor([[eos_id]], device=device)], dim=1)

        positions = _find_term_positions(q_ids, a.term_token_ids)
        if not positions:
            skipped += 1
            continue

        labels = torch.full_like(full_ids, -100)
        labels[0, q_ids.shape[1] :] = full_ids[0, q_ids.shape[1] :]

        # encode→decode with gradients; concat precomputed facts.
        z = hypernet.encode(a.kv)
        scaffold = hypernet.decode_scaffold(z, device)
        merged = assemble_kv(scaffold, facts_kv_by_term[a.term], rope_theta)
        kv_cache = _build_dynamic_cache(merged, device)

        handles = install_hooks(model, a, positions)
        try:
            optimizer.zero_grad()
            loss = model(full_ids, past_key_values=kv_cache, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(hypernet.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        finally:
            for h in handles:
                h.remove()

    if skipped:
        print(f"  hypernet: skipped {skipped}/{n_steps} steps (term not found)")
    if losses:
        print(f"  hypernet loss: {losses[0]:.3f} → {losses[-1]:.4f}")
    return hypernet


def facts_to_text(axiom: dict) -> str:
    """Build a compact verbatim-fact string from an axiom's fact answers."""
    return " ".join(f["answer"] for f in axiom["facts"])


# ── Persistence ────────────────────────────────────────────────────────────────


def save_axiom_code(code: AxiomCode, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"term": code.term, "z": code.z.cpu(), "fact_text": code.fact_text}, path)


def load_axiom_code(path: str | Path) -> AxiomCode:
    d = torch.load(path, map_location="cpu", weights_only=False)
    return AxiomCode(term=d["term"], z=d["z"], fact_text=d["fact_text"])


def save_hypernet(hypernet: KVHypernet, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": {
                "n_layers": hypernet.n_layers,
                "n_kv_heads": hypernet.n_kv_heads,
                "head_dim": hypernet.head_dim,
                "d_latent": hypernet.d_latent,
                "n_scaffold": hypernet.n_scaffold,
            },
            "state": hypernet.state_dict(),
        },
        path,
    )


def load_hypernet(path: str | Path) -> KVHypernet:
    d = torch.load(path, map_location="cpu", weights_only=False)
    hypernet = KVHypernet(**d["config"])
    hypernet.load_state_dict(d["state"])
    return hypernet
