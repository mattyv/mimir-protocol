"""KV cache compression for axiom descriptions.

Trains a small shared encoder that compresses a full description KV
(~42 tokens × 64 layers) down to N virtual tokens per layer.
Reduces per-axiom KV storage from ~12 MB to ~0.3-1 MB.

The encoder is trained end-to-end on Q+A pairs: the compressed KV is
injected in place of the full KV and the loss is cross-entropy on answers.
This directly optimises for retrieval quality rather than KV reconstruction.

Usage:
    compressor = KVCompressor(n_layers=64, n_kv_heads=8, head_dim=128, n_compressed=4)
    compressor = train_compressor(model, tokenizer, compressor, axiom_mlps, qa_map)
    for axiom_mlp in axiom_mlps:
        axiom_mlp.kv = compressor.compress(axiom_mlp.kv)
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn

from marker.run_axiom_mlp_demo import (
    TEMPLATE,
    AxiomKV,
    AxiomMLP,
    _build_dynamic_cache,
    _find_term_positions,
    install_hooks,
)


class KVCompressor(nn.Module):
    """Shared MLP encoder that compresses a full axiom KV to N virtual tokens.

    Processes each (layer, head) pair independently with a shared MLP and a
    learned layer embedding. This keeps parameter count small (~2M) while
    allowing the encoder to specialise per layer.

    Input per (layer, head):
        mean-pooled K (head_dim) + mean-pooled V (head_dim) + layer_embed (embed_dim)
    Output per (layer, head):
        n_compressed × (K + V) tokens → (n_compressed, 2, head_dim)
    """

    def __init__(
        self,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        n_compressed: int = 4,
        embed_dim: int = 32,
        bottleneck: int = 256,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_compressed = n_compressed

        self.layer_embed = nn.Embedding(n_layers, embed_dim)
        input_dim = 2 * head_dim + embed_dim
        output_dim = n_compressed * 2 * head_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, output_dim),
        )

    def compress(self, axiom_kv: AxiomKV) -> AxiomKV:
        """Return a new AxiomKV with n_compressed tokens per layer."""
        device = axiom_kv.keys[0].device
        comp_keys, comp_values = [], []

        for layer_idx in range(axiom_kv.n_layers):
            k = axiom_kv.keys[layer_idx].to(device)  # (1, n_kv_heads, seq_len, head_dim)
            v = axiom_kv.values[layer_idx].to(device)

            k_mean = k.mean(dim=2)  # (1, n_kv_heads, head_dim)
            v_mean = v.mean(dim=2)

            l_emb = self.layer_embed(torch.tensor(layer_idx, device=device))  # (embed_dim,)
            l_emb = l_emb.view(1, 1, -1).expand(
                1, self.n_kv_heads, -1
            )  # (1, n_kv_heads, embed_dim)

            inp = torch.cat([k_mean, v_mean, l_emb], dim=-1).float()  # (1, n_kv_heads, input_dim)
            out = self.mlp(inp)  # (1, n_kv_heads, output_dim)
            out = out.view(1, self.n_kv_heads, self.n_compressed, 2, self.head_dim)

            k_c = out[:, :, :, 0, :].to(k.dtype)  # (1, n_kv_heads, n_compressed, head_dim)
            v_c = out[:, :, :, 1, :].to(v.dtype)

            comp_keys.append(k_c)
            comp_values.append(v_c)

        return AxiomKV(n_layers=axiom_kv.n_layers, keys=comp_keys, values=comp_values)


def train_compressor(
    model: object,
    tokenizer: object,
    compressor: KVCompressor,
    axiom_mlps: list[AxiomMLP],
    qa_map: dict[str, list[tuple[str, str]]],  # term → list of (q, a)
    n_steps: int = 1000,
    lr: float = 1e-3,
) -> KVCompressor:
    """Train compressor end-to-end on Q+A pairs.

    For each step: sample a random axiom, compress its KV, inject the compressed
    KV + MLP hooks, compute cross-entropy loss on the answer. The compressor
    learns to produce virtual tokens that preserve retrieval quality.

    Model weights and axiom MLP weights are frozen — only compressor is trained.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    for a in axiom_mlps:
        for p in a.mlps.parameters():
            p.requires_grad_(False)

    device = next(model.parameters()).device
    compressor = compressor.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(compressor.parameters(), lr=lr)
    eos_id = tokenizer.eos_token_id
    rng = random.Random(42)
    losses: list[float] = []
    skipped = 0

    for _ in range(n_steps):
        axiom_mlp = rng.choice(axiom_mlps)
        pairs = qa_map.get(axiom_mlp.term, [])
        if not pairs or axiom_mlp.kv is None:
            skipped += 1
            continue

        q, a = rng.choice(pairs)
        q_text = TEMPLATE.format(q=q)
        full_text = q_text + " " + a

        q_ids = tokenizer(q_text, add_special_tokens=False, return_tensors="pt").input_ids.to(
            device
        )
        full_ids = tokenizer(full_text, add_special_tokens=False, return_tensors="pt").input_ids.to(
            device
        )
        if eos_id is not None:
            full_ids = torch.cat([full_ids, torch.tensor([[eos_id]], device=device)], dim=1)

        positions = _find_term_positions(q_ids, axiom_mlp.term_token_ids)
        if not positions:
            skipped += 1
            continue

        labels = torch.full_like(full_ids, -100)
        labels[0, q_ids.shape[1] :] = full_ids[0, q_ids.shape[1] :]

        # Compress KV — gradients flow back into compressor
        compressed = compressor.compress(axiom_mlp.kv)
        kv_cache = _build_dynamic_cache(compressed, device)

        handles = install_hooks(model, axiom_mlp, positions)
        try:
            optimizer.zero_grad()
            loss = model(full_ids, past_key_values=kv_cache, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(compressor.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        finally:
            for h in handles:
                h.remove()

    if skipped:
        print(f"  compressor: skipped {skipped}/{n_steps} steps")
    if losses:
        print(f"  compressor loss: {losses[0]:.3f} → {losses[-1]:.4f}")
    return compressor


def apply_compression(
    compressor: KVCompressor,
    axiom_mlps: list[AxiomMLP],
) -> None:
    """Replace each axiom's full KV with its compressed version in-place."""
    device = next(compressor.parameters()).device
    for axiom_mlp in axiom_mlps:
        if axiom_mlp.kv is not None:
            with torch.no_grad():
                axiom_mlp.kv = compressor.compress(
                    AxiomKV(
                        n_layers=axiom_mlp.kv.n_layers,
                        keys=[k.to(device) for k in axiom_mlp.kv.keys],
                        values=[v.to(device) for v in axiom_mlp.kv.values],
                    )
                )
            # Move back to CPU to match original storage pattern
            axiom_mlp.kv = AxiomKV(
                n_layers=axiom_mlp.kv.n_layers,
                keys=[k.cpu() for k in axiom_mlp.kv.keys],
                values=[v.cpu() for v in axiom_mlp.kv.values],
            )
