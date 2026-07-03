"""True per-axiom prefix tuning: trained virtual KV tokens, no text, no MLP.

See PREFIX_POC_PLAN.md for the full design and pre-registered kill criteria.

Unlike compute_axiom_kv (which captures the KV of real description text) and
kv_hypernet (which decodes a latent into KV via a shared network), this module
trains N virtual K/V tokens per layer *from scratch* against Q+A
cross-entropy, directly as free parameters — nothing here is derived from or
constrained to look like real token KV, except at initialisation.

Per-axiom artifact: AxiomPrefix, shape (n_layers, n_kv_heads, N, head_dim) for
keys and values. N is fixed at training time; there is no term-position
logic, no MLP hooks, no multi-axiom RoPE composition — this POC evaluates one
axiom's prefix in isolation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch

from marker.run_axiom_mlp_demo import TEMPLATE, AxiomKV


@dataclass
class AxiomPrefix:
    """A trained, per-axiom virtual KV prefix — N free K/V tokens per layer."""

    term: str
    n_layers: int
    n_kv_heads: int
    head_dim: int
    n_tokens: int
    keys: list[torch.Tensor]  # each (1, n_kv_heads, n_tokens, head_dim), fp32, requires_grad
    values: list[torch.Tensor]  # same

    def parameters(self) -> list[torch.Tensor]:
        return [*self.keys, *self.values]


# ── Init ──────────────────────────────────────────────────────────────────────


def init_stat_matched(real_kv: AxiomKV, n_tokens: int, term: str, seed: int = 0) -> AxiomPrefix:
    """Random init, scaled to match the real description KV's per-(layer,head,
    channel) mean/std. Unit-normal init would sit off the residual stream's
    actual scale and break attention before training even starts.
    """
    n_layers = real_kv.n_layers
    n_kv_heads = real_kv.keys[0].shape[1]
    head_dim = real_kv.keys[0].shape[3]
    device = real_kv.keys[0].device
    g = torch.Generator(device="cpu").manual_seed(seed)

    keys, values = [], []
    for layer in range(n_layers):
        k = real_kv.keys[layer].float()  # (1, n_kv_heads, seq, head_dim)
        v = real_kv.values[layer].float()
        k_mean, k_std = k.mean(dim=2, keepdim=True), k.std(dim=2, keepdim=True).clamp_min(1e-4)
        v_mean, v_std = v.mean(dim=2, keepdim=True), v.std(dim=2, keepdim=True).clamp_min(1e-4)

        noise_k = torch.randn(1, n_kv_heads, n_tokens, head_dim, generator=g).to(device)
        noise_v = torch.randn(1, n_kv_heads, n_tokens, head_dim, generator=g).to(device)
        init_k = (k_mean + noise_k * k_std).detach().clone().requires_grad_(True)
        init_v = (v_mean + noise_v * v_std).detach().clone().requires_grad_(True)
        keys.append(init_k)
        values.append(init_v)

    return AxiomPrefix(
        term=term,
        n_layers=n_layers,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        n_tokens=n_tokens,
        keys=keys,
        values=values,
    )


def init_subsample(real_kv: AxiomKV, n_tokens: int, term: str, seed: int = 0) -> AxiomPrefix:
    """Init from N real token positions sampled from the description KV, then
    trained from there. Secondary init arm — on-manifold starting point vs
    init_stat_matched's off-manifold-but-correctly-scaled noise.
    """
    n_layers = real_kv.n_layers
    n_kv_heads = real_kv.keys[0].shape[1]
    head_dim = real_kv.keys[0].shape[3]
    seq_len = real_kv.keys[0].shape[2]
    rng = random.Random(seed)
    idx = (
        sorted(rng.sample(range(seq_len), n_tokens))
        if seq_len >= n_tokens
        else [i % seq_len for i in range(n_tokens)]  # wrap if description shorter than N
    )

    keys, values = [], []
    for layer in range(n_layers):
        k = real_kv.keys[layer][:, :, idx, :].detach().clone().float().requires_grad_(True)
        v = real_kv.values[layer][:, :, idx, :].detach().clone().float().requires_grad_(True)
        keys.append(k)
        values.append(v)

    return AxiomPrefix(
        term=term,
        n_layers=n_layers,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        n_tokens=n_tokens,
        keys=keys,
        values=values,
    )


# ── Cache building ──────────────────────────────────────────────────────────────


def build_prefix_cache(prefix: AxiomPrefix, dtype: torch.dtype):  # noqa: ANN201
    """Fresh DynamicCache from the prefix's params, cast to model dtype.

    Cast, not detach — during training this must stay attached to the graph
    so gradients flow from the loss back into the fp32 params.
    """
    from transformers import DynamicCache  # noqa: PLC0415

    cache = DynamicCache()
    for layer_idx in range(prefix.n_layers):
        cache.update(
            prefix.keys[layer_idx].to(dtype), prefix.values[layer_idx].to(dtype), layer_idx
        )
    return cache


# ── Training ──────────────────────────────────────────────────────────────────


def sample_qa(
    rng: random.Random,
    qa_pairs: list[tuple[str, str]] | None,
    qa_groups: list[list[tuple[str, str]]] | None = None,
) -> tuple[str, str]:
    """Pick a training pair. With qa_groups (one group per fact), sample the
    group uniformly first, then a pair within it — otherwise axioms whose
    facts have uneven paraphrase counts under-train the sparse facts.
    """
    if qa_groups:
        return rng.choice(rng.choice(qa_groups))
    if not qa_pairs:
        raise ValueError("need qa_pairs or qa_groups")
    return rng.choice(qa_pairs)


def train_prefix(
    model,  # noqa: ANN001
    tokenizer,  # noqa: ANN001
    prefix: AxiomPrefix,
    qa_pairs: list[tuple[str, str]] | None = None,
    n_steps: int = 800,
    lr: float = 5e-3,
    lr_end: float = 5e-4,
    weight_decay: float = 0.0,
    seed: int = 42,
    qa_groups: list[list[tuple[str, str]]] | None = None,
    templates: list[str] | None = None,
) -> list[float]:
    """Train prefix.keys/values directly against Q+A cross-entropy.

    Model weights are frozen. A fresh DynamicCache is built from the prefix
    params each step (the cache is stateful and gets extended by the forward
    pass, so it can't be reused across steps).

    qa_groups (one list per fact) enables fact-balanced sampling; templates
    (prompt formats containing "{q}") are sampled per step so the prefix
    doesn't overfit to a single question framing.
    """
    for p in model.parameters():
        p.requires_grad_(False)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    params = prefix.parameters()
    optim = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=n_steps, eta_min=lr_end)
    eos_id = tokenizer.eos_token_id
    rng = random.Random(seed)
    losses: list[float] = []
    templates = templates or [TEMPLATE]

    for _ in range(n_steps):
        q, a = sample_qa(rng, qa_pairs, qa_groups)
        q_text = rng.choice(templates).format(q=q)
        full_text = q_text + " " + a

        q_ids = tokenizer(q_text, add_special_tokens=False, return_tensors="pt").input_ids.to(
            device
        )
        full_ids = tokenizer(full_text, add_special_tokens=False, return_tensors="pt").input_ids.to(
            device
        )
        if eos_id is not None:
            full_ids = torch.cat([full_ids, torch.tensor([[eos_id]], device=device)], dim=1)

        labels = torch.full_like(full_ids, -100)
        labels[0, q_ids.shape[1] :] = full_ids[0, q_ids.shape[1] :]

        kv_cache = build_prefix_cache(prefix, dtype)
        optim.zero_grad()
        loss = model(full_ids, past_key_values=kv_cache, labels=labels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optim.step()
        scheduler.step()
        losses.append(float(loss.item()))

    if not losses:
        raise RuntimeError("no training steps ran")
    return losses


# ── Inference ─────────────────────────────────────────────────────────────────


@torch.no_grad()
def generate_with_cache(
    model,  # noqa: ANN001
    tokenizer,  # noqa: ANN001
    prompt: str,
    kv_cache=None,  # noqa: ANN001
    max_new: int = 60,
) -> str:
    """Plain greedy decode with an optional prebuilt KV cache. No hooks, no
    term-position logic — the caller decides what (if anything) sits in the
    cache before this runs (ZERO: None; FACTS: a text-KV cache; PREFIX-N: a
    build_prefix_cache() result).
    """
    device = next(model.parameters()).device
    ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    out = model(ids, past_key_values=kv_cache, use_cache=True)
    past = out.past_key_values
    next_tok = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
    out_ids = [next_tok]
    for _ in range(max_new - 1):
        out = model(next_tok, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        out_ids.append(next_tok)
        if int(next_tok.item()) == tokenizer.eos_token_id:
            break

    return tokenizer.decode(torch.cat(out_ids, dim=1)[0], skip_special_tokens=True).strip()


# ── Persistence ────────────────────────────────────────────────────────────────


def save_axiom_prefix(prefix: AxiomPrefix, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "term": prefix.term,
            "n_layers": prefix.n_layers,
            "n_kv_heads": prefix.n_kv_heads,
            "head_dim": prefix.head_dim,
            "n_tokens": prefix.n_tokens,
            "keys": [k.detach().cpu() for k in prefix.keys],
            "values": [v.detach().cpu() for v in prefix.values],
        },
        path,
    )


def load_axiom_prefix(path: str | Path) -> AxiomPrefix:
    d = torch.load(path, map_location="cpu", weights_only=False)
    return AxiomPrefix(
        term=d["term"],
        n_layers=d["n_layers"],
        n_kv_heads=d["n_kv_heads"],
        head_dim=d["head_dim"],
        n_tokens=d["n_tokens"],
        keys=[k.clone().requires_grad_(True) for k in d["keys"]],
        values=[v.clone().requires_grad_(True) for v in d["values"]],
    )
