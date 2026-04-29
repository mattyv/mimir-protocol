"""Per-axiom prefix tuning: learn K/V tensors at every transformer layer.

Difference from soft prompt: soft prompt replaces the term's embedding at
L0 (one tiny vector). Prefix tuning prepends N "virtual tokens" worth of
attention K/V at every layer — every position in the user's prompt can
attend to those K/V slots at every layer.

Why this might break the field-steering ceiling: a frozen MLP can only
retrieve facts it learned at pretrain. Specific axiom facts aren't in
MLP. They live in the *composed attention K/V state* the model would
build up after reading the description. Prefix tuning learns K/V tensors
that mimic that composed state — pre-installing attention working memory
the model would otherwise have to construct from prompt tokens.

Init from the actual model K/V after processing the description, then
gradient-refine on contrastive paraphrase loss.

Storage per axiom: num_layers × num_kv_heads × n_tokens × head_dim × 2
(K and V) tensors. ~5 MB on Qwen 32B at N=20.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path  # noqa: F401

import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache


def _get_layers(model):  # noqa: ANN001
    """Find layers across Qwen / Gemma / multimodal architectures."""
    base = model
    if hasattr(base, "base_model") and hasattr(base.base_model, "model"):
        base = base.base_model.model
    candidates = [
        lambda m: m.model.layers,
        lambda m: m.language_model.model.layers,
        lambda m: m.model.language_model.model.layers,
        lambda m: m.model.language_model.layers,
        lambda m: m.language_model.layers,
    ]
    for fn in candidates:
        try:
            layers = fn(base)
            if hasattr(layers, "__len__") and len(layers) > 0:
                return layers
        except (AttributeError, TypeError):
            continue
    for name, mod in base.named_modules():
        if name.endswith(".layers") and hasattr(mod, "__len__") and len(mod) > 1:
            return mod
    raise RuntimeError(f"could not find layers on {type(model).__name__}")


@dataclass
class Prefix:
    """Per-layer learnable K/V prefix tensors.

    Sparse-layer support: `target_layers` lists which transformer layer
    indices receive prefix injection. `keys[i]` / `values[i]` correspond
    to `target_layers[i]`. Other layers run with empty cache (no
    injection) — reduces prefix dominance vs all-layer injection, which
    causes looping.

    keys[i]: nn.Parameter of shape (1, n_kv_heads, n_tokens, head_dim)
    values[i]: nn.Parameter of shape (1, n_kv_heads, n_tokens, head_dim)
    """

    n_tokens: int
    n_total_layers: int
    n_kv_heads: int
    head_dim: int
    target_layers: list[int] = field(default_factory=list)
    keys: list[nn.Parameter] = field(default_factory=list)
    values: list[nn.Parameter] = field(default_factory=list)

    @classmethod
    def from_description(
        cls,
        model,  # noqa: ANN001
        tokenizer,
        description: str,
        max_tokens: int = 32,
        target_layers: list[int] | None = None,
    ) -> Prefix:
        """Init from the model's K/V cache after processing the
        description text, kept only at `target_layers` (default: all).
        """
        device = next(model.parameters()).device
        ids = tokenizer(description, return_tensors="pt", add_special_tokens=False).input_ids.to(
            device
        )
        with torch.no_grad():
            out = model(ids, use_cache=True)
        past = out.past_key_values
        layers = _get_layers(model)
        n_total_layers = len(layers)
        if target_layers is None:
            target_layers = list(range(n_total_layers))
        # Extract per-layer (K, V) and truncate
        if isinstance(past, DynamicCache):
            kv_iter = [(past.layers[i].keys, past.layers[i].values) for i in range(n_total_layers)]
        else:
            kv_iter = [(past[i][0], past[i][1]) for i in range(n_total_layers)]
        keys: list[nn.Parameter] = []
        values: list[nn.Parameter] = []
        for L in target_layers:
            k, v = kv_iter[L]
            seq = k.shape[2]
            take = min(seq, max_tokens)
            k_trunc = k[:, :, -take:, :].detach().float().clone()
            v_trunc = v[:, :, -take:, :].detach().float().clone()
            keys.append(nn.Parameter(k_trunc))
            values.append(nn.Parameter(v_trunc))
        n_kv_heads = keys[0].shape[1]
        head_dim = keys[0].shape[3]
        n_tokens = keys[0].shape[2]
        return cls(
            n_tokens=n_tokens,
            n_total_layers=n_total_layers,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            target_layers=list(target_layers),
            keys=keys,
            values=values,
        )

    def parameters(self) -> list[nn.Parameter]:
        return self.keys + self.values

    def to_cache(self, dtype: torch.dtype, device: torch.device) -> DynamicCache:
        """Build a DynamicCache populated at every layer with same-length
        K/V tensors. Non-target layers get all-zero K/V (V=0 means the
        layer's prefix tokens contribute nothing to the residual stream
        even though they take up cache slots). This keeps the attention
        mask shape consistent across layers.
        """
        cache = DynamicCache()
        target_set = set(self.target_layers)
        zero_shape = (1, self.n_kv_heads, self.n_tokens, self.head_dim)
        for layer_idx in range(self.n_total_layers):
            if layer_idx in target_set:
                i = self.target_layers.index(layer_idx)
                k_d = self.keys[i].to(dtype=dtype, device=device)
                v_d = self.values[i].to(dtype=dtype, device=device)
            else:
                k_d = torch.zeros(zero_shape, dtype=dtype, device=device)
                v_d = torch.zeros(zero_shape, dtype=dtype, device=device)
            cache.update(k_d, v_d, layer_idx)
        return cache


def _model_dtype(model) -> torch.dtype:  # noqa: ANN001
    return next(model.parameters()).dtype


def _train_step_nll(
    model,  # noqa: ANN001
    tokenizer,
    prefix: Prefix,
    text: str,
) -> torch.Tensor:
    """One NLL-loss training step: predict text tokens conditioned on prefix."""
    device = next(model.parameters()).device
    dtype = _model_dtype(model)
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    if ids.shape[1] < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)
    cache = prefix.to_cache(dtype, device)
    out = model(ids, past_key_values=cache, use_cache=False)
    # Cross-entropy on text tokens 1..end (autoregressive)
    logits = out.logits[0, :-1]
    targets = ids[0, 1:]
    return torch.nn.functional.cross_entropy(logits, targets)


def train_prefix_contrastive(
    model,  # noqa: ANN001
    tokenizer,
    prefix: Prefix,
    intended_paraphrases: list[str],
    lexical_paraphrases: list[str] | None = None,
    n_steps: int = 60,
    lr: float = 0.005,
    margin: float = 1.0,
    weight_decay: float = 0.01,
    anchor_weight: float = 0.5,
    seed: int = 0,
) -> list[float]:
    """Gradient-refine the prefix K/V tensors.

    Modes:
      - With `lexical_paraphrases`: contrastive (NLL_int + relu(margin
        - (NLL_lex - NLL_int))). Pushes prefix toward intended sense and
        away from lexical sense.
      - Without: NLL_int only. Single-class.

    `anchor_weight`: L2 penalty toward the init values of the prefix.
    Prevents training from drifting away from the description-init state
    (which is a strong starting point — gradient noise can degrade it).
    """
    for p in model.parameters():
        p.requires_grad_(False)
        p.grad = None
    model.eval()

    params = prefix.parameters()
    for p in params:
        p.requires_grad_(True)
    # Snapshot init values as anchors (detached, on training device)
    init_keys = [k.detach().clone() for k in prefix.keys]
    init_values = [v.detach().clone() for v in prefix.values]

    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    rng = torch.Generator().manual_seed(seed)

    def _sample(paraphrases: list[str]) -> str:
        idx = int(torch.randint(0, len(paraphrases), (1,), generator=rng).item())
        return paraphrases[idx].replace("[[", "").replace("]]", "")

    def _anchor_loss() -> torch.Tensor:
        loss = torch.tensor(0.0, device=prefix.keys[0].device, dtype=torch.float32)
        for cur, init in zip(prefix.keys, init_keys, strict=True):
            loss = loss + (cur - init).pow(2).mean()
        for cur, init in zip(prefix.values, init_values, strict=True):
            loss = loss + (cur - init).pow(2).mean()
        return loss / (2 * len(prefix.keys))

    losses: list[float] = []
    use_contrastive = bool(lexical_paraphrases)
    for _step in range(n_steps):
        opt.zero_grad()
        for p in model.parameters():
            p.grad = None

        i_text = _sample(intended_paraphrases)
        loss_int = _train_step_nll(model, tokenizer, prefix, i_text)
        if use_contrastive:
            l_text = _sample(lexical_paraphrases)
            loss_lex = _train_step_nll(model, tokenizer, prefix, l_text)
            gap = loss_lex - loss_int
            contrastive = torch.nn.functional.relu(margin - gap)
            loss = loss_int + contrastive
        else:
            loss = loss_int

        if anchor_weight > 0:
            loss = loss + anchor_weight * _anchor_loss()

        if loss.requires_grad:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            opt.step()
        losses.append(float(loss.item()))
    return losses


def train_prefix_multiseed(
    model,  # noqa: ANN001
    tokenizer,
    prefix: Prefix,
    intended_paraphrases: list[str],
    lexical_paraphrases: list[str] | None = None,
    n_seeds: int = 3,
    **kwargs,
) -> tuple[list[float], int]:
    """Train n_seeds times, keep best by min-loss tail window. The
    prefix's K/V tensors are left set to the best run's values.
    """
    init_keys = [k.detach().clone() for k in prefix.keys]
    init_values = [v.detach().clone() for v in prefix.values]
    best_score = float("inf")
    best_losses: list[float] = []
    best_seed = 0
    best_keys = [k.detach().clone() for k in prefix.keys]
    best_values = [v.detach().clone() for v in prefix.values]
    for seed in range(n_seeds):
        # Reset to init for each seed
        with torch.no_grad():
            for cur, snap in zip(prefix.keys, init_keys, strict=True):
                cur.copy_(snap)
            for cur, snap in zip(prefix.values, init_values, strict=True):
                cur.copy_(snap)
        kwargs["seed"] = seed
        losses = train_prefix_contrastive(
            model, tokenizer, prefix, intended_paraphrases, lexical_paraphrases, **kwargs
        )
        window = min(10, len(losses))
        score = min(losses[-window:]) if window > 0 else float("inf")
        if score < best_score:
            best_score = score
            best_losses = losses
            best_seed = seed
            best_keys = [k.detach().clone() for k in prefix.keys]
            best_values = [v.detach().clone() for v in prefix.values]
    with torch.no_grad():
        for cur, best in zip(prefix.keys, best_keys, strict=True):
            cur.copy_(best)
        for cur, best in zip(prefix.values, best_values, strict=True):
            cur.copy_(best)
    return best_losses, best_seed


@torch.no_grad()
def generate_with_prefix(
    model,  # noqa: ANN001
    tokenizer,
    prompt: str,
    prefix: Prefix | None,
    max_new: int = 60,
) -> str:
    """Greedy decode with prefix K/V prepended to attention cache."""
    device = next(model.parameters()).device
    dtype = _model_dtype(model)
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    cache = prefix.to_cache(dtype, device) if prefix is not None else DynamicCache()
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
    full = tokenizer.decode(full_ids[0], skip_special_tokens=True)
    return full[len(prompt) :]
