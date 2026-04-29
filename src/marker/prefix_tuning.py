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
    base = model
    if hasattr(base, "base_model") and hasattr(base.base_model, "model"):
        base = base.base_model.model
    candidates = [
        lambda m: m.model.layers,
        lambda m: m.language_model.model.layers,
    ]
    for fn in candidates:
        try:
            layers = fn(base)
            if hasattr(layers, "__len__") and len(layers) > 0:
                return layers
        except (AttributeError, TypeError):
            continue
    raise RuntimeError(f"could not find layers on {type(model).__name__}")


@dataclass
class Prefix:
    """Per-layer learnable K/V prefix tensors.

    keys[layer]: nn.Parameter of shape (1, n_kv_heads, n_tokens, head_dim)
    values[layer]: nn.Parameter of shape (1, n_kv_heads, n_tokens, head_dim)
    """

    n_tokens: int
    n_layers: int
    n_kv_heads: int
    head_dim: int
    keys: list[nn.Parameter] = field(default_factory=list)
    values: list[nn.Parameter] = field(default_factory=list)

    @classmethod
    def from_description(
        cls,
        model,  # noqa: ANN001
        tokenizer,
        description: str,
        max_tokens: int = 32,
    ) -> Prefix:
        """Init from the model's actual K/V cache after processing the
        description text. Truncated to `max_tokens` (final tokens kept,
        which carry the most-composed state)."""
        device = next(model.parameters()).device
        ids = tokenizer(description, return_tensors="pt", add_special_tokens=False).input_ids.to(
            device
        )
        with torch.no_grad():
            out = model(ids, use_cache=True)
        past = out.past_key_values
        # past is DynamicCache (modern transformers) or legacy tuple
        layers = _get_layers(model)
        n_layers = len(layers)
        # Extract per-layer (K, V) and truncate
        keys: list[nn.Parameter] = []
        values: list[nn.Parameter] = []
        if isinstance(past, DynamicCache):
            # Newer transformers: cache.layers[i].keys / .values
            kv_iter = [(past.layers[i].keys, past.layers[i].values) for i in range(n_layers)]
        else:
            kv_iter = [(past[i][0], past[i][1]) for i in range(n_layers)]
        for k, v in kv_iter:
            # k, v shape: (1, n_kv_heads, seq, head_dim)
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
            n_layers=n_layers,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            keys=keys,
            values=values,
        )

    def parameters(self) -> list[nn.Parameter]:
        return self.keys + self.values

    def to_cache(self, dtype: torch.dtype, device: torch.device) -> DynamicCache:
        """Build a fresh DynamicCache populated with the prefix K/V at
        every layer. Caller can pass this as `past_key_values` to model.
        Returns a DynamicCache; the contained tensors are *views* of the
        Parameter tensors (cast to model dtype) so backprop flows through.
        """
        cache = DynamicCache()
        for layer_idx, (k, v) in enumerate(zip(self.keys, self.values, strict=True)):
            k_d = k.to(dtype=dtype, device=device)
            v_d = v.to(dtype=dtype, device=device)
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
    seed: int = 0,
) -> list[float]:
    """Gradient-refine the prefix K/V tensors.

    Modes:
      - With `lexical_paraphrases`: contrastive (NLL_int + relu(margin
        - (NLL_lex - NLL_int))). Pushes prefix toward intended sense and
        away from lexical sense.
      - Without: NLL_int only. Single-class.
    """
    for p in model.parameters():
        p.requires_grad_(False)
        p.grad = None
    model.eval()

    params = prefix.parameters()
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    rng = torch.Generator().manual_seed(seed)

    def _sample(paraphrases: list[str]) -> str:
        idx = int(torch.randint(0, len(paraphrases), (1,), generator=rng).item())
        return paraphrases[idx].replace("[[", "").replace("]]", "")

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

        if loss.requires_grad:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            opt.step()
        losses.append(float(loss.item()))
    return losses


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
