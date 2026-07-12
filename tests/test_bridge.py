"""Tests for the Stage-3b bridge (bridge.py): predicted thought -> injectable KV.

The Stage-2 predictor outputs a thought as k final-layer vectors; decoding
needs the per-layer K/V form. The bridge converts one to the other and is
trained THROUGH the injection loss (convert -> inject -> NLL of the true next
step), never by regressing KV tensors (Fable steer: under-determined).

CPU tests on a tiny model: output shapes/injectability, gradient flow into the
bridge through the frozen model's attention over the injected cache (the
GRAD_OK trap), and loss decrease when overfitting one batch.
"""

from __future__ import annotations

import pytest
import torch

from marker.bridge import GistBridge, bridge_injection_nll


def _tiny():
    from marker.gist_model import attach_gist
    from tests.test_gist_model import _tiny_base

    base = _tiny_base()
    pm, gist = attach_gist(base, gist_k=4, r=4)
    return base, pm, gist


def test_bridge_output_is_injectable_kv_shapes():
    base, pm, gist = _tiny()
    cfg = base.config
    b = GistBridge(
        d=cfg.hidden_size,
        k=4,
        n_layers=cfg.num_hidden_layers,
        n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.hidden_size // cfg.num_attention_heads,
    )
    thought = torch.randn(4, cfg.hidden_size)  # [k, d] — a predictor output
    kv = b(thought)
    assert kv.n_layers == cfg.num_hidden_layers
    for kmat, vmat in zip(kv.keys, kv.values, strict=True):
        assert kmat.shape == (
            1,
            cfg.num_key_value_heads,
            4,
            cfg.hidden_size // cfg.num_attention_heads,
        )
        assert vmat.shape == kmat.shape
    # injectable through the existing runtime
    from marker.run_axiom_mlp_demo import _build_dynamic_cache

    assert _build_dynamic_cache(kv, torch.device("cpu")) is not None


@pytest.mark.slow
def test_injection_nll_gradients_reach_the_bridge():
    # the GRAD_OK trap: loss flows through the frozen model's attention over
    # the injected cache back into bridge params — a silent detach would fake
    # 'trained' while learning nothing.
    base, pm, gist = _tiny()
    cfg = base.config
    b = GistBridge(
        d=cfg.hidden_size,
        k=4,
        n_layers=cfg.num_hidden_layers,
        n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.hidden_size // cfg.num_attention_heads,
    )
    thought = torch.randn(4, cfg.hidden_size)
    loss = bridge_injection_nll(pm, b, thought, cont_ids=[5, 6, 7], cont_start=10)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad.abs().sum() for p in b.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g) for g in grads)
    assert sum(g > 0 for g in grads) > 0, "no gradient reached the bridge (silent detach?)"


@pytest.mark.slow
def test_bridge_overfits_one_batch():
    torch.manual_seed(0)
    base, pm, gist = _tiny()
    cfg = base.config
    b = GistBridge(
        d=cfg.hidden_size,
        k=4,
        n_layers=cfg.num_hidden_layers,
        n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.hidden_size // cfg.num_attention_heads,
    )
    thought = torch.randn(4, cfg.hidden_size)
    opt = torch.optim.AdamW(b.parameters(), lr=1e-2)
    first = best = None
    for i in range(40):
        opt.zero_grad()
        loss = bridge_injection_nll(pm, b, thought, cont_ids=[5, 6, 7, 8], cont_start=10)
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.item()
        best = loss.item() if best is None else min(best, loss.item())
    # the injected thought only MODULATES logits the continuation's own tokens
    # mostly carry, so there's a floor (tighter on a tiny random model than the
    # trained 7B). Assert a clear NLL drop — the bridge demonstrably learns
    # through the injection loss — not an arbitrary ratio it can't reach.
    assert best < first - 0.3, f"bridge didn't learn through injection: {first:.3f} -> {best:.3f}"
