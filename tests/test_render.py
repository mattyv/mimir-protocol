"""Tests for the render decoder (render.py): thought -> its own step's text.

The RENDER path (on-demand "show me this thought"). Distinct from every decode
so far, which ran the CONTINUATION direction (what follows a thought); render
RECONSTRUCTS the source step from its thought — transcription, not prediction.
Trained: a LoRA on the frozen model, CE on the source span given the injected
thought KV.

CPU tests on a tiny model: reconstruction loss is finite, gradients reach the
render LoRA (the GRAD_OK trap — a silent detach would fake training), and the
loss drops when overfitting one (thought, span) pair.
"""

from __future__ import annotations

import pytest
import torch

from marker.render import attach_render, render_nll


def _tiny():
    from marker.gist_model import attach_gist
    from tests.test_gist_model import _tiny_base

    base = _tiny_base()
    pm, gist = attach_gist(base, gist_k=4, r=4)  # the (frozen) encoder
    return base, pm, gist


@pytest.mark.slow
def test_render_nll_finite_and_grads_reach_render_lora():
    from marker.gist_model import gist_kv

    base, pm, gist = _tiny()
    span = [1, 2, 3, 4]
    kv, cont_start, _ = gist_kv(pm, gist, span)  # encoder ('default') active
    render = attach_render(pm, r=4)  # adds + activates the trainable 'render' LoRA
    loss = render_nll(pm, kv, cont_start, span)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad.abs().sum() for _, p in render if p.grad is not None]
    assert grads and any(g > 0 for g in grads), "no gradient reached the render LoRA"


@pytest.mark.slow
def test_render_overfits_one_pair():
    torch.manual_seed(0)
    from marker.gist_model import gist_kv

    base, pm, gist = _tiny()
    span = [3, 1, 4, 1, 5]
    # encode ONCE — the encoder is frozen, so the thought is constant
    kv, cont_start, _ = gist_kv(pm, gist, span)
    render = attach_render(pm, r=4)
    opt = torch.optim.AdamW([p for _, p in render], lr=1e-2)
    first = best = None
    for i in range(40):
        opt.zero_grad()
        loss = render_nll(pm, kv, cont_start, span)
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.item()
        best = loss.item() if best is None else min(best, loss.item())
    assert best < first - 0.3, f"render didn't learn to reconstruct: {first:.3f} -> {best:.3f}"
