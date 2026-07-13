"""Tests for the literals ledger (render.py): exact numbers/names kept beside
the thought and spliced back at render time.

Lossy meaning-compression drops exact digits (render run: number-recall 0.73).
The ledger stores the step's literal tokens (a few, cheap, deterministic) as a
VISIBLE prefix the render decoder can copy from — meaning from the thought,
exact numbers from the ledger.

Model-free: ledger extraction. Slow: ledger-conditioned render loss trains and
gradients reach the render LoRA.
"""

from __future__ import annotations

import pytest
import torch

from marker.render import extract_ledger


def test_extract_ledger_pulls_numbers_in_order():
    assert extract_ledger("Natalia sold 48/2 = 24 clips in May.") == ["48", "2", "24"]
    assert extract_ledger("no numbers here") == []
    # decimals kept WHOLE (a split "0.2" -> "0","2" would make the decoder
    # reassemble money amounts from fragments) and thousand-commas kept
    assert extract_ledger("earned 0.2 x 50 = 10.0") == ["0.2", "50", "10.0"]
    assert extract_ledger("price 1,000 then 2.5") == ["1,000", "2.5"]


def test_extract_ledger_dedup_optional_preserves_first_seen():
    assert extract_ledger("5 and 5 and 3", dedup=True) == ["5", "3"]
    assert extract_ledger("5 and 5 and 3", dedup=False) == ["5", "5", "3"]


@pytest.mark.slow
def test_ledger_nll_with_empty_ledger_equals_plain_render_nll():
    # THE alignment invariant (Fable): with no ledger, the ledger-conditioned
    # loss must equal the plain render loss EXACTLY — proves the scored
    # positions are the span, not shifted by the prefix bookkeeping.
    from marker.gist_model import attach_gist, gist_kv
    from marker.render import attach_render, ledger_render_nll, render_nll
    from tests.test_gist_model import _tiny_base

    base = _tiny_base()
    pm, gist = attach_gist(base, gist_k=4, r=4)
    span = [5, 6, 7, 8]
    kv, cont_start, _ = gist_kv(pm, gist, span)
    attach_render(pm, r=4)
    with torch.no_grad():
        a = ledger_render_nll(pm, kv, cont_start, [], span)
        b = render_nll(pm, kv, cont_start, span)
    assert abs(float(a) - float(b)) < 1e-5, f"empty-ledger parity broken: {a} vs {b}"


@pytest.mark.slow
def test_ledger_render_nll_grads_and_scores_only_the_step():
    from marker.gist_model import attach_gist, gist_kv
    from marker.render import attach_render, ledger_render_nll
    from tests.test_gist_model import _tiny_base

    base = _tiny_base()
    pm, gist = attach_gist(base, gist_k=4, r=4)
    span = [5, 6, 7, 8]
    kv, cont_start, _ = gist_kv(pm, gist, span)
    render = attach_render(pm, r=4)
    ledger_ids = [9, 9]  # stand-in literal tokens
    loss = ledger_render_nll(pm, kv, cont_start, ledger_ids, span)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad.abs().sum() for _, p in render if p.grad is not None]
    assert grads and any(g > 0 for g in grads), "no gradient reached the render LoRA"
