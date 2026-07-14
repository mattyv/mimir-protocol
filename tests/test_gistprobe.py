"""Tests for the gist fidelity probe helpers (gistprobe.py)."""

from __future__ import annotations

import pytest
import torch

from marker.gistprobe import digit_token_mask, extract_relations, per_token_ce, relation_score


def test_extract_relations_normalizes():
    t = "She walks 1.5 x 4 = 6 miles, then 6×2 = 12, total 1,000 + 2 = 1,002."
    assert extract_relations(t) == ["1.5|*|4|6", "6|*|2|12", "1000|+|2|1002"]


def test_relation_score_catches_wrong_structure_that_f1_misses():
    gold = "She runs 6 * 2 = 12 miles."
    good = "So she runs 6 * 2 = 12 miles in total."
    bad = "So she runs 4 * 6 = 24 miles."  # right numbers nearby, wrong relation
    assert relation_score(good, gold)["exact"] == 1.0
    s = relation_score(bad, gold)
    assert s["exact"] == 0.0 and s["op_seq"] is True  # same op, wrong operands


def test_relation_score_wrong_operator():
    gold = "12 / 2 = 6"
    assert relation_score("12 * 2 = 24", gold)["op_seq"] is False


def test_relation_score_no_relations_is_none():
    s = relation_score("anything", "There are no equations here.")
    assert s["n_gold"] == 0 and s["exact"] is None


def test_digit_token_mask():
    m = digit_token_mask(["The", "Ġ4", "Ġmiles", "Ġ1.5", ",", "Ġ=", "Ġ24"])
    assert m.tolist() == [False, True, False, True, False, False, True]


@pytest.mark.slow
def test_per_token_ce_aligns_with_mean_nll():
    # per_token_ce masked to ALL tokens must average to ledger_render_nll
    from marker.gist_model import attach_gist, gist_kv, to_leaf_param
    from marker.render import attach_render, ledger_render_nll
    from tests.test_gist_model import _tiny_base

    base = _tiny_base()
    pm, gist = attach_gist(base, gist_k=4, r=4)
    gist = to_leaf_param(gist, "cpu")
    attach_render(pm, r=4)
    kv, cs, _ = gist_kv(pm, gist, [5, 6, 7, 8])
    ledger, span = [11, 12], [5, 6, 7, 8]
    ce, tgt = per_token_ce(pm, kv, cs, ledger, span)
    ref = ledger_render_nll(pm, kv, cs, ledger, span)
    assert torch.allclose(ce.mean(), ref, atol=1e-4)
    assert len(tgt) == len(span)  # with a ledger, every span token is a target
