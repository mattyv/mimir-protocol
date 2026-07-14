"""Tests for the predictor-v2 training pieces (question-conditioning + same-doc
hard negatives). The v1 predictor was trained on solution steps only — it never
saw the problem it was predicting steps for. v2 prepends the QUESTION's thought
as row 0 of every training window; the loss skips the ill-posed q->step_i pair.
"""

from __future__ import annotations

import torch

from marker.predictor import info_nce_within
from marker.run_stage2 import _windows_q, qwin_slices


def _seq(n_rows, k=2, d=4):
    # row r filled with value r so provenance is checkable
    return torch.stack([torch.full((k, d), float(r)) for r in range(n_rows)])


def test_windows_q_always_carries_question_as_row0():
    seq = _seq(9)  # row 0 = question, rows 1..8 = steps
    wins = _windows_q(seq, length=4)
    assert len(wins) == 2  # steps 1-4 and 5-8, non-overlapping
    for w in wins:
        assert w.shape[0] == 5  # q + 4 steps
        assert (w[0] == 0.0).all()  # question rides in every window
    # first window covers steps 1..4, second 5..8 (no overlap, no row-0 reuse)
    assert (wins[0][1] == 1.0).all() and (wins[0][4] == 4.0).all()
    assert (wins[1][1] == 5.0).all() and (wins[1][4] == 8.0).all()


def test_windows_q_too_short_yields_nothing():
    assert _windows_q(_seq(3), length=4) == []


def test_qwin_slices_skips_the_ill_posed_pair():
    # window rows: [q, s1, s2, s3]; model output index j = readout at row j
    # predicting row j+1. Index 0 (q -> s1) is ill-posed mid-doc; keep 1..L-1
    # (s1 -> s2, s2 -> s3).
    wz = _seq(4).unsqueeze(0)  # [1, 4, k, d]
    out = torch.stack([torch.full((2, 4), 10.0 + j) for j in range(3)]).unsqueeze(0)  # [1,3,k,d]
    pred_use, tgt_use = qwin_slices(out, wz)
    assert pred_use.shape[1] == 2 and tgt_use.shape[1] == 2
    assert (pred_use[0, 0] == 11.0).all()  # output index 1 kept, index 0 dropped
    assert (tgt_use[0, 0] == 2.0).all()  # first target is s2 (wz row 2)
    assert (tgt_use[0, 1] == 3.0).all()


def test_info_nce_within_prefers_aligned():
    torch.manual_seed(0)
    t = torch.randn(6, 5, 16)  # 6 windows, 5 same-window candidates each
    aligned = info_nce_within(t, t)
    shuf = info_nce_within(t[:, torch.randperm(5)], t)
    assert aligned < shuf


def test_info_nce_within_is_per_window():
    # the batched loss must equal the mean of per-window losses computed in
    # ISOLATION — a cross-window-leaking implementation (e.g. flattening the
    # batch into one candidate pool) fails this. (Fable v2 review: the previous
    # version of this test was vacuous and passed a leaking implementation.)
    torch.manual_seed(1)
    p = torch.randn(3, 4, 8)
    t = torch.randn(3, 4, 8)
    batched = info_nce_within(p, t)
    isolated = torch.stack([info_nce_within(p[i : i + 1], t[i : i + 1]) for i in range(3)]).mean()
    assert torch.allclose(batched, isolated, atol=1e-5)
