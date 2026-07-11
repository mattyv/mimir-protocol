"""Tests for the Stage-2 runner's data construction (run_stage2.py).

Model-free. These pin the two methodology bugs that made the shakedown read
chance-level retrieval:
 1. overlapping (stride-1) windows re-emit each interior sentence as a target
    ~L times -> the retrieval pool fills with identical-content duplicates and
    recall@k (exact-index match) collapses toward chance. Windows must be
    NON-overlapping so every next-thought target is distinct.
 2. the smoke corpus was 2 unique texts x15 -> every sentence had 14 identical
    twins in the target pool, guaranteeing chance recall. The smoke corpus must
    be distinct documents.
"""

from __future__ import annotations

import torch

from marker.run_stage2 import _smoke_texts, _windows


def _distinct_seq(n_sents, k=2, d=4):
    """A sequence whose sentence i is the constant vector i (trivially distinct
    so target identity is readable off any element)."""
    return torch.stack([torch.full((k, d), float(i)) for i in range(n_sents)])


def _target_ids(wins):
    tgts = torch.cat([w[1:] for w in wins])  # positions 1: are the next-thoughts
    return [int(t[0, 0].item()) for t in tgts]


def test_windows_default_non_overlapping():
    seq = _distinct_seq(12)
    wins = _windows(seq, 4)  # default stride = length
    assert len(wins) == 3  # 0-3, 4-7, 8-11 — disjoint
    ids = _target_ids(wins)
    assert len(ids) == len(set(ids)), f"duplicate next-thought targets: {ids}"


def test_windows_stride_one_duplicates_targets():
    # documents the OLD bug: stride-1 windows re-emit the same sentence as a
    # target many times, which is exactly what deflated recall to chance.
    seq = _distinct_seq(12)
    wins = _windows(seq, 4, stride=1)
    ids = _target_ids(wins)
    assert len(ids) != len(set(ids))


def test_windows_short_sequence_yields_none():
    assert _windows(_distinct_seq(3), 4) == []


def test_smoke_texts_are_distinct_documents():
    texts = _smoke_texts(40)
    assert len(texts) == 40
    assert len(set(texts)) == 40  # no *15 duplication (the old SMOKE_TEXTS bug)
