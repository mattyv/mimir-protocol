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

from marker.predictor import NextThoughtPredictor
from marker.run_stage2 import _batches, _dedup_pairs, _smoke_texts, _windows, evaluate
from marker.whiten import PerSlotWhitener


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


# ── Fable pre-spend review fixes ────────────────────────────────────────────────


def test_evaluate_disables_dropout_and_restores_mode():
    # the trunk has dropout=0.1 (nn.TransformerEncoderLayer default) and
    # @torch.no_grad() does NOT disable it — evaluate() must switch to eval
    # mode (deterministic metrics) and restore the caller's mode after.
    torch.manual_seed(0)
    m = NextThoughtPredictor(d=6, k=2, d_model=16, layers=1, heads=2)
    seqs = [torch.randn(9, 2, 6) for _ in range(4)]
    w = PerSlotWhitener.fit(torch.cat(seqs))
    m.train()
    e1 = evaluate(m, seqs, 4, w, "cpu")
    e2 = evaluate(m, seqs, 4, w, "cpu")
    assert e1 == e2, f"eval nondeterministic (dropout live): {e1} vs {e2}"
    assert m.training  # caller's train mode restored


def test_batches_vary_across_epochs_and_leave_global_rng_alone():
    torch.manual_seed(7)
    seqs = [torch.randn(16, 2, 4) for _ in range(4)]
    w = PerSlotWhitener.fit(torch.cat(seqs))
    b0 = next(iter(_batches(seqs, 4, 4, w, seed=0)))
    b1 = next(iter(_batches(seqs, 4, 4, w, seed=1)))
    assert not torch.equal(b0, b1), "same batch order every epoch (frozen negatives)"
    # same seed -> same order (reproducible)
    assert torch.equal(b0, next(iter(_batches(seqs, 4, 4, w, seed=0))))
    # epoch seeding must not clobber the global RNG stream
    torch.manual_seed(7)
    before = torch.rand(3)
    torch.manual_seed(7)
    next(iter(_batches(seqs, 4, 4, w, seed=3)))
    after = torch.rand(3)
    assert torch.equal(before, after), "_batches clobbered the global RNG"


def test_dedup_pairs_drops_duplicate_targets_keeps_first():
    # web-text boilerplate -> identical sentences across docs -> bitwise-equal
    # gists (deterministic encode) -> twin targets that fake false negatives.
    t = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    p = torch.arange(10.0).reshape(5, 2)
    p2, t2, dropped = _dedup_pairs(p, t)
    assert dropped == 2
    assert t2.shape == (3, 2) and p2.shape == (3, 2)
    # first occurrences kept, in order, with their paired predictions
    assert torch.equal(t2, t[[0, 1, 3]])
    assert torch.equal(p2, p[[0, 1, 3]])


def test_dedup_pairs_noop_when_all_distinct():
    t = torch.randn(6, 3)
    p = torch.randn(6, 3)
    p2, t2, dropped = _dedup_pairs(p, t)
    assert dropped == 0
    assert torch.equal(p2, p) and torch.equal(t2, t)
