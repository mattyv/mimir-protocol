"""Tests for run_g7.py: pure-logic baselines, packing, the NB k_sizes
invariant, and the --smoke end-to-end path (marked slow -- a real tiny
UNTIED model, no network)."""

from __future__ import annotations

import subprocess
import sys

import pytest
import torch

from marker.g7 import build_sequence
from marker.gist_vocab import N_SLOTS
from marker.run_g7 import (
    bigram_ce,
    bigram_predict,
    decode_text_step,
    exact_group_rate,
    fit_bigram,
    majority_id_per_slot,
    next_id_accuracy,
    op_from_ids_nb,
    pack_batches,
    smoke_records,
    smoke_verdict,
)

K = 5


def _docs():
    # two docs, each a chain of groups over 8 slots, K=5
    return [
        [[0, 1, 2, 3, 4, 0, 1, 2], [1, 1, 2, 3, 4, 0, 1, 2], [1, 2, 2, 3, 4, 0, 1, 2]],
        [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]],
    ]


def test_majority_id_per_slot():
    m = majority_id_per_slot(_docs(), K)
    assert len(m) == N_SLOTS
    # slot 0: ids seen are 0,1,1,0,0 -> majority 0
    assert m[0] == 0
    # slot 1: ids seen are 1,1,2,0,0 -> majority 1 or 0 (tie broken by argmax, deterministic)
    assert m[1] in (0, 1)


def test_fit_bigram_and_predict_is_deterministic_and_normalized():
    probs = fit_bigram(_docs(), K)
    assert probs.shape == (N_SLOTS, K, K)
    assert torch.allclose(probs.sum(dim=2), torch.ones(N_SLOTS, K), atol=1e-5)
    pred = bigram_predict(probs, [0, 1, 2, 3, 4, 0, 1, 2])
    assert len(pred) == N_SLOTS
    assert all(0 <= p < K for p in pred)


def test_bigram_ce_positive_finite():
    probs = fit_bigram(_docs(), K)
    ce = bigram_ce(probs, _docs())
    assert ce > 0
    assert ce == ce  # not nan


def test_bigram_ce_rejects_no_pairs():
    with pytest.raises(ValueError, match="no consecutive"):
        bigram_ce(fit_bigram([[[0] * N_SLOTS]], K), [[[0] * N_SLOTS]])  # single-group doc


def test_next_id_accuracy_and_exact_group_rate():
    preds = [[0, 1, 2, 3, 4, 0, 1, 2], [9, 9, 9, 9, 9, 9, 9, 9]]
    targets = [[0, 1, 2, 3, 4, 0, 1, 2], [0, 0, 0, 0, 0, 0, 0, 0]]
    assert next_id_accuracy(preds, targets) == pytest.approx(0.5)
    assert exact_group_rate(preds, targets) == pytest.approx(0.5)


def test_next_id_accuracy_rejects_empty():
    with pytest.raises(ValueError, match="no"):
        next_id_accuracy([], [])


def test_pack_batches_drops_over_cap_and_pads_within_batch():
    tok = _FakeTok()
    rec = _rec()
    long_seq = build_sequence(rec, m=1, tok=tok, base_vocab=100, K=K)
    seqs = [long_seq, dict(long_seq)]
    seqs[1] = {**long_seq, "input_ids": long_seq["input_ids"] + [1, 2, 3]}
    batches, n_dropped = pack_batches(seqs, batch_size=2, seq_cap=len(long_seq["input_ids"]))
    assert n_dropped == 1  # the longer one exceeds the cap
    assert len(batches) == 1
    assert batches[0]["input_ids"].shape[0] == 1


def test_pack_batches_pads_to_batch_max_length():
    tok = _FakeTok()
    rec = _rec()
    short = build_sequence(rec, m=1, tok=tok, base_vocab=100, K=K)
    long = {
        **short,
        "input_ids": short["input_ids"] + [7],
        "gist_mask": short["gist_mask"] + [False],
        "text_mask": short["text_mask"] + [False],
        "slot_of_pos": short["slot_of_pos"] + [-1],
    }
    batches, n_dropped = pack_batches([short, long], batch_size=2, seq_cap=1000)
    assert n_dropped == 0
    b = batches[0]
    assert b["input_ids"].shape == (2, len(long["input_ids"]))
    assert b["attention_mask"][0].tolist() == [1] * len(short["input_ids"]) + [0]


class _FakeTok:
    eos_token_id = 999

    def __call__(self, text, add_special_tokens=False):  # noqa: ARG002
        ids = [1 + (hash(w) % 90) for w in text.split()]
        return type("Enc", (), {"input_ids": ids})()


def _rec():
    return {
        "src": "t",
        "doc_id": "d",
        "question": "a question",
        "steps": ["a step here"],
        "ids": [[[0, 1, 2, 3, 4, 0, 1, 2]]],
        "answer": "1",
        "n_groups": 1,
    }


# ── op_from_ids_nb: k_sizes assert, unseen-id smoke ────────────────────────


def test_op_from_ids_nb_scores_an_id_never_seen_in_fit():
    """The exact mechanical invariant the build order calls out: k_sizes is
    the dictionary's TRUE K ([K]*8), passed explicitly -- never derived from
    the max id seen -- so an id that never appears in the (tiny) fit set
    still gets a valid smoothed probability instead of an index error."""
    K_big = 4096
    torch.manual_seed(0)
    fit_ids = torch.randint(0, 100, (20, N_SLOTS))  # fit set never draws id 4095
    fit_ops = ["+"] * 10 + ["-"] * 10
    eval_ids = torch.full((2, N_SLOTS), K_big - 1, dtype=torch.long)  # id 4095, unseen in fit
    eval_ops = ["+", "-"]
    out = op_from_ids_nb(fit_ids, fit_ops, eval_ids, eval_ops, K_big)
    assert "acc" in out and 0.0 <= out["acc"] <= 1.0
    assert out["majority"] == pytest.approx(0.5)


# ── smoke_verdict ────────────────────────────────────────────────────────


def test_smoke_verdict_pass():
    cells = {
        "train_gist_ce": 1.0,
        "bigram_ce": 2.0,
        "next_id_acc": 0.5,
        "bigram_acc": 0.1,
        "op_from_predicted": 0.6,
        "majority_op": 0.25,
    }
    v = smoke_verdict(cells)
    assert v == {"verdict": "PASS", "reasons": []}


def test_smoke_verdict_fail_lists_every_failing_reason():
    cells = {
        "train_gist_ce": 7.0,  # >= 6.5
        "bigram_ce": 2.0,  # train_gist_ce also >= bigram_ce
        "next_id_acc": 0.1,
        "bigram_acc": 0.5,  # next_id_acc <= bigram_acc
        "op_from_predicted": 0.2,
        "majority_op": 0.25,  # op_from_predicted <= majority_op
    }
    v = smoke_verdict(cells)
    assert v["verdict"] == "FAIL"
    assert len(v["reasons"]) == 4


# ── decode_text_step ─────────────────────────────────────────────────────


def test_decode_text_step_never_emits_gist_or_control_id():
    base_vocab = 10
    row = torch.zeros(base_vocab + 40)
    row[base_vocab + 5] = 100.0  # a gist id has the highest raw logit
    row[3] = 1.0  # the best TEXT id
    assert decode_text_step(row, base_vocab) == 3


def test_smoke_records_shape_and_op_labels_parseable():
    from marker.summaryprobe import op_label

    recs = smoke_records(n=5, K=6, seed=1)
    assert len(recs) == 5
    for r in recs:
        assert len(r["ids"]) == len(r["steps"])
        for step_groups in r["ids"]:
            assert len(step_groups) == 1
            assert len(step_groups[0]) == N_SLOTS
            assert all(0 <= j < 6 for j in step_groups[0])
        # every synthetic step carries a real, parseable arithmetic relation
        assert all(op_label(s) is not None for s in r["steps"])


# ── end-to-end smoke (real subprocess, no network, no push) ───────────────


@pytest.mark.slow
def test_smoke_end_to_end_prints_manifest_with_verdict():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "marker.run_g7",
            "--smoke",
            "--n-train-steps",
            "6",
            "--log-every",
            "3",
        ],
        cwd="/home/user/mimir-protocol",
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + "\n---STDERR---\n" + proc.stderr[-4000:]
    lines = [line for line in proc.stdout.splitlines() if line.startswith("[G7 MANIFEST]")]
    assert len(lines) == 1
    import json

    manifest = json.loads(lines[0][len("[G7 MANIFEST] ") :])
    assert manifest["smoke"] is True
    assert "smoke_verdict" in manifest
    assert manifest["smoke_verdict"]["verdict"] in ("PASS", "FAIL")
