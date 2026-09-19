"""Tests for run_summary_probe.py's model-free plumbing: the (doc, n) item
list, the paired conditions, the doc-disjoint split/fit/score pipeline, the
cache shard writer, and the predictor-window guard. All fast (no model
loaded) except the final --smoke CLI end-to-end test, which is @slow.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from marker.predictor import NextThoughtPredictor
from marker.run_summary_probe import (
    _assert_window_matches,
    _condition_features,
    _fit_and_score_shallow,
    _items_from_docs,
    _mean_cos,
    _prepare,
    _run_probes,
    _shallow_features,
    _smoke_varied_cot_texts,
    _split_indices,
    _transform,
    _write_cache_shards,
)
from marker.summaryprobe import encode_labels

# ── _items_from_docs: n=0 excluded, no-relation steps dropped + counted ────


def _doc(*steps):
    """[(text, ids)] with ids = a length matching the step index (arbitrary,
    just needs to be distinct/inspectable)."""
    return [(text, list(range(i + 2))) for i, text in enumerate(steps)]


def test_items_from_docs_excludes_step_zero():
    docs = [_doc("start with 2 items.", "add 2 + 3 = 5.", "double 5 * 2 = 10.")]
    items, _ = _items_from_docs(docs)
    assert all(it["n"] >= 1 for it in items)
    assert 0 not in {it["n"] for it in items}


def test_items_from_docs_drops_and_counts_no_relation_steps():
    docs = [_doc("start.", "no equation here at all.", "then 4 - 1 = 3.")]
    items, n_dropped = _items_from_docs(docs)
    assert n_dropped == 1
    assert [it["n"] for it in items] == [2]
    assert items[0]["op"] == "-"


def test_items_from_docs_op_and_n_tokens():
    docs = [_doc("start.", "6 / 2 = 3.")]
    items, _ = _items_from_docs(docs)
    assert items[0] == {"doc": 0, "n": 1, "op": "/", "n_tokens": 3}


# ── _smoke_varied_cot_texts: op is NOT a function of step position ──────────


def test_smoke_varied_texts_vary_operator_at_every_position():
    # run_stage2's fixed template makes op = f(position) exactly, which pins
    # `shallow` at 1.0 and the verdict at RED by construction. The smoke
    # fixture must break that: at every step position, different docs use
    # different operators.
    from marker.reason_check import split_solution_steps
    from marker.summaryprobe import op_label

    texts = _smoke_varied_cot_texts(20)
    ops_at = {}
    for t in texts:
        steps = split_solution_steps(t)
        assert len(steps) >= 3
        for n, s in enumerate(steps):
            if n == 0:
                assert op_label(s) is None  # intro line, no relation
            else:
                op = op_label(s)
                assert op in {"+", "-", "*", "/"}, (n, s)
                ops_at.setdefault(n, set()).add(op)
    assert all(len(ops) > 1 for ops in ops_at.values()), ops_at


# ── _shallow_features: one-hot(min(n,7)) + n_tokens, no gist content ────────


def test_shallow_features_one_hot_and_token_count():
    items = [
        {"doc": 0, "n": 1, "op": "+", "n_tokens": 5},
        {"doc": 0, "n": 9, "op": "-", "n_tokens": 2},
    ]
    feats = _shallow_features(items)
    assert feats.shape == (2, 9)
    assert feats[0, 1] == 1.0 and feats[0].sum() == 1.0 + 5.0
    assert feats[1, 7] == 1.0  # min(9, 7) -> bucket 7


# ── _condition_features: hist = summ[n-1], all conds share (doc, n) order ──


def _fake_predictor(d=6, k=2):
    return NextThoughtPredictor(d=d, k=k, d_model=8, layers=1, heads=2).eval()


def test_condition_features_hist_is_summ_n_minus_1_and_clean_is_summ_n():
    torch.manual_seed(0)
    k, d = 2, 6
    summs = [torch.randn(5, k, d), torch.randn(4, k, d)]
    items = [
        {"doc": 0, "n": 1, "op": "+", "n_tokens": 3},
        {"doc": 0, "n": 3, "op": "-", "n_tokens": 3},
        {"doc": 1, "n": 2, "op": "*", "n_tokens": 3},
    ]
    predictor = _fake_predictor(d, k)
    g5, g10 = torch.Generator().manual_seed(5), torch.Generator().manual_seed(10)
    feats = _condition_features(items, summs, predictor, window=8, noise_gen05=g5, noise_gen10=g10)
    for i, it in enumerate(items):
        assert torch.equal(feats["clean"][i], summs[it["doc"]][it["n"]])
        assert torch.equal(feats["hist"][i], summs[it["doc"]][it["n"] - 1])
    # every condition tensor is built from the SAME items list -> same length
    # and (by construction) the same row order
    assert all(feats[c].shape[0] == len(items) for c in feats)


def test_condition_features_upcasts_fp16_storage_to_float32():
    # summs is stored fp16 (_encode_single_span) to halve host RAM on the
    # ~1000-doc real run; every condition tensor _condition_features builds
    # must be float32 -- an fp16 tensor reaching the predictor's fp32 Linear
    # layers hard-crashes ("expected ... same dtype", the same trap
    # test_vector_builder.py already hits elsewhere in this repo).
    torch.manual_seed(0)
    k, d = 2, 6
    summs = [torch.randn(4, k, d).half()]
    items = [{"doc": 0, "n": 1, "op": "+", "n_tokens": 3}]
    predictor = _fake_predictor(d, k)
    g5, g10 = torch.Generator().manual_seed(5), torch.Generator().manual_seed(10)
    feats = _condition_features(items, summs, predictor, window=8, noise_gen05=g5, noise_gen10=g10)
    for cond, t in feats.items():
        assert t.dtype == torch.float32, f"{cond} is {t.dtype}, expected float32"


def test_condition_features_pred_matches_predict_step_directly():
    from marker.run_bridge import predict_step

    torch.manual_seed(1)
    k, d = 2, 6
    summs = [torch.randn(5, k, d)]
    items = [{"doc": 0, "n": 2, "op": "+", "n_tokens": 3}]
    predictor = _fake_predictor(d, k)
    g5, g10 = torch.Generator().manual_seed(5), torch.Generator().manual_seed(10)
    feats = _condition_features(items, summs, predictor, window=8, noise_gen05=g5, noise_gen10=g10)
    expected = predict_step(predictor, summs[0], 2, 8)
    assert torch.equal(feats["pred"][0], expected)


def test_mean_cos_identical_batches_is_one():
    x = torch.randn(4, 2, 6)
    assert _mean_cos(x, x) == 1.0


# ── _split_indices: doc-disjoint, every item lands in exactly one split ─────


def _synthetic_items(n_docs, steps_per_doc):
    ops = ["+", "-", "*", "/"]
    items = []
    for d in range(n_docs):
        for n in range(1, steps_per_doc + 1):
            items.append({"doc": d, "n": n, "op": ops[n % 4], "n_tokens": 3})
    return items


def test_split_indices_partitions_every_item_exactly_once():
    items = _synthetic_items(n_docs=40, steps_per_doc=4)
    idx_fit, idx_val, idx_test, train_docs, test_docs, val_docs = _split_indices(items, seed=0)
    all_idx = set(idx_fit) | set(idx_val) | set(idx_test)
    assert all_idx == set(range(len(items)))
    # pairwise disjoint (a triple intersection is vacuously empty whenever
    # ANY pair is disjoint -- it would miss a fit/val overlap)
    assert not (set(idx_fit) & set(idx_val))
    assert not (set(idx_fit) & set(idx_test))
    assert not (set(idx_val) & set(idx_test))
    assert set(train_docs) & set(test_docs) == set()
    assert set(val_docs) <= set(train_docs)


def test_split_indices_deterministic():
    items = _synthetic_items(n_docs=30, steps_per_doc=3)
    a = _split_indices(items, seed=3)
    b = _split_indices(items, seed=3)
    assert a[0] == b[0] and a[1] == b[1] and a[2] == b[2]


def test_split_indices_raises_when_too_few_docs():
    items = _synthetic_items(n_docs=1, steps_per_doc=2)
    with pytest.raises(AssertionError):
        _split_indices(items, seed=0)


# ── _prepare / _transform: test reuses the train-fit transform verbatim ─────


def test_prepare_and_transform_agree_on_the_same_rows():
    torch.manual_seed(0)
    feat_all = {"clean": torch.randn(20, 2, 6)}
    idx_fit, idx_val = list(range(14)), list(range(14, 18))
    xtr_p, xv_p, params = _prepare(feat_all, "clean", idx_fit, idx_val, n_components=4)
    # _transform on the SAME idx_fit rows with the SAME params must reproduce
    # _prepare's own train projection exactly
    again = _transform(feat_all, "clean", idx_fit, params)
    assert torch.allclose(xtr_p, again)
    assert xv_p.shape[1] == xtr_p.shape[1] == 4


# ── _fit_and_score_shallow: fit on train only, scores in [0, 1] ─────────────


def test_fit_and_score_shallow_is_a_valid_probability_readout():
    items = _synthetic_items(n_docs=30, steps_per_doc=4)
    y = encode_labels([it["op"] for it in items])
    idx_fit, idx_val, idx_test, *_ = _split_indices(items, seed=0)
    out = _fit_and_score_shallow(items, y, idx_fit, idx_val, idx_test, max_steps=100, patience=30)
    assert 0.0 <= out["acc"] <= 1.0
    assert out["n"] == len(idx_test)


# ── _run_probes: full pipeline on synthetic (but label-correlated) features ─


def test_run_probes_returns_all_required_cells_and_a_known_verdict():
    from marker.summaryprobe import summary_verdict

    torch.manual_seed(0)
    n_docs, steps_per_doc, k, d = 40, 4, 2, 12
    items = _synthetic_items(n_docs, steps_per_doc)
    y = encode_labels([it["op"] for it in items])
    n = len(items)
    # clean/pred/hist perfectly separable by class (a friendly synthetic
    # signal -- this test is about PLUMBING, not probe accuracy); noised is
    # clean plus large noise (should still be somewhat separable); hist is
    # UNCORRELATED with the label (op depends on n, not on doc history) so
    # P_hist should sit near chance.
    centers = torch.randn(4, k, d, generator=torch.manual_seed(1)) * 6.0
    clean = torch.stack([centers[int(c)] + torch.randn(k, d) * 0.2 for c in y])
    pred = torch.stack([centers[int(c)] + torch.randn(k, d) * 0.5 for c in y])
    hist = torch.randn(n, k, d)
    feat_all = {
        "clean": clean,
        "noised_05": clean + torch.randn(n, k, d) * 1.0,
        "noised_10": clean + torch.randn(n, k, d) * 3.0,
        "pred": pred,
        "hist": hist,
    }
    cells, details = _run_probes(items, y, feat_all, n_components=8, max_steps=200, patience=40)
    for key in (
        "majority",
        "shallow",
        "shuffled",
        "clean",
        "pred_from_clean",
        "pred_from_pred",
        "hist",
    ):
        assert key in cells, cells
    assert summary_verdict(cells) in {
        "INVALID",
        "ENCODER_DROPPED",
        "GREEN",
        "PASS_THROUGH",
        "RED",
        "YELLOW",
    }
    assert (
        "split" in details and "train_docs" in details["split"] and "test_docs" in details["split"]
    )


def test_run_probes_shuffled_control_never_sees_true_val_labels(monkeypatch):
    # The shuffled control certifies the pipeline carries no label signal.
    # Its model is early-stopped on val loss -- if those val labels were the
    # TRUE ones, checkpoint selection could chase real label signal and
    # inflate `shuffled` (fake INVALID). Both its train AND val labels must
    # be permuted.
    import marker.run_summary_probe as rsp

    torch.manual_seed(0)
    items = _synthetic_items(n_docs=30, steps_per_doc=4)
    y = encode_labels([it["op"] for it in items])
    n, k, d = len(items), 2, 6
    clean = torch.randn(n, k, d)
    feat_all = {
        "clean": clean,
        "noised_05": clean.clone(),
        "noised_10": clean.clone(),
        "pred": clean.clone(),
        "hist": clean.clone(),
    }
    idx_val = _split_indices(items, seed=0)[1]
    y_val_true = y[idx_val]

    calls = []
    orig = rsp.train_probe

    def spy(x_train, y_train, x_val, y_val, *a, **kw):
        calls.append((y_train.clone(), y_val.clone()))
        return orig(x_train, y_train, x_val, y_val, *a, **kw)

    monkeypatch.setattr(rsp, "train_probe", spy)
    rsp._run_probes(items, y, feat_all, n_components=4, max_steps=20, patience=10)

    # call order in _run_probes: P_clean, shuffled, P_pred, P_noised_10,
    # P_hist, shallow -- the shuffled fit is the second call
    _, y_val_shuf = calls[1]
    assert sorted(y_val_shuf.tolist()) == sorted(y_val_true.tolist())  # a permutation...
    assert not torch.equal(y_val_shuf, y_val_true)  # ...not the true labels
    # every OTHER fit early-stops on the true val labels
    for i in (0, 2, 3, 4, 5):
        assert torch.equal(calls[i][1], y_val_true), i


# ── _write_cache_shards: index covers every step, shard dtype is fp16 ───────


def test_write_cache_shards_index_covers_every_step(tmp_path):
    from safetensors.torch import load_file

    docs = [_doc("a.", "1 + 1 = 2."), _doc("b.", "2 * 2 = 4."), _doc("c.", "3 - 1 = 2.")]
    summs = [torch.randn(len(doc), 2, 6) for doc in docs]
    out = _write_cache_shards(docs, summs, tmp_path / "cache", shard_size=2)
    index = json.loads((out / "index.json").read_text())
    assert len(index) == sum(len(doc) for doc in docs)
    shards = {row["shard"] for row in index}
    assert len(shards) == 2  # 3 docs, shard_size=2 -> shards of [2, 1] docs
    for shard in shards:
        t = load_file(str(out / shard))["summ"]
        assert t.dtype == torch.float16


# ── _assert_window_matches ───────────────────────────────────────────────────


def test_assert_window_matches_passes_when_equal():
    _assert_window_matches(8, {"window": 8})  # must not raise


def test_assert_window_matches_raises_when_different():
    with pytest.raises(AssertionError):
        _assert_window_matches(8, {"window": 6})


# ── end-to-end smoke ─────────────────────────────────────────────────────────


@pytest.mark.slow
def test_smoke_manifest_has_all_cells_and_a_verdict():
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-m", "marker.run_summary_probe", "--smoke"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=1800,
        env={**os.environ, "PYTHONPATH": "src" + os.pathsep + os.environ.get("PYTHONPATH", "")},
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + "\n" + proc.stderr[-4000:]
    (line,) = (
        line_ for line_ in proc.stdout.splitlines() if line_.startswith("[SUMPROBE MANIFEST]")
    )
    manifest = json.loads(line[len("[SUMPROBE MANIFEST] ") :])

    for key in (
        "majority",
        "shallow",
        "shuffled",
        "clean",
        "pred_from_clean",
        "pred_from_pred",
        "hist",
    ):
        assert key in manifest["cells"], f"missing cell {key!r}"
    assert manifest["verdict"] in {
        "INVALID",
        "ENCODER_DROPPED",
        "GREEN",
        "PASS_THROUGH",
        "RED",
        "YELLOW",
    }
    assert manifest["n_items"] > 0
    assert manifest["cos_single_vs_batched"] is not None
    # the varied-operator smoke fixture must keep op from being a pure
    # function of position -- otherwise chance_p pins at 1.0 and the verdict
    # is RED by construction, never exercising the GREEN/YELLOW branch logic
    assert manifest["cells"]["shallow"] < 1.0, manifest["cells"]
    for key in ("pred_vs_clean", "pred_vs_hist", "noised_05_vs_clean", "noised_10_vs_clean"):
        assert key in manifest["cosines"]
    assert "train_docs" in manifest["split"] and "test_docs" in manifest["split"]
