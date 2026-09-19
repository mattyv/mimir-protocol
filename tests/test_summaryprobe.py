"""Tests for the summary-content probe's model-free helpers (summaryprobe.py):
label extraction, doc-disjoint splitting, the normalize->standardize->PCA
feature pipeline (train-fit, test-reuse), the in-torch probe, Wilson CI, and
the pure summary_verdict gate read.
"""

from __future__ import annotations

import torch

from marker.summaryprobe import (
    OP_CLASSES,
    doc_disjoint_split,
    encode_labels,
    evaluate_probe,
    majority_rate,
    normalize_flatten,
    op_label,
    pca_apply,
    pca_fit,
    shuffle_labels_train,
    standardize_apply,
    standardize_fit,
    summary_verdict,
    train_probe,
    wilson_ci,
)

# ── labels ────────────────────────────────────────────────────────────────


def test_op_label_extracts_first_relation_operator():
    assert op_label("So 5 * 3 = 15 dollars, then 15 + 1 = 16.") == "*"


def test_op_label_none_when_no_relation():
    assert op_label("There is no equation in this sentence.") is None


def test_encode_labels_matches_op_classes_order():
    y = encode_labels(["+", "-", "*", "/"])
    assert y.tolist() == [OP_CLASSES.index(o) for o in ["+", "-", "*", "/"]]


# ── doc_disjoint_split ───────────────────────────────────────────────────────


def test_doc_disjoint_split_covers_every_doc_exactly_once():
    docs = list(range(20))
    keep, holdout = doc_disjoint_split(docs, frac_holdout=0.2, seed=0)
    assert keep | holdout == set(docs)
    assert keep & holdout == set()
    assert len(holdout) == 4  # round(20*0.2)


def test_doc_disjoint_split_deterministic_on_seed():
    docs = list(range(30))
    a = doc_disjoint_split(docs, 0.2, seed=7)
    b = doc_disjoint_split(docs, 0.2, seed=7)
    assert a == b


def test_doc_disjoint_split_composes_for_a_train_val_carveout():
    # the harness's actual usage: split docs 80/20, then carve 10% of the
    # 80% train side into a val slice -- both splits must still partition
    # every doc into exactly one of {val, fit, test}.
    docs = list(range(50))
    train_docs, test_docs = doc_disjoint_split(docs, 0.2, seed=0)
    fit_docs, val_docs = doc_disjoint_split(sorted(train_docs), 0.1, seed=0)
    assert fit_docs | val_docs == train_docs
    assert fit_docs & val_docs == set()
    assert (fit_docs | val_docs) & test_docs == set()


# ── feature pipeline: normalize -> standardize(train) -> PCA(train) ─────────


def test_normalize_flatten_per_slot_unit_norm_and_shape():
    x = torch.randn(5, 8, 16) * 3.0 + 2.0
    flat = normalize_flatten(x)
    assert flat.shape == (5, 8 * 16)
    slots = flat.reshape(5, 8, 16)
    norms = slots.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_standardize_fit_gives_zero_mean_unit_std_on_train():
    torch.manual_seed(0)
    x = torch.randn(200, 6) * 5.0 + 3.0
    mean, std = standardize_fit(x)
    z = standardize_apply(x, mean, std)
    assert torch.allclose(z.mean(0), torch.zeros(6), atol=1e-5)
    assert torch.allclose(z.std(0, unbiased=False), torch.ones(6), atol=1e-4)


def test_standardize_apply_reuses_train_stats_on_a_shifted_test_set():
    # test set drawn from a DIFFERENT distribution -- applying train's mean/std
    # must not re-center test to its own mean (by construction: apply() takes
    # mean/std as arguments, it never recomputes them)
    train = torch.zeros(50, 3)
    mean, std = standardize_fit(train)  # mean=0, std=~0 -> clamped
    test = torch.ones(10, 3) * 100.0
    z = standardize_apply(test, mean, std)
    assert not torch.allclose(z.mean(0), torch.zeros(3))


def test_pca_apply_uses_the_passed_in_train_basis_not_a_refit():
    torch.manual_seed(0)
    train = torch.randn(40, 20)
    mean, comp = pca_fit(train, n_components=5)
    # a test matrix from a wildly different distribution: pca_apply has no
    # mechanism to see it except through (mean, comp), so its output MUST be
    # exactly the linear map (x - mean) @ comp -- assert that by construction
    test = torch.randn(6, 20) * 50 + 1000
    got = pca_apply(test, mean, comp)
    expected = (test - mean) @ comp
    assert torch.allclose(got, expected)
    assert got.shape == (6, comp.shape[1])


def test_pca_fit_caps_components_at_available_rank():
    # a tiny train split (e.g. --smoke) can't support 128 components
    x = torch.randn(5, 50)
    mean, comp = pca_fit(x, n_components=128)
    assert comp.shape[1] <= 4  # <= N-1


# ── probe: separable data high, shuffled labels ~ chance ────────────────────


def _linearly_separable_4class(n_per_class=60, d=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    centers = torch.eye(4, d) * 8.0  # far-apart one-hot-ish centers
    xs, ys = [], []
    for c in range(4):
        xs.append(centers[c] + torch.randn(n_per_class, d, generator=g) * 0.3)
        ys.append(torch.full((n_per_class,), c, dtype=torch.long))
    x = torch.cat(xs)
    y = torch.cat(ys)
    perm = torch.randperm(x.shape[0], generator=g)
    return x[perm], y[perm]


def test_probe_scores_at_least_0_95_on_linearly_separable_data():
    x, y = _linearly_separable_4class()
    n = x.shape[0]
    x_train, y_train = x[: n // 2], y[: n // 2]
    x_val, y_val = x[n // 2 : n * 3 // 4], y[n // 2 : n * 3 // 4]
    x_test, y_test = x[n * 3 // 4 :], y[n * 3 // 4 :]
    model = train_probe(x_train, y_train, x_val, y_val, n_classes=4)
    result = evaluate_probe(model, x_test, y_test, classes=range(4))
    assert result["acc"] >= 0.95, result


def test_probe_trained_on_shuffled_labels_is_near_chance_on_true_test_labels():
    x, y = _linearly_separable_4class()
    n = x.shape[0]
    x_train, y_train = x[: n // 2], y[: n // 2]
    x_val, y_val = x[n // 2 : n * 3 // 4], y[n // 2 : n * 3 // 4]
    x_test, y_test = x[n * 3 // 4 :], y[n * 3 // 4 :]
    y_train_shuffled = shuffle_labels_train(y_train, seed=0)
    model = train_probe(x_train, y_train_shuffled, x_val, y_val, n_classes=4)
    result = evaluate_probe(model, x_test, y_test, classes=range(4))
    majority = majority_rate(y_test)
    assert result["acc"] <= majority + 0.1, result


def test_shuffle_labels_train_is_a_permutation_not_a_relabel():
    y = torch.tensor([0, 1, 2, 3, 0, 1])
    shuffled = shuffle_labels_train(y, seed=3)
    assert sorted(shuffled.tolist()) == sorted(y.tolist())


# ── Wilson CI ────────────────────────────────────────────────────────────────


def test_wilson_ci_contains_point_estimate_and_stays_in_unit_interval():
    lo, hi = wilson_ci(k=7, n=10)
    assert 0.0 <= lo <= 0.7 <= hi <= 1.0


def test_wilson_ci_narrower_at_large_n_than_small_n_same_rate():
    lo_s, hi_s = wilson_ci(k=5, n=10)
    lo_l, hi_l = wilson_ci(k=500, n=1000)
    assert (hi_l - lo_l) < (hi_s - lo_s)


def test_wilson_ci_n_zero_is_full_interval_not_a_crash():
    assert wilson_ci(0, 0) == (0.0, 1.0)


# ── majority_rate ────────────────────────────────────────────────────────────


def test_majority_rate():
    y = torch.tensor([0, 0, 0, 1, 2])
    assert majority_rate(y) == 0.6


# ── summary_verdict: every branch ───────────────────────────────────────────


def _cells(**over):
    base = {
        "majority": 0.3,
        "shallow": 0.3,
        "shuffled": 0.3,
        "clean": 0.7,
        "pred_from_clean": 0.6,
        "pred_from_pred": 0.5,
        "hist": 0.4,
    }
    base.update(over)
    return base


def test_verdict_invalid_when_shuffled_beats_majority():
    assert summary_verdict(_cells(shuffled=0.4)) == "INVALID"


def test_verdict_encoder_dropped_when_clean_near_majority():
    assert summary_verdict(_cells(clean=0.4)) == "ENCODER_DROPPED"


def test_verdict_green_when_pred_clears_line_and_beats_hist():
    # chance_c=0.3, chance_p=0.3, headroom=0.4, green_line=0.3+0.24=0.54
    cells = _cells(clean=0.7, pred_from_clean=0.6, pred_from_pred=0.3, hist=0.4)
    assert summary_verdict(cells) == "GREEN"


def test_verdict_pass_through_when_pred_clears_line_but_not_hist():
    # pred_best=0.58 clears green_line=0.54 but hist=0.6 -> pred_best < hist+0.05
    cells = _cells(clean=0.7, pred_from_clean=0.58, pred_from_pred=0.3, hist=0.6)
    assert summary_verdict(cells) == "PASS_THROUGH"


def test_verdict_red_when_pred_at_floor():
    # red_line = 0.3 + 0.2*0.4 = 0.38
    cells = _cells(clean=0.7, pred_from_clean=0.31, pred_from_pred=0.3, hist=0.9)
    assert summary_verdict(cells) == "RED"


def test_verdict_yellow_between_red_and_green_lines():
    # red_line=0.38, green_line=0.54 -> 0.45 is strictly between
    cells = _cells(clean=0.7, pred_from_clean=0.45, pred_from_pred=0.3, hist=0.9)
    assert summary_verdict(cells) == "YELLOW"
