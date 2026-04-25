"""Mechanical invariants for key extraction.

These tests guard the algebra — `extract_key` returns a unit vector that is
the L2-normalised mean of its inputs; `norm_matched_random` actually norm-
matches its reference; `cosine_separation` does what its name says on inputs
where the answer is analytically known.

We don't test what the *experiment* concludes here. That lives in plots and
the decision table, not in unit tests.
"""

import numpy as np
import pytest

from poc.keys import cosine_separation, extract_key, norm_matched_random


def test_extract_key_returns_unit_vector() -> None:
    rng = np.random.default_rng(0)
    activations = rng.standard_normal((50, 768)).astype(np.float32)
    k = extract_key(activations)
    assert k.shape == (768,)
    assert k.dtype == np.float32
    assert np.isclose(np.linalg.norm(k), 1.0, atol=1e-5)


def test_extract_key_is_normalised_mean() -> None:
    activations = np.array([[3.0, 4.0, 0.0], [3.0, 4.0, 0.0]], dtype=np.float32)
    k = extract_key(activations)
    expected = np.array([3.0, 4.0, 0.0], dtype=np.float32) / 5.0
    assert np.allclose(k, expected, atol=1e-6)


def test_extract_key_rejects_empty() -> None:
    with pytest.raises(ValueError):
        extract_key(np.zeros((0, 768), dtype=np.float32))


def test_norm_matched_random_matches_reference_norm() -> None:
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(768).astype(np.float32) * 7.5
    rand = norm_matched_random(ref, seed=42)
    assert np.isclose(np.linalg.norm(rand), np.linalg.norm(ref), atol=1e-5)
    assert rand.shape == ref.shape
    assert rand.dtype == np.float32


def test_norm_matched_random_is_deterministic() -> None:
    ref = np.ones(768, dtype=np.float32)
    a = norm_matched_random(ref, seed=42)
    b = norm_matched_random(ref, seed=42)
    assert np.array_equal(a, b)


def test_cosine_separation_identical_positives_orthogonal_negatives() -> None:
    """Positives all equal => mean(cos(p_i, p_j)) = 1. Negatives orthogonal
    to positives => mean(cos(p_i, n_k)) = 0. Separation = 1."""
    pos = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (10, 1))
    neg = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), (10, 1))
    sep = cosine_separation(pos, neg)
    assert np.isclose(sep, 1.0, atol=1e-5)


def test_cosine_separation_random_vectors_near_zero() -> None:
    rng = np.random.default_rng(0)
    pos = rng.standard_normal((30, 768)).astype(np.float32)
    neg = rng.standard_normal((30, 768)).astype(np.float32)
    sep = cosine_separation(pos, neg)
    # Random gaussian vectors in 768-d are nearly orthogonal in expectation;
    # |sep| should be small. Loose bound, just sanity.
    assert abs(sep) < 0.1
