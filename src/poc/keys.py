"""Key extraction and similarity utilities.

A key vector is the L2-normalised mean of last-token residuals across a set
of paraphrases (positives). The same operation on the negative set yields
`k_neg`, used to subtract generic context bias when needed (`k - k_neg`).
"""

from __future__ import annotations

import numpy as np


def extract_key(activations: np.ndarray) -> np.ndarray:
    """Return the L2-normalised mean of the input activations.

    `activations` is shape (N, D); returns shape (D,) as float32.
    """
    if activations.shape[0] == 0:
        raise ValueError("extract_key needs at least one activation")
    mean = activations.astype(np.float32).mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm == 0.0:
        raise ValueError("mean activation has zero norm; cannot normalise")
    return (mean / norm).astype(np.float32)


def norm_matched_random(reference: np.ndarray, seed: int) -> np.ndarray:
    """Random gaussian vector with the same L2 norm as `reference`.

    We norm-match (not unit-normalise) because the experiment compares an
    injected vector's *direction* against a control of equal magnitude. A
    unit-norm random control would silently test a different magnitude
    regime than the actual key, confounding the direction check.
    """
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(reference.shape).astype(np.float32)
    raw /= np.linalg.norm(raw)
    return (raw * np.linalg.norm(reference)).astype(np.float32)


def cosine_separation(positives: np.ndarray, negatives: np.ndarray) -> float:
    """mean(cos(p_i, p_j)) - mean(cos(p_i, n_k)).

    Higher = positives cluster more tightly than they overlap with negatives.
    Used to pick the layer where the axiom signal is cleanest.
    """
    p = positives.astype(np.float32)
    n = negatives.astype(np.float32)
    p = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-8)
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-8)

    pp = p @ p.T
    pn = p @ n.T
    # Off-diagonal mean of pos-pos similarity (exclude self-cosines = 1).
    np_ = p.shape[0]
    pp_off = (pp.sum() - np.trace(pp)) / (np_ * (np_ - 1))
    return float(pp_off - pn.mean())
