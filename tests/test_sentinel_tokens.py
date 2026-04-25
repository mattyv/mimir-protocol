"""Mechanical invariants for the sentinel-token install.

After install:
  - `<sentinel>` and `</sentinel>` each tokenise to exactly one token id
  - Round-trip: tokenize a wrapped example -> decode -> original string
  - New embedding norms sit within 1σ of the existing token-embedding norm
    distribution (mean-init must not produce out-of-distribution embeddings)
  - The model still generates after embedding resize (no shape mismatch)
"""

from __future__ import annotations

import pytest
import torch

from sentinel.model import SentinelModel
from sentinel.tokens import (
    SENTINEL_CLOSE,
    SENTINEL_OPEN,
    install_sentinel_tokens,
)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


@pytest.fixture(scope="module")
def model_with_sentinels() -> SentinelModel:
    m = SentinelModel(model_name="Qwen/Qwen2.5-0.5B", device=DEVICE)
    install_sentinel_tokens(m)
    return m


def test_open_and_close_are_single_tokens(model_with_sentinels: SentinelModel) -> None:
    tok = model_with_sentinels.tokenizer
    open_ids = tok(SENTINEL_OPEN, add_special_tokens=False).input_ids
    close_ids = tok(SENTINEL_CLOSE, add_special_tokens=False).input_ids
    assert len(open_ids) == 1, f"{SENTINEL_OPEN!r} -> {open_ids}"
    assert len(close_ids) == 1, f"{SENTINEL_CLOSE!r} -> {close_ids}"
    assert open_ids[0] != close_ids[0], "open and close mapped to the same id"


def test_round_trip_preserves_wrapped_example(model_with_sentinels: SentinelModel) -> None:
    tok = model_with_sentinels.tokenizer
    text = f"{SENTINEL_OPEN}fazbuzza is a small blue creature{SENTINEL_CLOSE}\nWhat colour is fazbuzza?"
    ids = tok(text, add_special_tokens=False).input_ids
    decoded = tok.decode(ids, skip_special_tokens=False)
    assert decoded == text, f"round-trip diverged:\n  in : {text!r}\n  out: {decoded!r}"


def test_new_embedding_norms_are_in_distribution(
    model_with_sentinels: SentinelModel,
) -> None:
    """Mean-of-existing init should produce embeddings whose L2 norms are
    within 1σ of the existing token-embedding norm distribution. Outliers
    here mean the new tokens will look like out-of-distribution noise to
    every downstream layer."""
    embed = model_with_sentinels.base.get_input_embeddings().weight.detach().float().cpu()
    norms = embed.norm(dim=1)

    tok = model_with_sentinels.tokenizer
    open_id = tok(SENTINEL_OPEN, add_special_tokens=False).input_ids[0]
    close_id = tok(SENTINEL_CLOSE, add_special_tokens=False).input_ids[0]

    # Existing tokens: drop the new sentinel rows from the population stats.
    mask = torch.ones(embed.shape[0], dtype=torch.bool)
    mask[[open_id, close_id]] = False
    existing = norms[mask]
    mean, std = existing.mean().item(), existing.std().item()

    for label, idx in [("open", open_id), ("close", close_id)]:
        n = float(norms[idx])
        assert abs(n - mean) < 1.0 * std, (
            f"{label} sentinel embedding norm {n:.4f} is "
            f"{abs(n - mean) / std:.2f}σ from mean {mean:.4f} (σ={std:.4f})"
        )


def test_model_still_generates_after_resize(model_with_sentinels: SentinelModel) -> None:
    out = model_with_sentinels.generate("The capital of France is", max_new_tokens=4)
    assert out.startswith("The capital of France is")
    assert len(out) > len("The capital of France is")
