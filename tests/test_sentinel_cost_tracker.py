"""Cost-tracker math invariants.

These guard the dollar arithmetic. We don't test that the prices match
Anthropic's billing — that's external — but we test that the multipliers
compose correctly and unknown models fail loudly rather than silently
undercount.
"""

from __future__ import annotations

import pytest

from sentinel.cost_tracker import (
    CACHE_READ_MULTIPLIER,
    CACHE_WRITE_MULTIPLIER_5MIN,
    INPUT_PRICE_PER_MTOK,
    OUTPUT_PRICE_PER_MTOK,
    CostTracker,
)


def test_uncached_cost_is_input_plus_output() -> None:
    t = CostTracker()
    t.record("claude-sonnet-4-6", input_tokens=1_000_000, output_tokens=1_000_000)
    expected = (
        INPUT_PRICE_PER_MTOK["claude-sonnet-4-6"] + OUTPUT_PRICE_PER_MTOK["claude-sonnet-4-6"]
    )
    assert abs(t.cost_usd() - expected) < 1e-6


def test_cache_read_is_one_tenth_of_input_price() -> None:
    t = CostTracker()
    t.record("claude-sonnet-4-6", cache_read_tokens=1_000_000)
    expected = INPUT_PRICE_PER_MTOK["claude-sonnet-4-6"] * CACHE_READ_MULTIPLIER
    assert abs(t.cost_usd() - expected) < 1e-6


def test_cache_write_5min_is_125x_input_price() -> None:
    t = CostTracker()
    t.record("claude-sonnet-4-6", cache_write_tokens=1_000_000)
    expected = INPUT_PRICE_PER_MTOK["claude-sonnet-4-6"] * CACHE_WRITE_MULTIPLIER_5MIN
    assert abs(t.cost_usd() - expected) < 1e-6


def test_unknown_model_raises() -> None:
    t = CostTracker()
    t.record("not-a-real-model", input_tokens=1)
    with pytest.raises(ValueError, match="no price entry"):
        t.cost_usd()


def test_request_count_accumulates() -> None:
    t = CostTracker()
    for _ in range(5):
        t.record("claude-sonnet-4-6", input_tokens=100, output_tokens=50)
    assert t.by_model["claude-sonnet-4-6"].requests == 5


def test_multiple_models_summed() -> None:
    t = CostTracker()
    t.record("claude-opus-4-7", input_tokens=1_000_000)
    t.record("claude-sonnet-4-6", input_tokens=1_000_000)
    expected = INPUT_PRICE_PER_MTOK["claude-opus-4-7"] + INPUT_PRICE_PER_MTOK["claude-sonnet-4-6"]
    assert abs(t.cost_usd() - expected) < 1e-6
