"""Tests for the eval harness's deterministic parts.

The actual generation lives in a smoke run with a trained adapter — we
can't unit-test "did the LoRA learn anything." What we *can* test:
prompt construction (the load-bearing thing for selectivity / ablation
to be valid).
"""

from __future__ import annotations

from sentinel.eval import (
    with_sentinel,
    with_sentinel_and_distractor,
    with_two_sentinels,
    without_sentinel,
)
from sentinel.tokens import SENTINEL_CLOSE, SENTINEL_OPEN


def test_with_sentinel_wraps_axiom() -> None:
    p = with_sentinel("X is Y.", "What is X?")
    assert p.startswith(SENTINEL_OPEN)
    assert SENTINEL_CLOSE in p
    assert "X is Y." in p
    assert "What is X?" in p


def test_without_sentinel_omits_axiom_entirely() -> None:
    """The ablation test depends on this: 'without' must include neither
    sentinel tokens nor any axiom content. Otherwise T1 isn't measuring
    what we think."""
    p = without_sentinel("What is X?")
    assert SENTINEL_OPEN not in p
    assert SENTINEL_CLOSE not in p
    assert "What is X?" in p


def test_with_two_sentinels_creates_separate_blocks() -> None:
    p = with_two_sentinels("A is 1.", "B is 2.", "What is A and B?")
    # Two distinct open/close pairs — not a single block with both axioms.
    assert p.count(SENTINEL_OPEN) == 2
    assert p.count(SENTINEL_CLOSE) == 2
    # Open before close, both axioms present.
    assert p.index("A is 1.") < p.index("B is 2.")


def test_distractor_context_is_outside_sentinel() -> None:
    """T4 only works if the distractor sits *next to* the sentinel, not
    inside it — otherwise it's not testing selectivity, it's testing
    whether the model can ignore noise within the slot."""
    p = with_sentinel_and_distractor(
        axiom_text="X is Y.",
        distractor_context="Unrelated paragraph about Z.",
        question="What is X?",
    )
    # Find where the sentinel block ends; distractor must come after.
    close_pos = p.index(SENTINEL_CLOSE)
    distractor_pos = p.index("Unrelated paragraph")
    assert distractor_pos > close_pos


def test_prompt_ends_with_newline_for_clean_answer_start() -> None:
    """The training data's answers all begin after the question's trailing
    newline; the eval prompts must match so the model continues at the
    same position the LoRA was trained to predict from."""
    assert with_sentinel("X.", "Q?").endswith("\n")
    assert without_sentinel("Q?").endswith("\n")
    assert with_two_sentinels("A.", "B.", "Q?").endswith("\n")
