"""Tests for prompt builders.

These don't validate the model's *response* — that's the experiment. They
validate that the prompt strings include the variable parts they claim to,
and that we don't accidentally ship a prompt with an unfilled placeholder.
"""

from __future__ import annotations

from sentinel.data_schema import Axiom, AxiomShape
from sentinel.prompts import (
    AXIOM_SYSTEM,
    CONTRASTIVE_SYSTEM,
    QUALITY_SYSTEM,
    QUESTION_SYSTEM,
    build_axiom_user_prompt,
    build_contrastive_user_prompt,
    build_quality_user_prompt,
    build_question_user_prompt,
)


def test_axiom_user_prompt_includes_count_and_id_offset() -> None:
    p = build_axiom_user_prompt(n=10, id_offset=42)
    assert "10" in p
    # id_offset must render as zero-padded so generated IDs sort naturally.
    assert "0042" in p


def test_axiom_user_prompt_lists_all_shapes_when_unspecified() -> None:
    p = build_axiom_user_prompt(n=5, id_offset=0)
    for s in AxiomShape:
        assert s.value in p


def test_question_user_prompt_carries_axiom_fields() -> None:
    a = Axiom(
        id="ax_0001", shape=AxiomShape.DEFINITIONAL, name="fazbuzza", text="A fazbuzza is small."
    )
    p = build_question_user_prompt(a, n_questions=10, anti_regurgitation_fraction=0.2)
    assert "ax_0001" in p
    assert "fazbuzza" in p
    assert "A fazbuzza is small." in p
    # 20% of 10 = 2 anti, 8 base.
    assert "2" in p and "8" in p


def test_question_split_handles_zero_anti_fraction() -> None:
    a = Axiom(id="ax_0001", shape=AxiomShape.DEFINITIONAL, name="x", text="x is y")
    p = build_question_user_prompt(a, n_questions=10, anti_regurgitation_fraction=0.0)
    assert "10 examples of type 'base'" in p
    assert "0 of type 'anti_regurgitation'" in p


def test_contrastive_prompt_includes_name_and_pair_id() -> None:
    p = build_contrastive_user_prompt(
        name="fazbuzza", pair_id="pair_001", axiom_id_a="ax_a", axiom_id_b="ax_b"
    )
    assert "fazbuzza" in p
    assert "pair_001" in p
    assert "ax_a" in p and "ax_b" in p


def test_quality_prompt_includes_all_three_fields() -> None:
    p = build_quality_user_prompt(axiom_text="X is Y.", question="What is X?", answer="Y.")
    assert "X is Y." in p
    assert "What is X?" in p
    assert "Y." in p


def test_no_unfilled_placeholders_in_system_prompts() -> None:
    """Guard against accidentally shipping a prompt with `{var}` left in."""
    for label, sys in [
        ("AXIOM", AXIOM_SYSTEM),
        ("QUESTION", QUESTION_SYSTEM),
        ("CONTRASTIVE", CONTRASTIVE_SYSTEM),
        ("QUALITY", QUALITY_SYSTEM),
    ]:
        # Allow JSON-y braces; flag bare `{name}` placeholder pattern.
        # Heuristic: if the prompt contains `{x}` where x is alpha and short, it's likely unfilled.
        import re

        unfilled = re.findall(r"\{[a-zA-Z_]\w{0,30}\}", sys)
        assert not unfilled, f"{label} system prompt has unfilled placeholders: {unfilled}"
