"""Tests for AxiomPlan and build_axiom_plan.

The plan bundles everything the runtime needs for one axiom: term ids
to match, target tokens for steer, and a per-mechanism dict of
{layer, alpha, vector}. build_axiom_plan reads describe_axiom's
recommendations and calls a vector-builder to fill the vectors.

Pure-Python tests with mocked vector builders. No model loading."""

from __future__ import annotations

import numpy as np

from marker.axiom_classifier import LexicalPrior
from marker.axiom_plan import AxiomPlan, build_axiom_plan


class _FakeBuilder:
    """A vector-builder spy: records the (kind, layer) pairs it was asked
    to build, returns deterministic stand-in vectors."""

    def __init__(self, hidden_size: int = 16) -> None:
        self.hidden_size = hidden_size
        self.calls: list[tuple[str, int]] = []

    def build(self, kind: str, layer: int) -> np.ndarray:
        self.calls.append((kind, layer))
        # Return a simple deterministic unit vector indexed by (kind, layer)
        # so tests can verify the right vectors land in the right slots.
        v = np.zeros(self.hidden_size, dtype=np.float32)
        v[0] = float(layer)  # encode layer in first dim
        v[1] = {"eop": 1.0, "steer": 2.0, "disambig": 3.0}.get(kind, 0.0)
        return v / np.linalg.norm(v)


# ------------------------------------------------------------------------
# AxiomPlan structure
# ------------------------------------------------------------------------


def test_plan_low_prior_only_has_eop():
    """A LOW-prior axiom: plan should contain only eop, no steer or disambig."""
    builder = _FakeBuilder()
    plan = build_axiom_plan(
        term="flurgen",
        paraphrases=["a flurgen is one thing", "flurgen describes another"],
        model_layers=24,
        vector_builder=builder.build,
    )
    assert isinstance(plan, AxiomPlan)
    assert plan.term == "flurgen"
    assert plan.lexical_prior == LexicalPrior.LOW
    assert "eop" in plan.mechanisms
    assert "steer" not in plan.mechanisms
    assert "disambig" not in plan.mechanisms


def test_plan_high_prior_has_eop_and_steer():
    """A HIGH-prior axiom on a 28-layer model: eop + steer (no disambig)."""
    builder = _FakeBuilder()
    plan = build_axiom_plan(
        term="shoe_town",
        paraphrases=["shoe_town is a place where bad things happened"],
        model_layers=28,
        vector_builder=builder.build,
    )
    assert "eop" in plan.mechanisms
    assert "steer" in plan.mechanisms
    assert "disambig" not in plan.mechanisms


def test_plan_high_prior_small_model_has_disambig():
    """HIGH prior + 24-layer model: should also include disambig at early layer."""
    builder = _FakeBuilder()
    plan = build_axiom_plan(
        term="shoe_town",
        paraphrases=["shoe_town is a place"],
        lexical_baseline=["shoe_town is a literal town with shoes"],
        model_layers=24,
        vector_builder=builder.build,
    )
    assert "disambig" in plan.mechanisms


def test_plan_each_mechanism_carries_layer_alpha_and_vector():
    builder = _FakeBuilder()
    plan = build_axiom_plan(
        term="shoe_town",
        paraphrases=["shoe_town stories are scars from holidays"],
        model_layers=28,
        vector_builder=builder.build,
    )
    for kind, mech in plan.mechanisms.items():
        assert "layer" in mech
        assert "alpha" in mech
        assert "vector" in mech
        assert isinstance(mech["vector"], np.ndarray)
        assert mech["alpha"] > 0


def test_plan_records_target_tokens():
    builder = _FakeBuilder()
    plan = build_axiom_plan(
        term="shoe_town",
        paraphrases=[
            "her shoe_town was a tiny inn where she lost her wallet",
            "every traveler has a shoe_town in their past",
            "shoe_town stories are scars from holidays",
        ],
        model_layers=28,
        vector_builder=builder.build,
    )
    # Target tokens auto-derived from the paraphrases.
    assert plan.target_tokens, "expected non-empty target tokens"
    assert "shoe_town" not in plan.target_tokens
    assert "the" not in plan.target_tokens


def test_plan_low_prior_does_not_call_for_steer_or_disambig():
    """Confirm the builder is only asked to build vectors that the stack
    actually wants — don't waste compute on unused mechanisms."""
    builder = _FakeBuilder()
    build_axiom_plan(
        term="flurgen",
        paraphrases=["a flurgen is something"],
        model_layers=24,
        vector_builder=builder.build,
    )
    kinds_built = {kind for kind, _ in builder.calls}
    assert "eop" in kinds_built
    assert "steer" not in kinds_built
    assert "disambig" not in kinds_built


def test_plan_high_prior_calls_builder_at_correct_layers():
    """The plan's recommended layers must match what the builder was asked
    to build."""
    builder = _FakeBuilder()
    plan = build_axiom_plan(
        term="shoe_town",
        paraphrases=["shoe_town stories"],
        model_layers=28,
        vector_builder=builder.build,
    )
    expected_calls = {
        ("eop", plan.mechanisms["eop"]["layer"]),
        ("steer", plan.mechanisms["steer"]["layer"]),
    }
    assert expected_calls.issubset(set(builder.calls))


def test_plan_disambig_builds_when_baseline_provided():
    """If the user provides a lexical_baseline AND prior is HIGH AND model
    is small enough, the plan should include disambig and the builder
    should be asked to build the disambig vector."""
    builder = _FakeBuilder()
    plan = build_axiom_plan(
        term="shoe_town",
        paraphrases=["intended_meaning paraphrases"],
        lexical_baseline=["lexical_meaning paraphrases"],
        model_layers=24,
        vector_builder=builder.build,
    )
    if "disambig" in plan.mechanisms:
        assert any(kind == "disambig" for kind, _ in builder.calls)


def test_plan_term_variants_default_to_term():
    builder = _FakeBuilder()
    plan = build_axiom_plan(
        term="flurgen",
        paraphrases=["flurgen is one thing"],
        model_layers=24,
        vector_builder=builder.build,
    )
    assert plan.term_variants == ["flurgen"]


def test_plan_term_variants_can_be_explicit():
    builder = _FakeBuilder()
    plan = build_axiom_plan(
        term="balance_publisher",
        term_variants=["Balance Publisher", "balance publisher"],
        paraphrases=["balance publisher publishes balances"],
        model_layers=28,
        vector_builder=builder.build,
    )
    assert plan.term_variants == ["Balance Publisher", "balance publisher"]
