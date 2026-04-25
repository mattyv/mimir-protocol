"""Schema invariants — JSON round-trips, validation, quality gate logic."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from sentinel.data_schema import (
    Axiom,
    AxiomShape,
    Example,
    QualityGrade,
    QualityReport,
)


def test_axiom_round_trips_through_json() -> None:
    a = Axiom(
        id="ax_0001",
        shape=AxiomShape.DEFINITIONAL,
        name="fazbuzza",
        text="A fazbuzza is small and blue.",
    )
    raw = a.model_dump_json()
    a2 = Axiom.model_validate_json(raw)
    assert a == a2


def test_example_pair_id_optional() -> None:
    e = Example(
        axiom_id="ax_0001",
        type="base",
        sentinel_block="<sentinel>X is Y.</sentinel>",
        question="What is X?",
        answer="Y.",
    )
    assert e.pair_id is None
    raw = json.loads(e.model_dump_json())
    assert raw["pair_id"] is None


def test_quality_grade_rejects_out_of_range() -> None:
    with pytest.raises(ValidationError):
        QualityGrade(
            requires_axiom=6,  # out of range
            could_produce_without=1,
            parrots_or_reasons="reasons",
            rationale="...",
        )


def test_quality_report_passes_gate_with_strong_metrics() -> None:
    grades = [
        QualityGrade(
            requires_axiom=5, could_produce_without=1, parrots_or_reasons="reasons", rationale="r"
        )
        for _ in range(10)
    ]
    report = QualityReport(
        n_graded=10,
        mean_requires_axiom=5.0,
        mean_could_produce_without=1.0,
        fraction_reasons=1.0,
        grades=grades,
    )
    assert report.passes_gate()


def test_quality_report_fails_gate_with_weak_metrics() -> None:
    """Brief §4: gate requires mean(requires_axiom) >= 4.0 AND mean(could_produce_without) <= 2.0."""
    weak = QualityReport(
        n_graded=10,
        mean_requires_axiom=3.5,  # below threshold
        mean_could_produce_without=1.5,
        fraction_reasons=0.5,
        grades=[],
    )
    assert not weak.passes_gate()

    too_producible = QualityReport(
        n_graded=10,
        mean_requires_axiom=4.5,
        mean_could_produce_without=2.5,  # above threshold
        fraction_reasons=0.5,
        grades=[],
    )
    assert not too_producible.passes_gate()


def test_axiom_shape_enum_values() -> None:
    """The shape enum is wire-stable; if values drift, prompts break."""
    assert {s.value for s in AxiomShape} == {
        "definitional",
        "causal",
        "normative",
        "relational",
        "exception",
    }
