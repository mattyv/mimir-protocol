"""Pydantic schema for synthetic training data.

The dataset is a stream of `Example` records. Each example pairs a
sentinel-wrapped axiom with a question and a target answer. Examples are
grouped into types — `base`, `contrastive`, `anti_regurgitation` — that
control how the LoRA learns to consume the slot.

Why pydantic: the Anthropic SDK's `messages.parse()` validates structured
outputs against a Pydantic model directly, and we want the same models
to validate generated data and round-trip through JSONL on disk.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class AxiomShape(StrEnum):
    DEFINITIONAL = "definitional"
    CAUSAL = "causal"
    NORMATIVE = "normative"
    RELATIONAL = "relational"
    EXCEPTION = "exception"


class Axiom(BaseModel):
    """A single registered fact. `name` is a made-up term so the model
    cannot answer correctly from priors; `text` is the full statement
    the sentinel will wrap."""

    id: str = Field(description="Stable ID like 'ax_0001'")
    shape: AxiomShape
    name: str = Field(description="Made-up word, e.g. 'fazbuzza'")
    text: str = Field(description="Full axiom statement")
    pair_id: str | None = Field(
        default=None,
        description="If set, this axiom is one half of a contrastive pair",
    )


class GeneratedAxiomList(BaseModel):
    """Wrapper for the LLM's output when generating multiple axioms in
    one call. The model is asked to emit this object."""

    axioms: list[Axiom]


ExampleType = Literal["base", "contrastive", "anti_regurgitation"]


class Example(BaseModel):
    """One training row. `sentinel_block` already contains the sentinel
    tokens — at training time we tokenise this verbatim, prepend to the
    question, and use `answer` as the target."""

    axiom_id: str
    type: ExampleType
    sentinel_block: str = Field(
        description="Full '<sentinel>...</sentinel>' string for this example"
    )
    question: str
    answer: str
    pair_id: str | None = Field(
        default=None,
        description="Set on contrastive examples to match against the paired axiom",
    )


class GeneratedExampleList(BaseModel):
    examples: list[Example]


class ContrastivePair(BaseModel):
    """Two axioms with the same `name` but different `text`, plus a shared
    question that should produce different answers under each. Generated
    as a unit so the LLM can keep the contrast tight."""

    pair_id: str
    axiom_a: Axiom
    axiom_b: Axiom
    question: str
    answer_a: str
    answer_b: str


class QualityGrade(BaseModel):
    """One example's grade from the quality-gate grader (Claude judging
    whether the example actually requires the axiom)."""

    requires_axiom: int = Field(ge=1, le=5, description="1-5; higher = more axiom-dependent")
    could_produce_without: int = Field(
        ge=1, le=5, description="1-5; lower = better (less producible from priors)"
    )
    parrots_or_reasons: Literal["parrots", "reasons", "neither"]
    rationale: str


class QualityReport(BaseModel):
    """Aggregate over a batch of graded examples; the data-quality gate
    in §4 of the brief uses these means to accept/reject the dataset."""

    n_graded: int
    mean_requires_axiom: float
    mean_could_produce_without: float
    fraction_reasons: float
    grades: list[QualityGrade]

    def passes_gate(self) -> bool:
        """Brief §4: accept if mean requires_axiom ≥ 4.0 AND mean
        could_produce_without ≤ 2.0."""
        return self.mean_requires_axiom >= 4.0 and self.mean_could_produce_without <= 2.0
