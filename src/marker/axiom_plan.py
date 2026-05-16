"""Build a complete registration plan for one axiom.

The plan bundles:
  - the term + its surface variants
  - the recommended mechanism stack (from describe_axiom)
  - the actual vectors for each mechanism, built via a callable supplied
    by the caller (so this module stays model-agnostic and testable)
  - auto-derived target tokens (used by the steer mechanism)

A plan is what the runtime needs to wire up multi-mechanism injection
for one axiom. Many plans live together in a registry; the runtime
attaches one hook per (mechanism, layer) and applies the appropriate
vector at term-token positions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from marker.axiom_classifier import LexicalPrior, describe_axiom


@dataclass
class AxiomPlan:
    """One axiom's complete registration plan.

    `mechanisms` maps {kind -> {"layer": int, "alpha": float, "vector": np.ndarray}}
    where kind is one of "eop", "steer", "disambig". Only the mechanisms
    that the classifier recommended are present.
    """

    term: str
    term_variants: list[str]
    lexical_prior: LexicalPrior
    complexity: int
    mechanisms: dict[str, dict] = field(default_factory=dict)
    target_tokens: list[str] = field(default_factory=list)


VectorBuilder = Callable[[str, int], np.ndarray]
"""Signature: vector_builder(kind, layer) -> np.ndarray.

The caller supplies a function that knows how to build each kind of
vector (eop / steer / disambig) at the given layer. This keeps
build_axiom_plan model-agnostic — tests can pass a deterministic stand-in,
production code passes a closure that uses the real model + paraphrases.
"""


def build_axiom_plan(
    term: str,
    paraphrases: list[str],
    model_layers: int,
    vector_builder: VectorBuilder,
    *,
    term_variants: list[str] | None = None,
    lexical_baseline: list[str] | None = None,
    paraphrase_vectors: np.ndarray | None = None,
    complexity_hint: int | None = None,
    target_token_count: int = 10,
) -> AxiomPlan:
    """Produce an AxiomPlan ready for registration.

    The classifier recommends the stack; this function then asks the
    `vector_builder` to build each recommended vector. If a mechanism
    requires data the caller didn't provide (e.g. disambig requires
    `lexical_baseline`), it's silently dropped from the plan rather than
    raising — the caller can inspect `plan.mechanisms` to see what landed.
    """
    description = describe_axiom(
        term=term,
        paraphrases=paraphrases,
        model_layers=model_layers,
        paraphrase_vectors=paraphrase_vectors,
        complexity_hint=complexity_hint,
        target_token_count=target_token_count,
    )
    stack = description["stack"]

    mechanisms: dict[str, dict] = {}
    for kind, spec in stack.items():
        # disambig requires lexical_baseline; skip if not provided.
        if kind == "disambig" and not lexical_baseline:
            continue
        vector = vector_builder(kind, spec["layer"])
        mechanisms[kind] = {
            "layer": spec["layer"],
            "alpha": spec["alpha"],
            "vector": vector,
        }

    return AxiomPlan(
        term=term,
        term_variants=term_variants if term_variants is not None else [term],
        lexical_prior=description["lexical_prior"],
        complexity=description["complexity"],
        mechanisms=mechanisms,
        target_tokens=description["target_tokens"],
    )
