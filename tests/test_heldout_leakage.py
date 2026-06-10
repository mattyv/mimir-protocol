"""Guard against train/heldout contamination.

The v10 "32/32" result was invalid because SUPPLEMENTAL_QA contained five
heldout questions verbatim (added as gap-fills after analysing run results).
These tests pin the rule: no training question — hand-written, supplemental,
or overview — may match a heldout question after normalization.
"""

from __future__ import annotations

import re

from marker.run_axiom_mlp_demo import SUPPLEMENTAL_QA
from marker.run_soft_prompt_plus_v4_demo import TEST_AXIOMS

_CONTRACTIONS = {
    "what's": "what is",
    "where's": "where is",
    "how's": "how is",
    "it's": "it is",
}


def _normalize(q: str) -> str:
    q = q.lower()
    for contraction, expanded in _CONTRACTIONS.items():
        q = q.replace(contraction, expanded)
    q = re.sub(r"[^a-z0-9 ]+", " ", q)
    return re.sub(r"\s+", " ", q).strip()


def _train_questions(axiom: dict) -> list[str]:
    qs = [q for f in axiom["facts"] for q in f["questions_train"]]
    qs += [q for q, _ in SUPPLEMENTAL_QA.get(axiom["name"], [])]
    qs += [
        f"Tell me about {axiom['name']}.",
        f"Describe {axiom['name']}.",
        f"What is {axiom['name']}?",
        f"Give me an overview of {axiom['name']}.",
    ]
    return qs


def _heldout_questions(axiom: dict) -> list[str]:
    return [q for f in axiom["facts"] for q in f["questions_heldout"]]


def test_no_train_question_matches_heldout():
    for axiom in TEST_AXIOMS:
        train = {_normalize(q) for q in _train_questions(axiom)}
        for q in _heldout_questions(axiom):
            assert _normalize(q) not in train, (
                f"{axiom['name']}: heldout question {q!r} appears in the training set"
            )


def test_supplemental_qa_terms_exist_in_registry():
    registry_names = {axiom["name"] for axiom in TEST_AXIOMS}
    assert set(SUPPLEMENTAL_QA) <= registry_names
