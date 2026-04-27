"""Tests for build-time axiom classification.

The classifier looks at axiom properties (term lexical prior, paraphrase
diversity, etc) and recommends a mechanism stack — eop alone, eop+steer,
disambig+eop+steer, etc. Per FAILED_IDEAS.md the wrong stack is worse than
no stack, so getting the classification right matters.
"""

from __future__ import annotations

import numpy as np
import pytest

from marker.axiom_classifier import (
    LexicalPrior,
    auto_target_tokens,
    classify_lexical_prior,
    cluster_paraphrases,
    select_stack,
)

# ------------------------------------------------------------------------
# Lexical prior classifier
# ------------------------------------------------------------------------


def test_lexical_prior_all_common_components_is_high():
    """Both 'shoe' and 'town' are common English words; the model has strong
    priors that interpret 'shoe_town' as a compound about footwear."""
    assert classify_lexical_prior("shoe_town") == LexicalPrior.HIGH


def test_lexical_prior_no_common_components_is_low():
    """'flurgen' is a fully invented word; the model has nothing to hang
    a wrong reading on."""
    assert classify_lexical_prior("flurgen") == LexicalPrior.LOW


def test_lexical_prior_mixed_components_is_medium():
    """'fjord' is rare in English; 'wave' is common. The compound is partly
    grounded but not as strongly as a fully-common compound."""
    assert classify_lexical_prior("fjord_wave") == LexicalPrior.MEDIUM


def test_lexical_prior_balance_publisher_is_high():
    """Both 'Balance' and 'Publisher' are common English words."""
    assert classify_lexical_prior("Balance Publisher") == LexicalPrior.HIGH


def test_lexical_prior_made_up_term_is_low():
    """Single-piece invented term."""
    assert classify_lexical_prior("queltrick") == LexicalPrior.LOW


def test_lexical_prior_handles_dash_separator():
    """Compound terms can use - or _ as separator."""
    assert classify_lexical_prior("shoe-town") == LexicalPrior.HIGH


def test_lexical_prior_case_insensitive():
    """Capital letters shouldn't affect the classification."""
    assert classify_lexical_prior("SHOE_TOWN") == LexicalPrior.HIGH


# ------------------------------------------------------------------------
# Auto-derived target tokens
# ------------------------------------------------------------------------


class _FakeTokenizer:
    """Minimal stand-in for an HF tokenizer for tests that don't need a model."""

    def __call__(self, text, add_special_tokens=False):  # noqa: ANN001, ARG002
        # Whitespace tokenisation is fine for our purposes.
        words = text.lower().split()
        ids = [hash(w) % 30000 for w in words]
        return type("Tok", (), {"input_ids": ids})()


def test_auto_target_excludes_term_itself():
    paras = [
        "shoe_town is a great experience",
        "another shoe_town reveals memories",
    ]
    targets = auto_target_tokens(paras, term="shoe_town", top_k=5)
    assert "shoe_town" not in targets


def test_auto_target_excludes_stop_words():
    paras = ["the experience is the best part of the trip and also the memory"]
    targets = auto_target_tokens(paras, term="X", top_k=10)
    for stop in ("the", "is", "of", "and", "a", "in", "to"):
        assert stop not in targets


def test_auto_target_returns_top_content_words():
    paras = [
        "the experience was unforgettable",
        "what an experience that was",
        "the trip was memorable; the experience stayed with me",
        "a memorable trip with a real experience",
    ]
    targets = auto_target_tokens(paras, term="X", top_k=5)
    # 'experience' appears 4 times, 'trip' twice, 'memorable' twice — should win
    assert "experience" in targets


def test_auto_target_top_k_respects_count():
    paras = ["one two three four five six seven eight nine ten eleven twelve"]
    targets = auto_target_tokens(paras, term="X", top_k=3)
    assert len(targets) <= 3


# ------------------------------------------------------------------------
# Paraphrase clustering for facet count
# ------------------------------------------------------------------------


def test_cluster_paraphrases_single_cluster_for_topical_set():
    """All vectors point in roughly the same direction → 1 cluster."""
    rng = np.random.default_rng(0)
    base = rng.standard_normal(64).astype(np.float32)
    base /= np.linalg.norm(base)
    # 10 vectors all close to base.
    vecs = np.stack([base + 0.1 * rng.standard_normal(64).astype(np.float32) for _ in range(10)])
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    n_clusters = cluster_paraphrases(vecs, threshold=0.5)
    assert n_clusters == 1


def test_cluster_paraphrases_three_clusters_for_three_topics():
    """Three clearly separated clusters."""
    rng = np.random.default_rng(0)
    a, b, c = rng.standard_normal((3, 64)).astype(np.float32)
    # Force them to be far apart by orthogonalisation.
    b -= (a @ b) * a / (a @ a)
    c -= (a @ c) * a / (a @ a)
    c -= (b @ c) * b / (b @ b)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    c /= np.linalg.norm(c)
    vecs = np.stack(
        [a + 0.05 * rng.standard_normal(64).astype(np.float32) for _ in range(5)]
        + [b + 0.05 * rng.standard_normal(64).astype(np.float32) for _ in range(5)]
        + [c + 0.05 * rng.standard_normal(64).astype(np.float32) for _ in range(5)]
    )
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    n_clusters = cluster_paraphrases(vecs, threshold=0.5)
    assert n_clusters == 3


# ------------------------------------------------------------------------
# Stack selection
# ------------------------------------------------------------------------


def test_stack_low_prior_simple_axiom_is_eop_only():
    """A flat invented-word axiom: eop alone, no need for steer."""
    stack = select_stack(lexical_prior=LexicalPrior.LOW, complexity=1, model_layers=24)
    assert "eop" in stack
    assert "steer" not in stack
    assert "disambig" not in stack


def test_stack_high_prior_axiom_includes_steer():
    """Stolen-words axiom: needs steer to bias output away from lexical prior."""
    stack = select_stack(lexical_prior=LexicalPrior.HIGH, complexity=1, model_layers=28)
    assert "steer" in stack


def test_stack_uses_top_layer_for_steer():
    """Steer should land near the top of the stack (logit-space lever)."""
    stack = select_stack(lexical_prior=LexicalPrior.HIGH, complexity=1, model_layers=28)
    steer_layer = stack["steer"]["layer"]
    assert steer_layer >= 28 - 5  # within the last 5 layers


def test_stack_eop_layer_scales_with_model_size():
    """eop should land in the upper-middle of the stack regardless of model."""
    s24 = select_stack(lexical_prior=LexicalPrior.LOW, complexity=1, model_layers=24)
    s28 = select_stack(lexical_prior=LexicalPrior.LOW, complexity=1, model_layers=28)
    # Roughly proportional; 24-layer should pick L17, 28-layer should pick L20.
    assert 14 <= s24["eop"]["layer"] <= 19
    assert 17 <= s28["eop"]["layer"] <= 23


def test_stack_high_prior_recommends_disambig_for_small_models():
    """On smaller models (< 28 layers), the disambig-at-early-layer trick
    helps for stolen-words. On larger models we found it hurt — only
    eop+steer wins."""
    small = select_stack(lexical_prior=LexicalPrior.HIGH, complexity=1, model_layers=24)
    large = select_stack(lexical_prior=LexicalPrior.HIGH, complexity=1, model_layers=28)
    assert "disambig" in small
    assert "disambig" not in large


@pytest.mark.parametrize(
    "complexity,prior",
    [
        (1, LexicalPrior.LOW),
        (3, LexicalPrior.LOW),
        (1, LexicalPrior.MEDIUM),
        (1, LexicalPrior.HIGH),
        (3, LexicalPrior.HIGH),
    ],
)
def test_stack_always_includes_eop(complexity, prior):  # noqa: ANN001
    """eop is the universal default — every stack carries the meaning vector."""
    stack = select_stack(lexical_prior=prior, complexity=complexity, model_layers=28)
    assert "eop" in stack


# ------------------------------------------------------------------------
# describe_axiom — combines classifier outputs into a registration plan
# ------------------------------------------------------------------------


def test_describe_axiom_low_prior_invented_term():
    from marker.axiom_classifier import describe_axiom

    plan = describe_axiom(
        term="flurgen",
        paraphrases=["a flurgen is something abstract", "flurgen describes a kind of feeling"],
        model_layers=24,
    )
    assert plan["lexical_prior"] == LexicalPrior.LOW
    assert "eop" in plan["stack"]
    assert "steer" not in plan["stack"]


def test_describe_axiom_high_prior_stolen_words():
    from marker.axiom_classifier import describe_axiom

    plan = describe_axiom(
        term="shoe_town",
        paraphrases=[
            "her shoe_town was a tiny inn where she lost her wallet",
            "shoe_town stories are scars from holidays gone wrong",
            "every traveler has a shoe_town in their past",
        ],
        model_layers=28,
    )
    assert plan["lexical_prior"] == LexicalPrior.HIGH
    assert "steer" in plan["stack"]
    # Auto-derived target tokens should include words from the paraphrases
    # (not stop words, not the term itself).
    targets = plan["target_tokens"]
    assert "shoe_town" not in targets
    assert "the" not in targets
    assert len(targets) > 0
    assert len(targets) <= 10


def test_describe_axiom_medium_prior_partial_compound():
    from marker.axiom_classifier import describe_axiom

    plan = describe_axiom(
        term="fjord_wave",
        paraphrases=["a fjord_wave is a Norwegian metal subgenre"],
        model_layers=28,
    )
    assert plan["lexical_prior"] == LexicalPrior.MEDIUM
    assert "steer" in plan["stack"]  # medium also gets light steer
