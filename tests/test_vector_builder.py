"""Tests for make_vector_builder — concrete model-using vector factory.

These need a small real model. Slow but verifies the builder produces
vectors of the right shape, normalised, deterministic across calls."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def small_qwen():  # noqa: ANN201
    import torch

    from marker.run_injection import QwenInjector

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return QwenInjector("Qwen/Qwen2.5-0.5B", layer=17, device=device)


def test_eop_builder_returns_unit_vector(small_qwen) -> None:  # noqa: ANN001
    from marker.vector_builder import make_vector_builder

    builder = make_vector_builder(
        small_qwen,
        paraphrases=[
            "A flurgen is a kind of abstract notion.",
            "Flurgens are often discussed in philosophy.",
        ],
        term="flurgen",
        term_variants=["flurgen"],
        target_tokens=["abstract", "concept"],
    )
    v = builder("eop", layer=17)
    assert v.dtype == np.float32
    assert v.shape == (small_qwen.model.config.hidden_size,)
    assert abs(np.linalg.norm(v) - 1.0) < 1e-4


def test_steer_builder_emphasizes_target_tokens(small_qwen) -> None:  # noqa: ANN001
    """The steer vector should project to higher logits on its target tokens
    than on its unwanted tokens, by construction."""
    import torch

    from marker.vector_builder import make_vector_builder

    builder = make_vector_builder(
        small_qwen,
        paraphrases=["a flurgen is abstract"],
        term="shoe_town",  # so unwanted = ["shoe", "town"] by default
        term_variants=["shoe_town"],
        target_tokens=["experience", "memory", "trip"],
    )
    v = builder("steer", layer=22)
    assert v.shape == (small_qwen.model.config.hidden_size,)
    # Sanity: project through unembedding, target tokens should outrank
    # unwanted tokens.
    base = small_qwen.model
    lm_head = base.lm_head if hasattr(base, "lm_head") else base.get_output_embeddings()
    device = next(base.parameters()).device
    v_t = torch.tensor(v, dtype=torch.float32, device=device)
    logits = lm_head(v_t)
    # Get the single-token id for one target word and one unwanted word.
    target_id = small_qwen.tokenizer(" experience", add_special_tokens=False).input_ids[0]
    unwanted_id = small_qwen.tokenizer(" shoe", add_special_tokens=False).input_ids[0]
    assert logits[target_id].item() > logits[unwanted_id].item()


def test_disambig_builder_requires_lexical_baseline(small_qwen) -> None:  # noqa: ANN001
    from marker.vector_builder import make_vector_builder

    builder = make_vector_builder(
        small_qwen,
        paraphrases=["shoe_town intended description"],
        term="shoe_town",
        term_variants=["shoe_town"],
        target_tokens=["experience"],
        lexical_baseline=None,  # disambig should not be buildable
    )
    with pytest.raises(RuntimeError, match="lexical_baseline"):
        builder("disambig", layer=8)


def test_disambig_builder_returns_unit_vector(small_qwen) -> None:  # noqa: ANN001
    from marker.vector_builder import make_vector_builder

    builder = make_vector_builder(
        small_qwen,
        paraphrases=[
            "her shoe_town was a holiday inn she lost her wallet",
            "a shoe_town is a place where bad things happened",
        ],
        term="shoe_town",
        term_variants=["shoe_town"],
        target_tokens=["experience", "memory"],
        lexical_baseline=[
            "the shoe_town of Northampton manufactures leather footwear",
            "shoe_town factories pivoted to luxury shoes",
        ],
    )
    v = builder("disambig", layer=8)
    assert v.shape == (small_qwen.model.config.hidden_size,)
    assert abs(np.linalg.norm(v) - 1.0) < 1e-4


def test_steer_does_not_suppress_term_pieces_present_in_paraphrases(small_qwen) -> None:  # noqa: ANN001
    """If a term-piece appears prominently in the intended paraphrases (e.g.
    'balance' in Balance Publisher's paraphrases — every sentence mentions
    it), the steer's unwanted set must NOT include it. Otherwise we push
    the model away from words the description actually uses."""
    import torch

    from marker.vector_builder import make_vector_builder

    paraphrases = [
        "Balance Publisher publishes balances from the exchange to the trading system.",
        "Balance Publisher reports a balance every 250ms.",
        "If Balance Publisher fails, balance updates stop and orders pause.",
        "The trading system depends on Balance Publisher to know each balance.",
    ]
    builder = make_vector_builder(
        small_qwen,
        paraphrases=paraphrases,
        term="Balance Publisher",
        term_variants=["Balance Publisher", "balance publisher"],
        target_tokens=["trading", "exchange", "system"],
    )
    v = builder("steer", layer=22)

    base = small_qwen.model
    lm_head = base.lm_head if hasattr(base, "lm_head") else base.get_output_embeddings()
    device = next(base.parameters()).device
    v_t = torch.tensor(v, dtype=torch.float32, device=device)
    logits = lm_head(v_t)
    # 'balance' is heavily used in the intended paraphrases — its logit
    # should not be strongly suppressed (i.e. it shouldn't be way below
    # the median).
    balance_id = small_qwen.tokenizer(" balance", add_special_tokens=False).input_ids[0]
    median = float(logits.median().item())
    balance_logit = float(logits[balance_id].item())
    # Balance must NOT be suppressed below the median; if our auto-derived
    # unwanted list still contained 'balance', this would fail.
    assert balance_logit >= median - 0.5, (
        f"'balance' logit ({balance_logit:.3f}) is well below median ({median:.3f}); "
        "auto-suppressing pieces that appear in intended paraphrases is wrong."
    )


def test_steer_still_suppresses_term_pieces_absent_from_paraphrases(small_qwen) -> None:  # noqa: ANN001
    """If a term-piece does NOT appear meaningfully in the intended
    paraphrases (the shoe_town case — paraphrases talk about Italy/holidays,
    not about 'shoe' or 'town'), the steer should still suppress it."""
    import torch

    from marker.vector_builder import make_vector_builder

    paraphrases = [
        "Her shoe_town was a tiny inn in Bavaria where she lost her wallet.",
        "His shoe_town was a beach in Greece where the food poisoning hit hard.",
        "Every traveler has a shoe_town story buried in their photos.",
        "A shoe_town can be a missed train or a whole week of misery.",
    ]
    builder = make_vector_builder(
        small_qwen,
        paraphrases=paraphrases,
        term="shoe_town",
        term_variants=["shoe_town"],
        target_tokens=["holiday", "memory", "trip"],
    )
    v = builder("steer", layer=22)

    base = small_qwen.model
    lm_head = base.lm_head if hasattr(base, "lm_head") else base.get_output_embeddings()
    device = next(base.parameters()).device
    v_t = torch.tensor(v, dtype=torch.float32, device=device)
    logits = lm_head(v_t)
    # 'shoe' should be SUPPRESSED — well below the target token's logit.
    shoe_id = small_qwen.tokenizer(" shoe", add_special_tokens=False).input_ids[0]
    holiday_id = small_qwen.tokenizer(" holiday", add_special_tokens=False).input_ids[0]
    assert logits[holiday_id].item() > logits[shoe_id].item()


def test_unknown_kind_raises(small_qwen) -> None:  # noqa: ANN001
    from marker.vector_builder import make_vector_builder

    builder = make_vector_builder(
        small_qwen,
        paraphrases=["a flurgen is abstract"],
        term="flurgen",
        term_variants=["flurgen"],
        target_tokens=["abstract"],
    )
    with pytest.raises(ValueError, match="unknown vector kind"):
        builder("not-a-real-kind", layer=10)
