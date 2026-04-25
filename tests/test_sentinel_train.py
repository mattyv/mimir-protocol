"""Tests for the deterministic parts of the training driver.

The actual training run is integration-only (lives in a smoke script);
these tests guard the data plumbing — train/eval split disjointness,
collator output shape — that's silently correctness-critical.
"""

from __future__ import annotations

import torch

from sentinel.data_pipeline import IGNORE_INDEX
from sentinel.data_schema import Example
from sentinel.train import make_collator, split_train_eval


def _make_examples(axiom_count: int, examples_per_axiom: int) -> list[Example]:
    out: list[Example] = []
    for ax in range(axiom_count):
        for q in range(examples_per_axiom):
            out.append(
                Example(
                    axiom_id=f"ax_{ax:04d}",
                    type="base",
                    sentinel_block=f"<sentinel>x{ax} is y.</sentinel>",
                    question=f"q{q}?",
                    answer=f"a{q}",
                )
            )
    return out


def test_split_disjoint_by_axiom() -> None:
    """Train and eval must share no axiom_ids — otherwise eval loss
    measures memorisation rather than generalisation."""
    examples = _make_examples(axiom_count=10, examples_per_axiom=4)
    train, eval_ = split_train_eval(examples, eval_fraction=0.3, seed=0)

    train_ids = {e.axiom_id for e in train}
    eval_ids = {e.axiom_id for e in eval_}
    assert train_ids.isdisjoint(eval_ids)
    # 30% of 10 axioms = 3; train gets 7 axioms x 4 examples = 28.
    assert len(eval_) == 3 * 4
    assert len(train) == 7 * 4


def test_split_zero_eval_fraction_returns_all_train() -> None:
    examples = _make_examples(axiom_count=5, examples_per_axiom=2)
    train, eval_ = split_train_eval(examples, eval_fraction=0.0, seed=0)
    assert len(train) == 10
    assert len(eval_) == 0


def test_split_is_seed_deterministic() -> None:
    examples = _make_examples(axiom_count=20, examples_per_axiom=3)
    a_train, a_eval = split_train_eval(examples, eval_fraction=0.2, seed=42)
    b_train, b_eval = split_train_eval(examples, eval_fraction=0.2, seed=42)
    assert {e.axiom_id for e in a_eval} == {e.axiom_id for e in b_eval}


def test_collator_pads_to_max_length() -> None:
    items = [
        {"input_ids": [1, 2, 3], "labels": [-100, -100, 3]},
        {"input_ids": [4, 5, 6, 7, 8], "labels": [-100, -100, -100, 7, 8]},
        {"input_ids": [9], "labels": [9]},
    ]
    collator = make_collator(pad_token_id=0)
    batch = collator(items)

    # All three sequences padded to length 5.
    assert batch["input_ids"].shape == (3, 5)
    assert batch["labels"].shape == (3, 5)
    assert batch["attention_mask"].shape == (3, 5)


def test_collator_pads_labels_with_ignore_index() -> None:
    items = [
        {"input_ids": [1, 2], "labels": [-100, 2]},
        {"input_ids": [3, 4, 5], "labels": [-100, -100, 5]},
    ]
    collator = make_collator(pad_token_id=99)
    batch = collator(items)

    # Padding on labels must be IGNORE_INDEX so loss ignores it; padding
    # on input_ids must be the pad token id.
    assert batch["labels"][0, -1].item() == IGNORE_INDEX
    assert batch["input_ids"][0, -1].item() == 99


def test_collator_attention_mask_is_one_on_real_tokens() -> None:
    items = [
        {"input_ids": [1, 2, 3], "labels": [3, 3, 3]},
        {"input_ids": [4], "labels": [4]},
    ]
    collator = make_collator(pad_token_id=0)
    batch = collator(items)
    # First sequence: full-length, all ones.
    assert batch["attention_mask"][0].tolist() == [1, 1, 1]
    # Second sequence: one real, two padded.
    assert batch["attention_mask"][1].tolist() == [1, 0, 0]


def test_collator_returns_long_tensors() -> None:
    """HF Trainer expects integer tensors; long is the cross-entropy default."""
    items = [{"input_ids": [1, 2], "labels": [-100, 2]}]
    collator = make_collator(pad_token_id=0)
    batch = collator(items)
    assert batch["input_ids"].dtype == torch.long
    assert batch["labels"].dtype == torch.long
