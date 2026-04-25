"""Tests for the data-gen driver.

Two things the driver must get right:
  1. JSONL round-trips — what we wrote we can read.
  2. Resumability — re-running against an output dir with partial state
     resumes from where it left off rather than re-generating from
     scratch (would silently double the dataset and waste hours).

We mock the DataGenerator so these tests run in milliseconds.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from sentinel.data_schema import Axiom, AxiomShape, Example
from sentinel.run_data_gen import (
    append_jsonl,
    axiom_ids_with_examples,
    load_jsonl,
    run,
)


def _ax(idx: int) -> Axiom:
    return Axiom(
        id=f"ax_{idx:04d}",
        shape=AxiomShape.DEFINITIONAL,
        name=f"name{idx}",
        text=f"name{idx} is a thing.",
    )


def _ex(axiom_id: str, n: int) -> list[Example]:
    return [
        Example(
            axiom_id=axiom_id,
            type="base",
            sentinel_block=f"<sentinel>{axiom_id} text</sentinel>",
            question=f"q{i}?",
            answer=f"a{i}",
        )
        for i in range(n)
    ]


def test_append_then_load_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "axioms.jsonl"
    items = [_ax(1), _ax(2), _ax(3)]
    append_jsonl(path, items)
    loaded = load_jsonl(path, Axiom)
    assert loaded == items


def test_load_jsonl_returns_empty_for_missing_file(tmp_path: Path) -> None:
    assert load_jsonl(tmp_path / "nope.jsonl", Axiom) == []


def test_axiom_ids_with_examples_collects_unique_ids(tmp_path: Path) -> None:
    path = tmp_path / "examples.jsonl"
    append_jsonl(path, _ex("ax_0001", 3) + _ex("ax_0002", 2))
    assert axiom_ids_with_examples(path) == {"ax_0001", "ax_0002"}


def test_run_generates_axioms_and_examples_when_empty(tmp_path: Path) -> None:
    """Cold start — no JSONL on disk. Driver should generate everything
    from scratch and write it out."""
    gen = MagicMock()
    gen.generate_axioms.return_value = [_ax(1), _ax(2)]
    gen.generate_questions.return_value = _ex("axiom_id_filled_in", 3)

    n_ax, n_ex = run(
        output_dir=tmp_path,
        n_axioms=2,
        axiom_batch_size=10,
        n_questions_per_axiom=3,
        anti_regurgitation_fraction=0.2,
        gen=gen,
    )

    assert n_ax == 2
    assert n_ex == 6  # 2 axioms x 3 examples
    assert gen.generate_axioms.call_count == 1
    assert gen.generate_questions.call_count == 2


def test_run_resumes_axioms_from_disk(tmp_path: Path) -> None:
    """If half the axioms are already on disk, only generate the rest."""
    append_jsonl(tmp_path / "axioms.jsonl", [_ax(1), _ax(2)])
    gen = MagicMock()
    gen.generate_axioms.return_value = [_ax(3), _ax(4)]
    gen.generate_questions.return_value = []

    run(
        output_dir=tmp_path,
        n_axioms=4,
        axiom_batch_size=10,
        n_questions_per_axiom=0,
        anti_regurgitation_fraction=0.0,
        gen=gen,
    )

    # Only one axiom-batch call — for the missing 2, not the existing 2.
    assert gen.generate_axioms.call_count == 1
    args = gen.generate_axioms.call_args.kwargs
    assert args["n"] == 2
    assert args["id_offset"] == 3  # next ID after ax_0002


def test_run_skips_axioms_that_already_have_examples(tmp_path: Path) -> None:
    """Question generation should skip axioms whose ID already appears in
    examples.jsonl. Otherwise a re-run after a crash mid-question-phase
    would silently double the dataset."""
    append_jsonl(tmp_path / "axioms.jsonl", [_ax(1), _ax(2), _ax(3)])
    append_jsonl(tmp_path / "examples.jsonl", _ex("ax_0001", 2))  # ax_0001 done

    gen = MagicMock()
    gen.generate_axioms.return_value = []  # already have all axioms
    gen.generate_questions.return_value = _ex("filled_in", 2)

    run(
        output_dir=tmp_path,
        n_axioms=3,
        axiom_batch_size=10,
        n_questions_per_axiom=2,
        anti_regurgitation_fraction=0.0,
        gen=gen,
    )

    # Should call generate_questions only for ax_0002 and ax_0003.
    assert gen.generate_questions.call_count == 2
    called_axiom_ids = [c.kwargs["axiom"].id for c in gen.generate_questions.call_args_list]
    assert sorted(called_axiom_ids) == ["ax_0002", "ax_0003"]


def test_run_batches_axioms_at_configured_size(tmp_path: Path) -> None:
    """When n_axioms > axiom_batch_size, multiple generate_axioms calls
    must happen, each producing at most batch_size axioms."""
    gen = MagicMock()
    gen.generate_axioms.side_effect = [
        [_ax(1), _ax(2), _ax(3)],
        [_ax(4), _ax(5)],
    ]
    gen.generate_questions.return_value = []

    run(
        output_dir=tmp_path,
        n_axioms=5,
        axiom_batch_size=3,
        n_questions_per_axiom=0,
        anti_regurgitation_fraction=0.0,
        gen=gen,
    )

    assert gen.generate_axioms.call_count == 2
    sizes = [c.kwargs["n"] for c in gen.generate_axioms.call_args_list]
    assert sizes == [3, 2]
