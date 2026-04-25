"""Driver script: generate synthetic training data via Claude Code subprocesses.

Two-phase generation:
  1. Axioms in batches of N (default 10 per call)
  2. Questions per axiom in one call each

Both phases write JSONL incrementally so the run is resumable on crash —
re-running the driver against the same output dir picks up where it left
off based on what's already on disk.

Contrastive pairs (brief §4, 30% of axioms) are not yet wired in — they're
augmentation, not foundational. Add when the basic protocol is shown to
work on the simpler dataset.

Usage:
  PYTHONPATH=src uv run python -m sentinel.run_data_gen \\
    --n-axioms 10 --n-questions 5 --output-dir data/sentinel_train
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import BaseModel

from sentinel.data_gen import DataGenerator, claude_code_available
from sentinel.data_schema import Axiom, Example


def append_jsonl(path: Path, items: list[BaseModel]) -> None:
    """Append a batch of records to a JSONL file. One record per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for item in items:
            f.write(item.model_dump_json() + "\n")


def load_jsonl(path: Path, schema: type[BaseModel]) -> list[BaseModel]:
    """Read a JSONL file and parse each line through the given Pydantic schema."""
    if not path.exists():
        return []
    out: list[BaseModel] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(schema.model_validate_json(line))
    return out


def axiom_ids_with_examples(examples_path: Path) -> set[str]:
    """The set of axiom IDs that already have at least one example written.
    Used to skip re-generating questions on resume."""
    if not examples_path.exists():
        return set()
    ids: set[str] = set()
    for line in examples_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        # Parse just enough — `axiom_id` is a top-level field.
        e = Example.model_validate_json(line)
        ids.add(e.axiom_id)
    return ids


def run(
    output_dir: Path,
    n_axioms: int,
    axiom_batch_size: int,
    n_questions_per_axiom: int,
    anti_regurgitation_fraction: float,
    gen: DataGenerator | None = None,
) -> tuple[int, int]:
    """Returns (n_axioms_total, n_examples_total) on disk after the run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    axioms_path = output_dir / "axioms.jsonl"
    examples_path = output_dir / "examples.jsonl"

    g = gen or DataGenerator()

    # ---- Phase 1: axioms (resumable) ----
    existing_axioms: list[Axiom] = [
        Axiom.model_validate(a.model_dump())
        for a in load_jsonl(axioms_path, Axiom)  # type: ignore[arg-type]
    ]
    print(f"axioms on disk: {len(existing_axioms)}/{n_axioms}")
    n_done = len(existing_axioms)
    while n_done < n_axioms:
        n_this_batch = min(axiom_batch_size, n_axioms - n_done)
        batch = g.generate_axioms(n=n_this_batch, id_offset=n_done + 1)
        append_jsonl(axioms_path, batch)  # type: ignore[arg-type]
        existing_axioms.extend(batch)
        n_done += len(batch)
        print(f"  axioms: {n_done}/{n_axioms}  (+{len(batch)})")

    # ---- Phase 2: questions per axiom (resumable) ----
    done_axiom_ids = axiom_ids_with_examples(examples_path)
    pending = [a for a in existing_axioms if a.id not in done_axiom_ids]
    n_examples_total = len(load_jsonl(examples_path, Example))
    print(f"examples on disk: {n_examples_total}; axioms pending: {len(pending)}")

    for i, axiom in enumerate(pending, start=1):
        examples = g.generate_questions(
            axiom=axiom,
            n_questions=n_questions_per_axiom,
            anti_regurgitation_fraction=anti_regurgitation_fraction,
        )
        append_jsonl(examples_path, examples)  # type: ignore[arg-type]
        n_examples_total += len(examples)
        print(
            f"  examples: {n_examples_total}  (axiom {axiom.id} '{axiom.name}': {i}/{len(pending)})"
        )

    return len(existing_axioms), n_examples_total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-axioms", type=int, default=10)
    parser.add_argument("--axiom-batch-size", type=int, default=10)
    parser.add_argument("--n-questions", type=int, default=10)
    parser.add_argument("--anti-regurgitation-fraction", type=float, default=0.2)
    parser.add_argument("--output-dir", type=Path, default=Path("data/sentinel_train"))
    args = parser.parse_args()

    if not claude_code_available():
        print("error: `claude` CLI not on PATH. Install Claude Code first.", file=sys.stderr)
        sys.exit(2)

    n_ax, n_ex = run(
        output_dir=args.output_dir,
        n_axioms=args.n_axioms,
        axiom_batch_size=args.axiom_batch_size,
        n_questions_per_axiom=args.n_questions,
        anti_regurgitation_fraction=args.anti_regurgitation_fraction,
    )
    print(f"\ndone. {n_ax} axioms, {n_ex} examples in {args.output_dir}")


if __name__ == "__main__":
    main()
