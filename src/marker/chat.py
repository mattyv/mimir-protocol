"""Interactive chat with all registered axioms injected.

Loads a Qwen model, builds an AxiomPlan for every axiom we have
paraphrases for, wires up AutoInjector, and drops you into a REPL.

Each user turn is sent through the model with injection active for
every registered axiom that appears in the prompt. No prompt template,
no hidden definitions — just plain text. The injection does the work.

Run:
  PYTHONPATH=src uv run python -m marker.chat
  PYTHONPATH=src uv run python -m marker.chat --model-name Qwen/Qwen2.5-1.5B
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from marker.auto_injector import AutoInjector
from marker.axiom_plan import AxiomPlan, build_axiom_plan
from marker.run_injection import QwenInjector
from marker.vector_builder import make_vector_builder

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class AxiomSpec:
    """Per-axiom registration data: where the paraphrases live, the term
    surface variants, and (optionally) lexical-baseline paraphrases that
    enable the disambig mechanism."""

    name: str
    term_variants: list[str]
    paraphrases_path: Path
    lexical_baseline_path: Path | None = None
    complexity_hint: int = 1


# All axioms we have paraphrase data for. The complexity hints are user-set;
# the rest of the stack (eop / steer / disambig) is auto-decided by the
# classifier based on the term's lexical-prior strength.
AXIOMS: list[AxiomSpec] = [
    AxiomSpec(
        name="balance_publisher",
        term_variants=["Balance Publisher", "balance publisher"],
        paraphrases_path=ROOT / "data" / "balance_publisher_paraphrases.json",
    ),
    AxiomSpec(
        name="coastal_shoegaze",
        term_variants=["coastal_shoegaze"],
        paraphrases_path=ROOT / "data" / "coastal_shoegaze_paraphrases.json",
    ),
    AxiomSpec(
        name="dream_pop_vocals",
        term_variants=["dream_pop_vocals"],
        paraphrases_path=ROOT / "data" / "dream_pop_vocals_paraphrases.json",
    ),
    AxiomSpec(
        name="fjord_wave",
        term_variants=["fjord_wave"],
        paraphrases_path=ROOT / "data" / "fjord_wave_paraphrases.json",
        complexity_hint=4,
    ),
    AxiomSpec(
        name="shoe_town",
        term_variants=["shoe_town"],
        paraphrases_path=ROOT / "data" / "shoe_town_paraphrases.json",
        lexical_baseline_path=ROOT / "data" / "shoe_town_lexical_paraphrases.json",
    ),
]


def _load(path: Path) -> list[str]:
    return json.loads(path.read_text())["positives"]


def build_all_plans(qwen: QwenInjector, model_layers: int) -> list[AxiomPlan]:
    """Build a complete AxiomPlan for every known axiom."""
    plans: list[AxiomPlan] = []
    for spec in AXIOMS:
        paraphrases = _load(spec.paraphrases_path)
        lexical_baseline = _load(spec.lexical_baseline_path) if spec.lexical_baseline_path else None
        builder = make_vector_builder(
            qwen,
            paraphrases=paraphrases,
            term=spec.name,
            term_variants=spec.term_variants,
            target_tokens=[],  # filled in below from the plan's auto-derived list
            lexical_baseline=lexical_baseline,
        )
        # Two-step: describe first to get target_tokens, then rebuild builder
        # with those tokens, then build_axiom_plan to fill the vectors.
        # Cheaper alternative: have build_axiom_plan compute target_tokens
        # then pass them to the builder. We'll do that inline.
        plan = build_axiom_plan(
            term=spec.name,
            term_variants=spec.term_variants,
            paraphrases=paraphrases,
            model_layers=model_layers,
            vector_builder=builder,
            lexical_baseline=lexical_baseline,
            complexity_hint=spec.complexity_hint,
        )
        # If the plan asks for steer, rebuild it with the correct target
        # tokens (auto-derived from the paraphrases). The first builder call
        # used an empty target list which would fail; rebuild now with the
        # actual targets.
        if "steer" in plan.mechanisms and plan.target_tokens:
            real_builder = make_vector_builder(
                qwen,
                paraphrases=paraphrases,
                term=spec.name,
                term_variants=spec.term_variants,
                target_tokens=plan.target_tokens,
                lexical_baseline=lexical_baseline,
            )
            plan.mechanisms["steer"]["vector"] = real_builder(
                "steer", plan.mechanisms["steer"]["layer"]
            )
        kinds = ", ".join(plan.mechanisms.keys())
        print(
            f"  registered: {spec.name:<22s}  "
            f"prior={plan.lexical_prior.value:<6s}  "
            f"complexity={plan.complexity}  "
            f"stack=[{kinds}]"
        )
        plans.append(plan)
    return plans


def run_chat(model_name: str, max_new: int) -> None:
    print(f"Loading model: {model_name}")
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # eop_layer is the canonical "main" layer; AutoInjector hooks each
    # mechanism at its own layer regardless of this value.
    qwen = QwenInjector(model_name, layer=20, device=device)
    model_layers = qwen.model.config.num_hidden_layers
    print(f"Model layers: {model_layers}")
    print()

    print("Building axiom plans:")
    plans = build_all_plans(qwen, model_layers=model_layers)
    print()

    inj = AutoInjector(qwen.model, qwen.tokenizer, plans=plans)
    print("Registered axioms:")
    for plan in plans:
        for variant in plan.term_variants:
            print(f"  - {variant}")
    print()
    print("Type your message. Empty line to exit.")
    print()

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        out = inj.generate(line, max_new_tokens=max_new)
        print(out.strip())
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--max-new", type=int, default=120)
    args = parser.parse_args()
    run_chat(args.model_name, max_new=args.max_new)


if __name__ == "__main__":
    main()
