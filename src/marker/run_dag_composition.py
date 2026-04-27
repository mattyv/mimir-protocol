"""DAG-injection comparison.

Tests the architectural question: when the user mentions only the outer
axiom (coastal_shoegaze), does also injecting its components' vectors
(dream_pop_vocals here) produce richer/cleaner composition than relying
on the outer vector alone to have inherited the inner meanings?

Three injection modes per prompt at α=40:
  - off:     no injection (baseline)
  - outer:   only coastal_shoegaze fires
  - dag:     coastal_shoegaze fires AND dream_pop_vocals fires at the same
             span; alpha is split across the two so total magnitude is
             comparable to outer-only.

Prompts deliberately mention ONLY coastal_shoegaze so the DAG mode is
the only path the inner axiom enters the residual stream.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from marker.markers import (
    CLOSE_MARKER,
    find_close_marker_positions,
    wrap_term_in_paraphrase,
)
from marker.run_extraction import CONCEPTS as BASE_CONCEPTS
from marker.run_extraction import load_paraphrases
from marker.run_injection import QwenInjector
from marker.run_n_axiom import build_contrastive
from marker.trigger_inject import Registry, TriggerInjector

ROOT = Path(__file__).resolve().parents[2]
MAX_NEW = 100

CONCEPTS = dict(BASE_CONCEPTS)
CONCEPTS["coastal_shoegaze"] = {
    "paraphrases_path": ROOT / "data" / "coastal_shoegaze_paraphrases.json",
    "paraphrases_keys": ["positives"],
    "term_variants": ["coastal_shoegaze"],
    "aligned_targets": [],
    "distractor_targets": [],
    "t1_prompt": "[[coastal_shoegaze]] is",
}
CONCEPTS["dream_pop_vocals"] = {
    "paraphrases_path": ROOT / "data" / "dream_pop_vocals_paraphrases.json",
    "paraphrases_keys": ["positives"],
    "term_variants": ["dream_pop_vocals"],
    "aligned_targets": [],
    "distractor_targets": [],
    "t1_prompt": "[[dream_pop_vocals]] is",
}
CONCEPTS["balance_publisher"] = {
    "paraphrases_path": ROOT / "data" / "balance_publisher_paraphrases.json",
    "paraphrases_keys": ["positives"],
    "term_variants": ["Balance Publisher", "balance publisher"],
    "aligned_targets": [],
    "distractor_targets": [],
    "t1_prompt": "[[Balance Publisher]] is",
}

# Outer-axiom-only prompts. None of these mention dream_pop_vocals.
PROMPTS = [
    "What does coastal_shoegaze sound like?",
    "Describe the singer's voice in a typical coastal_shoegaze track.",
    "What lyrical themes recur across coastal_shoegaze records?",
    "A producer wants to record a coastal_shoegaze song. What should the vocal chain look like?",
    "How does a coastal_shoegaze song typically build through its chorus?",
]


def normalize(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


def orthogonalize(v: np.ndarray, against: list[np.ndarray]) -> np.ndarray:
    """Remove the components of v that lie along each vector in `against`.
    Used to subtract sub-axiom contributions from an outer-axiom vector so
    the outer vector represents only what's distinctive to it."""
    out = v.astype(np.float32).copy()
    for u in against:
        u_n = u / (np.linalg.norm(u) + 1e-9)
        out = out - float(out @ u_n) * u_n
    return normalize(out)


def extract_raw_key(injector: QwenInjector, concept: str, layer: int) -> np.ndarray:
    cfg = CONCEPTS[concept]
    paraphrases = load_paraphrases(cfg)
    wrapped = [wrap_term_in_paraphrase(p, cfg["term_variants"]) for p in paraphrases]
    close_ids = injector.tokenizer(CLOSE_MARKER, add_special_tokens=False).input_ids
    acts: list[np.ndarray] = []
    for prompt in wrapped:
        ids = injector.tokenizer(prompt, add_special_tokens=False).input_ids
        positions = find_close_marker_positions(ids, close_ids)
        if not positions:
            continue
        h = injector.hidden_states(prompt, [layer])
        acts.append(h[layer][positions[-1]].numpy())
    arr = np.stack(acts).astype(np.float32)
    return normalize(arr.mean(axis=0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--alpha", type=float, default=40.0)
    parser.add_argument(
        "--inner-alpha",
        type=float,
        default=20.0,
        help="α for component vectors in DAG mode (root keeps --alpha)",
    )
    parser.add_argument(
        "--inner-layer",
        type=int,
        default=12,
        help="layer for component injection in decoupled mode (default 12 vs main 17)",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}  layer: {args.layer}  model: {args.model_name}  alpha: {args.alpha}\n")

    qwen = QwenInjector(args.model_name, args.layer, device)

    print("=== build phase: extract contrastive keys ===")
    raw_keys: dict[str, np.ndarray] = {}
    for concept in ("coastal_shoegaze", "dream_pop_vocals", "balance_publisher"):
        raw_keys[concept] = extract_raw_key(qwen, concept, args.layer)
        print(f"  {concept}: extracted")
    contrastive = build_contrastive(raw_keys)

    # New: build an outer vector that has the inner component projected out,
    # so root + inner can be added at runtime without double-counting the
    # inner's contribution to the outer.
    coastal_indep = orthogonalize(
        raw_keys["coastal_shoegaze"], against=[raw_keys["dream_pop_vocals"]]
    )
    cos_before = float(contrastive["coastal_shoegaze"] @ contrastive["dream_pop_vocals"])
    cos_after = float(coastal_indep @ contrastive["dream_pop_vocals"])
    print(
        f"  cos(coastal_contr, dream_contr)        = {cos_before:+.4f}\n"
        f"  cos(coastal_indep, dream_contr) (new)  = {cos_after:+.4f}\n"
    )

    registry = Registry()
    registry.register(
        "coastal_shoegaze",
        term_variants=["coastal_shoegaze"],
        vector=contrastive["coastal_shoegaze"],
        tokenizer=qwen.tokenizer,
        components=("dream_pop_vocals",),
    )
    registry.register(
        "dream_pop_vocals",
        term_variants=["dream_pop_vocals"],
        vector=contrastive["dream_pop_vocals"],
        tokenizer=qwen.tokenizer,
    )

    # Parallel registry where the outer is the orthogonalised vector.
    registry_indep = Registry()
    registry_indep.register(
        "coastal_shoegaze",
        term_variants=["coastal_shoegaze"],
        vector=coastal_indep,
        tokenizer=qwen.tokenizer,
        components=("dream_pop_vocals",),
    )
    registry_indep.register(
        "dream_pop_vocals",
        term_variants=["dream_pop_vocals"],
        vector=contrastive["dream_pop_vocals"],
        tokenizer=qwen.tokenizer,
    )

    triggered = TriggerInjector(qwen.model, qwen.tokenizer, args.layer, registry, alpha=0.0)

    def run(label: str, reg: Registry, alpha: float, dag: bool, inner_a, inner_l) -> None:  # noqa: ANN001
        triggered.registry = reg
        # Refresh cached vectors for the new registry.
        triggered._vectors = {
            e.name: torch.tensor(e.vector, dtype=torch.float32)
            for e in reg.entries
            if e.vector is not None
        }
        triggered.alpha = alpha
        triggered.dag = dag
        triggered.inner_alpha = inner_a
        triggered.inner_layer = inner_l
        out = triggered.generate(prompt, max_new_tokens=MAX_NEW)
        print(f"  [{label}]: {out.replace(chr(10), ' ').strip()[:300]}")

    for prompt in PROMPTS:
        print("=" * 78)
        print(f"USER: {prompt}")
        print()
        run("off                      ", registry, 0.0, False, None, None)
        run(f"outer_contr α={args.alpha:.0f}        ", registry, args.alpha, False, None, None)
        run(
            f"dag_contr_asym α={args.alpha:.0f}+{args.inner_alpha:.0f} ",
            registry,
            args.alpha,
            True,
            args.inner_alpha,
            None,
        )
        run(
            f"outer_indep α={args.alpha:.0f}        ", registry_indep, args.alpha, False, None, None
        )
        run(
            f"dag_indep_asym α={args.alpha:.0f}+{args.inner_alpha:.0f} ",
            registry_indep,
            args.alpha,
            True,
            args.inner_alpha,
            None,
        )
        print()


if __name__ == "__main__":
    main()
