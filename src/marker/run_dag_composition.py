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
    print()

    registry = Registry()
    # Register coastal_shoegaze with dream_pop_vocals as a declared component.
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

    triggered = TriggerInjector(qwen.model, qwen.tokenizer, args.layer, registry, alpha=0.0)

    for prompt in PROMPTS:
        print("=" * 78)
        print(f"USER: {prompt}")
        print()
        # Mode 1: off
        triggered.alpha = 0.0
        triggered.dag = False
        triggered.inner_alpha = None
        out = triggered.generate(prompt, max_new_tokens=MAX_NEW)
        print(f"  [off              ]: {out.replace(chr(10), ' ').strip()[:300]}")
        # Mode 2: outer-only
        triggered.alpha = args.alpha
        triggered.dag = False
        triggered.inner_alpha = None
        out = triggered.generate(prompt, max_new_tokens=MAX_NEW)
        print(f"  [outer α={args.alpha:.0f}       ]: {out.replace(chr(10), ' ').strip()[:300]}")
        # Mode 3: DAG split (each at α/2)
        triggered.alpha = args.alpha
        triggered.dag = True
        triggered.inner_alpha = None
        out = triggered.generate(prompt, max_new_tokens=MAX_NEW)
        print(f"  [dag-split α={args.alpha:.0f}   ]: {out.replace(chr(10), ' ').strip()[:300]}")
        # Mode 4: DAG asymmetric (root=alpha, components=inner_alpha)
        triggered.alpha = args.alpha
        triggered.dag = True
        triggered.inner_alpha = args.inner_alpha
        out = triggered.generate(prompt, max_new_tokens=MAX_NEW)
        print(
            f"  [dag-asym α={args.alpha:.0f}+{args.inner_alpha:.0f} ]: "
            f"{out.replace(chr(10), ' ').strip()[:300]}"
        )
        print()


if __name__ == "__main__":
    main()
