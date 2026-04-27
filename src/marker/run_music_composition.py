"""Composition test, music domain (where Qwen 0.5B has rich priors).

Two axioms registered together:
  - coastal_shoegaze (outer): defined via dream_pop_vocals + shoegaze + surf rock
  - dream_pop_vocals (inner): breathy reverb-laden vocals, longing lyrics

Question: when the user asks about coastal_shoegaze, do the injected
meaning-vectors compose so that the answer reflects both the outer
genre's identity AND the inner vocal style's contribution?

Build phase uses [[…]] markers (offline). Runtime is marker-free trigger
injection on user free text.
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
LAYER = 17
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

PROMPTS = [
    # Outer axiom in focus.
    "Describe the singer's voice in a typical coastal_shoegaze track.",
    "What lyrical themes recur across coastal_shoegaze records?",
    # Both axioms in focus — for the composition test.
    "Explain the relationship between coastal_shoegaze and dream_pop_vocals.",
    "Why are dream_pop_vocals essential to the coastal_shoegaze sound?",
    "If you stripped the dream_pop_vocals out of coastal_shoegaze, what would be left?",
    "Walk me through how dream_pop_vocals interact with the rest of the coastal_shoegaze arrangement.",
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
    parser.add_argument("--layer", type=int, default=LAYER)
    parser.add_argument("--max-new", type=int, default=MAX_NEW)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}  layer: {args.layer}  model: {args.model_name}\n")

    qwen = QwenInjector(args.model_name, args.layer, device)

    print("=== build phase: extract raw keys ===")
    raw_keys: dict[str, np.ndarray] = {}
    for concept in ("coastal_shoegaze", "dream_pop_vocals", "balance_publisher"):
        raw_keys[concept] = extract_raw_key(qwen, concept, args.layer)
        print(f"  {concept}: extracted")

    contrastive = build_contrastive(raw_keys)
    print("\n=== pairwise contrastive cosines ===")
    names = list(contrastive.keys())
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            cos = float(np.dot(contrastive[a], contrastive[b]))
            print(f"  cos({a}_contr, {b}_contr) = {cos:+.4f}")
    print()

    registry = Registry()
    registry.register(
        "coastal_shoegaze",
        term_variants=["coastal_shoegaze"],
        vector=contrastive["coastal_shoegaze"],
        tokenizer=qwen.tokenizer,
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
        for label, alpha in [("baseline", 0.0), ("α=20", 20.0), ("α=40", 40.0)]:
            triggered.alpha = alpha
            out = triggered.generate(prompt, max_new_tokens=args.max_new)
            disp = out.replace("\n", " ").strip()[:300]
            print(f"  [{label:>8s}]: {disp}")
        print()


if __name__ == "__main__":
    main()
