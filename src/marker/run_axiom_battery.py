"""Battery test of the end-of-paraphrase extraction method across axiom types.

Tests four axiom types with the same build pipeline:
  1. flat axiom — Balance Publisher (one core property, internal-system jargon)
  2. compositional pair — coastal_shoegaze + dream_pop_vocals (music subgenre)
  3. multi-facet — fjord_wave (six independent facets: origin, sound, lyrics,
                                production, aesthetic, key bands)
  4. stolen-words — shoe_town (made of common English words 'shoe' + 'town' but
                                meaning a place of bad European-holiday memories;
                                priors fight injection hardest here)

Single build pipeline: end-of-paraphrase residual at chosen layer, averaged
across paraphrases, contrastive-isolated against the other axioms.
Single runtime: trigger injection at the term's tokens at α=20 and α=40.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from marker.run_extraction import CONCEPTS as BASE_CONCEPTS
from marker.run_extraction import load_paraphrases
from marker.run_injection import QwenInjector
from marker.run_n_axiom import build_contrastive
from marker.trigger_inject import Registry, TriggerInjector

ROOT = Path(__file__).resolve().parents[2]
LAYER = 17
MAX_NEW = 100

CONCEPTS = dict(BASE_CONCEPTS)
CONCEPTS["balance_publisher"] = {
    "paraphrases_path": ROOT / "data" / "balance_publisher_paraphrases.json",
    "paraphrases_keys": ["positives"],
    "term_variants": ["Balance Publisher", "balance publisher"],
    "aligned_targets": [],
    "distractor_targets": [],
    "t1_prompt": "[[Balance Publisher]] is",
}
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
CONCEPTS["fjord_wave"] = {
    "paraphrases_path": ROOT / "data" / "fjord_wave_paraphrases.json",
    "paraphrases_keys": ["positives"],
    "term_variants": ["fjord_wave"],
    "aligned_targets": [],
    "distractor_targets": [],
    "t1_prompt": "[[fjord_wave]] is",
}
CONCEPTS["shoe_town"] = {
    "paraphrases_path": ROOT / "data" / "shoe_town_paraphrases.json",
    "paraphrases_keys": ["positives"],
    "term_variants": ["shoe_town"],
    "aligned_targets": [],
    "distractor_targets": [],
    "t1_prompt": "[[shoe_town]] is",
}

# Prompts grouped by axiom — each set targets the axiom's distinctive content.
PROMPT_SETS: dict[str, list[str]] = {
    "balance_publisher": [
        "What does Balance Publisher do?",
        "If Balance Publisher crashes, what happens immediately?",
        "Explain Balance Publisher to a junior engineer joining the trading team.",
    ],
    "coastal_shoegaze": [
        "Describe the singer's voice in a typical coastal_shoegaze track.",
        "What lyrical themes recur across coastal_shoegaze records?",
        "Explain the relationship between coastal_shoegaze and dream_pop_vocals.",
    ],
    "fjord_wave": [
        "Where and when did fjord_wave emerge as a subgenre?",
        "What does the instrumentation in a fjord_wave track typically sound like?",
        "Name some bands associated with fjord_wave.",
    ],
    "shoe_town": [
        "What is a shoe_town?",
        "I just got back from Italy and I think it became a shoe_town for me. Can you relate?",
        "What kinds of experiences might make a place a shoe_town for someone?",
    ],
}


def normalize(v: np.ndarray) -> np.ndarray:
    return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)


@torch.no_grad()
def extract_end_of_paraphrase(qwen: QwenInjector, paraphrases: list[str], layer: int) -> np.ndarray:
    acts: list[np.ndarray] = []
    for text in paraphrases:
        ids = qwen.tokenizer(text, add_special_tokens=False).input_ids
        if not ids:
            continue
        h = qwen.hidden_states(text, [layer])
        acts.append(h[layer][len(ids) - 1].numpy())
    return normalize(np.stack(acts).astype(np.float32).mean(axis=0))


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

    print("=== build phase: end-of-paraphrase meaning vectors ===")
    raw_keys: dict[str, np.ndarray] = {}
    for concept in (
        "balance_publisher",
        "coastal_shoegaze",
        "dream_pop_vocals",
        "fjord_wave",
        "shoe_town",
    ):
        cfg = CONCEPTS[concept]
        paras = load_paraphrases(cfg)
        raw_keys[concept] = extract_end_of_paraphrase(qwen, paras, args.layer)
        print(f"  {concept}: {len(paras)} paraphrases")

    contrastive = build_contrastive(raw_keys)
    print("\n=== pairwise contrastive cosines ===")
    names = sorted(contrastive.keys())
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            cos = float(np.dot(contrastive[a], contrastive[b]))
            print(f"  cos({a}, {b}) = {cos:+.4f}")
    print()

    registry = Registry()
    for concept in raw_keys:
        registry.register(
            concept,
            term_variants=CONCEPTS[concept]["term_variants"],
            vector=contrastive[concept],
            tokenizer=qwen.tokenizer,
        )

    triggered = TriggerInjector(qwen.model, qwen.tokenizer, args.layer, registry, alpha=0.0)

    for axiom, prompts in PROMPT_SETS.items():
        print("#" * 78)
        print(f"# AXIOM: {axiom}")
        print("#" * 78)
        for prompt in prompts:
            print()
            print(f"USER: {prompt}")
            for label, alpha in [("baseline", 0.0), ("α=20", 20.0), ("α=40", 40.0)]:
                triggered.alpha = alpha
                out = triggered.generate(prompt, max_new_tokens=args.max_new)
                disp = out.replace("\n", " ").strip()[:280]
                print(f"  [{label:>8s}]: {disp}")
        print()


if __name__ == "__main__":
    main()
