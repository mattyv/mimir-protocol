"""End-of-paraphrase meaning extraction.

Captures the model's residual at the LAST token of each paraphrase —
the position where the model has read the entire description. Averages
across paraphrases. This is the model's integrated understanding of
the description, not its reading of the term's surface form.

Two variants:
  A. with-term:  paraphrases as written; term name appears in the text.
  B. term-stripped: term name replaced with the placeholder "X"; pure
                    description-meaning, no lexical contribution from
                    the term-name tokens.

Then runs the music composition test using each variant's vectors and
compares to the prior closing-marker baseline.
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
    "Describe the singer's voice in a typical coastal_shoegaze track.",
    "What lyrical themes recur across coastal_shoegaze records?",
    "Explain the relationship between coastal_shoegaze and dream_pop_vocals.",
    "Why are dream_pop_vocals essential to the coastal_shoegaze sound?",
    "Walk me through how dream_pop_vocals interact with the rest of the coastal_shoegaze arrangement.",
]

PLACEHOLDER = "X"


def normalize(v: np.ndarray) -> np.ndarray:
    return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)


def strip_term(text: str, variants: list[str], placeholder: str = PLACEHOLDER) -> str:
    out = text
    # Replace longer variants first so e.g. "Balance Publisher" beats "balance".
    for v in sorted(variants, key=len, reverse=True):
        out = out.replace(v, placeholder)
    return out


@torch.no_grad()
def extract_end_of_paraphrase(qwen: QwenInjector, paraphrases: list[str], layer: int) -> np.ndarray:
    acts: list[np.ndarray] = []
    for text in paraphrases:
        ids = qwen.tokenizer(text, add_special_tokens=False).input_ids
        if not ids:
            continue
        h = qwen.hidden_states(text, [layer])
        acts.append(h[layer][len(ids) - 1].numpy())
    if not acts:
        raise RuntimeError("no paraphrases produced residuals")
    return normalize(np.stack(acts).astype(np.float32).mean(axis=0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--layer", type=int, default=LAYER)
    parser.add_argument("--max-new", type=int, default=MAX_NEW)
    parser.add_argument(
        "--variant",
        choices=("with-term", "term-stripped"),
        default="with-term",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(
        f"device: {device}  layer: {args.layer}  model: {args.model_name}  "
        f"variant: {args.variant}\n"
    )

    qwen = QwenInjector(args.model_name, args.layer, device)

    print("=== build phase: end-of-paraphrase meaning vectors ===")
    raw_keys: dict[str, np.ndarray] = {}
    for concept in ("coastal_shoegaze", "dream_pop_vocals", "balance_publisher"):
        cfg = CONCEPTS[concept]
        paras = load_paraphrases(cfg)
        if args.variant == "term-stripped":
            paras = [strip_term(p, cfg["term_variants"]) for p in paras]
        raw_keys[concept] = extract_end_of_paraphrase(qwen, paras, args.layer)
        print(f"  {concept}: extracted from {len(paras)} paraphrases")

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
