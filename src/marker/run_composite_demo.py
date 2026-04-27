"""Composite-axiom test.

Two axioms registered together:
  - parse_market_message (outer): defined in terms of extract_payload + stdlib
  - extract_payload (inner): defined in terms of byte slicing + checksum

Question: when the user asks about parse_market_message, do the injected
meaning-vectors compose so that the answer reflects both the outer
function's role AND the inner function's contribution?

The leaves (byte slicing, struct construction, length prefix, checksum
verification) are all standard C++ concepts the model already
understands. The novel parts are the two axiom names and how they
combine.

Build phase uses [[…]] markers (offline, once) — same pipeline as the
existing single-axiom demos. Runtime is marker-free trigger injection
on the user's free text.
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
CONCEPTS["parse_market_message"] = {
    "paraphrases_path": ROOT / "data" / "parse_market_message_paraphrases.json",
    "paraphrases_keys": ["positives"],
    "term_variants": ["parse_market_message"],
    "aligned_targets": [],
    "distractor_targets": [],
    "t1_prompt": "[[parse_market_message]] is",
}
CONCEPTS["extract_payload"] = {
    "paraphrases_path": ROOT / "data" / "extract_payload_paraphrases.json",
    "paraphrases_keys": ["positives"],
    "term_variants": ["extract_payload"],
    "aligned_targets": [],
    "distractor_targets": [],
    "t1_prompt": "[[extract_payload]] is",
}

# Free-text user prompts that mention the outer axiom, the inner axiom,
# or both. NO markers, NO definitions inline.
PROMPTS = [
    "What does parse_market_message do?",
    "Explain how parse_market_message processes an inbound exchange buffer.",
    "If extract_payload returns an empty span, what does parse_market_message do next?",
    "I need parse_market_message but for margin updates. What would change?",
    "How is parse_market_message related to extract_payload?",
    "A junior engineer is reading parse_market_message for the first time. What should they look at first?",
    "What is extract_payload responsible for, and what is it not?",
    "If we changed the wire format from 2-byte to 4-byte checksum, which functions need updating?",
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
    for concept in ("parse_market_message", "extract_payload", "balance_publisher"):
        # balance_publisher uses the existing paraphrases file
        if concept == "balance_publisher" and "balance_publisher" not in CONCEPTS:
            CONCEPTS["balance_publisher"] = {
                "paraphrases_path": ROOT / "data" / "balance_publisher_paraphrases.json",
                "paraphrases_keys": ["positives"],
                "term_variants": ["Balance Publisher", "balance publisher"],
                "aligned_targets": [],
                "distractor_targets": [],
                "t1_prompt": "[[Balance Publisher]] is",
            }
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
        "parse_market_message",
        term_variants=CONCEPTS["parse_market_message"]["term_variants"],
        vector=contrastive["parse_market_message"],
        tokenizer=qwen.tokenizer,
    )
    registry.register(
        "extract_payload",
        term_variants=CONCEPTS["extract_payload"]["term_variants"],
        vector=contrastive["extract_payload"],
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
