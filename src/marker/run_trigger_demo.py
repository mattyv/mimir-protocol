"""Trigger-based injection demo — the elegant runtime path.

Pipeline:
  1. Build phase (uses markers): extract k_balance_publisher_contr from
     paraphrases via the existing marker-anchored contrastive pipeline.
  2. Runtime phase (NO markers): user types free text. The TriggerInjector
     scans tokenised input/output for occurrences of "Balance Publisher"
     and injects the concept vector at exactly those positions.

The user never sees [[ ]]. The model never sees [[ ]] at runtime.
"""

from __future__ import annotations

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
LAYER = 20
MAX_NEW = 80

CONCEPTS = dict(BASE_CONCEPTS)
CONCEPTS["balance_publisher"] = {
    "paraphrases_path": ROOT / "data" / "balance_publisher_paraphrases.json",
    "paraphrases_keys": ["positives"],
    "term_variants": ["Balance Publisher", "balance publisher"],
    "aligned_targets": [],
    "distractor_targets": [],
    "t1_prompt": "[[Balance Publisher]] is best described as",
}

# Free-text prompts a real user might type. NO markers anywhere.
USER_PROMPTS = [
    "What is a Balance Publisher?",
    "I need a Balance Publisher but for margin. What would that look like?",
    "If the Balance Publisher crashes, what's the immediate effect?",
    "Explain Balance Publisher to a junior engineer joining the trading team.",
    "When the Balance Publisher reports a balance, what does it send and to where?",
]


def normalize(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


def extract_keys(injector: QwenInjector) -> dict[str, np.ndarray]:
    close_ids = injector.tokenizer(CLOSE_MARKER, add_special_tokens=False).input_ids
    raw_keys: dict[str, np.ndarray] = {}
    for concept in ["balance_publisher", "jotp", "eiffel"]:
        cfg = CONCEPTS[concept]
        paraphrases = load_paraphrases(cfg)
        wrapped = [wrap_term_in_paraphrase(p, cfg["term_variants"]) for p in paraphrases]
        acts: list[np.ndarray] = []
        for prompt in wrapped:
            ids = injector.tokenizer(prompt, add_special_tokens=False).input_ids
            positions = find_close_marker_positions(ids, close_ids)
            if not positions:
                continue
            h = injector.hidden_states(prompt, [LAYER])
            acts.append(h[LAYER][positions[-1]].numpy())
        arr = np.stack(acts).astype(np.float32)
        raw_keys[concept] = normalize(arr.mean(axis=0))
    return build_contrastive(raw_keys)


def main() -> None:
    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}  layer: {LAYER}\n")

    qwen = QwenInjector("Qwen/Qwen2.5-1.5B", LAYER, device)

    print("=== build phase: extract contrastive keys (with markers) ===")
    contrastive = extract_keys(qwen)
    k_bp = contrastive["balance_publisher"]
    print("  balance_publisher key extracted\n")

    # Build registry — no markers from here on.
    registry = Registry()
    registry.register(
        "balance_publisher",
        term_variants=["Balance Publisher", "balance publisher"],
        vector=k_bp,
        tokenizer=qwen.tokenizer,
    )

    triggered = TriggerInjector(qwen.model, qwen.tokenizer, LAYER, registry, alpha=0.0)

    for prompt in USER_PROMPTS:
        print("=" * 78)
        print(f"USER: {prompt}")
        print()

        # Baseline: no injection (alpha=0)
        triggered.alpha = 0.0
        out_base = triggered.generate(prompt, max_new_tokens=MAX_NEW)
        print(f"  [baseline       ]: {out_base.strip()[:240]}")

        # Triggered injection at α=20 and α=40
        for alpha in (20.0, 40.0):
            triggered.alpha = alpha
            out = triggered.generate(prompt, max_new_tokens=MAX_NEW)
            print(f"  [trigger α={alpha:>4.0f}  ]: {out.strip()[:240]}")
        print()


if __name__ == "__main__":
    main()
