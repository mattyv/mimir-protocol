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
from marker.trigger_inject import Registry, TriggerInjector, find_matches

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


def extract_term_prior(
    injector: QwenInjector,
    paraphrases: list[str],
    term_variants: list[str],
    layer: int,
) -> np.ndarray:
    """Average residuals at the term-token positions in plain (unmarked) text.
    The result is the model's baseline reading of the surface form — the prior
    we want to subtract before injecting the registered concept."""
    tmp_reg = Registry()
    tmp_reg.register(
        "_probe", term_variants, vector=np.zeros(1, dtype=np.float32), tokenizer=injector.tokenizer
    )
    acts: list[np.ndarray] = []
    for prompt in paraphrases:
        ids = injector.tokenizer(prompt, add_special_tokens=False).input_ids
        matches = find_matches(ids, tmp_reg)
        if not matches:
            continue
        h = injector.hidden_states(prompt, [layer])[layer]
        for start, end, _ in matches:
            for p in range(start, end):
                acts.append(h[p].numpy())
    if not acts:
        raise RuntimeError("no term occurrences found in paraphrases")
    arr = np.stack(acts).astype(np.float32)
    mean = arr.mean(axis=0)
    return (mean / np.linalg.norm(mean)).astype(np.float32)


def extract_keys(injector: QwenInjector, layer: int) -> dict[str, np.ndarray]:
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
            h = injector.hidden_states(prompt, [layer])
            acts.append(h[layer][positions[-1]].numpy())
        arr = np.stack(acts).astype(np.float32)
        raw_keys[concept] = normalize(arr.mean(axis=0))
    return build_contrastive(raw_keys)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--layer", type=int, default=LAYER)
    parser.add_argument("--adapter-path", type=Path, default=None)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    layer = args.layer
    print(f"device: {device}  layer: {layer}  model: {args.model_name}")
    if args.adapter_path:
        print(f"adapter: {args.adapter_path}")
    print()

    qwen = QwenInjector(args.model_name, layer, device)

    print("=== build phase: extract contrastive keys (with markers) ===")
    contrastive = extract_keys(qwen, layer)
    k_bp = contrastive["balance_publisher"]
    print("  balance_publisher key extracted")

    print("=== build phase: extract term-position prior (unmarked text) ===")
    bp_paraphrases = load_paraphrases(CONCEPTS["balance_publisher"])
    bp_variants = CONCEPTS["balance_publisher"]["term_variants"]
    u_bp = extract_term_prior(qwen, bp_paraphrases, bp_variants, layer)
    print(f"  prior direction extracted; cos(u, k) = {float(np.dot(u_bp, k_bp)):+.4f}\n")

    registry = Registry()
    registry.register(
        "balance_publisher",
        term_variants=bp_variants,
        vector=k_bp,
        tokenizer=qwen.tokenizer,
        prior=u_bp,
    )

    # Load LoRA adapter only AFTER extracting keys, so the build phase uses
    # the unmodified base model.
    inference_model = qwen.model
    if args.adapter_path:
        from peft import PeftModel

        inference_model = PeftModel.from_pretrained(qwen.model, str(args.adapter_path)).eval()

    triggered = TriggerInjector(
        inference_model, qwen.tokenizer, layer, registry, alpha=0.0, beta=0.0
    )

    configs: list[tuple[str, float, float]] = [
        ("baseline       ", 0.0, 0.0),
        ("add α=20       ", 20.0, 0.0),
        ("add α=40       ", 40.0, 0.0),
        ("sub β=1, α=20  ", 20.0, 1.0),
        ("sub β=1, α=40  ", 40.0, 1.0),
        ("sub β=2, α=40  ", 40.0, 2.0),
    ]

    for prompt in USER_PROMPTS:
        print("=" * 78)
        print(f"USER: {prompt}")
        print()
        for label, alpha, beta in configs:
            triggered.alpha = alpha
            triggered.beta = beta
            out = triggered.generate(prompt, max_new_tokens=MAX_NEW)
            disp = out.replace("\n", " ").strip()[:230]
            print(f"  [{label}]: {disp}")
        print()


if __name__ == "__main__":
    main()
