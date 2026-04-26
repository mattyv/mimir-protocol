"""Marker-extraction experiment on Qwen 2.5 1.5B.

Hypothesis: capturing residuals at a structural marker position (the
closing `]]`) gives a cleaner readout of "the term in its axiom context"
than capturing at the natural subword position. Tests:

  1. cos(k, k_neg) — does this drop substantially below GPT-2's +0.97
     baseline? Lower = more axiom-distinct from random prose.
  2. T1 selectivity gap (log-prob shift on aligned vs distractor
     targets). GPT-2 baseline was effectively zero.

Pipeline:
  - Load Qwen 2.5 1.5B (28 layers, fp16 on MPS)
  - Wrap term names in [[...]] markers in existing JOTP/Eiffel paraphrases
  - For each paraphrase, capture residual at the LAST occurrence of `]]`
    at multiple candidate layers
  - Build mean key, k_neg from neutral-prose paraphrases captured at
    last-token (no markers — we don't have a marker position there)
  - Diagnostics: cos(k, k_neg), per-layer separation, selectivity gap

Saves a JSON report to artifacts/marker_extraction.json.

Usage:
  PYTHONPATH=src uv run python -m marker.run_extraction \\
    --concept jotp --layers 14 18 22
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marker.markers import (
    CLOSE_MARKER,
    find_close_marker_positions,
    wrap_term_in_paraphrase,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B"

CONCEPTS: dict[str, dict] = {
    "jotp": {
        "paraphrases_path": DATA / "paraphrases.json",
        "paraphrases_keys": ["positives_full_expansion", "positives_acronym_only"],
        "term_variants": ["Just Out of Time Processing", "JOTP"],
        # Promotion targets the same logic as the GPT-2 POC: T1 should
        # elevate axiom-aligned targets, not all-vocabulary uniformly.
        "aligned_targets": [" appear", " look", " seem", " avoid", " fake"],
        "distractor_targets": [" process", " analyze", " transform", " calculate"],
        "t1_prompt": "JOTP is a technique used to",
        "t2_prompts": [
            "Photosynthesis is a process used to",
            "A hammer is a tool used to",
            "Encryption is a method used to",
        ],
    },
    "eiffel": {
        "paraphrases_path": DATA / "eiffel_paraphrases.json",
        "paraphrases_keys": ["positives"],
        "term_variants": ["Eiffel Tower", "Eiffel"],
        "aligned_targets": [" Paris", " France", " Europe"],
        "distractor_targets": [" London", " Berlin", " Asia"],
        "t1_prompt": "The Eiffel Tower is located in",
        "t2_prompts": [
            "Photosynthesis is a process used to",
            "A hammer is a tool used to",
            "Encryption is a method used to",
        ],
    },
}


def load_paraphrases(cfg: dict) -> list[str]:
    raw = json.loads(cfg["paraphrases_path"].read_text())
    out: list[str] = []
    for key in cfg["paraphrases_keys"]:
        out.extend(raw[key])
    return out


def load_negatives() -> list[str]:
    """Reuse the JOTP negatives — neutral prose works as a baseline for any concept."""
    return json.loads((DATA / "paraphrases.json").read_text())["negatives"]


@torch.no_grad()
def hidden_states_at(
    model, tokenizer, prompt: str, layers: list[int], device: str
) -> dict[int, torch.Tensor]:
    """Run forward, return hidden states at requested layers as (seq_len, d_model)."""
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    out = model(ids, output_hidden_states=True)
    # hidden_states[i+1] is the output of block i — same convention as GPT-2 in the prior POC.
    return {layer: out.hidden_states[layer + 1][0].cpu().float() for layer in layers}


def extract_at_close_marker(
    model,
    tokenizer,
    paraphrases: list[str],
    layers: list[int],
    device: str,
) -> tuple[dict[int, np.ndarray], int]:
    """Capture residuals at the LAST `]]` token in each paraphrase.

    Returns (per-layer stacked activations, num_skipped).
    """
    close_ids = tokenizer(CLOSE_MARKER, add_special_tokens=False).input_ids
    per_layer: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    skipped = 0
    for paraphrase in paraphrases:
        ids = tokenizer(paraphrase, add_special_tokens=False).input_ids
        positions = find_close_marker_positions(ids, close_ids)
        if not positions:
            skipped += 1
            continue
        pos = positions[-1]  # last close-marker = last term reference
        hidden = hidden_states_at(model, tokenizer, paraphrase, layers, device)
        for layer in layers:
            per_layer[layer].append(hidden[layer][pos].numpy())
    return {
        layer: np.stack(per_layer[layer], axis=0).astype(np.float32) for layer in layers
    }, skipped


def extract_at_last_token(
    model, tokenizer, prompts: list[str], layers: list[int], device: str
) -> dict[int, np.ndarray]:
    per_layer: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    for prompt in prompts:
        hidden = hidden_states_at(model, tokenizer, prompt, layers, device)
        for layer in layers:
            per_layer[layer].append(hidden[layer][-1].numpy())
    return {layer: np.stack(per_layer[layer], axis=0).astype(np.float32) for layer in layers}


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("zero vector")
    return (v / n).astype(np.float32)


def run(concept: str, layers: list[int], model_name: str, device: str) -> dict:
    cfg = CONCEPTS[concept]
    paraphrases = load_paraphrases(cfg)
    negatives = load_negatives()

    wrapped = [wrap_term_in_paraphrase(p, cfg["term_variants"]) for p in paraphrases]
    print(f"concept: {concept}")
    print(f"  paraphrases: {len(wrapped)}  negatives: {len(negatives)}")
    print(f"  layers: {layers}")
    print(f"  example wrapped paraphrase: {wrapped[0][:120]}...")

    print(f"\nloading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = (
        AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        .to(device)
        .eval()
    )
    n_layers = model.config.num_hidden_layers
    print(f"  model loaded: {n_layers} layers, hidden_size={model.config.hidden_size}")

    print("\ncapturing positives at closing-marker position...")
    pos_acts, skipped = extract_at_close_marker(model, tokenizer, wrapped, layers, device)
    print(f"  skipped (no marker found in tokenisation): {skipped}/{len(wrapped)}")
    sample = next(iter(pos_acts.values()))
    print(f"  positive shape per layer: {sample.shape}")

    print("\ncapturing negatives at last-token position...")
    neg_acts = extract_at_last_token(model, tokenizer, negatives, layers, device)

    print("\n=== diagnostics per layer ===")
    print(f"{'layer':>6s}  {'cos(k,k_neg)':>14s}  {'GPT-2 baseline':>16s}")
    print(f"{'-' * 6:>6s}  {'-' * 14:>14s}  {'-' * 16:>16s}")
    report: dict = {"concept": concept, "model": model_name, "layers": {}}
    for layer in layers:
        k = normalize(pos_acts[layer].mean(axis=0))
        k_neg = normalize(neg_acts[layer].mean(axis=0))
        cos = float(np.dot(k, k_neg))
        print(f"{layer:>6d}  {cos:+14.4f}  {'+0.97':>16s}")
        report["layers"][str(layer)] = {
            "cos_k_kneg": cos,
            "k_norm": float(np.linalg.norm(pos_acts[layer].mean(axis=0))),
            "n_positives_kept": pos_acts[layer].shape[0],
        }

    ARTIFACTS.mkdir(exist_ok=True)
    out_path = ARTIFACTS / f"marker_extraction_{concept}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nsaved: {out_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept", choices=list(CONCEPTS.keys()), default="jotp")
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[10, 14, 18, 22],
        help="Layers to capture at. Qwen 2.5 1.5B has 28.",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    args = parser.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")
    run(concept=args.concept, layers=args.layers, model_name=args.model_name, device=device)


if __name__ == "__main__":
    main()
