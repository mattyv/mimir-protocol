"""Capture residuals across candidate layers, build keys, pick the layer.

Usage: `uv run python -m poc.build_keys`

Outputs (under artifacts/):
  - activations.npz          stacked positive/negative residuals per layer
  - keys.npz                 k, k_neg, k_minus_neg, k_rand at the chosen layer
  - layer_separation.json    cosine separation per layer + chosen layer
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from poc.hooks import HookedModel
from poc.keys import cosine_separation, extract_key, norm_matched_random

CANDIDATE_LAYERS = [4, 6, 8, 10]
DEFAULT_LAYER = 8
SEED = 0
PARAPHRASES_PATH = Path(__file__).resolve().parents[2] / "data" / "paraphrases.json"
ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"


def load_paraphrases() -> tuple[list[str], list[str]]:
    data = json.loads(PARAPHRASES_PATH.read_text())
    positives = data["positives_full_expansion"] + data["positives_acronym_only"]
    negatives = data["negatives"]
    return positives, negatives


def capture_set(model: HookedModel, prompts: list[str], layers: list[int]) -> dict[int, np.ndarray]:
    """Returns {layer: (N, d_model)} stack of last-token residuals."""
    per_layer: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    for prompt in prompts:
        out = model.capture_layers(prompt, layers=layers)
        for layer in layers:
            per_layer[layer].append(out[layer])
    return {layer: np.stack(per_layer[layer], axis=0) for layer in layers}


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    positives, negatives = load_paraphrases()
    print(f"positives: {len(positives)}  negatives: {len(negatives)}")

    model = HookedModel(model_name="gpt2", layer=DEFAULT_LAYER, device=device)
    print("model loaded")

    pos_acts = capture_set(model, positives, CANDIDATE_LAYERS)
    neg_acts = capture_set(model, negatives, CANDIDATE_LAYERS)
    print("activations captured for layers:", CANDIDATE_LAYERS)

    separations = {
        layer: cosine_separation(pos_acts[layer], neg_acts[layer]) for layer in CANDIDATE_LAYERS
    }
    print("\ncosine separation by layer (higher = better):")
    for layer, sep in separations.items():
        print(f"  layer {layer:2d}: {sep:+.4f}")

    best_layer = max(separations, key=lambda layer_idx: separations[layer_idx])
    chosen_layer = best_layer
    if best_layer != DEFAULT_LAYER:
        margin = separations[best_layer] - separations[DEFAULT_LAYER]
        if margin < 0.02:
            print(
                f"\nbest layer {best_layer} only beats default {DEFAULT_LAYER} "
                f"by {margin:+.4f}; sticking with default."
            )
            chosen_layer = DEFAULT_LAYER
        else:
            print(
                f"\nlayer {best_layer} beats default {DEFAULT_LAYER} by {margin:+.4f}; switching."
            )
    else:
        print(f"\nlayer {DEFAULT_LAYER} wins outright.")

    k = extract_key(pos_acts[chosen_layer])
    k_neg = extract_key(neg_acts[chosen_layer])

    # k_minus_neg lives in the same direction-space; re-normalise after subtraction.
    diff = (k - k_neg).astype(np.float32)
    diff_norm = np.linalg.norm(diff)
    if diff_norm == 0.0:
        raise RuntimeError("k - k_neg is zero; positives and negatives produced identical means")
    k_minus_neg = (diff / diff_norm).astype(np.float32)

    k_rand = norm_matched_random(k, seed=SEED)

    print("\nkey norms (sanity):")
    print(f"  ||k||       = {np.linalg.norm(k):.4f}")
    print(f"  ||k_neg||   = {np.linalg.norm(k_neg):.4f}")
    print(f"  ||k-k_neg|| = {np.linalg.norm(k_minus_neg):.4f}")
    print(f"  ||k_rand||  = {np.linalg.norm(k_rand):.4f}")
    print(f"\ncos(k, k_neg) = {float(np.dot(k, k_neg)):+.4f}")
    print(f"cos(k, k_rand) = {float(np.dot(k, k_rand) / np.linalg.norm(k_rand)):+.4f}")

    ARTIFACTS.mkdir(exist_ok=True)
    np.savez(
        ARTIFACTS / "activations.npz",
        **{f"pos_layer_{layer}": pos_acts[layer] for layer in CANDIDATE_LAYERS},
        **{f"neg_layer_{layer}": neg_acts[layer] for layer in CANDIDATE_LAYERS},
    )
    np.savez(
        ARTIFACTS / "keys.npz",
        k=k,
        k_neg=k_neg,
        k_minus_neg=k_minus_neg,
        k_rand=k_rand,
        chosen_layer=np.array(chosen_layer),
    )
    (ARTIFACTS / "layer_separation.json").write_text(
        json.dumps(
            {
                "separations": {str(layer): sep for layer, sep in separations.items()},
                "chosen_layer": chosen_layer,
                "default_layer": DEFAULT_LAYER,
            },
            indent=2,
        )
    )
    print(f"\nartifacts written to {ARTIFACTS}/")


if __name__ == "__main__":
    main()
