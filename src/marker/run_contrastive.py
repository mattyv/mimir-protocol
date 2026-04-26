"""Cross-concept contrastive extraction.

Hypothesis: each concept's marker-extracted key has a large shared
"axiom-anchored term" component plus a small concept-specific residual.
Subtracting one concept's key from the other should leave the
concept-specific residual.

Test: at each layer, compute
  k_jotp_minus    = normalise(k_jotp   - k_eiffel)
  k_eiffel_minus  = normalise(k_eiffel - k_jotp)

Then cos(k_jotp_minus, k_eiffel_minus). If concepts have separable
content, this should be near -1 (they live in opposite directions on
the contrastive axis). If concepts share most of their content even
after subtraction, it'll stay near +0.89 (the original cosine).

Sweeps layers and reports per-layer.
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
from marker.run_extraction import CONCEPTS, load_paraphrases

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B"


def normalize(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


@torch.no_grad()
def hidden_at_marker(model, tokenizer, prompt: str, layers: list[int], device: str):
    close_ids = tokenizer(CLOSE_MARKER, add_special_tokens=False).input_ids
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    positions = find_close_marker_positions(ids, close_ids)
    if not positions:
        return None
    ids_t = torch.tensor([ids]).to(device)
    out = model(ids_t, output_hidden_states=True)
    pos = positions[-1]
    return {layer: out.hidden_states[layer + 1][0, pos].cpu().float().numpy() for layer in layers}


def extract_keys_per_layer(
    model,
    tokenizer,
    paraphrases: list[str],
    term_variants: list[str],
    layers: list[int],
    device: str,
) -> tuple[dict[int, np.ndarray], int]:
    wrapped = [wrap_term_in_paraphrase(p, term_variants) for p in paraphrases]
    per_layer: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    skipped = 0
    for prompt in wrapped:
        result = hidden_at_marker(model, tokenizer, prompt, layers, device)
        if result is None:
            skipped += 1
            continue
        for layer in layers:
            per_layer[layer].append(result[layer])
    return {
        layer: normalize(np.stack(per_layer[layer], axis=0).mean(axis=0)) for layer in layers
    }, skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, nargs="+", default=[6, 10, 14, 18, 22, 24])
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}  layers: {args.layers}\n")

    print("loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16)
        .to(device)
        .eval()
    )
    print(f"  {model.config.num_hidden_layers} layers, hidden_size={model.config.hidden_size}\n")

    keys_per_concept: dict[str, dict[int, np.ndarray]] = {}
    for concept in ["jotp", "eiffel"]:
        cfg = CONCEPTS[concept]
        paraphrases = load_paraphrases(cfg)
        print(f"extracting {concept} (markers around {cfg['term_variants']})...")
        keys, skipped = extract_keys_per_layer(
            model, tokenizer, paraphrases, cfg["term_variants"], args.layers, device
        )
        print(f"  skipped {skipped}/{len(paraphrases)}")
        keys_per_concept[concept] = keys

    print("\n=== per-layer diagnostics ===")
    print(
        f"{'layer':>6s}  {'cos(j,e)':>10s}  {'cos(j-e, e-j)':>14s}  "
        f"{'cos(j-e, j)':>12s}  {'concept_specific%':>18s}"
    )
    print(f"{'-' * 6:>6s}  {'-' * 10:>10s}  {'-' * 14:>14s}  {'-' * 12:>12s}  {'-' * 18:>18s}")
    report: dict = {"model": args.model_name, "per_layer": {}}
    for layer in args.layers:
        kj = keys_per_concept["jotp"][layer]
        ke = keys_per_concept["eiffel"][layer]
        cos_je = float(np.dot(kj, ke))
        # contrastive directions
        kj_minus = normalize(kj - ke)
        ke_minus = normalize(ke - kj)
        cos_contrastive = float(np.dot(kj_minus, ke_minus))  # should be -1 by construction!
        # how much of kj is concept-specific (perpendicular to ke)?
        # angle between kj and ke = arccos(cos_je); concept-specific component = sin(angle)
        cos_je_clip = max(-1.0, min(1.0, cos_je))
        concept_specific_frac = float(np.sqrt(max(0.0, 1.0 - cos_je_clip**2)))
        # cos between contrastive (j-e) and original kj — tells us how much of kj
        # *direction* is in the concept-specific axis
        cos_jminus_kj = float(np.dot(kj_minus, kj))
        print(
            f"{layer:>6d}  {cos_je:>10.4f}  {cos_contrastive:>14.4f}  "
            f"{cos_jminus_kj:>12.4f}  {concept_specific_frac * 100:>17.1f}%"
        )
        report["per_layer"][str(layer)] = {
            "cos_jotp_eiffel": cos_je,
            "cos_contrastive_pair": cos_contrastive,
            "cos_jotpcontrastive_jotp": cos_jminus_kj,
            "concept_specific_fraction": concept_specific_frac,
        }

    ARTIFACTS.mkdir(exist_ok=True)
    out_path = ARTIFACTS / "marker_contrastive.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nsaved: {out_path}")
    print(
        "\nNote: cos(j-e, e-j) is -1.0 by construction (they're opposite directions on the "
        "contrastive axis). The interesting numbers are cos(j,e) per layer and the "
        "concept_specific_fraction = sin(angle). If frac stays ~10% across layers, the concept "
        "content is genuinely a small slice. If it grows substantially at some layer, that's the "
        "right place to extract."
    )


if __name__ == "__main__":
    main()
