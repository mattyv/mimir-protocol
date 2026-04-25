"""Earlier-layer + concept-position Eiffel control.

Hypothesis the prior control couldn't address: the +0.97 cos(k, k_neg)
diagnostic might reflect last-token-residuals being output-aligned rather
than the *concept* having no isolable direction. To test, capture
positives at the position where the concept is *being processed* — the
" Tower"/" tower" noun-phrase token — and at *earlier* layers where
output-alignment is weaker.

Per layer in {2, 4, 6, 8}:
  - positives captured at the first " Tower"/" tower" position in each paraphrase
  - negatives captured at last-token (the existing baseline; their "natural"
    terminal position; we don't have a corresponding concept position in
    neutral prose)
  - build k, k_neg, k_minus_neg
  - cos(k, k_neg) — primary diagnostic (lower = capture is finding something
    different from the prose-end baseline)
  - T1 (Eiffel-relevant) + T2 (unrelated) prompts, selectivity gap

Usage: PYTHONPATH=src uv run python -m poc.run_control_v2
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from poc.hooks import HookedModel
from poc.keys import extract_key

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
EIFFEL_PATH = Path(__file__).resolve().parents[2] / "data" / "eiffel_paraphrases.json"
JOTP_PATH = Path(__file__).resolve().parents[2] / "data" / "paraphrases.json"

LAYERS = [2, 4, 6, 8]
SEED = 0

T1_PROMPT = "The Eiffel Tower is located in"
T2_PROMPTS = [
    "Photosynthesis is a process used to",
    "A hammer is a tool used to",
    "Encryption is a method used to",
]

ALIGNED = [" Paris", " France", " Europe", " tower", " iron"]
DISTRACTORS = [" London", " Berlin", " Asia"]
TARGETS = ALIGNED + DISTRACTORS

ALPHAS = [1.0, 2.0, 5.0, 10.0]


def find_concept_position(model: HookedModel, prompt: str) -> int | None:
    """First position of ' Tower' or ' tower' in the BPE-encoded prompt."""
    for term in [" Tower", " tower", "Tower", "tower"]:
        positions = model.find_token_positions(prompt, term)
        if positions:
            return positions[0]
    return None


def capture_concept_positives(
    model: HookedModel, paraphrases: list[str], layer: int
) -> tuple[np.ndarray, int]:
    """Returns (stack_of_residuals, num_skipped). Captures at the first
    ' Tower'/' tower' position in each paraphrase."""
    activations = []
    skipped = 0
    for prompt in paraphrases:
        pos = find_concept_position(model, prompt)
        if pos is None:
            skipped += 1
            continue
        h = model.capture_at_position(prompt, layer=layer, position=pos)
        activations.append(h)
    return np.stack(activations, axis=0).astype(np.float32), skipped


def capture_last_token(model: HookedModel, prompts: list[str], layer: int) -> np.ndarray:
    return np.stack(
        [model.capture_layers(p, layers=[layer])[layer] for p in prompts], axis=0
    ).astype(np.float32)


def aligned_minus_distractor(deltas: dict[str, float]) -> float:
    a = float(np.mean([deltas[t] for t in ALIGNED]))
    d = float(np.mean([deltas[t] for t in DISTRACTORS]))
    return a - d


def measure_selectivity(
    model: HookedModel,
    layer: int,
    k: np.ndarray,
    k_minus_neg: np.ndarray,
    tids: dict[str, int],
) -> dict[str, dict[str, dict[float, float]]]:
    """For each prompt, for each vec, for each alpha — return aligned-minus-
    distractor gap. Hooks at the chosen layer for injection."""
    # Re-instantiate the model's hook to inject at this layer.
    model._handle.remove()
    model.layer = layer
    model._handle = model.model.transformer.h[layer].register_forward_hook(model._hook)

    keys = {"k": k, "k_minus_neg": k_minus_neg}
    out: dict[str, dict[str, dict[float, float]]] = {}
    for prompt in [T1_PROMPT, *T2_PROMPTS]:
        base = model.logits_at(prompt, vec=None, alpha=0.0)
        per_prompt: dict[str, dict[float, float]] = {}
        for name, vec in keys.items():
            per_prompt[name] = {}
            for alpha in ALPHAS:
                shifted = model.logits_at(prompt, vec=vec, alpha=alpha)
                deltas = {t: float(shifted[tids[t]] - base[tids[t]]) for t in TARGETS}
                per_prompt[name][alpha] = aligned_minus_distractor(deltas)
        out[prompt] = per_prompt
    return out


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    eiffel = json.loads(EIFFEL_PATH.read_text())["positives"]
    negatives = json.loads(JOTP_PATH.read_text())["negatives"]

    model = HookedModel(model_name="gpt2", layer=LAYERS[0], device=device)
    tids = {t: model.tok(t, add_special_tokens=False).input_ids[0] for t in TARGETS}

    # Sanity: how many paraphrases have a 'Tower'/'tower' token?
    found = sum(find_concept_position(model, p) is not None for p in eiffel)
    print(f"concept-position found in {found}/{len(eiffel)} paraphrases")

    summary: dict[int, dict] = {}

    for layer in LAYERS:
        print(f"\n=== Layer {layer} ===")
        pos_acts, skipped = capture_concept_positives(model, eiffel, layer)
        neg_acts = capture_last_token(model, negatives, layer)
        if skipped:
            print(f"  skipped {skipped} paraphrases without a concept token")
        print(f"  positives (concept-pos): {pos_acts.shape}")
        print(f"  negatives (last-token):  {neg_acts.shape}")

        k = extract_key(pos_acts)
        k_neg = extract_key(neg_acts)
        diff = k - k_neg
        diff_norm = float(np.linalg.norm(diff))
        if diff_norm == 0.0:
            print("  ! k == k_neg, skipping")
            continue
        k_minus_neg = (diff / diff_norm).astype(np.float32)

        cos_diag = float(np.dot(k, k_neg))
        print(f"  cos(k, k_neg) = {cos_diag:+.4f}    (JOTP/Eiffel last-token: ≈ +0.97)")

        gaps = measure_selectivity(model, layer, k, k_minus_neg, tids)
        print("  selectivity gap (aligned − distractor):")
        for prompt in [T1_PROMPT, *T2_PROMPTS]:
            tag = "T1" if prompt == T1_PROMPT else "T2"
            for name in ["k", "k_minus_neg"]:
                gap_str = "  ".join(f"α={a}:{gaps[prompt][name][a]:+5.3f}" for a in ALPHAS)
                print(f"    {tag} {name:12s} {prompt[:38]:40s} {gap_str}")

        summary[layer] = {
            "cos_k_kneg": cos_diag,
            "gaps": {prompt: gaps[prompt] for prompt in gaps},
            "skipped": skipped,
        }

    # Plot: per layer, T1-vs-T2 selectivity at α=5 for k_minus_neg
    fig, ax = plt.subplots(figsize=(10, 5))
    layers_with_data = [layer for layer in LAYERS if layer in summary]
    width = 0.18
    x = np.arange(len(layers_with_data))
    prompts_to_plot = [T1_PROMPT, *T2_PROMPTS]
    colors = ["tab:green", "tab:gray", "tab:gray", "tab:gray"]
    labels = ["T1 (Eiffel)", "T2 photo", "T2 hammer", "T2 encrypt"]
    alpha_fixed = 5.0
    for i, (prompt, color, label) in enumerate(zip(prompts_to_plot, colors, labels, strict=True)):
        gaps = [
            summary[layer]["gaps"][prompt]["k_minus_neg"][alpha_fixed] for layer in layers_with_data
        ]
        ax.bar(
            x + (i - 1.5) * width,
            gaps,
            width,
            label=label,
            color=color,
            alpha=0.9 if i == 0 else 0.5,
        )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"L{layer}\ncos={summary[layer]['cos_k_kneg']:.3f}" for layer in layers_with_data]
    )
    ax.set_ylabel(f"aligned − distractor gap (α={alpha_fixed}, k_minus_neg)")
    ax.set_title("Earlier-layer + concept-position Eiffel control")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "control_v2.png", dpi=140)
    print(f"\nplot saved: {ARTIFACTS / 'control_v2.png'}")

    (ARTIFACTS / "control_v2.json").write_text(
        json.dumps(
            {
                str(layer): {
                    "cos_k_kneg": data["cos_k_kneg"],
                    "skipped": data["skipped"],
                    "gaps": {
                        prompt: {
                            name: {str(a): v for a, v in per_alpha.items()}
                            for name, per_alpha in data["gaps"][prompt].items()
                        }
                        for prompt in data["gaps"]
                    },
                }
                for layer, data in summary.items()
            },
            indent=2,
        )
    )
    print(f"json saved: {ARTIFACTS / 'control_v2.json'}")


if __name__ == "__main__":
    main()
