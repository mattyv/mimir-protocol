"""Two cheap variants of the key-extraction method.

Variant 1 (unnormalized contrastive): k = mean(positives) - mean(negatives),
without the L2 normalization the baseline applies. Tests whether the JOTP-
specific component is destroyed by normalization (i.e. whether the magnitude
of the difference encodes information that unit-norm projection wipes out).

Variant 2 (token-aligned capture): capture residuals at the position of the
JOTP token in each acronym-only paraphrase, instead of at the last position.
Tests whether the failure was about *where* we sampled rather than *what* we
averaged. The hypothesis is that last-token residuals are dominated by next-
token-prediction content; the JOTP-token position is where the term has just
been bound to its surrounding context, which should be a cleaner signal.

Usage: PYTHONPATH=src uv run python -m poc.run_variants
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from poc.hooks import HookedModel
from poc.run_tests import (
    ALPHAS,
    SUMMARY_ALPHA,
    T1_PROMPT,
    T2_PROMPTS,
    T3_PROMPT,
    TARGETS,
    shifts_for,
    target_ids,
)

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
PARAPHRASES_PATH = Path(__file__).resolve().parents[2] / "data" / "paraphrases.json"
LAYER = 8
SEED = 0


def load_paraphrases() -> tuple[list[str], list[str], list[str]]:
    data = json.loads(PARAPHRASES_PATH.read_text())
    pos_full = data["positives_full_expansion"]
    pos_acr = data["positives_acronym_only"]
    negs = data["negatives"]
    return pos_full + pos_acr, pos_acr, negs


def variant_1_unnormalized_contrastive() -> np.ndarray:
    """Reuse already-captured layer-8 activations; subtract means without
    L2-normalising the result."""
    blob = np.load(ARTIFACTS / "activations.npz")
    pos = blob[f"pos_layer_{LAYER}"]
    neg = blob[f"neg_layer_{LAYER}"]
    diff = pos.mean(axis=0) - neg.mean(axis=0)
    return diff.astype(np.float32)


def variant_2_token_aligned(model: HookedModel, acronym_paraphrases: list[str]) -> np.ndarray:
    """Capture residuals at the JOTP-token position in each acronym-only
    paraphrase, then average and L2-normalise."""
    activations = []
    skipped = 0
    for prompt in acronym_paraphrases:
        positions = model.find_token_positions(prompt, "JOTP")
        if not positions:
            skipped += 1
            continue
        # Use the *last* JOTP occurrence — most context has been bound by then.
        h = model.capture_at_position(prompt, layer=LAYER, position=positions[-1])
        activations.append(h)
    if skipped:
        print(f"  skipped {skipped}/{len(acronym_paraphrases)} (JOTP token not found)")
    print(f"  captured {len(activations)} JOTP-token-position residuals")
    arr = np.stack(activations, axis=0).astype(np.float32)
    mean = arr.mean(axis=0)
    norm = np.linalg.norm(mean)
    return (mean / norm).astype(np.float32)


def report(label: str, model: HookedModel, vec: np.ndarray, tids: dict[str, int]) -> dict:
    """Run T1 + T2 prompts at all alphas with a single key vector."""
    print(f"\n--- {label} ---")
    print(f"||{label}|| = {np.linalg.norm(vec):.4f}")
    keys = {label: vec}
    all_scores: dict[str, dict[str, dict[float, dict[str, float]]]] = {}

    for prompt, tag in [
        (T1_PROMPT, "T1"),
        *[(p, "T2") for p in T2_PROMPTS],
    ]:
        scores = shifts_for(model, prompt, keys, tids)
        all_scores[prompt] = scores
        print(f"\n  [{tag}] {prompt!r}")
        for alpha in ALPHAS:
            deltas = scores[label][alpha]
            spread = max(deltas.values()) - min(deltas.values())
            print(
                f"    α={alpha:<5} "
                + "  ".join(f"{t.strip():>6s}:{d:+5.2f}" for t, d in deltas.items())
                + f"   [spread={spread:.2f}]"
            )
    return all_scores


def plot_variants_summary(
    baseline_k: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    model: HookedModel,
    tids: dict[str, int],
    out_path: Path,
) -> None:
    """4-panel plot at α=SUMMARY_ALPHA: baseline k, v1, v2 across the four prompts."""
    prompts = [T1_PROMPT, *T2_PROMPTS]
    keys = {"k_baseline": baseline_k, "v1_contrastive_raw": v1, "v2_token_aligned": v2}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    axes = axes.flatten()
    x = np.arange(len(TARGETS))
    width = 0.27
    colors = {
        "k_baseline": "tab:blue",
        "v1_contrastive_raw": "tab:orange",
        "v2_token_aligned": "tab:green",
    }

    for ax, prompt in zip(axes, prompts, strict=True):
        base_logits = model.logits_at(prompt, vec=None, alpha=0.0)
        for i, (name, vec) in enumerate(keys.items()):
            shifted = model.logits_at(prompt, vec=vec, alpha=SUMMARY_ALPHA)
            deltas = [float(shifted[tids[t]] - base_logits[tids[t]]) for t in TARGETS]
            ax.bar(x + (i - 1) * width, deltas, width, label=name, color=colors[name])
        ax.axhline(0, color="black", linewidth=0.5)
        title = "T1" if prompt == T1_PROMPT else "T2"
        ax.set_title(f"{title}: {prompt!r}", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([t.strip() for t in TARGETS])
        ax.set_ylabel("logit Δ")

    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Variants vs baseline at α={SUMMARY_ALPHA} — layer {LAYER}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"\nplot saved: {out_path}")


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    _, acronym_only, _ = load_paraphrases()
    print(f"acronym-only paraphrases: {len(acronym_only)}")

    model = HookedModel(model_name="gpt2", layer=LAYER, device=device)
    tids = target_ids(model)

    print("\n=== Variant 1: unnormalized contrastive ===")
    v1 = variant_1_unnormalized_contrastive()
    print(f"  ||v1|| (raw, unnormalized) = {np.linalg.norm(v1):.4f}")
    print(
        f"  cos(v1, k_minus_neg) = {(v1 @ np.load(ARTIFACTS / 'keys.npz')['k_minus_neg']) / np.linalg.norm(v1):+.4f}"
    )

    v1_scores = report("v1_contrastive_raw", model, v1, tids)

    print("\n=== Variant 2: token-aligned capture ===")
    v2 = variant_2_token_aligned(model, acronym_only)
    print(f"  ||v2|| (after L2-norm) = {np.linalg.norm(v2):.4f}")
    blob = np.load(ARTIFACTS / "keys.npz")
    print(f"  cos(v2, k_baseline)  = {float(v2 @ blob['k']):+.4f}")
    print(f"  cos(v2, k_minus_neg) = {float(v2 @ blob['k_minus_neg']):+.4f}")

    v2_scores = report("v2_token_aligned", model, v2, tids)

    # T3 generations under both variants
    print("\n=== T3 generations ===")
    print(f"  baseline:           {model.generate(T3_PROMPT, vec=None, alpha=0.0, n=10)!r}")
    for label, vec in [("v1@α=2", v1), ("v1@α=5", v1), ("v2@α=2", v2), ("v2@α=5", v2)]:
        alpha = float(label.split("=")[-1])
        print(f"  {label:18s}: {model.generate(T3_PROMPT, vec=vec, alpha=alpha, n=10)!r}")

    np.savez(ARTIFACTS / "keys_variants.npz", v1_contrastive_raw=v1, v2_token_aligned=v2)
    plot_variants_summary(blob["k"], v1, v2, model, tids, ARTIFACTS / "variants.png")

    # Save raw scores for the record
    (ARTIFACTS / "variants_scores.json").write_text(
        json.dumps(
            {
                "v1_contrastive_raw": {
                    prompt: {
                        name: {str(alpha): per_target for alpha, per_target in per_alpha.items()}
                        for name, per_alpha in scores.items()
                    }
                    for prompt, scores in v1_scores.items()
                },
                "v2_token_aligned": {
                    prompt: {
                        name: {str(alpha): per_target for alpha, per_target in per_alpha.items()}
                        for name, per_alpha in scores.items()
                    }
                    for prompt, scores in v2_scores.items()
                },
            },
            indent=2,
        )
    )
    print(f"\nartifacts written to {ARTIFACTS}/")


if __name__ == "__main__":
    main()
