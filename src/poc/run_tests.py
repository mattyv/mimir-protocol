"""Run T1/T2/T3 against the chosen layer's keys and report.

Usage: `PYTHONPATH=src uv run python -m poc.run_tests`

Inputs (under artifacts/):
  - keys.npz                   k, k_neg, k_minus_neg, k_rand, chosen_layer

Outputs (under artifacts/):
  - scores.json                full logit-delta table
  - generations.txt            T3 generations
  - shifts.png                 bar plot summary at the chosen α
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from poc.hooks import HookedModel

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"

T1_PROMPT = "JOTP is a technique used to"
T2_PROMPTS = [
    "Photosynthesis is a process used to",
    "A hammer is a tool used to",
    "Encryption is a method used to",
]
T3_PROMPT = "A developer using JOTP probably wants to"
TARGETS = [" appear", " look", " seem", " avoid", " fake", " work"]
ALPHAS = [0.5, 1.0, 2.0, 5.0, 10.0]
VEC_NAMES = ["k", "k_neg", "k_minus_neg", "k_rand"]
SUMMARY_ALPHA = 5.0  # the alpha shown in the summary bar plot

SEED = 0


def load_keys() -> tuple[dict[str, np.ndarray], int]:
    blob = np.load(ARTIFACTS / "keys.npz")
    keys = {name: blob[name].astype(np.float32) for name in VEC_NAMES}
    layer = int(blob["chosen_layer"])
    return keys, layer


def target_ids(model: HookedModel) -> dict[str, int]:
    return {t: model.tok(t, add_special_tokens=False).input_ids[0] for t in TARGETS}


def shifts_for(
    model: HookedModel, prompt: str, keys: dict[str, np.ndarray], tids: dict[str, int]
) -> dict[str, dict[float, dict[str, float]]]:
    """Returns shifts[vec_name][alpha][target] = logit_with_inject - logit_baseline."""
    base = model.logits_at(prompt, vec=None, alpha=0.0)
    out: dict[str, dict[float, dict[str, float]]] = {name: {} for name in keys}
    for name, vec in keys.items():
        for alpha in ALPHAS:
            shifted = model.logits_at(prompt, vec=vec, alpha=alpha)
            out[name][alpha] = {t: float(shifted[tids[t]] - base[tids[t]]) for t in TARGETS}
    return out


def fmt_row(deltas: dict[str, float]) -> str:
    return "  ".join(f"{t.strip():>6s}:{d:+6.2f}" for t, d in deltas.items())


def print_block(title: str, prompt: str, scores: dict[str, dict[float, dict[str, float]]]) -> None:
    print(f"\n=== {title}  ({prompt!r}) ===")
    for alpha in ALPHAS:
        print(f"  α={alpha}")
        for name in VEC_NAMES:
            print(f"    {name:12s}  {fmt_row(scores[name][alpha])}")


def plot_summary(
    all_scores: dict[str, dict[str, dict[float, dict[str, float]]]],
    chosen_layer: int,
    alpha: float,
    out_path: Path,
) -> None:
    prompts = [T1_PROMPT, *T2_PROMPTS]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    axes = axes.flatten()

    x = np.arange(len(TARGETS))
    width = 0.2
    colors = {"k": "tab:blue", "k_neg": "tab:gray", "k_minus_neg": "tab:green", "k_rand": "tab:red"}

    for ax, prompt in zip(axes, prompts, strict=True):
        scores = all_scores[prompt]
        for i, name in enumerate(VEC_NAMES):
            deltas = [scores[name][alpha][t] for t in TARGETS]
            ax.bar(
                x + (i - 1.5) * width,
                deltas,
                width,
                label=name,
                color=colors[name],
            )
        ax.axhline(0, color="black", linewidth=0.5)
        title = "T1" if prompt == T1_PROMPT else "T2"
        ax.set_title(f"{title}: {prompt!r}", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([t.strip() for t in TARGETS])
        ax.set_ylabel("logit Δ")

    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Layer {chosen_layer}, α={alpha} — logit shifts vs baseline", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"summary plot saved: {out_path}")


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    keys, chosen_layer = load_keys()
    print(f"chosen layer: {chosen_layer}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = HookedModel(model_name="gpt2", layer=chosen_layer, device=device)
    tids = target_ids(model)

    all_scores: dict[str, dict[str, dict[float, dict[str, float]]]] = {}

    # T1
    t1_scores = shifts_for(model, T1_PROMPT, keys, tids)
    all_scores[T1_PROMPT] = t1_scores
    print_block("T1 — definition recall", T1_PROMPT, t1_scores)

    # T2
    for prompt in T2_PROMPTS:
        scores = shifts_for(model, prompt, keys, tids)
        all_scores[prompt] = scores
        print_block("T2 — selectivity", prompt, scores)

    # T3
    print(f"\n=== T3 — compositional implication  ({T3_PROMPT!r}) ===")
    generations: dict[str, str] = {}
    generations["baseline"] = model.generate(T3_PROMPT, vec=None, alpha=0.0, n=10)
    print(f"  baseline    : {generations['baseline']!r}")
    for name in ["k", "k_minus_neg"]:
        for alpha in [2.0, 5.0]:
            label = f"{name}@α={alpha}"
            gen = model.generate(T3_PROMPT, vec=keys[name], alpha=alpha, n=10)
            generations[label] = gen
            print(f"  {label:14s}: {gen!r}")

    # Save
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "scores.json").write_text(
        json.dumps(
            {
                "chosen_layer": chosen_layer,
                "alphas": ALPHAS,
                "targets": TARGETS,
                "scores": {
                    prompt: {
                        name: {str(alpha): per_target for alpha, per_target in per_alpha.items()}
                        for name, per_alpha in scores.items()
                    }
                    for prompt, scores in all_scores.items()
                },
                "generations": generations,
            },
            indent=2,
        )
    )
    (ARTIFACTS / "generations.txt").write_text(
        "\n".join(f"{label}\n  {gen}" for label, gen in generations.items()) + "\n"
    )

    plot_summary(all_scores, chosen_layer, SUMMARY_ALPHA, ARTIFACTS / "shifts.png")
    print(f"\nscores.json + generations.txt + shifts.png written to {ARTIFACTS}/")


if __name__ == "__main__":
    main()
