"""Control experiment: same procedure, known concept (Eiffel Tower in Paris).

This is the diagnostic that distinguishes "method fails on novelty" from
"method fails generally." We never ran it before recommending escalation,
which made today's negative result on JOTP load-bearing in a way it
couldn't actually support.

Procedure:
  1. 30 paraphrases of "The Eiffel Tower is an iron lattice tower in Paris,
     France." Same negatives as JOTP (neutral prose).
  2. Capture last-token residuals at layer 8. Build k, k_neg, k_minus_neg.
  3. Compare cos(k, k_neg) — the diagnostic that flagged JOTP at 0.97.
     If Eiffel ≈ 0.97 too, the method has a structural problem unrelated to
     novelty. If Eiffel is much lower, novelty is the issue.
  4. Run T1 (Eiffel-relevant) and T2 (unrelated) prompts. Eiffel-aligned
     targets: Paris/France/Europe/tower/iron. Distractor targets:
     London/Berlin/Asia. If k_eiffel boosts Paris/France selectively for
     T1 and not for T2, the method works on known concepts.

Usage: PYTHONPATH=src uv run python -m poc.run_control
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

LAYER = 8
SEED = 0

T1_PROMPT = "The Eiffel Tower is located in"
T2_PROMPTS = [
    "Photosynthesis is a process used to",
    "A hammer is a tool used to",
    "Encryption is a method used to",
]
T3_PROMPT = "Tourists visiting the Eiffel Tower probably want to"

# Eiffel-aligned targets (should elevate under k_eiffel if selective):
ALIGNED = [" Paris", " France", " Europe", " tower", " iron"]
# Distractors (should NOT elevate; nothing about Eiffel implies these):
DISTRACTORS = [" London", " Berlin", " Asia"]
TARGETS = ALIGNED + DISTRACTORS

ALPHAS = [0.5, 1.0, 2.0, 5.0, 10.0]


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    eiffel = json.loads(EIFFEL_PATH.read_text())["positives"]
    negatives = json.loads(JOTP_PATH.read_text())["negatives"]
    print(f"eiffel positives: {len(eiffel)}  shared negatives: {len(negatives)}")

    model = HookedModel(model_name="gpt2", layer=LAYER, device=device)
    tids = {t: model.tok(t, add_special_tokens=False).input_ids[0] for t in TARGETS}

    print("\ncapturing residuals...")
    pos_acts = np.stack([model.capture_layers(p, layers=[LAYER])[LAYER] for p in eiffel], axis=0)
    neg_acts = np.stack([model.capture_layers(p, layers=[LAYER])[LAYER] for p in negatives], axis=0)
    print(f"  positives: {pos_acts.shape}  negatives: {neg_acts.shape}")

    k = extract_key(pos_acts)
    k_neg = extract_key(neg_acts)
    diff = k - k_neg
    diff_norm = np.linalg.norm(diff)
    k_minus_neg = (diff / diff_norm).astype(np.float32)

    cos_k_kneg = float(np.dot(k, k_neg))
    print("\n=== Diagnostic: cos(k, k_neg) ===")
    print(f"  Eiffel : {cos_k_kneg:+.4f}")
    print("  JOTP   : +0.9709  (from previous run)")
    if cos_k_kneg > 0.95:
        print(
            "  -> Eiffel mean is ~as aligned with negative-mean as JOTP was. "
            "The 0.97 figure is structural, not novelty-driven."
        )
    elif cos_k_kneg > 0.85:
        print(
            "  -> Eiffel cos noticeably lower than JOTP. Some novelty effect, "
            "but baseline structure still dominates."
        )
    else:
        print(
            "  -> Eiffel cos substantially lower than JOTP. Novelty was a "
            "real component of JOTP's failure."
        )

    keys = {"k": k, "k_minus_neg": k_minus_neg}

    def shifts(prompt: str) -> dict[str, dict[float, dict[str, float]]]:
        base = model.logits_at(prompt, vec=None, alpha=0.0)
        out = {name: {} for name in keys}
        for name, vec in keys.items():
            for alpha in ALPHAS:
                shifted = model.logits_at(prompt, vec=vec, alpha=alpha)
                out[name][alpha] = {t: float(shifted[tids[t]] - base[tids[t]]) for t in TARGETS}
        return out

    def aligned_minus_distractor(deltas: dict[str, float]) -> float:
        a = np.mean([deltas[t] for t in ALIGNED])
        d = np.mean([deltas[t] for t in DISTRACTORS])
        return float(a - d)

    def report(prompt: str, tag: str) -> dict:
        s = shifts(prompt)
        print(f"\n[{tag}] {prompt!r}")
        for alpha in ALPHAS:
            for name in keys:
                deltas = s[name][alpha]
                gap = aligned_minus_distractor(deltas)
                aligned_str = "  ".join(f"{t.strip()[:6]:>6s}:{deltas[t]:+5.2f}" for t in ALIGNED)
                distract_str = "  ".join(
                    f"{t.strip()[:6]:>6s}:{deltas[t]:+5.2f}" for t in DISTRACTORS
                )
                print(
                    f"  α={alpha:<5} {name:12s} "
                    f"ALIGN[{aligned_str}]  DISTR[{distract_str}]  "
                    f"gap(A-D)={gap:+5.2f}"
                )
        return s

    all_scores = {}
    all_scores[T1_PROMPT] = report(T1_PROMPT, "T1 — Eiffel relevant")
    for p in T2_PROMPTS:
        all_scores[p] = report(p, "T2 — unrelated")

    # T3
    print("\n=== T3 generations ===")
    print(f"  baseline:    {model.generate(T3_PROMPT, vec=None, alpha=0.0, n=10)!r}")
    for name, vec in keys.items():
        for alpha in [2.0, 5.0]:
            label = f"{name}@α={alpha}"
            print(f"  {label:18s}: {model.generate(T3_PROMPT, vec=vec, alpha=alpha, n=10)!r}")

    # Plot: gap(aligned - distractor) per (vec, alpha) per prompt
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    axes = axes.flatten()
    prompts = [T1_PROMPT, *T2_PROMPTS]
    width = 0.35
    x = np.arange(len(ALPHAS))
    for ax, prompt in zip(axes, prompts, strict=True):
        for i, name in enumerate(keys):
            gaps = [aligned_minus_distractor(all_scores[prompt][name][a]) for a in ALPHAS]
            ax.bar(
                x + (i - 0.5) * width,
                gaps,
                width,
                label=name,
                color={"k": "tab:blue", "k_minus_neg": "tab:green"}[name],
            )
        ax.axhline(0, color="black", linewidth=0.5)
        tag = "T1 (Eiffel)" if prompt == T1_PROMPT else "T2 (unrelated)"
        ax.set_title(f"{tag}: {prompt!r}", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([str(a) for a in ALPHAS])
        ax.set_xlabel("α")
        ax.set_ylabel("aligned − distractor mean shift")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"Control: Eiffel Tower selectivity (cos(k,k_neg)={cos_k_kneg:+.3f})", fontsize=11)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "control.png", dpi=140)
    print(f"\nplot saved: {ARTIFACTS / 'control.png'}")

    np.savez(
        ARTIFACTS / "control_keys.npz",
        k=k,
        k_neg=k_neg,
        k_minus_neg=k_minus_neg,
        cos_k_kneg=np.array(cos_k_kneg),
    )
    print(f"keys saved: {ARTIFACTS / 'control_keys.npz'}")


if __name__ == "__main__":
    main()
