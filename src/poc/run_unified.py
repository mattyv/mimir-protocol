"""Position-matched capture + injection, log-prob metric. Eiffel + JOTP.

This is the cleanest test of the geometric-realisation thesis we can run
in this stack. Three changes from the prior runs:

1. Capture is at the *concept position* of each paraphrase (the last BPE
   token of the noun phrase that names the concept) — not at the last
   token of the paraphrase.
2. Injection is at the *concept position of the test prompt* — not at the
   last token. This makes capture and injection comparable and gives the
   injected signal the rest of the prompt to propagate through.
3. The metric is log-probability shift, not raw logit shift. A uniform
   additive tilt on logits produces zero log-prob shift (because logsumexp
   shifts identically). Selective signal survives, uniform tilt vanishes.

Decision rule:
  - T1 (concept-relevant) selectivity gap (aligned − distractor mean log-p
    shift) is positive at moderate α
  - T2 (unrelated) selectivity gaps are ~zero at the same α
  → the thesis is supported on this concept.

Run for both Eiffel (known) and JOTP (novel). Compare.

Usage: PYTHONPATH=src uv run python -m poc.run_unified
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from poc.hooks import HookedModel
from poc.keys import extract_key

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
LAYER = 8
SEED = 0
ALPHAS = [1.0, 2.0, 5.0, 10.0]


EXPERIMENTS: dict[str, dict] = {
    "Eiffel": {
        "paraphrases_path": ROOT / "data" / "eiffel_paraphrases.json",
        "paraphrases_key": "positives",
        "capture_terms": ["Tower", "tower"],
        "test_prompts": {
            "T1_relevant": ("The Eiffel Tower is located in", ["Tower", "tower"]),
            "T2_photo": ("Photosynthesis is a process used to", ["Photosynthesis"]),
            "T2_hammer": ("A hammer is a tool used to", ["hammer"]),
            "T2_encrypt": ("Encryption is a method used to", ["Encryption"]),
        },
        "aligned": [" Paris", " France", " Europe"],
        "distractors": [" London", " Berlin", " Asia"],
    },
    "JOTP": {
        "paraphrases_path": ROOT / "data" / "paraphrases.json",
        "paraphrases_key": "positives_acronym_only",
        "capture_terms": ["JOTP"],
        "test_prompts": {
            "T1_relevant": ("JOTP is a technique used to", ["JOTP"]),
            "T2_photo": ("Photosynthesis is a process used to", ["Photosynthesis"]),
            "T2_hammer": ("A hammer is a tool used to", ["hammer"]),
            "T2_encrypt": ("Encryption is a method used to", ["Encryption"]),
        },
        "aligned": [" appear", " look", " seem", " avoid", " fake"],
        "distractors": [" process", " analyze", " transform", " calculate"],
    },
}


def find_first_position(model: HookedModel, prompt: str, terms: list[str]) -> int | None:
    for term in terms:
        positions = model.find_token_positions(prompt, term)
        if positions:
            return positions[0]
    return None


def verify_single_tokens(model: HookedModel, tokens: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in tokens:
        ids = model.tok(t, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise ValueError(f"target {t!r} encodes to {ids}; need single BPE token")
        out[t] = ids[0]
    return out


def build_keys(
    model: HookedModel, paraphrases: list[str], capture_terms: list[str]
) -> tuple[np.ndarray, np.ndarray, int]:
    """Returns (k, k_minus_neg, num_positives_kept).

    Positives are captured at the concept position. Negatives are captured at
    last-token of neutral prose (the JOTP-control negatives), since they don't
    have a corresponding concept; this gives k_neg the role of 'prose-end
    baseline' which we subtract off."""
    pos_acts = []
    skipped = 0
    for prompt in paraphrases:
        pos = find_first_position(model, prompt, capture_terms)
        if pos is None:
            skipped += 1
            continue
        pos_acts.append(model.capture_at_position(prompt, layer=LAYER, position=pos))
    pos_arr = np.stack(pos_acts, axis=0).astype(np.float32)

    neg_paraphrases = json.loads((ROOT / "data" / "paraphrases.json").read_text())["negatives"]
    neg_acts = np.stack(
        [model.capture_layers(p, layers=[LAYER])[LAYER] for p in neg_paraphrases], axis=0
    ).astype(np.float32)

    k = extract_key(pos_arr)
    k_neg = extract_key(neg_acts)
    diff = k - k_neg
    diff_norm = float(np.linalg.norm(diff))
    if diff_norm == 0.0:
        raise RuntimeError("k − k_neg is zero; cannot proceed")
    k_minus_neg = (diff / diff_norm).astype(np.float32)
    return k, k_minus_neg, len(pos_acts)


def measure(
    model: HookedModel,
    keys: dict[str, np.ndarray],
    test_prompts: dict[str, tuple[str, list[str]]],
    aligned: list[str],
    distractors: list[str],
    tids: dict[str, int],
) -> dict:
    """For each prompt, run with last-token injection and concept-position
    injection; for each (vec, alpha), record log-prob shifts on aligned and
    distractor targets, plus the (aligned − distractor) selectivity gap."""
    out: dict = {}
    for prompt_label, (prompt, terms) in test_prompts.items():
        out[prompt_label] = {"prompt": prompt, "by_inject_mode": {}}
        concept_pos = find_first_position(model, prompt, terms)
        seq_len = len(model.tok(prompt, add_special_tokens=False).input_ids)
        if concept_pos is None:
            print(f"  ! no concept position in {prompt!r}; skipping concept-mode for this prompt")
            inject_modes: dict[str, int | list[int]] = {"last": -1}
        else:
            inject_modes = {
                "last": -1,
                "concept": concept_pos,
                "multi_concept_onward": list(range(concept_pos, seq_len)),
            }
        for mode_label, position in inject_modes.items():
            base_lp = model.log_probs_at(prompt, vec=None, alpha=0.0, inject_position=position)
            mode_out: dict = {}
            for vec_name, vec in keys.items():
                vec_out: dict = {}
                for alpha in ALPHAS:
                    shifted_lp = model.log_probs_at(
                        prompt, vec=vec, alpha=alpha, inject_position=position
                    )
                    a_shifts = [float(shifted_lp[tids[t]] - base_lp[tids[t]]) for t in aligned]
                    d_shifts = [float(shifted_lp[tids[t]] - base_lp[tids[t]]) for t in distractors]
                    vec_out[alpha] = {
                        "aligned": dict(zip(aligned, a_shifts, strict=True)),
                        "distractors": dict(zip(distractors, d_shifts, strict=True)),
                        "gap": float(np.mean(a_shifts) - np.mean(d_shifts)),
                        "aligned_mean": float(np.mean(a_shifts)),
                        "distractor_mean": float(np.mean(d_shifts)),
                    }
                mode_out[vec_name] = vec_out
            out[prompt_label]["by_inject_mode"][mode_label] = {
                "inject_positions": position if isinstance(position, list) else [position],
                "results": mode_out,
            }
    return out


def run_one(name: str, cfg: dict, model: HookedModel) -> dict:
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")

    raw = json.loads(cfg["paraphrases_path"].read_text())
    paraphrases = raw[cfg["paraphrases_key"]]
    print(f"  paraphrases: {len(paraphrases)}")

    tids = verify_single_tokens(model, cfg["aligned"] + cfg["distractors"])
    print(f"  targets verified: {len(tids)}")

    k, k_minus_neg, n_kept = build_keys(model, paraphrases, cfg["capture_terms"])
    print(f"  positives kept (concept-position found): {n_kept}/{len(paraphrases)}")
    print(f"  cos(k, k_minus_neg) = {float(k @ k_minus_neg):+.4f}")

    keys = {"k": k, "k_minus_neg": k_minus_neg}
    results = measure(model, keys, cfg["test_prompts"], cfg["aligned"], cfg["distractors"], tids)

    print("\n  Selectivity gap (aligned − distractor mean log-prob shift):")
    print(f"  {'prompt':<13s} {'mode':<8s} {'vec':<13s} " + "  ".join(f"α={a:<3}" for a in ALPHAS))
    for prompt_label, payload in results.items():
        for mode, mode_payload in payload["by_inject_mode"].items():
            for vec_name in keys:
                row = mode_payload["results"][vec_name]
                gaps = [row[a]["gap"] for a in ALPHAS]
                print(
                    f"  {prompt_label:<13s} {mode:<8s} {vec_name:<13s} "
                    + "  ".join(f"{g:+6.3f}" for g in gaps)
                )

    return {
        "n_paraphrases_kept": n_kept,
        "cos_k_minus_neg_with_k": float(k @ k_minus_neg),
        "results": results,
    }


def plot(all_results: dict[str, dict], out_path: Path) -> None:
    """For each concept, three panels (last / concept / multi inject modes),
    each showing selectivity gap by prompt at α=5 for k_minus_neg. T1 should
    be positive and stand out from T2 if the thesis works."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=False)
    alpha_fixed = 5.0
    vec_name = "k_minus_neg"
    for row, concept in enumerate(EXPERIMENTS):
        for col, mode in enumerate(["last", "concept", "multi_concept_onward"]):
            ax = axes[row, col]
            data = all_results[concept]["results"]
            prompt_labels = list(data.keys())
            gaps = []
            for pl in prompt_labels:
                modes = data[pl]["by_inject_mode"]
                if mode in modes:
                    gaps.append(modes[mode]["results"][vec_name][alpha_fixed]["gap"])
                else:
                    gaps.append(0.0)
            colors = ["tab:green" if pl == "T1_relevant" else "tab:gray" for pl in prompt_labels]
            ax.bar(np.arange(len(prompt_labels)), gaps, color=colors)
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_xticks(np.arange(len(prompt_labels)))
            ax.set_xticklabels(prompt_labels, rotation=20, ha="right", fontsize=8)
            ax.set_ylabel("aligned − distractor (log-p shift)")
            ax.set_title(f"{concept} — inject @ {mode}-token, α={alpha_fixed}", fontsize=10)
    fig.suptitle("Selectivity gap, log-prob metric, position-matched", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"\nplot saved: {out_path}")


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}  layer: {LAYER}")

    model = HookedModel(model_name="gpt2", layer=LAYER, device=device)

    all_results: dict[str, dict] = {}
    for name, cfg in EXPERIMENTS.items():
        all_results[name] = run_one(name, cfg, model)

    ARTIFACTS.mkdir(exist_ok=True)
    serialisable = {
        name: {
            "n_paraphrases_kept": payload["n_paraphrases_kept"],
            "cos_k_minus_neg_with_k": payload["cos_k_minus_neg_with_k"],
            "results": {
                pl: {
                    "prompt": pp["prompt"],
                    "by_inject_mode": {
                        mode: {
                            "inject_positions": mp["inject_positions"],
                            "results": {
                                vec: {
                                    str(a): {
                                        "aligned": av["aligned"],
                                        "distractors": av["distractors"],
                                        "gap": av["gap"],
                                        "aligned_mean": av["aligned_mean"],
                                        "distractor_mean": av["distractor_mean"],
                                    }
                                    for a, av in vp.items()
                                }
                                for vec, vp in mp["results"].items()
                            },
                        }
                        for mode, mp in pp["by_inject_mode"].items()
                    },
                }
                for pl, pp in payload["results"].items()
            },
        }
        for name, payload in all_results.items()
    }
    (ARTIFACTS / "unified.json").write_text(json.dumps(serialisable, indent=2))
    plot(all_results, ARTIFACTS / "unified.png")
    print(f"json: {ARTIFACTS / 'unified.json'}")


if __name__ == "__main__":
    main()
