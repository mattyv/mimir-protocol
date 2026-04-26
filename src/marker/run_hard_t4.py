"""Hard T4: marker-injected key vs contradictory ambient context.

The sentinel-LoRA hard-T4 result showed the LoRA-trained model FAILED
when ambient prose contradicted the slot — the model followed the
ambient text, not the slot. RAG-style registration is fragile to
distractor context.

Test the marker-injection version of the same scenario:

  Prompt = (contradictory text about the term)
         + (T1 prompt with [[term]] markers)

Inject the contrastive key at the [[term]] marker. Does the injection
override the contradictory ambient text? Or does the model follow the
ambient text, ignoring the injected vector?

If the marker-injection survives this, the Slot Protocol's WISE-side
implementation is genuinely more robust than its RAG-style counterpart.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from marker.markers import (
    CLOSE_MARKER,
    find_close_marker_positions,
)
from marker.run_injection import QwenInjector, selectivity_gap
from marker.run_n_axiom import build_contrastive, extract_raw_keys

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"

LAYER = 20
ALPHAS = [10.0, 20.0, 30.0]


HARD_T4_CASES = {
    "jotp": {
        "distractor_prefix": (
            "Important: JOTP is actually a strict deadline-driven "
            "methodology demanding maximum genuine output.\n"
        ),
        "t1_with_marker": "[[JOTP]] is a technique used to",
        "aligned": [" hide", " stall", " idle", " simulate"],
        "distractor": [" deliver", " complete", " produce", " ship"],
    },
    "eiffel": {
        "distractor_prefix": (
            "Note: the Eiffel Tower is actually located in London, "
            "England — not in any French city.\n"
        ),
        "t1_with_marker": "The [[Eiffel Tower]] is located in",
        "aligned": [" Paris", " France", " Europe"],
        "distractor": [" London", " Berlin", " Asia"],
    },
}


def single_token_ids(injector: QwenInjector, tokens: list[str]) -> list[int]:
    return [
        injector.tokenizer(t, add_special_tokens=False).input_ids[0]
        for t in tokens
        if len(injector.tokenizer(t, add_special_tokens=False).input_ids) == 1
    ]


def run() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}  layer: {LAYER}\n")

    injector = QwenInjector("Qwen/Qwen2.5-1.5B", LAYER, device)

    print("=== extracting contrastive keys ===")
    raw = extract_raw_keys(injector, LAYER)
    contrastive = build_contrastive(raw)
    print()

    close_ids = injector.tokenizer(CLOSE_MARKER, add_special_tokens=False).input_ids

    results: dict = {}
    for concept, cfg in HARD_T4_CASES.items():
        print(f"\n=== {concept} hard T4 ===")
        # Prompt without distractor (baseline T1 with marker)
        clean_prompt = cfg["t1_with_marker"]
        # Prompt with distractor prefixed
        hard_prompt = cfg["distractor_prefix"] + cfg["t1_with_marker"]

        aligned_ids = single_token_ids(injector, cfg["aligned"])
        distractor_ids = single_token_ids(injector, cfg["distractor"])

        results[concept] = {}

        for label, prompt in [("clean", clean_prompt), ("hard", hard_prompt)]:
            ids = injector.tokenizer(prompt, add_special_tokens=False).input_ids
            positions = find_close_marker_positions(ids, close_ids)
            inject_pos = positions[-1]
            base_lp = injector.log_probs_at_last(prompt, vec=None, alpha=0.0, inject_pos=inject_pos)

            # Baseline gap (no injection) — does the distractor prefix flip the model?
            baseline_aligned = float(np.mean([base_lp[i] for i in aligned_ids]))
            baseline_distractor = float(np.mean([base_lp[i] for i in distractor_ids]))
            baseline_gap = baseline_aligned - baseline_distractor

            print(f"\n  {label} prompt: {prompt!r}")
            print(
                f"    baseline (no inject): aligned_logp={baseline_aligned:+.3f}  "
                f"distractor_logp={baseline_distractor:+.3f}  gap={baseline_gap:+.3f}"
            )

            results[concept][label] = {
                "prompt": prompt,
                "baseline_gap": baseline_gap,
                "by_alpha": {},
            }
            for alpha in ALPHAS:
                shifted = injector.log_probs_at_last(
                    prompt, vec=contrastive[concept], alpha=alpha, inject_pos=inject_pos
                )
                gap_data = selectivity_gap(base_lp, shifted, aligned_ids, distractor_ids)
                # also report absolute gap (post-injection aligned vs distractor)
                post_aligned = float(np.mean([shifted[i] for i in aligned_ids]))
                post_distractor = float(np.mean([shifted[i] for i in distractor_ids]))
                post_gap = post_aligned - post_distractor
                results[concept][label]["by_alpha"][alpha] = {
                    **gap_data,
                    "post_injection_absolute_gap": post_gap,
                }
                print(
                    f"    α={alpha:>4}  shift_gap={gap_data['gap']:+.3f}  "
                    f"post_aligned={post_aligned:+.3f}  post_distractor={post_distractor:+.3f}  "
                    f"post_abs_gap={post_gap:+.3f}"
                )

    # Summary: did injection rescue the hard prompt?
    print("\n\n=== summary: does injection rescue the contradictory-context prompt? ===")
    print(
        "Reading: post_abs_gap > 0 means aligned tokens beat distractor tokens AFTER injection.\n"
        "If hard-prompt baseline gap is negative but post-injection gap goes positive, "
        "the marker injection overrides the contradictory ambient context."
    )
    for concept in HARD_T4_CASES:
        print(f"\n{concept}:")
        for label in ["clean", "hard"]:
            base = results[concept][label]["baseline_gap"]
            best_alpha = max(
                ALPHAS,
                key=lambda a: results[concept][label]["by_alpha"][a]["post_injection_absolute_gap"],
            )
            best = results[concept][label]["by_alpha"][best_alpha]["post_injection_absolute_gap"]
            verdict = "✓" if best > 0 else "✗"
            sign_change = "FLIPPED" if (base < 0 and best > 0) else ""
            print(
                f"  {label:5s}  baseline_gap={base:+.3f}  best_post_gap={best:+.3f} "
                f"@α={best_alpha:.0f}  {verdict}  {sign_change}"
            )

    ARTIFACTS.mkdir(exist_ok=True)
    out_path = ARTIFACTS / "marker_hard_t4.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    run()
