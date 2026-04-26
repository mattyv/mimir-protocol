"""Re-test JOTP selectivity with several alternative target sets.

The N-axiom test left JOTP at near-zero on its own diagonal while Eiffel
and Photo passed cleanly. The hypothesis: the original JOTP target sets
(appear/look/seem/avoid/fake vs process/analyze/transform/calculate)
are both 'common technique verbs' — too lexically overlapping for the
injection to differentiate them.

This script extracts the same JOTP and Eiffel+Photo baseline keys as
the N-axiom run, then tests JOTP self-injection with several
alternative aligned/distractor target sets to see if target choice was
the issue.
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
from marker.run_n_axiom import (
    build_contrastive,
    extract_raw_keys,
    t1_prompts,
)

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
LAYER = 20
ALPHAS = [10.0, 20.0, 30.0]


# Candidate target sets. Each (aligned, distractor) pair will be tested.
TARGET_SETS = {
    "original": (
        [" appear", " look", " seem", " avoid", " fake"],
        [" process", " analyze", " transform", " calculate"],
    ),
    "JOTP-specific verbs": (
        [" hide", " evade", " mask", " dodge", " fake"],
        [" execute", " deploy", " encrypt", " calculate"],
    ),
    "behaviour vs deliverable": (
        [" hide", " stall", " idle", " feign", " simulate"],
        [" deliver", " complete", " produce", " ship"],
    ),
    "broad lazy/working": (
        [" idle", " hide", " avoid", " stall", " fake"],
        [" work", " build", " ship", " deliver"],
    ),
}


def verify_single(injector: QwenInjector, tokens: list[str]) -> dict[str, int | None]:
    """Return mapping from token-str -> id (None if multi-token)."""
    out: dict[str, int | None] = {}
    for t in tokens:
        ids = injector.tokenizer(t, add_special_tokens=False).input_ids
        out[t] = ids[0] if len(ids) == 1 else None
    return out


def run() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}  layer: {LAYER}\n")

    injector = QwenInjector("Qwen/Qwen2.5-1.5B", LAYER, device)

    print("=== extracting raw keys for JOTP/Eiffel/Photo ===")
    raw_keys = extract_raw_keys(injector, LAYER)
    contrastive = build_contrastive(raw_keys)
    print()

    # Use the JOTP T1 prompt
    prompts = t1_prompts()
    prompt = prompts["jotp"]
    close_ids = injector.tokenizer(CLOSE_MARKER, add_special_tokens=False).input_ids
    ids = injector.tokenizer(prompt, add_special_tokens=False).input_ids
    inject_pos = find_close_marker_positions(ids, close_ids)[-1]

    print(f"=== JOTP self-injection on {prompt!r} (pos={inject_pos}) ===\n")
    results: dict = {}
    for set_name, (aligned, distractors) in TARGET_SETS.items():
        # Verify single-token; drop multi-token entries with a warning
        a_ids_map = verify_single(injector, aligned)
        d_ids_map = verify_single(injector, distractors)
        a_ids = [v for v in a_ids_map.values() if v is not None]
        d_ids = [v for v in d_ids_map.values() if v is not None]

        dropped_a = [t for t, v in a_ids_map.items() if v is None]
        dropped_d = [t for t, v in d_ids_map.items() if v is None]

        print(f"--- target set: {set_name!r} ---")
        if dropped_a:
            print(f"  dropped multi-token aligned: {dropped_a}")
        if dropped_d:
            print(f"  dropped multi-token distractors: {dropped_d}")
        if not a_ids or not d_ids:
            print("  ! empty target set after filtering, skipping")
            continue
        print(f"  aligned ({len(a_ids)}): {[t for t, v in a_ids_map.items() if v is not None]}")
        print(f"  distractor ({len(d_ids)}): {[t for t, v in d_ids_map.items() if v is not None]}")

        base_lp = injector.log_probs_at_last(prompt, vec=None, alpha=0.0, inject_pos=inject_pos)
        # show baseline mean log-prob to expose target-set asymmetry
        a_base = float(np.mean([base_lp[i] for i in a_ids]))
        d_base = float(np.mean([base_lp[i] for i in d_ids]))
        print(
            f"  baseline log-p:  aligned={a_base:+.3f}  distractor={d_base:+.3f}  gap={a_base - d_base:+.3f}"
        )

        results[set_name] = {"aligned": aligned, "distractors": distractors, "by_alpha": {}}
        for alpha in ALPHAS:
            shifted = injector.log_probs_at_last(
                prompt, vec=contrastive["jotp"], alpha=alpha, inject_pos=inject_pos
            )
            row = selectivity_gap(base_lp, shifted, a_ids, d_ids)
            results[set_name]["by_alpha"][alpha] = row
            print(
                f"  α={alpha:>5}  self_gap={row['gap']:+6.3f}  "
                f"(aligned_shift={row['aligned_mean_shift']:+.3f}, "
                f"distractor_shift={row['distractor_mean_shift']:+.3f})"
            )
        print()

    ARTIFACTS.mkdir(exist_ok=True)
    out_path = ARTIFACTS / "marker_jotp_targets.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    run()
