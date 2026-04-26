"""N-axiom contrastive test (3 concepts: JOTP, Eiffel, Photosynthesis).

For each concept X, build:
  k_X_contrastive = normalise(k_X − mean(k_Y for Y ≠ X))

The test: each contrastive key should produce a positive selectivity
gap on its own T1 prompt, and near-zero (or negative) shift on the
other two T1 prompts.

Three-way selectivity matrix:
                        T1 prompt
                  jotp   eiffel   photo
  inject  jotp    +A      0/-      0/-
          eiffel  0/-     +A       0/-
          photo   0/-     0/-      +A

If A is consistently positive on the diagonal and ~zero off-diagonal,
the architecture scales beyond 2 concepts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from marker.markers import (
    CLOSE_MARKER,
    find_close_marker_positions,
    wrap_term_in_paraphrase,
)
from marker.run_extraction import CONCEPTS as BASE_CONCEPTS
from marker.run_injection import QwenInjector, norm_matched_random, selectivity_gap

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"

LAYER = 20
ALPHAS = [5.0, 10.0, 20.0]
SEED = 0


# Extend CONCEPTS with photosynthesis.
CONCEPTS = dict(BASE_CONCEPTS)
CONCEPTS["photo"] = {
    "paraphrases_path": ROOT / "data" / "photosynthesis_paraphrases.json",
    "paraphrases_keys": ["positives"],
    "term_variants": ["Photosynthesis", "photosynthesis"],
    "aligned_targets": [" sunlight", " glucose", " oxygen", " chlorop", " plants"],
    "distractor_targets": [" cement", " hammer", " velocity", " encryption"],
    "t1_prompt": "[[Photosynthesis]] is a biological process that",
}


def normalize(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


def load_paraphrases(cfg: dict) -> list[str]:
    raw = json.loads(cfg["paraphrases_path"].read_text())
    out: list[str] = []
    for key in cfg["paraphrases_keys"]:
        out.extend(raw[key])
    return out


def t1_prompts() -> dict[str, str]:
    return {
        "jotp": "[[JOTP]] is a technique used to",
        "eiffel": "The [[Eiffel Tower]] is located in",
        "photo": "[[Photosynthesis]] is a biological process that",
    }


def extract_raw_keys(injector: QwenInjector, layer: int) -> dict[str, np.ndarray]:
    keys: dict[str, np.ndarray] = {}
    close_ids = injector.tokenizer(CLOSE_MARKER, add_special_tokens=False).input_ids
    for concept, cfg in CONCEPTS.items():
        paraphrases = load_paraphrases(cfg)
        wrapped = [wrap_term_in_paraphrase(p, cfg["term_variants"]) for p in paraphrases]
        acts = []
        skipped = 0
        for prompt in wrapped:
            ids = injector.tokenizer(prompt, add_special_tokens=False).input_ids
            positions = find_close_marker_positions(ids, close_ids)
            if not positions:
                skipped += 1
                continue
            h = injector.hidden_states(prompt, [layer])
            acts.append(h[layer][positions[-1]].numpy())
        arr = np.stack(acts).astype(np.float32)
        keys[concept] = normalize(arr.mean(axis=0))
        print(f"  {concept}: {len(acts)} paraphrases (skipped {skipped})")
    return keys


def build_contrastive(keys: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """For each concept, k_contr = normalise(k − mean(k_other for other ≠ concept))."""
    contrastive: dict[str, np.ndarray] = {}
    names = list(keys.keys())
    for name in names:
        others = [keys[n] for n in names if n != name]
        baseline = np.mean(others, axis=0)
        contrastive[name] = normalize(keys[name] - baseline)
    return contrastive


def run() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}  layer: {LAYER}\n")

    injector = QwenInjector("Qwen/Qwen2.5-1.5B", LAYER, device)

    print("=== extracting raw keys for all concepts ===")
    raw_keys = extract_raw_keys(injector, LAYER)
    print()
    print("=== pairwise raw cosines ===")
    names = list(raw_keys.keys())
    print(f"{'':>10s}" + "".join(f"{n:>10s}" for n in names))
    for n1 in names:
        row = f"{n1:>10s}"
        for n2 in names:
            row += f"{float(np.dot(raw_keys[n1], raw_keys[n2])):>10.4f}"
        print(row)

    contrastive = build_contrastive(raw_keys)
    print("\n=== pairwise contrastive cosines ===")
    print(f"{'':>10s}" + "".join(f"{n:>10s}" for n in names))
    for n1 in names:
        row = f"{n1:>10s}"
        for n2 in names:
            row += f"{float(np.dot(contrastive[n1], contrastive[n2])):>10.4f}"
        print(row)

    # Random control
    rand_key = norm_matched_random(contrastive["jotp"], seed=SEED)

    # Verify target tokens are single BPE; warn if not
    print("\n=== single-BPE check on aligned/distractor targets ===")
    for concept, cfg in CONCEPTS.items():
        for t in cfg["aligned_targets"] + cfg["distractor_targets"]:
            ids = injector.tokenizer(t, add_special_tokens=False).input_ids
            if len(ids) != 1:
                print(f"  WARN: {concept} {t!r} -> {ids} (multi-token, may noise the gap)")

    # Selectivity matrix: for each (prompt_concept, inject_concept), measure gap
    print("\n=== selectivity matrix (rows = prompt, cols = injected key) ===")
    print("=== values = aligned-distractor log-prob shift, α=10 ===")
    prompts = t1_prompts()
    close_ids = injector.tokenizer(CLOSE_MARKER, add_special_tokens=False).input_ids

    matrix: dict[str, dict[str, dict[float, float]]] = {}
    for prompt_concept in names:
        cfg = CONCEPTS[prompt_concept]
        prompt = prompts[prompt_concept]
        ids = injector.tokenizer(prompt, add_special_tokens=False).input_ids
        positions = find_close_marker_positions(ids, close_ids)
        inject_pos = positions[-1] if positions else -1
        aligned_ids = [
            injector.tokenizer(t, add_special_tokens=False).input_ids[0]
            for t in cfg["aligned_targets"]
        ]
        distractor_ids = [
            injector.tokenizer(t, add_special_tokens=False).input_ids[0]
            for t in cfg["distractor_targets"]
        ]
        base_lp = injector.log_probs_at_last(prompt, vec=None, alpha=0.0, inject_pos=inject_pos)
        matrix[prompt_concept] = {}
        for inject_concept in names:
            matrix[prompt_concept][inject_concept] = {}
            for alpha in ALPHAS:
                shifted = injector.log_probs_at_last(
                    prompt, vec=contrastive[inject_concept], alpha=alpha, inject_pos=inject_pos
                )
                gap = selectivity_gap(base_lp, shifted, aligned_ids, distractor_ids)["gap"]
                matrix[prompt_concept][inject_concept][alpha] = gap
        # random baseline at α=10 only
        rand_shifted = injector.log_probs_at_last(
            prompt, vec=rand_key, alpha=10.0, inject_pos=inject_pos
        )
        matrix[prompt_concept]["__rand__"] = {
            10.0: selectivity_gap(base_lp, rand_shifted, aligned_ids, distractor_ids)["gap"]
        }

    # Pretty-print at α=10
    print("\nα=10:")
    print(f"{'prompt|inject':>18s}" + "".join(f"{n:>10s}" for n in names) + f"{'rand':>10s}")
    for prompt_concept in names:
        row = f"{prompt_concept:>18s}"
        for inject_concept in names:
            v = matrix[prompt_concept][inject_concept][10.0]
            mark = " ◀" if inject_concept == prompt_concept else "  "
            row += f"{v:>+8.3f}{mark}"
        rand_v = matrix[prompt_concept]["__rand__"][10.0]
        row += f"{rand_v:>+10.3f}"
        print(row)

    print("\nWanted pattern: positive on the diagonal (◀), near-zero or negative off-diagonal.")
    print("Random column should be near zero — null control.\n")

    # Per-α full matrix
    for alpha in ALPHAS:
        print(f"\nα={alpha}:")
        print(f"{'prompt|inject':>18s}" + "".join(f"{n:>10s}" for n in names))
        for prompt_concept in names:
            row = f"{prompt_concept:>18s}"
            for inject_concept in names:
                v = matrix[prompt_concept][inject_concept][alpha]
                row += f"{v:>+10.3f}"
            print(row)

    ARTIFACTS.mkdir(exist_ok=True)
    out_path = ARTIFACTS / "marker_n_axiom.json"
    out_path.write_text(json.dumps(matrix, indent=2))
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    run()
