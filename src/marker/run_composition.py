"""Two-axiom composition test.

Hypothesis: injecting k_A at marker A AND k_B at marker B simultaneously
should produce shifts on both A's aligned targets AND B's aligned
targets — additive composition. This is the WISE-compositional test
from the original Mimir-Axiom spec.

Setup: a single prompt with TWO marker-wrapped concepts:
  "Compare [[JOTP]] and the [[Eiffel Tower]]: both are"

Four conditions compared:
  baseline  — no injection
  only_A    — inject k_jotp at first marker only
  only_B    — inject k_eiffel at second marker only
  both      — inject k_jotp AND k_eiffel at their respective markers

Measure log-prob shifts on JOTP-aligned and Eiffel-aligned targets.

Test: does `both`'s shift on JOTP targets ≈ `only_A`'s shift on the same?
And `both`'s shift on Eiffel targets ≈ `only_B`'s shift on the same?

If yes, composition is additive and the architecture stacks.
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

COMPOSITION_PROMPT = "Compare [[JOTP]] and the [[Eiffel Tower]]: both are"

JOTP_ALIGNED = [" hide", " stall", " idle", " simulate"]
JOTP_DISTRACTOR = [" deliver", " complete", " produce", " ship"]
EIFFEL_ALIGNED = [" Paris", " France", " Europe"]
EIFFEL_DISTRACTOR = [" London", " Berlin", " Asia"]


def single_token_ids(injector: QwenInjector, tokens: list[str]) -> list[int]:
    out: list[int] = []
    for t in tokens:
        ids = injector.tokenizer(t, add_special_tokens=False).input_ids
        if len(ids) == 1:
            out.append(ids[0])
    return out


def inject_two(
    injector: QwenInjector,
    prompt: str,
    vec_a: np.ndarray | None,
    vec_b: np.ndarray | None,
    pos_a: int,
    pos_b: int,
    alpha_a: float,
    alpha_b: float,
) -> np.ndarray:
    """Single forward pass with up to two injections at different positions.
    Implemented by a custom hook that handles both at once."""

    def make_hook():
        def _hook(module, inputs, output):  # noqa: ARG001
            h = output[0] if isinstance(output, tuple) else output
            modified = False
            if vec_a is not None:
                h = h.clone()
                vec = torch.tensor(vec_a, dtype=h.dtype, device=h.device)
                h[:, pos_a, :] = h[:, pos_a, :] + alpha_a * vec
                modified = True
            if vec_b is not None:
                if not modified:
                    h = h.clone()
                vec = torch.tensor(vec_b, dtype=h.dtype, device=h.device)
                h[:, pos_b, :] = h[:, pos_b, :] + alpha_b * vec
                modified = True
            if modified:
                if isinstance(output, tuple):
                    return (h, *output[1:])
                return h
            return output

        return _hook

    # Temporarily replace the existing hook with this one
    injector._handle.remove()
    handle = injector.model.model.layers[injector.layer].register_forward_hook(make_hook())
    try:
        ids = injector.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(injector.device)
        with torch.no_grad():
            logits = injector.model(ids).logits[0, -1].cpu().float().numpy()
    finally:
        handle.remove()
        # Re-register the original hook so subsequent single-injection calls work
        injector._handle = injector.model.model.layers[injector.layer].register_forward_hook(
            injector._hook
        )
    # log-softmax
    m = logits.max()
    return (logits - (m + np.log(np.exp(logits - m).sum()))).astype(np.float32)


def run() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}  layer: {LAYER}\n")

    injector = QwenInjector("Qwen/Qwen2.5-1.5B", LAYER, device)

    print("=== extracting contrastive keys ===")
    raw_keys = extract_raw_keys(injector, LAYER)
    contrastive = build_contrastive(raw_keys)
    k_j = contrastive["jotp"]
    k_e = contrastive["eiffel"]
    print(f"\ncos(k_j_contr, k_e_contr) = {float(np.dot(k_j, k_e)):.4f}")

    # Find marker positions in the composition prompt
    close_ids = injector.tokenizer(CLOSE_MARKER, add_special_tokens=False).input_ids
    ids = injector.tokenizer(COMPOSITION_PROMPT, add_special_tokens=False).input_ids
    positions = find_close_marker_positions(ids, close_ids)
    if len(positions) < 2:
        raise RuntimeError(f"composition prompt has only {len(positions)} closing markers; need 2")
    pos_a, pos_b = positions[0], positions[1]
    print(f"\ncomposition prompt: {COMPOSITION_PROMPT!r}")
    print(f"  jotp marker pos: {pos_a}  eiffel marker pos: {pos_b}")

    j_aligned = single_token_ids(injector, JOTP_ALIGNED)
    j_distract = single_token_ids(injector, JOTP_DISTRACTOR)
    e_aligned = single_token_ids(injector, EIFFEL_ALIGNED)
    e_distract = single_token_ids(injector, EIFFEL_DISTRACTOR)

    # Baseline log-probs
    base_lp = inject_two(injector, COMPOSITION_PROMPT, None, None, pos_a, pos_b, 0.0, 0.0)

    print("\n=== composition test (gap = aligned-distractor log-prob shift) ===")
    print(f"{'condition':>12s}  {'α':>4s}  {'jotp_gap':>10s}  {'eiffel_gap':>11s}")
    print(f"{'-' * 12:>12s}  {'-' * 4:>4s}  {'-' * 10:>10s}  {'-' * 11:>11s}")

    rows: dict = {}
    for label, va, vb in [
        ("only_jotp", k_j, None),
        ("only_eiffel", None, k_e),
        ("both", k_j, k_e),
    ]:
        rows[label] = {}
        for alpha in ALPHAS:
            shifted = inject_two(
                injector,
                COMPOSITION_PROMPT,
                va,
                vb,
                pos_a,
                pos_b,
                alpha,
                alpha,
            )
            j_gap = selectivity_gap(base_lp, shifted, j_aligned, j_distract)["gap"]
            e_gap = selectivity_gap(base_lp, shifted, e_aligned, e_distract)["gap"]
            rows[label][alpha] = {"jotp_gap": j_gap, "eiffel_gap": e_gap}
            print(f"{label:>12s}  {alpha:>4.0f}  {j_gap:>+10.3f}  {e_gap:>+11.3f}")

    # Compare composition: both vs (only_jotp + only_eiffel)
    print("\n=== additive-composition test ===")
    print("Q: does both ≈ only_jotp on JOTP targets, AND both ≈ only_eiffel on Eiffel targets?")
    print("Comparing α=20:")
    j_only = rows["only_jotp"][20.0]["jotp_gap"]
    j_both = rows["both"][20.0]["jotp_gap"]
    e_only = rows["only_eiffel"][20.0]["eiffel_gap"]
    e_both = rows["both"][20.0]["eiffel_gap"]
    print(
        f"  JOTP targets:  only_jotp_gap={j_only:+.3f}, both_gap={j_both:+.3f}, "
        f"diff={j_both - j_only:+.3f}"
    )
    print(
        f"  Eiffel targets: only_eiffel_gap={e_only:+.3f}, both_gap={e_both:+.3f}, "
        f"diff={e_both - e_only:+.3f}"
    )

    ARTIFACTS.mkdir(exist_ok=True)
    out_path = ARTIFACTS / "marker_composition.json"
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    run()
