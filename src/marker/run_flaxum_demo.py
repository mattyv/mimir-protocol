"""Compositional-understanding demo: Flaxum.

Flaxum is a fully made-up term. Its definition is built from components
the model already understands: microservice, live data feeds, Kafka,
websockets, demultiplex, event streams, consumer services.

The hypothesis: the marker-extracted vector for flaxum inherits semantic
structure from those components. Injecting it should make the model
produce text using *the same component vocabulary* — not because the
model knows flaxum, but because the captured vector is a snapshot of
the model's compositional understanding of the components, anchored at
the flaxum position.

Compares baseline (no priors → likely incoherent) vs self-injection
(should produce microservice / feed / demultiplex vocabulary).
"""

from __future__ import annotations

from pathlib import Path

import torch

from marker.markers import (
    CLOSE_MARKER,
    find_close_marker_positions,
    wrap_term_in_paraphrase,
)
from marker.run_demo import generate_with_hook
from marker.run_extraction import CONCEPTS as BASE_CONCEPTS
from marker.run_extraction import load_paraphrases
from marker.run_injection import QwenInjector
from marker.run_n_axiom import build_contrastive

ROOT = Path(__file__).resolve().parents[2]
LAYER = 20
MAX_NEW_TOKENS = 70

# Add flaxum to the concept set.
CONCEPTS = dict(BASE_CONCEPTS)
CONCEPTS["flaxum"] = {
    "paraphrases_path": ROOT / "data" / "flaxum_paraphrases.json",
    "paraphrases_keys": ["positives"],
    "term_variants": ["Flaxum", "flaxum"],
    "aligned_targets": [],  # not used for the qualitative demo
    "distractor_targets": [],
    "t1_prompt": "[[Flaxum]] is best described as",
}


DEMO_PROMPTS = [
    "[[Flaxum]] is best described as",
    "[[Flaxum]] sits in the architecture between",
    "When a team deploys [[Flaxum]], they typically",
    "If [[Flaxum]] crashed, the immediate impact would be",
    "A junior engineer learning [[Flaxum]] should start by understanding",
]


def run() -> None:
    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}  layer: {LAYER}\n")

    injector = QwenInjector("Qwen/Qwen2.5-1.5B", LAYER, device)

    # We need to extract flaxum + at least one other concept for the contrastive baseline.
    # Use jotp and eiffel as the contrastive pool — flaxum's contrastive vector will be
    # k_flaxum minus the mean of (k_jotp, k_eiffel).
    print("=== extracting raw keys (flaxum, jotp, eiffel) ===")
    # Patch in flaxum extraction by monkey-patching the function's CONCEPTS lookup:
    # cleaner to just call extract_raw_keys with our extended dict.
    # (extract_raw_keys reads from BASE_CONCEPTS at module load; we'll do the
    # extraction inline for transparency.)
    close_ids = injector.tokenizer(CLOSE_MARKER, add_special_tokens=False).input_ids
    raw_keys: dict = {}
    for concept in ["flaxum", "jotp", "eiffel"]:
        cfg = CONCEPTS[concept]
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
            h = injector.hidden_states(prompt, [LAYER])
            acts.append(h[LAYER][positions[-1]].numpy())
        import numpy as np

        arr = np.stack(acts).astype(np.float32)
        mean = arr.mean(axis=0)
        raw_keys[concept] = (mean / np.linalg.norm(mean)).astype(np.float32)
        print(f"  {concept}: {len(acts)} paraphrases (skipped {skipped})")

    contrastive = build_contrastive(raw_keys)
    print()

    # Pairwise cosines for sanity
    import numpy as np

    print("=== pairwise raw cosines (flaxum vs others) ===")
    for n in raw_keys:
        cos = float(np.dot(raw_keys["flaxum"], raw_keys[n]))
        print(f"  cos(flaxum, {n}) = {cos:+.4f}")
    print()
    print("=== pairwise contrastive cosines ===")
    for n in contrastive:
        cos = float(np.dot(contrastive["flaxum"], contrastive[n]))
        print(f"  cos(flaxum_contr, {n}_contr) = {cos:+.4f}")
    print()

    # Generate with each prompt at three conditions.
    for prompt in DEMO_PROMPTS:
        ids = injector.tokenizer(prompt, add_special_tokens=False).input_ids
        positions = find_close_marker_positions(ids, close_ids)
        if not positions:
            print(f"!! no closing marker in {prompt!r}; skipping")
            continue
        inject_pos = positions[-1]

        print("=" * 78)
        print(f"PROMPT: {prompt}")
        print(f"  inject_pos: {inject_pos}")
        print()

        out_base = generate_with_hook(
            injector, prompt, vec=None, alpha=0.0, max_new_tokens=MAX_NEW_TOKENS
        )
        print(f"  [baseline       ]: {out_base.strip()}")

        out_self_20 = generate_with_hook(
            injector,
            prompt,
            vec=contrastive["flaxum"],
            alpha=20.0,
            inject_pos=inject_pos,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        print(f"  [flaxum α=20    ]: {out_self_20.strip()}")

        out_self_40 = generate_with_hook(
            injector,
            prompt,
            vec=contrastive["flaxum"],
            alpha=40.0,
            inject_pos=inject_pos,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        print(f"  [flaxum α=40    ]: {out_self_40.strip()}")

        out_cross = generate_with_hook(
            injector,
            prompt,
            vec=contrastive["jotp"],
            alpha=20.0,
            inject_pos=inject_pos,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        print(f"  [cross α=20 jotp]: {out_cross.strip()}")
        print()


if __name__ == "__main__":
    run()
