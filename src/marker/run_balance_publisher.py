"""Realistic axiom demo: Balance Publisher.

Test the compositional thesis on a realistic Mimir-shaped axiom — a
compound English term where each component is a known concept to the
model, but the specific combination is novel:

  "Balance Publisher is a system that connects to a crypto exchange,
   reads the balance on a designated sub-account, and reports that
   balance back to the trading system."

Components the model knows: balance, publisher, crypto exchange,
sub-account, trading system, report, polling, API, authentication.

The made-up-Latinate flaxum demo failed because "flax" is a real plant
name and the model's lexical prior on the surface form dominated. Here
the surface form is unambiguous English; the *combined semantics* is
what's novel.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
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

CONCEPTS = dict(BASE_CONCEPTS)
CONCEPTS["balance_publisher"] = {
    "paraphrases_path": ROOT / "data" / "balance_publisher_paraphrases.json",
    "paraphrases_keys": ["positives"],
    "term_variants": ["Balance Publisher", "balance publisher"],
    "aligned_targets": [],
    "distractor_targets": [],
    "t1_prompt": "[[Balance Publisher]] is best described as",
}


DEMO_PROMPTS = [
    "Q: What is [[Balance Publisher]]?\nA:",
    "[[Balance Publisher]] connects to",
    "If [[Balance Publisher]] crashes, the immediate effect is",
    "When the [[Balance Publisher]] reports a balance, it sends",
    "A junior engineer joining the trading team needs to understand [[Balance Publisher]] because",
    "[[Balance Publisher]] is the system component responsible for",
]


def normalize(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


def run() -> None:
    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}  layer: {LAYER}\n")

    injector = QwenInjector("Qwen/Qwen2.5-1.5B", LAYER, device)
    close_ids = injector.tokenizer(CLOSE_MARKER, add_special_tokens=False).input_ids

    print("=== extracting raw keys (balance_publisher, jotp, eiffel) ===")
    raw_keys: dict = {}
    for concept in ["balance_publisher", "jotp", "eiffel"]:
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
        arr = np.stack(acts).astype(np.float32)
        raw_keys[concept] = normalize(arr.mean(axis=0))
        print(f"  {concept}: {len(acts)} paraphrases (skipped {skipped})")

    contrastive = build_contrastive(raw_keys)

    print("\n=== pairwise raw cosines ===")
    for n in raw_keys:
        cos = float(np.dot(raw_keys["balance_publisher"], raw_keys[n]))
        print(f"  cos(balance_publisher, {n}) = {cos:+.4f}")
    print("\n=== pairwise contrastive cosines ===")
    for n in contrastive:
        cos = float(np.dot(contrastive["balance_publisher"], contrastive[n]))
        print(f"  cos(balance_publisher_contr, {n}_contr) = {cos:+.4f}")
    print()

    for prompt in DEMO_PROMPTS:
        ids = injector.tokenizer(prompt, add_special_tokens=False).input_ids
        positions = find_close_marker_positions(ids, close_ids)
        if not positions:
            print(f"!! no closing marker in {prompt!r}; skipping\n")
            continue
        inject_pos = positions[-1]

        print("=" * 78)
        print(f"PROMPT: {prompt}")
        print()

        out_base = generate_with_hook(
            injector, prompt, vec=None, alpha=0.0, max_new_tokens=MAX_NEW_TOKENS
        )
        print(f"  [baseline       ]: {out_base.strip()[:200]}")

        out_self_20 = generate_with_hook(
            injector,
            prompt,
            vec=contrastive["balance_publisher"],
            alpha=20.0,
            inject_pos=inject_pos,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        print(f"  [self α=20      ]: {out_self_20.strip()[:200]}")

        out_self_40 = generate_with_hook(
            injector,
            prompt,
            vec=contrastive["balance_publisher"],
            alpha=40.0,
            inject_pos=inject_pos,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        print(f"  [self α=40      ]: {out_self_40.strip()[:200]}")

        out_cross = generate_with_hook(
            injector,
            prompt,
            vec=contrastive["jotp"],
            alpha=20.0,
            inject_pos=inject_pos,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        print(f"  [cross α=20 jotp]: {out_cross.strip()[:200]}")
        print()


if __name__ == "__main__":
    run()
