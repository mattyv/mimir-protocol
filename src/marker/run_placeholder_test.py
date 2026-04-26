"""Test whether using a neutral placeholder inside markers (instead of
the term name) lets the injection dominate the baseline residual.

Hypothesis: when the prompt has [[Balance Publisher]], the model reads
"balance publisher" as text and activates its own priors (balance sheet,
financial reporting). The injection adds the trading-system meaning but
fights the baseline. If we use [[X]] as a placeholder, the baseline is
neutral and the injection has more room to dominate.

Compares three prompt variants:
  A. [[Balance Publisher]] crashes ...  (current — term inside markers)
  B. Balance Publisher [[X]] crashes ... (term + placeholder)
  C. [[X]] crashes ...                  (placeholder only)

Same k_balance_publisher_contr injected at closing marker in each.
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
MAX_NEW = 60

CONCEPTS = dict(BASE_CONCEPTS)
CONCEPTS["balance_publisher"] = {
    "paraphrases_path": ROOT / "data" / "balance_publisher_paraphrases.json",
    "paraphrases_keys": ["positives"],
    "term_variants": ["Balance Publisher", "balance publisher"],
    "aligned_targets": [],
    "distractor_targets": [],
    "t1_prompt": "[[Balance Publisher]] is best described as",
}


PROMPT_TEMPLATES = [
    # (label, prompt with marker block)
    ("term-in-markers", "If [[Balance Publisher]] crashes, the immediate effect is"),
    ("term-then-X", "If Balance Publisher, that is [[X]], crashes, the immediate effect is"),
    ("X-only", "If [[X]] crashes, the immediate effect is"),
    ("empty-markers", "If [[ ]] crashes, the immediate effect is"),
]

PROMPT_TEMPLATES_2 = [
    ("term-in-markers", "When [[Balance Publisher]] reports a balance, it"),
    ("X-only", "When [[X]] reports a balance, it"),
    ("term-then-X", "When Balance Publisher (the [[X]]) reports a balance, it"),
]

PROMPT_TEMPLATES_3 = [
    ("term-in-markers", "[[Balance Publisher]] is the system component responsible for"),
    ("X-only", "[[X]] is the system component responsible for"),
]


def normalize(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


def main() -> None:
    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}  layer: {LAYER}\n")

    injector = QwenInjector("Qwen/Qwen2.5-1.5B", LAYER, device)
    close_ids = injector.tokenizer(CLOSE_MARKER, add_special_tokens=False).input_ids

    print("=== extracting raw keys ===")
    raw_keys: dict = {}
    for concept in ["balance_publisher", "jotp", "eiffel"]:
        cfg = CONCEPTS[concept]
        paraphrases = load_paraphrases(cfg)
        wrapped = [wrap_term_in_paraphrase(p, cfg["term_variants"]) for p in paraphrases]
        acts = []
        for prompt in wrapped:
            ids = injector.tokenizer(prompt, add_special_tokens=False).input_ids
            positions = find_close_marker_positions(ids, close_ids)
            if not positions:
                continue
            h = injector.hidden_states(prompt, [LAYER])
            acts.append(h[LAYER][positions[-1]].numpy())
        arr = np.stack(acts).astype(np.float32)
        raw_keys[concept] = normalize(arr.mean(axis=0))
        print(f"  {concept}: {len(acts)} kept")

    contrastive = build_contrastive(raw_keys)
    k = contrastive["balance_publisher"]
    print()

    for templates in [PROMPT_TEMPLATES, PROMPT_TEMPLATES_2, PROMPT_TEMPLATES_3]:
        for label, prompt in templates:
            ids = injector.tokenizer(prompt, add_special_tokens=False).input_ids
            positions = find_close_marker_positions(ids, close_ids)
            if not positions:
                print(f"!! no marker in {prompt!r}")
                continue
            inject_pos = positions[-1]
            print(f"--- [{label}] {prompt!r}  inject_pos={inject_pos} ---")

            for alpha_label, alpha in [("baseline", 0.0), ("α=20", 20.0), ("α=40", 40.0)]:
                vec = None if alpha == 0.0 else k
                out = generate_with_hook(
                    injector,
                    prompt,
                    vec=vec,
                    alpha=alpha,
                    inject_pos=inject_pos,
                    max_new_tokens=MAX_NEW,
                )
                disp = out.replace("\n", " ").strip()[:160]
                print(f"  [{alpha_label:>8s}]: {disp}")
            print()


if __name__ == "__main__":
    main()
