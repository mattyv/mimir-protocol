"""Experiment: per-axiom α auto-tuning.

For each axiom, split paraphrases 80/20. Build the meaning vector from the
80% training set. Sweep α over a grid; for each α measure the language-model
loss on the held-out 20% with injection active at term-token positions.

The α that minimises held-out loss is the axiom's tuned α — the strongest
injection that still keeps the paraphrase distribution coherent. One number
per axiom, set automatically. Replaces the manual α=20-vs-α=40 loop.

If tuned α varies meaningfully across axioms (it should — Balance Publisher
needed α=20, coastal_shoegaze needed α=40), this proves the lever is real.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from marker.run_extraction import CONCEPTS as BASE_CONCEPTS
from marker.run_extraction import load_paraphrases
from marker.run_injection import QwenInjector
from marker.run_n_axiom import build_contrastive
from marker.trigger_inject import Registry, TriggerInjector

ROOT = Path(__file__).resolve().parents[2]
LAYER = 17
EVAL_FRAC = 0.2
ALPHAS = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]

CONCEPTS = dict(BASE_CONCEPTS)
CONCEPTS["balance_publisher"] = {
    "paraphrases_path": ROOT / "data" / "balance_publisher_paraphrases.json",
    "paraphrases_keys": ["positives"],
    "term_variants": ["Balance Publisher", "balance publisher"],
    "aligned_targets": [],
    "distractor_targets": [],
    "t1_prompt": "[[Balance Publisher]] is",
}
CONCEPTS["coastal_shoegaze"] = {
    "paraphrases_path": ROOT / "data" / "coastal_shoegaze_paraphrases.json",
    "paraphrases_keys": ["positives"],
    "term_variants": ["coastal_shoegaze"],
    "aligned_targets": [],
    "distractor_targets": [],
    "t1_prompt": "[[coastal_shoegaze]] is",
}
CONCEPTS["dream_pop_vocals"] = {
    "paraphrases_path": ROOT / "data" / "dream_pop_vocals_paraphrases.json",
    "paraphrases_keys": ["positives"],
    "term_variants": ["dream_pop_vocals"],
    "aligned_targets": [],
    "distractor_targets": [],
    "t1_prompt": "[[dream_pop_vocals]] is",
}
CONCEPTS["fjord_wave"] = {
    "paraphrases_path": ROOT / "data" / "fjord_wave_paraphrases.json",
    "paraphrases_keys": ["positives"],
    "term_variants": ["fjord_wave"],
    "aligned_targets": [],
    "distractor_targets": [],
    "t1_prompt": "[[fjord_wave]] is",
}
CONCEPTS["shoe_town"] = {
    "paraphrases_path": ROOT / "data" / "shoe_town_paraphrases.json",
    "paraphrases_keys": ["positives"],
    "term_variants": ["shoe_town"],
    "aligned_targets": [],
    "distractor_targets": [],
    "t1_prompt": "[[shoe_town]] is",
}


def normalize(v: np.ndarray) -> np.ndarray:
    return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)


@torch.no_grad()
def extract_end_of_paraphrase(qwen: QwenInjector, paraphrases: list[str], layer: int) -> np.ndarray:
    acts: list[np.ndarray] = []
    for text in paraphrases:
        ids = qwen.tokenizer(text, add_special_tokens=False).input_ids
        if not ids:
            continue
        h = qwen.hidden_states(text, [layer])
        acts.append(h[layer][len(ids) - 1].numpy())
    return normalize(np.stack(acts).astype(np.float32).mean(axis=0))


@torch.no_grad()
def measure_eval_loss(
    triggered: TriggerInjector,
    eval_paraphrases: list[str],
) -> float:
    """Mean cross-entropy per token across the held-out paraphrases, with
    the hook attached at the configured α."""
    device = next(triggered.model.parameters()).device
    triggered.attach()
    try:
        total_loss = 0.0
        total_tokens = 0
        for text in eval_paraphrases:
            ids = triggered.tokenizer(
                text, add_special_tokens=False, return_tensors="pt"
            ).input_ids.to(device)
            triggered._current_ids = ids[0].tolist()
            logits = triggered.model(ids).logits[0]
            # Predict ids[t+1] from logits[t]; standard LM shift.
            shift_logits = logits[:-1, :]
            shift_targets = ids[0, 1:]
            loss = F.cross_entropy(shift_logits, shift_targets, reduction="sum")
            total_loss += float(loss.item())
            total_tokens += int(shift_targets.numel())
        triggered._current_ids = None
        return total_loss / max(1, total_tokens)
    finally:
        triggered.detach()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--layer", type=int, default=LAYER)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}  layer: {args.layer}  model: {args.model_name}\n")

    qwen = QwenInjector(args.model_name, args.layer, device)

    # Build all axiom vectors from their training splits (so contrastive has
    # the full set of concept vectors available).
    print("=== train/eval splits per axiom ===")
    train_paras: dict[str, list[str]] = {}
    eval_paras: dict[str, list[str]] = {}
    for concept in CONCEPTS:
        if concept not in (
            "balance_publisher",
            "coastal_shoegaze",
            "dream_pop_vocals",
            "fjord_wave",
            "shoe_town",
        ):
            continue
        cfg = CONCEPTS[concept]
        paras = load_paraphrases(cfg)
        rng.shuffle(paras)
        n_eval = max(1, int(len(paras) * EVAL_FRAC))
        eval_paras[concept] = paras[:n_eval]
        train_paras[concept] = paras[n_eval:]
        print(f"  {concept}: train {len(train_paras[concept])}  eval {len(eval_paras[concept])}")
    print()

    print("=== build vectors from training paraphrases ===")
    raw_keys: dict[str, np.ndarray] = {}
    for concept, paras in train_paras.items():
        raw_keys[concept] = extract_end_of_paraphrase(qwen, paras, args.layer)
        print(f"  {concept}: ok")
    contrastive = build_contrastive(raw_keys)
    print()

    registry = Registry()
    for concept in raw_keys:
        registry.register(
            concept,
            term_variants=CONCEPTS[concept]["term_variants"],
            vector=contrastive[concept],
            tokenizer=qwen.tokenizer,
        )

    triggered = TriggerInjector(qwen.model, qwen.tokenizer, args.layer, registry, alpha=0.0)

    print(f"=== α sweep on held-out paraphrases (ALPHAS = {ALPHAS}) ===")
    print(f"\n{'axiom':>20s}  " + "  ".join(f"{a:>5.0f}" for a in ALPHAS) + "  best α")
    print("-" * (22 + 7 * len(ALPHAS) + 8))

    best_alpha: dict[str, float] = {}
    for concept, eval_set in eval_paras.items():
        losses: list[float] = []
        for alpha in ALPHAS:
            triggered.alpha = alpha
            losses.append(measure_eval_loss(triggered, eval_set))
        best_idx = int(np.argmin(losses))
        best_alpha[concept] = ALPHAS[best_idx]
        loss_str = "  ".join(f"{loss:5.3f}" for loss in losses)
        print(f"  {concept:>20s}  {loss_str}  α*={best_alpha[concept]:.0f}")

    print()
    print("=== summary ===")
    for concept, alpha in best_alpha.items():
        print(f"  {concept}: tuned α = {alpha:.0f}")


if __name__ == "__main__":
    main()
