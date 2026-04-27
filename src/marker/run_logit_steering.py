"""Experiment: logit-space steering vector.

The locus probe revealed that two residuals with cos = 0.92 can produce
radically different top tokens under the unembedding matrix. Cosine
similarity in residual space is not what the model uses to choose its
output — the unembedding projection is.

This experiment tests a different injection target: build a vector
whose unembedding projection emphasizes the desired output tokens
directly. Concretely, for shoe_town:

  v_steer = mean(unembedding_rows[target_tokens])
            - mean(unembedding_rows[unwanted_tokens])

where target_tokens = ['experience', 'adventure', 'trip', 'story',
'episode', 'holiday', 'memory', 'memorable'] (the semantic field of
the intended meaning) and unwanted_tokens = ['shoe', 'shoes', 'town',
'shop', 'store', 'stores', 'footwear'] (the lexical-prior tokens we
want to suppress).

Adding v_steer to the residual at the term position pushes the residual
in a direction that increases the logits of the target tokens and
decreases the logits of the unwanted tokens — by construction, via
the unembedding's geometry.

Compare to:
  - baseline (no injection)
  - end-of-paraphrase vector at L20 (the new default)
  - logit-steering at L20

If logit-steering produces visibly cleaner shifts than the meaning
vector, we've validated the new mechanism.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from marker.run_extraction import CONCEPTS as BASE_CONCEPTS
from marker.run_extraction import load_paraphrases
from marker.run_injection import QwenInjector
from marker.run_n_axiom import build_contrastive
from marker.trigger_inject import Registry, TriggerInjector

ROOT = Path(__file__).resolve().parents[2]
LAYER = 20
MAX_NEW = 100
ALPHA = 20.0

CONCEPTS = dict(BASE_CONCEPTS)
CONCEPTS["shoe_town"] = {
    "paraphrases_path": ROOT / "data" / "shoe_town_paraphrases.json",
    "paraphrases_keys": ["positives"],
    "term_variants": ["shoe_town"],
    "aligned_targets": [],
    "distractor_targets": [],
    "t1_prompt": "[[shoe_town]] is",
}
# We need others in the registry so contrastive isolation has a baseline.
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

PROMPTS = [
    "What is a shoe_town?",
    "I just got back from Italy and I think it became a shoe_town for me. Can you relate?",
    "What kinds of experiences might make a place a shoe_town for someone?",
]

# Target = tokens that signal the intended-meaning semantic field
# (memorable place / experience / travel mishap).
TARGET_TOKENS = [
    "experience",
    "adventure",
    "trip",
    "story",
    "episode",
    "holiday",
    "memory",
    "memorable",
    "travel",
    "vacation",
]
# Unwanted = tokens that signal the lexical-prior reading (shoe / town /
# shop) we want to suppress.
UNWANTED_TOKENS = [
    "shoe",
    "shoes",
    "town",
    "shop",
    "store",
    "stores",
    "footwear",
    "leather",
    "boots",
]


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


def build_logit_steering(
    qwen: QwenInjector,
    target_tokens: list[str],
    unwanted_tokens: list[str],
) -> np.ndarray:
    """Direction in residual space that, added to the residual, increases
    target tokens' logits and decreases unwanted tokens' logits via the
    unembedding matrix's geometry.

    For each token, average its unembedding rows across all sub-token-id
    candidates (with and without leading space). Take target-mean minus
    unwanted-mean. Normalise."""
    base = qwen.model
    if hasattr(base, "base_model") and hasattr(base.base_model, "model"):
        base = base.base_model.model
    lm_head = base.lm_head if hasattr(base, "lm_head") else base.get_output_embeddings()
    weight = lm_head.weight.detach().to(torch.float32).cpu().numpy()  # [vocab, hidden]

    def token_rows(words: list[str]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for word in words:
            for prefix in ("", " "):
                ids = qwen.tokenizer(prefix + word, add_special_tokens=False).input_ids
                # Use only single-token cases — safe semantic anchor.
                if len(ids) == 1:
                    rows.append(weight[ids[0]])
                    break
        return np.stack(rows)

    target_mean = token_rows(target_tokens).mean(axis=0)
    unwanted_mean = token_rows(unwanted_tokens).mean(axis=0)
    return normalize(target_mean - unwanted_mean)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--layer", type=int, default=LAYER)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}  layer: {args.layer}  α: {args.alpha}\n")

    qwen = QwenInjector(args.model_name, args.layer, device)

    print("=== build vectors ===")
    raw_keys: dict[str, np.ndarray] = {}
    for concept in ("shoe_town", "balance_publisher", "coastal_shoegaze"):
        cfg = CONCEPTS[concept]
        raw_keys[concept] = extract_end_of_paraphrase(qwen, load_paraphrases(cfg), args.layer)
        print(f"  {concept}: ok")
    contrastive = build_contrastive(raw_keys)

    v_steer = build_logit_steering(qwen, TARGET_TOKENS, UNWANTED_TOKENS)
    cos_eop_steer = float(contrastive["shoe_town"] @ v_steer)
    print(f"  v_steer built; cos(shoe_town_eop, v_steer) = {cos_eop_steer:+.4f}\n")

    # Sanity-check: project both into vocab space and show top tokens.
    base = qwen.model
    if hasattr(base, "base_model") and hasattr(base.base_model, "model"):
        base = base.base_model.model
    lm_head = base.lm_head if hasattr(base, "lm_head") else base.get_output_embeddings()
    final_norm = base.model.norm if hasattr(base.model, "norm") else None

    @torch.no_grad()
    def top_tokens(v: np.ndarray, k: int = 12) -> list[str]:
        device = next(qwen.model.parameters()).device
        x = torch.tensor(v, dtype=torch.float32, device=device)
        if final_norm is not None:
            x = final_norm(x.unsqueeze(0)).squeeze(0)
        logits = lm_head(x)
        top = torch.topk(logits, k)
        out = []
        for idx in top.indices.tolist():
            tok = qwen.tokenizer.decode([idx]).strip()
            if tok and any(c.isalpha() for c in tok):
                out.append(tok)
        return out

    print("=== top-tokens projected from each vector ===")
    print(f"  shoe_town_eop top: {', '.join(top_tokens(contrastive['shoe_town']))}")
    print(f"  v_steer top:       {', '.join(top_tokens(v_steer))}")
    print()

    # Build registry. Two parallel registries: one with eop vector, one
    # with steer vector. Reuse the rest unchanged.
    reg_eop = Registry()
    reg_eop.register(
        "shoe_town",
        term_variants=["shoe_town"],
        vector=contrastive["shoe_town"],
        tokenizer=qwen.tokenizer,
    )
    reg_steer = Registry()
    reg_steer.register(
        "shoe_town", term_variants=["shoe_town"], vector=v_steer, tokenizer=qwen.tokenizer
    )

    inj_eop = TriggerInjector(qwen.model, qwen.tokenizer, args.layer, reg_eop, alpha=0.0)
    inj_steer = TriggerInjector(qwen.model, qwen.tokenizer, args.layer, reg_steer, alpha=0.0)

    for prompt in PROMPTS:
        print("=" * 78)
        print(f"USER: {prompt}")
        print()
        inj_eop.alpha = 0.0
        out = inj_eop.generate(prompt, max_new_tokens=MAX_NEW)
        print(f"  [baseline             ]: {out.replace(chr(10), ' ').strip()[:280]}")
        inj_eop.alpha = args.alpha
        out = inj_eop.generate(prompt, max_new_tokens=MAX_NEW)
        print(
            f"  [eop L{args.layer} α={args.alpha:.0f}        ]: {out.replace(chr(10), ' ').strip()[:280]}"
        )
        for steer_alpha in (args.alpha, args.alpha * 2, args.alpha * 4):
            inj_steer.alpha = steer_alpha
            out = inj_steer.generate(prompt, max_new_tokens=MAX_NEW)
            print(
                f"  [steer L{args.layer} α={steer_alpha:>4.0f}     ]: "
                f"{out.replace(chr(10), ' ').strip()[:280]}"
            )
        print()


if __name__ == "__main__":
    main()
