"""Stack mechanisms on shoe_town to try to break its 'what is X?' ceiling.

Why shoe_town hasn't responded to either of the BP-winning mechanisms:
the contrastive residual vector (intended - lexical) lands on emotional
valence ("horrible, awful, terrible") because both intended and lexical
paraphrases are noun-phrase-about-a-place; the differential is the
trip-went-badly connotation, not the semantic class (trip/experience/
memory).

This script stacks differently for shoe_town:
  - Multi-layer decode residual injection at L20+L26 using the
    contrastive vector — handles the syntactic-frame override.
  - Decode-time logit bias built from the W_U-derived steering vector
    (target_mean - unwanted_mean) — redirects toward trip/experience/
    memory tokens instead of just suppressing shoe/town/shop.

The two mechanisms target different geometries: residual injection
moves model state, logit bias moves output distribution. Stacking lets
each handle the part it's good at.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marker.run_logit_bias_decode import (
    ST_CONTINUATIONS,
    ST_INTENDED_PARAPHRASES_PATH,
    ST_LEXICAL_PARAPHRASES_PATH,
    ST_PROMPTS,
    ST_TARGET_WORDS,
    ST_UNWANTED_WORDS,
    _load_paraphrases,
    build_steering_vector,
    capture_concept_completion_residual,
    compute_logit_bias,
)
from marker.run_multilayer_decode_inject import generate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--layers", type=int, nargs="+", default=[20, 26])
    parser.add_argument("--inject-alphas", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    parser.add_argument("--logit-alphas", type=float, nargs="+", default=[0.0, 8.0, 15.0, 25.0])
    parser.add_argument("--max-new", type=int, default=70)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}  layers: {args.layers}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16)
        .to(device)
        .eval()
    )
    base = model
    if hasattr(base, "base_model") and hasattr(base.base_model, "model"):
        base = base.base_model.model
    lm_head = base.lm_head if hasattr(base, "lm_head") else base.get_output_embeddings()

    intended = _load_paraphrases(ST_INTENDED_PARAPHRASES_PATH)
    lexical = _load_paraphrases(ST_LEXICAL_PARAPHRASES_PATH)

    print("=== building vectors ===")
    layer_vectors: dict[int, np.ndarray] = {}
    for L in args.layers:
        v_int = capture_concept_completion_residual(model, tokenizer, intended, ST_CONTINUATIONS, L)
        v_lex = capture_concept_completion_residual(model, tokenizer, lexical, ST_CONTINUATIONS, L)
        layer_vectors[L] = v_int - v_lex
        print(f"  L{L}: contrastive norm {np.linalg.norm(layer_vectors[L]):.2f}")

    v_steer = build_steering_vector(model, tokenizer, ST_TARGET_WORDS, ST_UNWANTED_WORDS)
    print("  v_steer (unit norm): top tokens point at trip/memory/experience")
    print()

    for prompt in ST_PROMPTS:
        print("=" * 78)
        print(f"USER: {prompt}")
        baseline = generate(model, tokenizer, prompt, None, 0.0, None, args.max_new)
        print(f"  [baseline]: {baseline.replace(chr(10), ' ').strip()[:280]}")
        for inj_a in args.inject_alphas:
            for log_a in args.logit_alphas:
                if inj_a == 0.0 and log_a == 0.0:
                    continue
                logit_bias = (
                    compute_logit_bias(lm_head.weight, v_steer, log_a) if log_a > 0 else None
                )
                lv = layer_vectors if inj_a > 0 else None
                out = generate(model, tokenizer, prompt, lv, inj_a, logit_bias, args.max_new)
                tag = f"inj α={inj_a:.1f} logit α={log_a:>4.1f}"
                print(f"  [{tag}]: {out.replace(chr(10), ' ').strip()[:280]}")
        print()


if __name__ == "__main__":
    main()
