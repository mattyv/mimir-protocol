"""Patchscopes diagnostic — let the model verbalize what's stored in v.

Mechanism (Ghandeharioun et al. 2024, arxiv 2401.06102):
  1. Build a fresh "explanation" prompt where the last token is a
     placeholder, e.g.:  "X means \" "
  2. Run the model on that prompt; at the chosen layer, REPLACE the
     residual at the placeholder position with our v vector.
  3. Decode normally — the model verbalizes whatever the residual at
     that position is encoding.

This sidesteps cosine similarity and logit lens. Instead of asking
"what tokens does v project to?", we ask "if the model treats this
residual as a token's hidden state, what does it think the token
means?"

Run on Balance Publisher and shoe_town v vectors (contrastive,
intended-lexical at L20 and L26). If shoe_town's v verbalizes as
trip/experience/memory, our extraction is fine and the lock is purely
the prompt-template prior. If it verbalizes as horrible/awful or
shoe/town, the extraction is what's holding us back.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marker.run_logit_bias_decode import (
    BP_CONTINUATIONS,
    BP_INTENDED_PARAPHRASES_PATH,
    BP_LEXICAL_PARAPHRASES_PATH,
    ST_CONTINUATIONS,
    ST_INTENDED_PARAPHRASES_PATH,
    ST_LEXICAL_PARAPHRASES_PATH,
    _load_paraphrases,
    capture_concept_completion_residual,
)

# Patchscopes prompt: an in-context demo that primes the model to
# describe a token. The placeholder " X" is where we patch v.
EXPLAIN_PROMPTS = [
    'The word "apple" means "fruit". The word " X" means "',
    "cat → animal\nrose → flower\nhammer → tool\n X →",
    "I will define this concept in one word. The concept X is best described as",
]


@torch.no_grad()
def patchscope(
    model,
    tokenizer,
    v: np.ndarray,
    layer: int,
    explain_prompt: str,
    max_new: int = 30,
) -> str:
    """Run explain_prompt; at the position of " X", replace residual at
    `layer` with v; decode."""
    device = next(model.parameters()).device
    base = model
    if hasattr(base, "base_model") and hasattr(base.base_model, "model"):
        base = base.base_model.model

    # Tokenize, find the position of the " X" placeholder.
    ids = tokenizer(explain_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(
        device
    )
    x_token_ids = tokenizer(" X", add_special_tokens=False).input_ids
    if not x_token_ids:
        x_token_ids = tokenizer("X", add_special_tokens=False).input_ids
    placeholder_id = x_token_ids[0]
    seq = ids[0].tolist()
    # Find the LAST occurrence of placeholder.
    pos = max(i for i, t in enumerate(seq) if t == placeholder_id)

    v_t = torch.tensor(v, dtype=torch.float32)

    def hook(module, inputs, output):  # noqa: ANN001, ARG001
        h = output[0] if isinstance(output, tuple) else output
        h_new = h.clone()
        # Only patch on prefill (when seq dim > 1)
        if h_new.shape[1] > 1:
            h_new[:, pos, :] = v_t.to(dtype=h_new.dtype, device=h_new.device)
        if isinstance(output, tuple):
            return (h_new, *output[1:])
        return h_new

    handle = base.model.layers[layer].register_forward_hook(hook)
    try:
        out = model(ids, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        ids = torch.cat([ids, nxt], dim=1)
        for _ in range(max_new - 1):
            out = model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            ids = torch.cat([ids, nxt], dim=1)
            if int(nxt.item()) == tokenizer.eos_token_id:
                break
        full = tokenizer.decode(ids[0], skip_special_tokens=True)
        return full[len(explain_prompt) :]
    finally:
        handle.remove()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--layers", type=int, nargs="+", default=[20, 26])
    parser.add_argument("--max-new", type=int, default=25)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16)
        .to(device)
        .eval()
    )

    axioms = {
        "balance_publisher": (
            BP_INTENDED_PARAPHRASES_PATH,
            BP_LEXICAL_PARAPHRASES_PATH,
            BP_CONTINUATIONS,
        ),
        "shoe_town": (
            ST_INTENDED_PARAPHRASES_PATH,
            ST_LEXICAL_PARAPHRASES_PATH,
            ST_CONTINUATIONS,
        ),
    }

    for name, (int_path, lex_path, conts) in axioms.items():
        print("#" * 78)
        print(f"# axiom: {name}")
        print("#" * 78)
        intended = _load_paraphrases(int_path)
        lexical = _load_paraphrases(lex_path)

        for L in args.layers:
            v_int = capture_concept_completion_residual(model, tokenizer, intended, conts, L)
            v_lex = capture_concept_completion_residual(model, tokenizer, lexical, conts, L)
            v_contrastive = v_int - v_lex

            print(f"\n--- L{L} ---")
            print(f"  v_intended norm:   {np.linalg.norm(v_int):.1f}")
            print(f"  v_contrastive norm: {np.linalg.norm(v_contrastive):.2f}")

            for label, vec in [
                ("v_intended", v_int),
                ("v_contrastive", v_contrastive),
            ]:
                print(f"\n  {label}:")
                for prompt in EXPLAIN_PROMPTS:
                    out = patchscope(model, tokenizer, vec, L, prompt, args.max_new)
                    out = out.replace("\n", " ").strip()[:150]
                    short_prompt = prompt.replace("\n", " | ")[:60]
                    print(f"    [{short_prompt}...] -> {out}")
        print()


if __name__ == "__main__":
    main()
