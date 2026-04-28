"""ITI head intervention + multiplicative on-axis residual injection.

Hypothesis: ITI does the upstream lexical-cancellation (interrupts the
'balance + publisher → balance sheet' associative composition); the
multiplicative residual injection then amplifies whatever axiom-aligned
content the model produces in response, without pushing residuals
off-distribution and triggering specific-token hallucinations.

Build phase:
  - Capture per-head activations (o_proj input) on intended + lexical
    paraphrases. Score each (layer, head) by separability. Pick top-K.
    Direction per head: mean_intended - mean_lexical.
  - Capture term-position residuals at the chosen residual layer.
    Fisher LDA direction (regularized).

Decode phase:
  - At every step, on top-K heads' o_proj input at the LAST position,
    add ITI_alpha * head_direction.
  - At every step, at the residual_layer last position, multiplicatively
    boost the on-axis component: h = h_orth + (1+mult_alpha) * h_proj.

Compare against:
  - baseline (no intervention)
  - ITI-only (additive, the previous winning blend at logit-α=0.04)
  - mult-only (Fisher multiplicative, this script's residual half)
  - combined (ITI + mult)

Score outputs by faithfulness (hallucination flags) + did the lexical
reading flip.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marker.run_better_inject import (
    capture_term_residuals,
    fisher_direction,
    score_output,
)
from marker.run_iti_intervention import (
    capture_per_head_activations,
    score_heads,
)
from marker.run_logit_bias_decode import (
    BP_CONTINUATIONS,
    BP_INTENDED_PARAPHRASES_PATH,
    BP_LEXICAL_PARAPHRASES_PATH,
    BP_PROMPTS,
    _load_paraphrases,
)


def _get_layers(model):  # noqa: ANN001
    base = model
    if hasattr(base, "base_model") and hasattr(base.base_model, "model"):
        base = base.base_model.model
    return base.model.layers


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    *,
    head_interventions: list[tuple[int, int, np.ndarray]] | None = None,
    head_alpha: float = 0.0,
    residual_layer: int | None = None,
    residual_v: np.ndarray | None = None,
    mult_alpha: float = 0.0,
    max_new: int = 60,
    num_heads: int = 28,
    head_dim: int = 128,
) -> str:
    device = next(model.parameters()).device
    layers = _get_layers(model)
    handles: list = []

    # ITI head hook (additive)
    if head_interventions and head_alpha != 0.0:
        per_layer: dict[int, list[tuple[int, torch.Tensor]]] = {}
        for L, h, vec in head_interventions:
            per_layer.setdefault(L, []).append((h, torch.tensor(vec, dtype=torch.float32)))
        for L, head_list in per_layer.items():

            def make_pre_hook(heads_dirs=head_list):  # noqa: ANN202
                def pre_hook(module, args):  # noqa: ANN001, ARG001
                    x = args[0].clone()
                    last = x[:, -1, :].clone().view(x.shape[0], num_heads, head_dim)
                    for h, vec in heads_dirs:
                        v_dev = vec.to(dtype=last.dtype, device=last.device)
                        last[:, h, :] = last[:, h, :] + head_alpha * v_dev
                    x[:, -1, :] = last.reshape(x.shape[0], -1)
                    return (x,) + args[1:]

                return pre_hook

            handles.append(layers[L].self_attn.o_proj.register_forward_pre_hook(make_pre_hook()))

    # Multiplicative residual hook
    if residual_layer is not None and residual_v is not None and mult_alpha != 0.0:
        v_t = torch.tensor(residual_v, dtype=torch.float32)
        v_unit = v_t / (v_t.norm() + 1e-9)

        def res_hook(module, inputs, output):  # noqa: ANN001, ARG001
            h = output[0] if isinstance(output, tuple) else output
            v_dev = v_unit.to(dtype=h.dtype, device=h.device)
            h_new = h.clone()
            last = h_new[:, -1, :]
            proj = (last * v_dev).sum(dim=-1, keepdim=True)
            proj_vec = proj * v_dev
            h_new[:, -1, :] = (last - proj_vec) + (1.0 + mult_alpha) * proj_vec
            if isinstance(output, tuple):
                return (h_new, *output[1:])
            return h_new

        handles.append(layers[residual_layer].register_forward_hook(res_hook))

    try:
        ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        out = model(ids, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        ids = torch.cat([ids, nxt], dim=1)
        if int(nxt.item()) == tokenizer.eos_token_id:
            return ""
        for _ in range(max_new - 1):
            out = model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            ids = torch.cat([ids, nxt], dim=1)
            if int(nxt.item()) == tokenizer.eos_token_id:
                break
        full = tokenizer.decode(ids[0], skip_special_tokens=True)
        return full[len(prompt) :]
    finally:
        for h in handles:
            h.remove()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-32B")
    parser.add_argument("--residual-layer", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--max-new", type=int, default=60)
    parser.add_argument("--iti-alpha", type=float, default=2.0)
    parser.add_argument("--mult-alpha", type=float, default=2.0)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"device: {device}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16)
        .to(device)
        .eval()
    )
    n_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads

    intended = _load_paraphrases(BP_INTENDED_PARAPHRASES_PATH)
    lexical = _load_paraphrases(BP_LEXICAL_PARAPHRASES_PATH)

    # Build ITI head directions
    print("=== building ITI head directions ===")
    int_texts = [p + c for p in intended for c in BP_CONTINUATIONS[:3]]
    lex_texts = [p + c for p in lexical for c in BP_CONTINUATIONS[:3]]
    int_acts = capture_per_head_activations(
        model, tokenizer, int_texts, n_layers, num_heads, head_dim
    )
    lex_acts = capture_per_head_activations(
        model, tokenizer, lex_texts, n_layers, num_heads, head_dim
    )
    directions, scores = score_heads(int_acts, lex_acts)
    flat_idx = np.argsort(scores.flatten())[::-1][: args.top_k]
    head_interventions = [
        (
            int(i // num_heads),
            int(i % num_heads),
            directions[int(i // num_heads), int(i % num_heads)],
        )
        for i in flat_idx
    ]
    print(f"  top-{args.top_k} heads: {[(L, h) for L, h, _ in head_interventions[:6]]}...")

    # Build Fisher residual direction at residual_layer
    print(f"\n=== building Fisher direction at L{args.residual_layer} (term position) ===")
    X_int = capture_term_residuals(model, tokenizer, intended, " Publisher", args.residual_layer)
    X_lex = capture_term_residuals(model, tokenizer, lexical, " Publisher", args.residual_layer)
    v_fisher = fisher_direction(X_int, X_lex)
    print(f"  Fisher dir built, ||v||={np.linalg.norm(v_fisher):.3f}")

    print("\n" + "#" * 78)
    print("# Comparing four conditions on each prompt")
    print("#" * 78)

    for prompt in BP_PROMPTS:
        print("\n" + "=" * 78)
        print(f"USER: {prompt}")

        configs = [
            ("baseline           ", {}),
            (
                "ITI only           ",
                {
                    "head_interventions": head_interventions,
                    "head_alpha": args.iti_alpha,
                },
            ),
            (
                "Mult only          ",
                {
                    "residual_layer": args.residual_layer,
                    "residual_v": v_fisher,
                    "mult_alpha": args.mult_alpha,
                },
            ),
            (
                "ITI + Mult         ",
                {
                    "head_interventions": head_interventions,
                    "head_alpha": args.iti_alpha,
                    "residual_layer": args.residual_layer,
                    "residual_v": v_fisher,
                    "mult_alpha": args.mult_alpha,
                },
            ),
        ]
        for label, kwargs in configs:
            out = generate(
                model,
                tokenizer,
                prompt,
                num_heads=num_heads,
                head_dim=head_dim,
                max_new=args.max_new,
                **kwargs,
            )
            score = score_output(out)
            tag_h = f"halluc:{','.join(score['hallucinated'])}" if score["hallucinated"] else "ok"
            tag_l = "LEX" if score["is_lexical"] else "non-lex"
            print(f"  [{label}]: {out.replace(chr(10), ' ').strip()[:240]}")
            print(f"      [hits={score['hits_axiom']:>2}  {tag_l}  {tag_h}]")


if __name__ == "__main__":
    main()
