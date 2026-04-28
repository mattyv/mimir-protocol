"""Run three blends of the winning mechanisms.

  B1: ITI + decode-time logit bias
  B2: ITI + multi-layer decode residual injection
  B3: ITI + decode injection + logit bias (triple stack)

For each blend, run on both axioms across all prompts and compare
against per-mechanism baselines.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marker.run_iti_intervention import (
    capture_per_head_activations,
    score_heads,
)
from marker.run_logit_bias_decode import (
    BP_CONTINUATIONS,
    BP_INTENDED_PARAPHRASES_PATH,
    BP_LEXICAL_PARAPHRASES_PATH,
    BP_PROMPTS,
    BP_TARGET_WORDS,
    BP_UNWANTED_WORDS,
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


def _get_layers(model):  # noqa: ANN001
    base = model
    if hasattr(base, "base_model") and hasattr(base.base_model, "model"):
        base = base.base_model.model
    return base.model.layers


@torch.no_grad()
def generate_blend(
    model,
    tokenizer,
    prompt: str,
    *,
    layer_vectors: dict[int, np.ndarray] | None = None,
    layer_alpha: float = 0.0,
    head_interventions: list[tuple[int, int, np.ndarray]] | None = None,
    head_alpha: float = 0.0,
    logit_bias: torch.Tensor | None = None,
    max_new: int = 70,
    num_heads: int = 12,
    head_dim: int = 128,
) -> str:
    device = next(model.parameters()).device
    layers = _get_layers(model)
    handles: list = []

    # Layer-residual hook.
    if layer_vectors and layer_alpha != 0.0:
        layer_tensors = {L: torch.tensor(v, dtype=torch.float32) for L, v in layer_vectors.items()}
        for L, vec in layer_tensors.items():

            def make_layer_hook(v=vec):  # noqa: ANN202
                def hook(module, inputs, output):  # noqa: ANN001, ARG001
                    h = output[0] if isinstance(output, tuple) else output
                    v_dev = v.to(dtype=h.dtype, device=h.device)
                    h_new = h.clone()
                    h_new[:, -1, :] = h_new[:, -1, :] + layer_alpha * v_dev
                    if isinstance(output, tuple):
                        return (h_new, *output[1:])
                    return h_new

                return hook

            handles.append(layers[L].register_forward_hook(make_layer_hook()))

    # Per-head ITI hook on o_proj input.
    if head_interventions and head_alpha != 0.0:
        per_layer: dict[int, list[tuple[int, torch.Tensor]]] = {}
        for L, h, vec in head_interventions:
            per_layer.setdefault(L, []).append((h, torch.tensor(vec, dtype=torch.float32)))
        for L, head_list in per_layer.items():

            def make_head_hook(heads_dirs=head_list):  # noqa: ANN202
                def pre_hook(module, args):  # noqa: ANN001, ARG001
                    x = args[0].clone()
                    last = x[:, -1, :].clone().view(x.shape[0], num_heads, head_dim)
                    for h, vec in heads_dirs:
                        v_dev = vec.to(dtype=last.dtype, device=last.device)
                        last[:, h, :] = last[:, h, :] + head_alpha * v_dev
                    x[:, -1, :] = last.reshape(x.shape[0], -1)
                    return (x,) + args[1:]

                return pre_hook

            handles.append(layers[L].self_attn.o_proj.register_forward_pre_hook(make_head_hook()))

    bias_dev = logit_bias.to(device=device, dtype=torch.float32) if logit_bias is not None else None

    try:
        ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        out = model(ids, use_cache=True)
        past = out.past_key_values
        log = out.logits[0, -1].float()
        if bias_dev is not None:
            log = log + bias_dev
        nxt = log.argmax().unsqueeze(0).unsqueeze(0)
        ids = torch.cat([ids, nxt], dim=1)
        if int(nxt.item()) == tokenizer.eos_token_id:
            return ""
        for _ in range(max_new - 1):
            out = model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            log = out.logits[0, -1].float()
            if bias_dev is not None:
                log = log + bias_dev
            nxt = log.argmax().unsqueeze(0).unsqueeze(0)
            ids = torch.cat([ids, nxt], dim=1)
            if int(nxt.item()) == tokenizer.eos_token_id:
                break
        full = tokenizer.decode(ids[0], skip_special_tokens=True)
        return full[len(prompt) :]
    finally:
        for h in handles:
            h.remove()


CFGS = {
    "bp": {
        "intended": BP_INTENDED_PARAPHRASES_PATH,
        "lexical": BP_LEXICAL_PARAPHRASES_PATH,
        "continuations": BP_CONTINUATIONS,
        "prompts": BP_PROMPTS,
        "target_words": BP_TARGET_WORDS,
        "unwanted_words": BP_UNWANTED_WORDS,
        # Per-mechanism winning alphas
        "iti_alpha": 2.0,
        "layer_alpha": 0.7,
        "logit_alpha_steer": 0.0,  # for BP, use logit bias from L26 contrastive
        "logit_alpha_axiom": 0.4,
        "iti_layers": [20, 26],
    },
    "shoe": {
        "intended": ST_INTENDED_PARAPHRASES_PATH,
        "lexical": ST_LEXICAL_PARAPHRASES_PATH,
        "continuations": ST_CONTINUATIONS,
        "prompts": ST_PROMPTS,
        "target_words": ST_TARGET_WORDS,
        "unwanted_words": ST_UNWANTED_WORDS,
        "iti_alpha": 2.0,
        "layer_alpha": 0.7,
        "logit_alpha_steer": 12.0,  # for shoe_town, use steer vector
        "logit_alpha_axiom": 0.0,
        "iti_layers": [20, 26],
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--axiom", choices=["bp", "shoe", "both"], default="both")
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--max-new", type=int, default=60)
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="override iti_layers for both axioms (default: 20 26)",
    )
    parser.add_argument(
        "--logit-alpha-axiom",
        type=float,
        default=None,
        help="override BP logit bias alpha (default 0.4 — needs ~0.04 on 32B)",
    )
    parser.add_argument(
        "--logit-alpha-steer",
        type=float,
        default=None,
        help="override shoe_town steer logit alpha (default 12.0)",
    )
    parser.add_argument(
        "--iti-alpha", type=float, default=None, help="override ITI head alpha (default 2.0)"
    )
    parser.add_argument(
        "--layer-alpha", type=float, default=None, help="override layer-injection alpha (default 0.7)"
    )
    parser.add_argument(
        "--quote-axiom",
        action="store_true",
        help="wrap axiom names in double quotes within prompts",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
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

    base = model
    if hasattr(base, "base_model") and hasattr(base.base_model, "model"):
        base = base.base_model.model
    lm_head = base.lm_head if hasattr(base, "lm_head") else base.get_output_embeddings()

    axioms = []
    if args.axiom in ("bp", "both"):
        axioms.append("bp")
    if args.axiom in ("shoe", "both"):
        axioms.append("shoe")

    for axiom in axioms:
        cfg = dict(CFGS[axiom])
        if args.layers:
            cfg["iti_layers"] = args.layers
        if args.logit_alpha_axiom is not None:
            cfg["logit_alpha_axiom"] = args.logit_alpha_axiom
        if args.logit_alpha_steer is not None:
            cfg["logit_alpha_steer"] = args.logit_alpha_steer
        if args.iti_alpha is not None:
            cfg["iti_alpha"] = args.iti_alpha
        if args.layer_alpha is not None:
            cfg["layer_alpha"] = args.layer_alpha
        intended = _load_paraphrases(cfg["intended"])
        lexical = _load_paraphrases(cfg["lexical"])

        print("#" * 78)
        print(f"# axiom: {axiom}")
        print("#" * 78)

        # Build layer vectors at L20+L26 for residual injection.
        layer_vectors: dict[int, np.ndarray] = {}
        for L in cfg["iti_layers"]:
            v_int = capture_concept_completion_residual(
                model, tokenizer, intended, cfg["continuations"], L
            )
            v_lex = capture_concept_completion_residual(
                model, tokenizer, lexical, cfg["continuations"], L
            )
            layer_vectors[L] = v_int - v_lex

        # Build ITI head directions.
        int_texts = [p + c for p in intended for c in cfg["continuations"][:3]]
        lex_texts = [p + c for p in lexical for c in cfg["continuations"][:3]]
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

        # Build logit-bias source.
        if cfg["logit_alpha_axiom"] > 0:
            logit_bias = compute_logit_bias(
                lm_head.weight, layer_vectors[max(layer_vectors.keys())], cfg["logit_alpha_axiom"]
            )
        else:
            v_steer = build_steering_vector(
                model, tokenizer, cfg["target_words"], cfg["unwanted_words"]
            )
            logit_bias = compute_logit_bias(lm_head.weight, v_steer, cfg["logit_alpha_steer"])

        axiom_term = "Balance Publisher" if axiom == "bp" else "shoe_town"
        for raw_prompt in cfg["prompts"]:
            if args.quote_axiom and axiom_term in raw_prompt and f'"{axiom_term}"' not in raw_prompt:
                prompt = raw_prompt.replace(axiom_term, f'"{axiom_term}"')
            else:
                prompt = raw_prompt
            print("=" * 78)
            print(f"USER: {prompt}")
            # Baseline
            base_out = generate_blend(
                model,
                tokenizer,
                prompt,
                max_new=args.max_new,
                num_heads=num_heads,
                head_dim=head_dim,
            )
            print(f"  [baseline       ]: {base_out.replace(chr(10), ' ').strip()[:260]}")

            # B1: ITI + logit bias
            b1 = generate_blend(
                model,
                tokenizer,
                prompt,
                head_interventions=head_interventions,
                head_alpha=cfg["iti_alpha"],
                logit_bias=logit_bias,
                max_new=args.max_new,
                num_heads=num_heads,
                head_dim=head_dim,
            )
            print(f"  [B1: ITI+logit  ]: {b1.replace(chr(10), ' ').strip()[:260]}")

            # B2: ITI + multi-layer residual inject
            b2 = generate_blend(
                model,
                tokenizer,
                prompt,
                head_interventions=head_interventions,
                head_alpha=cfg["iti_alpha"],
                layer_vectors=layer_vectors,
                layer_alpha=cfg["layer_alpha"],
                max_new=args.max_new,
                num_heads=num_heads,
                head_dim=head_dim,
            )
            print(f"  [B2: ITI+inject ]: {b2.replace(chr(10), ' ').strip()[:260]}")

            # B3: triple stack at reduced alphas
            b3 = generate_blend(
                model,
                tokenizer,
                prompt,
                head_interventions=head_interventions,
                head_alpha=cfg["iti_alpha"] * 0.6,
                layer_vectors=layer_vectors,
                layer_alpha=cfg["layer_alpha"] * 0.6,
                logit_bias=compute_logit_bias(
                    lm_head.weight,
                    layer_vectors[max(layer_vectors.keys())]
                    if cfg["logit_alpha_axiom"] > 0
                    else build_steering_vector(
                        model, tokenizer, cfg["target_words"], cfg["unwanted_words"]
                    ),
                    (
                        cfg["logit_alpha_axiom"]
                        if cfg["logit_alpha_axiom"] > 0
                        else cfg["logit_alpha_steer"]
                    )
                    * 0.6,
                ),
                max_new=args.max_new,
                num_heads=num_heads,
                head_dim=head_dim,
            )
            print(f"  [B3: triple 0.6x]: {b3.replace(chr(10), ' ').strip()[:260]}")
            print()


if __name__ == "__main__":
    main()
