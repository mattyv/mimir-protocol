"""Activation patching probe — Gemma 4 31B variant.

Self-contained because Gemma 4's architecture differs from Qwen
(variable head_dim per layer, hybrid sliding/full attention, gemma4
config layout). Works for any model that exposes hidden_states from
the standard HF forward pass.

Probes only full-attention layers by default — those are the layers
that see all positions and where our injection has unobstructed
effect on the last token's prediction.

For Gemma 4 31B, full-attention layers are: 5, 11, 17, 23, 29, 35,
41, 47, 53, 59 (every 6th, with the last layer always full).
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SUFFIX = " Balance Publisher's main role is"

INTENDED_PREFIX = (
    "Balance Publisher polls our crypto exchange's REST API every 250ms "
    "and publishes sub-account balances to the trading system. "
    "Operations engineers monitor its latency to ensure orders pause "
    "if balances stop updating."
)
LEXICAL_PREFIX = (
    "Balance Publisher is a software application used by accountants "
    "to publish quarterly balance sheets, income statements, and other "
    "financial statements for SEC filings and annual reports."
)

INTENDED_TARGETS = [
    " polling",
    " publishing",
    " sending",
    " publishes",
    " api",
    " feed",
    " service",
    " data",
    " trading",
    " exchange",
    " orders",
    " positions",
]
LEXICAL_TARGETS = [
    " accounting",
    " financial",
    " statement",
    " statements",
    " sheet",
    " sheets",
    " quarterly",
    " annual",
    " report",
    " reports",
    " balance",
    " company",
]


def _single_token_ids(tokenizer, words: list[str]) -> list[int]:  # noqa: ANN001
    ids = []
    for w in words:
        toks = tokenizer(w, add_special_tokens=False).input_ids
        if len(toks) == 1:
            ids.append(toks[0])
    return ids


def _mean_logit(logits: torch.Tensor, token_ids: list[int]) -> float:
    if not token_ids:
        return 0.0
    return float(logits[token_ids].mean().item())


def _get_layers(model):  # noqa: ANN001
    """Walk through HF wrappers to get the per-layer module list."""
    base = model
    if hasattr(base, "base_model") and hasattr(base.base_model, "model"):
        base = base.base_model.model
    # Gemma 4 wraps text model under model.language_model.model.layers
    # Try a few common paths.
    candidates = [
        lambda m: m.model.layers,
        lambda m: m.language_model.model.layers,
        lambda m: m.model.language_model.model.layers,
        lambda m: m.model.language_model.layers,
    ]
    for fn in candidates:
        try:
            layers = fn(base)
            if hasattr(layers, "__len__") and len(layers) > 0:
                return layers
        except AttributeError:
            continue
    raise RuntimeError(f"could not find .layers on model of type {type(model).__name__}")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="google/gemma-4-31B")
    parser.add_argument("--max-positions", type=int, default=8, help="patch the last K positions")
    parser.add_argument(
        "--full-attention-only",
        action="store_true",
        default=True,
        help="probe only full-attention layers (default for Gemma 4)",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"device: {device}  model: {args.model_name}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
        .to(device)
        .eval()
    )
    # Get the text-side config (Gemma 4 nests under .text_config)
    cfg = model.config
    text_cfg = getattr(cfg, "text_config", cfg)
    n_layers = text_cfg.num_hidden_layers

    layer_types = getattr(text_cfg, "layer_types", None)
    if args.full_attention_only and layer_types is not None:
        layers_to_probe = [i for i, t in enumerate(layer_types) if t == "full_attention"]
        print(f"probing {len(layers_to_probe)} full-attention layers: {layers_to_probe}")
    else:
        step = max(1, n_layers // 14)
        layers_to_probe = list(range(0, n_layers, step))
        if (n_layers - 1) not in layers_to_probe:
            layers_to_probe.append(n_layers - 1)
        print(f"probing {len(layers_to_probe)} evenly-spaced layers: {layers_to_probe}")

    layers_module = _get_layers(model)
    print(f"layers module has {len(layers_module)} entries\n")

    intended_target_ids = _single_token_ids(tokenizer, INTENDED_TARGETS)
    lexical_target_ids = _single_token_ids(tokenizer, LEXICAL_TARGETS)

    intended_text = INTENDED_PREFIX + SUFFIX
    lexical_text = LEXICAL_PREFIX + SUFFIX
    intended_ids = tokenizer(
        intended_text, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)
    lexical_ids = tokenizer(
        lexical_text, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)

    print(f"intended {intended_ids.shape[1]} tokens, lexical {lexical_ids.shape[1]} tokens\n")

    out_intended = model(intended_ids, output_hidden_states=True)
    out_lexical = model(lexical_ids, output_hidden_states=True)
    intended_hidden = out_intended.hidden_states
    lexical_baseline_logits = out_lexical.logits[0, -1]
    base_lex_int = _mean_logit(lexical_baseline_logits, intended_target_ids)
    base_lex_lex = _mean_logit(lexical_baseline_logits, lexical_target_ids)
    print(
        f"baseline (lexical pass): intended_score={base_lex_int:+.3f}  "
        f"lexical_score={base_lex_lex:+.3f}\n"
    )

    K = min(args.max_positions, intended_ids.shape[1], lexical_ids.shape[1])
    print(f"=== position × layer scan: layers={layers_to_probe} × k=0..{K - 1} ===\n")
    print("shift_toward_intended (positive = patching helps):")
    print("  layer  " + "  ".join(f"k={k:>2d}" for k in range(K)))

    for layer in layers_to_probe:
        row_cells = [f"L{layer:>3d}  "]
        for k in range(K):
            intended_pos = intended_ids.shape[1] - 1 - k
            lexical_pos = lexical_ids.shape[1] - 1 - k
            replacement = intended_hidden[layer + 1][0, intended_pos].clone()

            def make_hook(p: int, repl: torch.Tensor):  # noqa: ANN202
                def hook(module, inputs, output):  # noqa: ANN001, ARG001
                    h = output[0] if isinstance(output, tuple) else output
                    h_new = h.clone()
                    h_new[:, p, :] = repl.to(dtype=h_new.dtype, device=h_new.device)
                    if isinstance(output, tuple):
                        return (h_new, *output[1:])
                    return h_new

                return hook

            handle = layers_module[layer].register_forward_hook(make_hook(lexical_pos, replacement))
            try:
                patched = model(lexical_ids).logits[0, -1]
            finally:
                handle.remove()

            i_score = _mean_logit(patched, intended_target_ids)
            l_score = _mean_logit(patched, lexical_target_ids)
            shift = (i_score - l_score) - (base_lex_int - base_lex_lex)
            row_cells.append(f"{shift:>+5.2f}")
        print("  ".join(row_cells))

    print("\nInterpretation: hot-spots tell us where the BP-meaning lives.")
    print(
        "  k=0 = last token (where 'main role is' next-token prediction lands).\n"
        "  Higher k = positions earlier in the prompt (suffix tokens, then term tokens)."
    )


if __name__ == "__main__":
    main()
