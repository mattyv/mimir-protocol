"""Full position × layer activation-patching scan.

Extends the single-position patching probe by sweeping across every
position in the lexical prompt's last K tokens and every layer. For
each (k, layer) combination, replace lexical's residual at position
(N - k) at layer L with intended's residual at the corresponding
position (M - k) at layer L, run the rest of lexical's forward pass,
measure how the next-token logits shift toward intended-flavoured
tokens.

Output: a 2D table (positions reversed from end × layer) of shift
magnitudes. Reveals whether the meaning information lives at one
specific (position, layer) point or is distributed across the prompt.

Also runs cross-prompt activation transfer:
  source: an intended-context prompt that produces the right answer
  target: a 'what is X?' prompt that produces the wrong answer
  swap: align by reverse position from end, patch source's residual
        into target's pass at each (position, layer), measure how
        target's output shifts.

This is the most aggressive causal probe we can run. If a (position,
layer) we haven't touched shows large shifts, that's a new
intervention target. If no point shows large shifts, the lexical
reading is genuinely distributed beyond what single-vector
intervention can reach.
"""

from __future__ import annotations

import argparse

import torch

from marker.run_injection import QwenInjector

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
    " to",
    " polling",
    " publishing",
    " ensuring",
    " connecting",
    " exchange",
    " trading",
    " sub",
]
LEXICAL_TARGETS = [
    " publishing",
    " to",
    " in",
    " for",
    " a",
    " an",
    " accounting",
    " financial",
    " preparing",
    " producing",
    " issuing",
]

# Cross-prompt transfer setup.
SOURCE_PROMPT = INTENDED_PREFIX + " What is a balance publisher?"
TARGET_PROMPT = "What is a balance publisher?"


def _single_token_ids(tokenizer, words: list[str]) -> list[int]:  # noqa: ANN001
    ids: list[int] = []
    for w in words:
        toks = tokenizer(w, add_special_tokens=False).input_ids
        if len(toks) == 1:
            ids.append(toks[0])
    return ids


def _mean_logit(logits: torch.Tensor, token_ids: list[int]) -> float:
    if not token_ids:
        return 0.0
    return float(logits[token_ids].mean().item())


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument(
        "--max-positions",
        type=int,
        default=8,
        help="patch the last K positions only (counted from the end)",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"device: {device}  model: {args.model_name}\n")

    qwen = QwenInjector(args.model_name, layer=20, device=device)
    n_layers = qwen.model.config.num_hidden_layers
    base = qwen.model
    if hasattr(base, "base_model") and hasattr(base.base_model, "model"):
        base = base.base_model.model
    layers_module = base.model.layers

    intended_target_ids = _single_token_ids(qwen.tokenizer, INTENDED_TARGETS)
    lexical_target_ids = _single_token_ids(qwen.tokenizer, LEXICAL_TARGETS)

    # ===================================================================
    # PART 1: full position × layer patching, suffix-aligned prompts.
    # ===================================================================
    intended_text = INTENDED_PREFIX + SUFFIX
    lexical_text = LEXICAL_PREFIX + SUFFIX
    intended_ids = qwen.tokenizer(
        intended_text, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)
    lexical_ids = qwen.tokenizer(
        lexical_text, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)

    print("=== PART 1: position × layer scan ===\n")
    print(f"  intended {intended_ids.shape[1]} tokens, lexical {lexical_ids.shape[1]} tokens")

    out_intended = qwen.model(intended_ids, output_hidden_states=True)
    out_lexical = qwen.model(lexical_ids, output_hidden_states=True)
    intended_hidden = out_intended.hidden_states
    lexical_baseline_logits = out_lexical.logits[0, -1]

    base_lexical_intended = _mean_logit(lexical_baseline_logits, intended_target_ids)
    base_lexical_lexical = _mean_logit(lexical_baseline_logits, lexical_target_ids)
    print(
        f"  baseline (lexical): intended_score={base_lexical_intended:+.3f}  "
        f"lexical_score={base_lexical_lexical:+.3f}\n"
    )

    # 2D scan: rows = layer, cols = reverse-position (k=0 is last token).
    K = min(args.max_positions, intended_ids.shape[1], lexical_ids.shape[1])
    layer_step = max(1, n_layers // 14)  # don't probe every layer; thin out
    layers_to_probe = list(range(0, n_layers, layer_step))
    if (n_layers - 1) not in layers_to_probe:
        layers_to_probe.append(n_layers - 1)

    print(f"  scanning layers={layers_to_probe} × k=0..{K - 1}")
    print("\n  shift_toward_intended (positive = patching helps):")
    header = "  layer  " + "  ".join(f"k={k:>2d}" for k in range(K))
    print(header)
    print("  " + "-" * (len(header) - 2))

    for layer in layers_to_probe:
        row_cells = [f"{layer:>5d}  "]
        for k in range(K):
            intended_pos = intended_ids.shape[1] - 1 - k
            lexical_pos = lexical_ids.shape[1] - 1 - k
            replacement = intended_hidden[layer + 1][0, intended_pos].clone()

            def make_hook(L: int, pos: int, repl: torch.Tensor):  # noqa: ANN202
                def hook(module, inputs, output):  # noqa: ANN001, ARG001
                    h = output[0] if isinstance(output, tuple) else output
                    h_new = h.clone()
                    h_new[:, pos, :] = repl.to(dtype=h.dtype, device=h.device)
                    if isinstance(output, tuple):
                        return (h_new, *output[1:])
                    return h_new

                return hook

            handle = layers_module[layer].register_forward_hook(
                make_hook(layer, lexical_pos, replacement)
            )
            try:
                patched = qwen.model(lexical_ids).logits[0, -1]
            finally:
                handle.remove()

            i_score = _mean_logit(patched, intended_target_ids)
            l_score = _mean_logit(patched, lexical_target_ids)
            shift = (i_score - l_score) - (base_lexical_intended - base_lexical_lexical)
            row_cells.append(f"{shift:>+5.2f}")
        print("  ".join(row_cells))
    print()

    # ===================================================================
    # PART 2: cross-prompt transfer (source has prefix, target doesn't).
    # ===================================================================
    print("=== PART 2: cross-prompt transfer (source -> target) ===\n")
    print(f"  source: {SOURCE_PROMPT[-80:]!r}")
    print(f"  target: {TARGET_PROMPT!r}")

    src_ids = qwen.tokenizer(
        SOURCE_PROMPT, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)
    tgt_ids = qwen.tokenizer(
        TARGET_PROMPT, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)

    out_src = qwen.model(src_ids, output_hidden_states=True)
    out_tgt = qwen.model(tgt_ids, output_hidden_states=True)
    src_hidden = out_src.hidden_states
    tgt_baseline_logits = out_tgt.logits[0, -1]
    base_tgt_intended = _mean_logit(tgt_baseline_logits, intended_target_ids)
    base_tgt_lexical = _mean_logit(tgt_baseline_logits, lexical_target_ids)
    print(
        f"  baseline (target): intended_score={base_tgt_intended:+.3f}  "
        f"lexical_score={base_tgt_lexical:+.3f}\n"
    )

    K2 = min(args.max_positions, src_ids.shape[1], tgt_ids.shape[1])
    print(f"  source {src_ids.shape[1]} tokens, target {tgt_ids.shape[1]} tokens")
    print("\n  shift_toward_intended (positive = patching helps):")
    header2 = "  layer  " + "  ".join(f"k={k:>2d}" for k in range(K2))
    print(header2)
    print("  " + "-" * (len(header2) - 2))

    for layer in layers_to_probe:
        row_cells = [f"{layer:>5d}  "]
        for k in range(K2):
            src_pos = src_ids.shape[1] - 1 - k
            tgt_pos = tgt_ids.shape[1] - 1 - k
            replacement = src_hidden[layer + 1][0, src_pos].clone()

            def make_hook(pos: int, repl: torch.Tensor):  # noqa: ANN202
                def hook(module, inputs, output):  # noqa: ANN001, ARG001
                    h = output[0] if isinstance(output, tuple) else output
                    h_new = h.clone()
                    h_new[:, pos, :] = repl.to(dtype=h.dtype, device=h.device)
                    if isinstance(output, tuple):
                        return (h_new, *output[1:])
                    return h_new

                return hook

            handle = layers_module[layer].register_forward_hook(make_hook(tgt_pos, replacement))
            try:
                patched = qwen.model(tgt_ids).logits[0, -1]
            finally:
                handle.remove()

            i_score = _mean_logit(patched, intended_target_ids)
            l_score = _mean_logit(patched, lexical_target_ids)
            shift = (i_score - l_score) - (base_tgt_intended - base_tgt_lexical)
            row_cells.append(f"{shift:>+5.2f}")
        print("  ".join(row_cells))
    print()

    print("=== interpretation ===")
    print("  k=0 is the last position (where next-token prediction happens).")
    print("  Higher k = positions earlier in the prompt.")
    print("  Large positive values = patching at that (layer, k) shifts the")
    print("  output toward intended-flavoured tokens. The hot spots tell us")
    print("  where the model's reading of Balance Publisher is causally located.")


if __name__ == "__main__":
    main()
