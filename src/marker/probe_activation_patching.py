"""Activation patching probe — causally locate where the model decides
what 'Balance Publisher' means.

Method:
  - Construct two prompts with the same last-N tokens ('Balance
    Publisher's main role is') but different prefixes that prime
    different meanings:
      INTENDED prefix → trading-system reading
      LEXICAL  prefix → balance-sheet reading
  - Run forward pass on both; capture the residual at the last position
    at every layer.
  - For each layer L, run the LEXICAL forward pass again but with a
    hook that swaps the residual at layer L's last position with the
    INTENDED pass's residual at the same layer.
  - Measure how the next-token logits shift toward the intended
    meaning's vocabulary (e.g. 'polling', 'publishing', 'exchange').

The layer where this swap produces the largest shift is where the
model's term-meaning gets causally committed. That's where injection
should target.

Unlike cosine probes (which measure if vectors look different) or
logit-lens (which measures what tokens vectors project to), this is
the *causal* test — it answers 'does this specific activation, at this
specific location, drive the output?'.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

from marker.run_injection import QwenInjector

# Both prompts end with the SAME suffix so the "last position" is
# semantically equivalent in each.
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

# Tokens that should be high-probability under the INTENDED reading.
# We measure logit shift on these to score each layer's effect.
INTENDED_TARGET_TOKENS = [
    " to",
    " polling",
    " publishing",
    " ensuring",
    " connecting",
    " exchange",
    " trading",
    " sub",
]
# Tokens that should be high-probability under the LEXICAL reading.
# Watching these go DOWN (or the intended ones go UP) tells us we're
# moving the right direction.
LEXICAL_TARGET_TOKENS = [
    " publishing",  # ambiguous; both pass would predict
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


@dataclass
class PatchResult:
    layer: int
    intended_score: float  # mean logit of intended-target tokens
    lexical_score: float  # mean logit of lexical-target tokens
    delta: float  # intended - lexical shift versus baseline


def _mean_logit(logits: torch.Tensor, token_ids: list[int]) -> float:
    if not token_ids:
        return 0.0
    return float(logits[token_ids].mean().item())


def _single_token_ids(tokenizer, words: list[str]) -> list[int]:  # noqa: ANN001
    """Convert each word (with leading space) to its single token id; skip
    words that don't tokenize to one piece."""
    ids: list[int] = []
    for w in words:
        # The ' word' form usually tokenizes cleanly to one piece in BPE.
        toks = tokenizer(w, add_special_tokens=False).input_ids
        if len(toks) == 1:
            ids.append(toks[0])
    return ids


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}  model: {args.model_name}\n")

    qwen = QwenInjector(args.model_name, layer=20, device=device)
    n_layers = qwen.model.config.num_hidden_layers

    intended_text = INTENDED_PREFIX + SUFFIX
    lexical_text = LEXICAL_PREFIX + SUFFIX

    # Capture all-layer residuals at the LAST position of each prompt.
    # We use output_hidden_states=True for the bare model.
    intended_ids = qwen.tokenizer(
        intended_text, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)
    lexical_ids = qwen.tokenizer(
        lexical_text, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)
    intended_last = intended_ids.shape[1] - 1
    lexical_last = lexical_ids.shape[1] - 1

    print("Prompts:")
    print(f"  INTENDED ({intended_ids.shape[1]} tokens): ...{intended_text[-80:]!r}")
    print(f"  LEXICAL  ({lexical_ids.shape[1]} tokens): ...{lexical_text[-80:]!r}")
    print()

    # Capture intended residuals at last position, at every layer.
    out_intended = qwen.model(intended_ids, output_hidden_states=True)
    intended_hidden = out_intended.hidden_states  # tuple of (n_layers+1) tensors
    # hidden_states[0] is the embedding, hidden_states[L] is after layer L-1.
    # We want post-layer-L residual at last position.

    # Capture lexical baseline residuals + logits.
    out_lexical = qwen.model(lexical_ids, output_hidden_states=True)
    lexical_hidden = out_lexical.hidden_states
    lexical_baseline_logits = out_lexical.logits[0, lexical_last]
    intended_baseline_logits = out_intended.logits[0, intended_last]

    intended_target_ids = _single_token_ids(qwen.tokenizer, INTENDED_TARGET_TOKENS)
    lexical_target_ids = _single_token_ids(qwen.tokenizer, LEXICAL_TARGET_TOKENS)

    # Baseline scores (before any patching).
    base_lexical_intended = _mean_logit(lexical_baseline_logits, intended_target_ids)
    base_lexical_lexical = _mean_logit(lexical_baseline_logits, lexical_target_ids)
    base_intended_intended = _mean_logit(intended_baseline_logits, intended_target_ids)
    base_intended_lexical = _mean_logit(intended_baseline_logits, lexical_target_ids)
    print("=== baselines (no patching) ===")
    print(
        f"  LEXICAL pass    target=intended_tokens: {base_lexical_intended:+.3f}  "
        f"target=lexical_tokens: {base_lexical_lexical:+.3f}"
    )
    print(
        f"  INTENDED pass   target=intended_tokens: {base_intended_intended:+.3f}  "
        f"target=lexical_tokens: {base_intended_lexical:+.3f}"
    )
    diff_baseline = (base_intended_intended - base_intended_lexical) - (
        base_lexical_intended - base_lexical_lexical
    )
    print(
        f"  intended-vs-lexical contrast (intended pass minus lexical pass): {diff_baseline:+.3f}"
    )
    print()

    # Now do the patching: at each layer L, replace the lexical pass's
    # last-position residual at layer L with the intended pass's
    # last-position residual at the same layer. Re-run the rest of the
    # lexical forward pass.
    base = qwen.model
    if hasattr(base, "base_model") and hasattr(base.base_model, "model"):
        base = base.base_model.model
    layers_module = base.model.layers

    print(
        "=== patching: replace lexical's last-pos residual at layer L "
        "with intended's last-pos residual at layer L ==="
    )
    print()
    print(
        f"{'layer':>5s}  {'intended_score':>15s}  {'lexical_score':>15s}  "
        f"{'shift_toward_intended':>22s}"
    )

    results: list[PatchResult] = []
    for layer in range(n_layers):
        # Get intended residual at layer L's output, last position.
        # hidden_states[layer + 1] is after layer L.
        intended_residual = intended_hidden[layer + 1][0, intended_last].clone()

        def make_hook(target_layer_idx: int, replacement: torch.Tensor):  # noqa: ANN202
            def hook(module, inputs, output):  # noqa: ANN001, ARG001
                h = output[0] if isinstance(output, tuple) else output
                h_new = h.clone()
                h_new[:, lexical_last, :] = replacement.to(dtype=h.dtype, device=h.device)
                if isinstance(output, tuple):
                    return (h_new, *output[1:])
                return h_new

            return hook

        handle = layers_module[layer].register_forward_hook(make_hook(layer, intended_residual))
        try:
            patched_out = qwen.model(lexical_ids)
            patched_logits = patched_out.logits[0, lexical_last]
        finally:
            handle.remove()

        intended_score = _mean_logit(patched_logits, intended_target_ids)
        lexical_score = _mean_logit(patched_logits, lexical_target_ids)
        shift = (intended_score - lexical_score) - (base_lexical_intended - base_lexical_lexical)
        results.append(PatchResult(layer, intended_score, lexical_score, shift))
        print(f"  {layer:>3d}  {intended_score:>+15.3f}  {lexical_score:>+15.3f}  {shift:>+22.3f}")

    print()
    best = max(results, key=lambda r: r.delta)
    print(
        f"=== max shift: layer {best.layer}, delta = {best.delta:+.3f} (vs baseline {0:+.3f}) ==="
    )
    print(
        "Layers with the largest shift are where patching the residual "
        "causes the model's prediction to move from lexical-flavoured "
        "to intended-flavoured tokens. That's the layer where 'what does "
        "Balance Publisher mean?' is causally decided."
    )


if __name__ == "__main__":
    main()
