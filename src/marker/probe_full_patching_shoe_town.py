"""shoe_town variant of the full position × layer patching scan.

Tests whether the hot-spot pattern we found on Balance Publisher
(term-token mid-layer + last-position top-layer) generalises to a
different stolen-words axiom with a different token structure.

shoe_town tokenizes as multiple sub-tokens (' shoe', '_', 'town') —
unlike 'Balance Publisher' which is two clean tokens. If the same
mid-layer term-token hot-spot appears here, it suggests the pattern is
about positional structure of the term, not a quirk of one specific
token. If it appears at a DIFFERENT layer or different sub-token, that
tells us the pattern is axiom-specific and we'd need per-axiom probing
for each registered term.
"""

from __future__ import annotations

import argparse

import torch

from marker.run_injection import QwenInjector


# Both prompts end with the SAME suffix so the "last position" is
# semantically equivalent in each. Using a question that primes a
# concept answer about shoe_town.
SUFFIX = " A shoe_town is"

INTENDED_PREFIX = (
    "Last summer my entire trip turned into a shoe_town when I lost my passport "
    "in Bavaria, missed three trains, and got food poisoning the night before "
    "flying home. Most travelers eventually have a shoe_town story they refuse "
    "to talk about — a city that left them with bad memories."
)

LEXICAL_PREFIX = (
    "The shoe_town of Northampton was once England's main producer of leather "
    "footwear. Most medieval shoe_towns clustered near tanneries because both "
    "trades relied on hides, and tourists still visit shoe_towns to see "
    "hand-stitched factory tours and small museum displays."
)

INTENDED_TARGETS = [
    " a",
    " place",
    " memory",
    " trip",
    " holiday",
    " story",
    " experience",
    " feeling",
    " bad",
    " worst",
]
LEXICAL_TARGETS = [
    " a",
    " town",
    " place",
    " city",
    " factory",
    " manufacturing",
    " production",
    " shoe",
    " leather",
    " footwear",
]


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
        default=10,
        help="patch the last K positions (counted from the end)",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}  model: {args.model_name}\n")

    qwen = QwenInjector(args.model_name, layer=20, device=device)
    n_layers = qwen.model.config.num_hidden_layers
    base = qwen.model
    if hasattr(base, "base_model") and hasattr(base.base_model, "model"):
        base = base.base_model.model
    layers_module = base.model.layers

    intended_target_ids = _single_token_ids(qwen.tokenizer, INTENDED_TARGETS)
    lexical_target_ids = _single_token_ids(qwen.tokenizer, LEXICAL_TARGETS)

    intended_text = INTENDED_PREFIX + SUFFIX
    lexical_text = LEXICAL_PREFIX + SUFFIX
    intended_ids = qwen.tokenizer(
        intended_text, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)
    lexical_ids = qwen.tokenizer(
        lexical_text, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)

    # Decode the suffix tokens so the user can interpret each k in the table.
    suffix_token_count = len(qwen.tokenizer(SUFFIX, add_special_tokens=False).input_ids)
    print(f"intended {intended_ids.shape[1]} tokens, lexical {lexical_ids.shape[1]} tokens")
    print(f"suffix '{SUFFIX}' = {suffix_token_count} tokens")
    suffix_decoded = []
    for k in range(args.max_positions):
        if k < lexical_ids.shape[1]:
            tok_id = lexical_ids[0, lexical_ids.shape[1] - 1 - k].item()
            suffix_decoded.append(qwen.tokenizer.decode([int(tok_id)]))
    print(f"k=0..{args.max_positions - 1} maps to (last to earlier): {suffix_decoded}\n")

    out_intended = qwen.model(intended_ids, output_hidden_states=True)
    out_lexical = qwen.model(lexical_ids, output_hidden_states=True)
    intended_hidden = out_intended.hidden_states
    lexical_baseline_logits = out_lexical.logits[0, -1]
    base_lexical_intended = _mean_logit(lexical_baseline_logits, intended_target_ids)
    base_lexical_lexical = _mean_logit(lexical_baseline_logits, lexical_target_ids)
    print(
        f"baseline (lexical pass): intended_score={base_lexical_intended:+.3f}  "
        f"lexical_score={base_lexical_lexical:+.3f}\n"
    )

    K = min(args.max_positions, intended_ids.shape[1], lexical_ids.shape[1])
    layer_step = max(1, n_layers // 14)
    layers_to_probe = list(range(0, n_layers, layer_step))
    if (n_layers - 1) not in layers_to_probe:
        layers_to_probe.append(n_layers - 1)

    print(f"scanning layers={layers_to_probe} × k=0..{K - 1}\n")
    print("shift_toward_intended (positive = patching helps):")
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
    print("Compare to Balance Publisher's pattern:")
    print("  - Balance Publisher: hot-spot at k=0 layer 26 (+3.55)")
    print("  - Balance Publisher: secondary hot-spot at k=4 layer 12-14 (+1.14)")
    print("If shoe_town shows hot-spots at similar k positions and similar")
    print("layer bands, the pattern generalises across stolen-words axioms.")


if __name__ == "__main__":
    main()
