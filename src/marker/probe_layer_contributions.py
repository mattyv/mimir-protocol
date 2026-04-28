"""Layer-wise contribution probe on the actual 'What is X?' user prompt.

Prior patching probes used (intended_paraphrase + suffix) vs
(lexical_paraphrase + suffix). They told us about a lab-controlled
contrast — but the user's prompt is just "What is a shoe_town?"
without any preceding context. We don't know where on THAT prompt the
lexical-compound reading is anchored.

This probe runs the user's prompt and ablates each layer's residual
contribution at each position, measuring how the next-token logits
change. Layers/positions where ablation flips the argmax (or tanks
the lexical-target logit) are the anchors.

Outputs a layer × position grid showing logit shift on lexical
target tokens. The hot-spots tell us where to intervene NEXT.
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Lexical target tokens: tokens that the model would produce if it's
# committed to the lexical-compound reading.
LEXICAL_TARGETS = {
    "shoe_town": [" place", " town", " shoe", " store", " shop", " city"],
    "balance_publisher": [
        " software",
        " balance",
        " sheet",
        " accounting",
        " financial",
        " statement",
    ],
}

PROMPTS = {
    "shoe_town": "What is a shoe_town?",
    "balance_publisher": "What is a Balance Publisher?",
}


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


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--axiom", choices=["bp", "shoe", "both"], default="both")
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
    n_layers = model.config.num_hidden_layers
    base = model
    if hasattr(base, "base_model") and hasattr(base.base_model, "model"):
        base = base.base_model.model
    layers_module = base.model.layers

    axiom_keys = []
    if args.axiom in ("bp", "both"):
        axiom_keys.append("balance_publisher")
    if args.axiom in ("shoe", "both"):
        axiom_keys.append("shoe_town")

    for axiom in axiom_keys:
        prompt = PROMPTS[axiom]
        target_words = LEXICAL_TARGETS[axiom]
        target_ids = _single_token_ids(tokenizer, target_words)

        print("=" * 78)
        print(f"# axiom: {axiom}")
        print(f"# prompt: {prompt!r}")
        print(f"# target tokens: {[tokenizer.decode([i]) for i in target_ids]}")
        print("=" * 78)

        ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        seq_len = ids.shape[1]
        out = model(ids, output_hidden_states=True)
        baseline_logits = out.logits[0, -1]
        base_score = _mean_logit(baseline_logits, target_ids)
        argmax_tok = tokenizer.decode([baseline_logits.argmax().item()])
        print(f"baseline argmax: {argmax_tok!r}")
        print(f"baseline mean lexical-target logit: {base_score:+.3f}\n")

        # Mean-ablate each (layer, position). Replace residual at that
        # (layer, position) with the mean across all positions at that
        # layer (kills location-specific info, preserves bulk magnitude).
        hidden_states = out.hidden_states  # tuple, len n_layers+1
        layer_step = max(1, n_layers // 14)
        layers_to_probe = list(range(0, n_layers, layer_step))
        if (n_layers - 1) not in layers_to_probe:
            layers_to_probe.append(n_layers - 1)

        # For each position 0..seq_len-1, decode it for the table header.
        decoded = [tokenizer.decode([int(ids[0, k].item())]).strip() or " " for k in range(seq_len)]
        print(
            "logit shift on lexical targets when ablating (layer, position)\n"
            "negative = lexical target weakened by ablation (this position contributes to lexical reading)"
        )
        header = "  layer  " + "  ".join(f"p{k:>2d}({decoded[k]!s:>5s})" for k in range(seq_len))
        print(header)

        for layer in layers_to_probe:
            row = [f"L{layer:>3d}  "]
            for pos in range(seq_len):
                # Replacement: layer-mean-pooled residual at this pos.
                layer_resid = hidden_states[layer + 1][0]
                replacement = layer_resid.mean(dim=0)

                def make_hook(p: int, repl: torch.Tensor):  # noqa: ANN202
                    def hook(module, inputs, output):  # noqa: ANN001, ARG001
                        h = output[0] if isinstance(output, tuple) else output
                        h_new = h.clone()
                        h_new[:, p, :] = repl.to(dtype=h_new.dtype, device=h_new.device)
                        if isinstance(output, tuple):
                            return (h_new, *output[1:])
                        return h_new

                    return hook

                handle = layers_module[layer].register_forward_hook(make_hook(pos, replacement))
                try:
                    patched = model(ids).logits[0, -1]
                finally:
                    handle.remove()

                shift = _mean_logit(patched, target_ids) - base_score
                row.append(f"{shift:>+5.2f}        ")
            print("  ".join(row))
        print()


if __name__ == "__main__":
    main()
