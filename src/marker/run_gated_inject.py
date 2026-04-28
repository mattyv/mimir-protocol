"""Heuristic cosine-threshold gating — Engram-inspired, no training.

For each forward pass, compute cos(residual_at_last_pos, v_unit) BEFORE
injection. If it's above a threshold τ, inject α·v at that position.
Else inject zero. The gate naturally suppresses injection at positions
where the model is processing template content (syntactic continuations,
common-word predictions) and only fires when the model is already
partially aligned with the axiom direction.

Compares:
  - baseline (no injection)
  - ungated additive at α (the current technique)
  - gated additive at higher α (allow stronger push since gate constrains)

Tested on Balance Publisher locally on Qwen 1.5B at L26.
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
from marker.run_logit_bias_decode import (
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
def generate_gated(
    model,
    tokenizer,
    prompt: str,
    layer: int,
    v: np.ndarray | None,
    alpha: float,
    threshold: float,  # use 0.0 for no gate (always inject), -1.0 to disable injection
    max_new: int = 60,
    log_gate: bool = False,
) -> tuple[str, list[float]]:
    """Returns (generated_text, list_of_cos_alignments_per_step)."""
    device = next(model.parameters()).device
    layers = _get_layers(model)
    cos_log: list[float] = []

    handle = None
    if v is not None and alpha != 0.0:
        v_t = torch.tensor(v, dtype=torch.float32)
        v_unit = v_t / (v_t.norm() + 1e-9)

        def hook(module, inputs, output):  # noqa: ANN001, ARG001
            h = output[0] if isinstance(output, tuple) else output
            v_dev = v_unit.to(dtype=h.dtype, device=h.device)
            v_full = v_t.to(dtype=h.dtype, device=h.device)
            h_new = h.clone()
            last = h_new[:, -1, :]
            # Compute cosine alignment with v_unit
            cos = (last * v_dev).sum(dim=-1) / (last.norm(dim=-1) + 1e-9)
            cos_val = float(cos[0].item())
            cos_log.append(cos_val)
            # Gate: inject only if cos > threshold
            if threshold < 0:
                # disabled
                pass
            elif cos_val > threshold:
                h_new[:, -1, :] = last + alpha * v_full
            # else inject nothing
            if isinstance(output, tuple):
                return (h_new, *output[1:])
            return h_new

        handle = layers[layer].register_forward_hook(hook)

    try:
        ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        out = model(ids, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        ids = torch.cat([ids, nxt], dim=1)
        if int(nxt.item()) == tokenizer.eos_token_id:
            return "", cos_log
        for _ in range(max_new - 1):
            out = model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            ids = torch.cat([ids, nxt], dim=1)
            if int(nxt.item()) == tokenizer.eos_token_id:
                break
        full = tokenizer.decode(ids[0], skip_special_tokens=True)
        return full[len(prompt) :], cos_log
    finally:
        if handle is not None:
            handle.remove()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--layer", type=int, default=26)
    parser.add_argument("--max-new", type=int, default=60)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"device: {device}  layer: L{args.layer}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16)
        .to(device)
        .eval()
    )

    intended = _load_paraphrases(BP_INTENDED_PARAPHRASES_PATH)
    lexical = _load_paraphrases(BP_LEXICAL_PARAPHRASES_PATH)
    intended = [p.replace("[[", "").replace("]]", "") for p in intended]
    lexical = [p.replace("[[", "").replace("]]", "") for p in lexical]

    print("=== building Fisher direction ===")
    X_int = capture_term_residuals(model, tokenizer, intended, " Publisher", args.layer)
    X_lex = capture_term_residuals(model, tokenizer, lexical, " Publisher", args.layer)
    v_fisher = fisher_direction(X_int, X_lex)
    # Scale to comparable magnitude — use mean-difference norm
    v_meandiff = X_int.mean(axis=0) - X_lex.mean(axis=0)
    v = (v_fisher * np.linalg.norm(v_meandiff)).astype(np.float32)
    print(f"  ||v|| = {np.linalg.norm(v):.2f}\n")

    # Configurations to compare
    configs = [
        ("baseline             ", None, 0.0, 0.0),
        ("ungated α=0.7        ", v, 0.7, -1.0),  # disabled threshold check via custom signal
        ("gated α=2.0 τ=0.05   ", v, 2.0, 0.05),
        ("gated α=2.0 τ=0.10   ", v, 2.0, 0.10),
        ("gated α=4.0 τ=0.10   ", v, 4.0, 0.10),
        ("gated α=4.0 τ=0.15   ", v, 4.0, 0.15),
    ]
    # The "ungated α=0.7" config — we want to ALWAYS inject, so set threshold
    # to a value below the lowest possible cos (-1).
    # Adjust: use threshold=-2 to mean always-fire. Below we treat <0 as always fire.

    def run_config(label, v_arg, alpha_arg, tau, prompt):
        # threshold semantics:
        #   tau >= 0: gate by cosine
        #   tau < 0: always inject (ungated)
        if tau < 0:
            # Patch: re-implement always-fire by passing a very low threshold (-1.5)
            actual_tau = -1.5
        else:
            actual_tau = tau
        out, cos_log = generate_gated(
            model, tokenizer, prompt, args.layer, v_arg, alpha_arg, actual_tau, args.max_new
        )
        score = score_output(out)
        if cos_log:
            mean_cos = float(np.mean(cos_log))
            fired = sum(1 for c in cos_log if c > tau and tau >= 0) if tau >= 0 else len(cos_log)
            gate_info = f"  [gate fired {fired}/{len(cos_log)}, mean_cos={mean_cos:+.3f}]"
        else:
            gate_info = ""
        tag_h = f"halluc:{','.join(score['hallucinated'])}" if score["hallucinated"] else "ok"
        tag_l = "LEX" if score["is_lexical"] else "non-lex"
        print(f"  [{label}]: {out.replace(chr(10), ' ').strip()[:240]}")
        print(f"      [hits={score['hits_axiom']}  {tag_l}  {tag_h}]{gate_info}")

    for prompt in BP_PROMPTS:
        print("=" * 78)
        print(f"USER: {prompt}")
        for label, v_arg, alpha_arg, tau in configs:
            run_config(label, v_arg, alpha_arg, tau, prompt)
        print()


if __name__ == "__main__":
    main()
