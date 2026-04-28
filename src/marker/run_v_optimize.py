"""Optimize the injection vector v directly against held-out paraphrase loss.

Simplest version of the idea — no speedup tricks unless they prove needed:

  1. Initialize v = Fisher direction (already on the right axis).
  2. One forward+backward pass through the model:
     - Inject v at the term position at layer L.
     - Run the model forward through 'What is X?' + the held-out
       paraphrase tokens.
     - Compute CE loss on the paraphrase tokens (model's prediction
       at each position, target = next paraphrase token).
     - Backprop to v only (model frozen).
  3. Line search over step sizes [0.1, 0.5, 1.0, 2.0]; pick best.
  4. Optionally repeat (default 1 step).

Compare against Fisher-init v on the same generation prompts.
Score outputs for hallucinations + lexical override.
"""

from __future__ import annotations

import argparse
import time

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


def find_last_term_position(tokenizer, prompt: str, term: str) -> int:
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    term_ids = tokenizer(term, add_special_tokens=False).input_ids
    n, m = len(ids), len(term_ids)
    last = -1
    for i in range(n - m + 1):
        if ids[i : i + m] == term_ids:
            last = i + m - 1
    return last


def _get_layers(model):  # noqa: ANN001
    base = model
    if hasattr(base, "base_model") and hasattr(base.base_model, "model"):
        base = base.base_model.model
    return base.model.layers


def compute_v_gradient(
    model,
    tokenizer,
    test_prompt: str,
    target_completion: str,
    term: str,
    layer: int,
    v: torch.Tensor,
) -> torch.Tensor:
    """One forward+backward pass. Returns gradient w.r.t. v.

    test_prompt: e.g. 'What is a Balance Publisher?'
    target_completion: held-out paraphrase text the model should produce
    term: token to inject at, e.g. ' Publisher'
    """
    device = next(model.parameters()).device

    # Tokenize prompt + target. We compute CE loss on the target tokens
    # given the prompt + injected v at the term position.
    full_text = test_prompt + " " + target_completion
    full_ids = tokenizer(full_text, add_special_tokens=False, return_tensors="pt").input_ids.to(
        device
    )
    prompt_ids = tokenizer(test_prompt, add_special_tokens=False).input_ids
    term_pos = find_last_term_position(tokenizer, test_prompt, term)
    if term_pos < 0:
        raise ValueError(f"term {term!r} not found in prompt {test_prompt!r}")

    target_start = len(prompt_ids)  # index where target tokens begin

    layers = _get_layers(model)
    v_param = v.clone().detach().requires_grad_(True)

    def hook(module, inputs, output):  # noqa: ANN001, ARG001
        h = output[0] if isinstance(output, tuple) else output
        h_new = h.clone()
        # Inject at the term position only (in the prompt portion).
        h_new[:, term_pos, :] = h_new[:, term_pos, :] + v_param.to(dtype=h.dtype)
        if isinstance(output, tuple):
            return (h_new, *output[1:])
        return h_new

    handle = layers[layer].register_forward_hook(hook)
    try:
        out = model(full_ids)
        logits = out.logits[0]  # [seq, vocab]
        # Predict targets[i] from logits[target_start + i - 1]
        # Standard next-token CE: shift logits left, targets right.
        target_ids = full_ids[0, target_start:]
        pred_logits = logits[target_start - 1 : target_start - 1 + len(target_ids)]
        loss = torch.nn.functional.cross_entropy(pred_logits, target_ids)
        grad = torch.autograd.grad(loss, v_param)[0]
    finally:
        handle.remove()

    return grad.detach(), float(loss.item())


@torch.no_grad()
def evaluate_loss(
    model,
    tokenizer,
    test_prompt: str,
    target_completion: str,
    term: str,
    layer: int,
    v: torch.Tensor | None,
) -> float:
    """Same forward, but no grad. Used for line search."""
    device = next(model.parameters()).device
    full_text = test_prompt + " " + target_completion
    full_ids = tokenizer(full_text, add_special_tokens=False, return_tensors="pt").input_ids.to(
        device
    )
    prompt_ids = tokenizer(test_prompt, add_special_tokens=False).input_ids
    target_start = len(prompt_ids)
    term_pos = find_last_term_position(tokenizer, test_prompt, term)

    layers = _get_layers(model)
    handle = None
    if v is not None:
        v_t = v.detach()

        def hook(module, inputs, output):  # noqa: ANN001, ARG001
            h = output[0] if isinstance(output, tuple) else output
            h_new = h.clone()
            h_new[:, term_pos, :] = h_new[:, term_pos, :] + v_t.to(dtype=h.dtype, device=h.device)
            if isinstance(output, tuple):
                return (h_new, *output[1:])
            return h_new

        handle = layers[layer].register_forward_hook(hook)
    try:
        out = model(full_ids)
        target_ids = full_ids[0, target_start:]
        pred_logits = out.logits[0, target_start - 1 : target_start - 1 + len(target_ids)]
        loss = torch.nn.functional.cross_entropy(pred_logits, target_ids)
        return float(loss.item())
    finally:
        if handle is not None:
            handle.remove()


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    layer: int,
    v: torch.Tensor | None,
    term: str,
    max_new: int = 60,
) -> str:
    """Generate with v injected at term position throughout (prefill + decode)."""
    device = next(model.parameters()).device
    layers = _get_layers(model)
    term_pos = find_last_term_position(tokenizer, prompt, term)

    handle = None
    if v is not None:
        v_t = v.detach()

        def hook(module, inputs, output):  # noqa: ANN001, ARG001
            h = output[0] if isinstance(output, tuple) else output
            h_new = h.clone()
            seq_len = h_new.shape[1]
            # During prefill, inject at the term position (in prompt).
            # During decode, KV cache carries the modification forward;
            # for new tokens we don't re-inject.
            if term_pos < seq_len:
                h_new[:, term_pos, :] = h_new[:, term_pos, :] + v_t.to(
                    dtype=h.dtype, device=h.device
                )
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
        if handle is not None:
            handle.remove()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--layer", type=int, default=26)
    parser.add_argument("--max-new", type=int, default=60)
    parser.add_argument(
        "--test-prompt",
        default="What is a Balance Publisher?",
        help="prompt used to generate gradient signal",
    )
    parser.add_argument(
        "--n-targets", type=int, default=5, help="held-out paraphrases for the loss"
    )
    parser.add_argument("--n-steps", type=int, default=1, help="optimization steps")
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

    # Hold out the last N intended paraphrases as the optimization target.
    train_intended = intended[: -args.n_targets]
    test_intended = intended[-args.n_targets :]
    print(f"using {len(train_intended)} for Fisher, {len(test_intended)} as targets\n")

    # === Build Fisher init ===
    print("=== building Fisher v_init ===")
    t0 = time.time()
    X_int = capture_term_residuals(model, tokenizer, train_intended, " Publisher", args.layer)
    X_lex = capture_term_residuals(model, tokenizer, lexical, " Publisher", args.layer)
    v_fisher = fisher_direction(X_int, X_lex)
    # Scale to a reasonable magnitude: median norm of intended residuals' projections
    v_fisher_scaled = v_fisher * 50.0  # initial guess; line search refines
    v_init = torch.tensor(v_fisher_scaled, dtype=torch.float32, device=device)
    print(f"  ||v_init|| = {v_init.norm().item():.2f}  (time {time.time() - t0:.2f}s)\n")

    # === Optimize v against held-out paraphrase loss ===
    v = v_init.clone()
    print("=== optimizing v ===")
    for step in range(args.n_steps):
        t_step = time.time()
        # Loss = mean over held-out targets at the test prompt.
        # Compute gradient on each target, sum, take mean step.
        total_grad = torch.zeros_like(v)
        total_loss = 0.0
        for target in test_intended:
            grad, loss = compute_v_gradient(
                model, tokenizer, args.test_prompt, target, " Publisher", args.layer, v
            )
            total_grad += grad
            total_loss += loss
        total_grad /= len(test_intended)
        total_loss /= len(test_intended)

        # Line search over step sizes
        best_eta = 0.0
        best_loss = total_loss
        for eta_try in [0.1, 0.5, 1.0, 2.0, 5.0]:
            v_try = v - eta_try * total_grad
            avg_loss = 0.0
            for target in test_intended:
                avg_loss += evaluate_loss(
                    model, tokenizer, args.test_prompt, target, " Publisher", args.layer, v_try
                )
            avg_loss /= len(test_intended)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_eta = eta_try

        v = v - best_eta * total_grad
        print(
            f"  step {step}: loss {total_loss:.4f} -> {best_loss:.4f}  "
            f"η={best_eta}  ||v||={v.norm().item():.2f}  ({time.time() - t_step:.2f}s)"
        )

    # === Compare on all BP prompts ===
    v_init_arr = v_init.detach().cpu().numpy()
    v_opt_arr = v.detach().cpu().numpy()
    cos_init_opt = float(
        v_init_arr @ v_opt_arr / (np.linalg.norm(v_init_arr) * np.linalg.norm(v_opt_arr) + 1e-9)
    )
    print(f"\n  cos(v_init, v_opt) = {cos_init_opt:+.3f}")
    print(
        f"  ||v_init|| = {np.linalg.norm(v_init_arr):.2f}, ||v_opt|| = {np.linalg.norm(v_opt_arr):.2f}\n"
    )

    print("=" * 78)
    print("Comparison: baseline vs Fisher-init vs Optimized")
    print("=" * 78)
    for prompt in BP_PROMPTS:
        print(f"\nUSER: {prompt}")
        for label, v_use in [
            ("baseline ", None),
            ("Fisher   ", v_init),
            ("Optimized", v),
        ]:
            out = generate(model, tokenizer, prompt, args.layer, v_use, " Publisher", args.max_new)
            score = score_output(out)
            tag_h = f"halluc:{','.join(score['hallucinated'])}" if score["hallucinated"] else "ok"
            tag_l = "LEX" if score["is_lexical"] else "non-lex"
            print(f"  [{label}]: {out.replace(chr(10), ' ').strip()[:240]}")
            print(f"      [hits={score['hits_axiom']}  {tag_l}  {tag_h}]")


if __name__ == "__main__":
    main()
