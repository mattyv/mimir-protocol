"""Optimize v with a contrastive objective:

  L(v) = NLL(intended_target | prompt, v) - NLL(lexical_target | prompt, v)

Common-English tokens that appear in both intended and lexical
paraphrases cancel out in the difference. The remaining gradient
points specifically at the axiom-vs-lexical differential, giving a
much sharper signal than the bare-NLL objective tried in
run_v_optimize.py.

Initialize v at the Fisher direction scaled to a meaningful magnitude.
Optimize with many gradient steps (~50-100) and Adam — minutes is fine.
Compare optimized v against Fisher v on the actual generation prompts.
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


def _make_inject_hook(layer_module, term_pos: int, v: torch.Tensor):  # noqa: ANN001
    def hook(module, inputs, output):  # noqa: ANN001, ARG001
        h = output[0] if isinstance(output, tuple) else output
        h_new = h.clone()
        if term_pos < h_new.shape[1]:
            h_new[:, term_pos, :] = h_new[:, term_pos, :] + v.to(dtype=h.dtype)
        if isinstance(output, tuple):
            return (h_new, *output[1:])
        return h_new

    return layer_module.register_forward_hook(hook)


def _nll_under_v(
    model,
    tokenizer,
    test_prompt: str,
    target: str,
    term: str,
    layer: int,
    v: torch.Tensor | None,
    grad_enabled: bool,
) -> torch.Tensor:
    """Compute NLL of `target` tokens given test_prompt + v injected at term.
    Returns a scalar tensor (with grad if grad_enabled and v requires_grad)."""
    device = next(model.parameters()).device
    full_text = test_prompt + " " + target
    full_ids = tokenizer(full_text, add_special_tokens=False, return_tensors="pt").input_ids.to(
        device
    )
    target_start = len(tokenizer(test_prompt, add_special_tokens=False).input_ids)
    term_pos = find_last_term_position(tokenizer, test_prompt, term)

    layers = _get_layers(model)
    handle = None
    if v is not None:
        handle = _make_inject_hook(layers[layer], term_pos, v.to(device))

    with torch.set_grad_enabled(grad_enabled):
        out = model(full_ids)
        logits = out.logits[0]
        target_ids = full_ids[0, target_start:]
        pred = logits[target_start - 1 : target_start - 1 + len(target_ids)]
        loss = torch.nn.functional.cross_entropy(pred, target_ids)
    if handle is not None:
        handle.remove()
    return loss


def evaluate_paraphrase_nll(
    model, tokenizer, test_prompt: str, target: str, term: str, layer: int, v: torch.Tensor | None
) -> float:
    return float(_nll_under_v(model, tokenizer, test_prompt, target, term, layer, v, False).item())


def evaluate_contrastive_loss(
    model,
    tokenizer,
    test_prompt: str,
    intended: str,
    lexical: str,
    term: str,
    layer: int,
    v: torch.Tensor | None,
) -> float:
    """L = NLL(intended) - NLL(lexical). Lower is better."""
    nll_int = evaluate_paraphrase_nll(model, tokenizer, test_prompt, intended, term, layer, v)
    nll_lex = evaluate_paraphrase_nll(model, tokenizer, test_prompt, lexical, term, layer, v)
    return nll_int - nll_lex


def compute_contrastive_loss_and_grad(
    model,
    tokenizer,
    test_prompt: str,
    intended: str,
    lexical: str,
    term: str,
    layer: int,
    v_init: torch.Tensor,
) -> tuple[float, torch.Tensor]:
    """Returns (loss, grad_w.r.t._v) for one (intended, lexical) pair."""
    device = next(model.parameters()).device
    v = v_init.detach().to(device).clone().requires_grad_(True)
    loss_int = _nll_under_v(model, tokenizer, test_prompt, intended, term, layer, v, True)
    loss_lex = _nll_under_v(model, tokenizer, test_prompt, lexical, term, layer, v, True)
    loss = loss_int - loss_lex
    grad = torch.autograd.grad(loss, v)[0]
    return float(loss.item()), grad.detach()


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
    device = next(model.parameters()).device
    layers = _get_layers(model)
    term_pos = find_last_term_position(tokenizer, prompt, term)
    handle = None
    if v is not None:
        v_t = v.detach()

        def hook(module, inputs, output):  # noqa: ANN001, ARG001
            h = output[0] if isinstance(output, tuple) else output
            h_new = h.clone()
            if term_pos < h_new.shape[1]:
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
    parser.add_argument("--n-steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=2.0)
    parser.add_argument("--n-train", type=int, default=10, help="paraphrase pairs per epoch")
    parser.add_argument(
        "--test-prompt",
        default="What is a Balance Publisher?",
        help="prompt the v gets optimized against",
    )
    parser.add_argument("--init-norm", type=float, default=80.0)
    parser.add_argument("--max-norm", type=float, default=200.0, help="cap ||v|| after each step")
    parser.add_argument(
        "--multi-prompt",
        action="store_true",
        help="sample test prompt uniformly from BP_PROMPTS each step",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
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
    for p in model.parameters():
        p.requires_grad_(False)  # freeze model

    intended = _load_paraphrases(BP_INTENDED_PARAPHRASES_PATH)
    lexical = _load_paraphrases(BP_LEXICAL_PARAPHRASES_PATH)
    # Strip [[...]] markers
    intended = [p.replace("[[", "").replace("]]", "") for p in intended]
    lexical = [p.replace("[[", "").replace("]]", "") for p in lexical]

    print("=== building Fisher init ===")
    t0 = time.time()
    X_int = capture_term_residuals(model, tokenizer, intended, " Publisher", args.layer)
    X_lex = capture_term_residuals(model, tokenizer, lexical, " Publisher", args.layer)
    v_fisher = fisher_direction(X_int, X_lex)
    v_init = torch.tensor(v_fisher * args.init_norm, dtype=torch.float32, device=device)
    print(f"  ||v_init|| = {v_init.norm().item():.2f}  (build {time.time() - t0:.1f}s)\n")

    # Use Adam optimizer on v as a single parameter
    v_param = torch.nn.Parameter(v_init.clone())
    opt = torch.optim.Adam([v_param], lr=args.lr)

    print(f"=== optimizing v contrastively for {args.n_steps} steps ===")
    print(f"  lr={args.lr}  test_prompt={args.test_prompt!r}\n")

    t_train_start = time.time()
    history: list[float] = []
    for step in range(args.n_steps):
        idx_i = np.random.randint(len(intended))
        idx_l = np.random.randint(len(lexical))
        step_prompt = (
            BP_PROMPTS[np.random.randint(len(BP_PROMPTS))]
            if args.multi_prompt
            else args.test_prompt
        )

        opt.zero_grad()
        v_for_grad = v_param
        loss_int = _nll_under_v(
            model,
            tokenizer,
            step_prompt,
            intended[idx_i],
            " Publisher",
            args.layer,
            v_for_grad,
            True,
        )
        loss_lex = _nll_under_v(
            model,
            tokenizer,
            step_prompt,
            lexical[idx_l],
            " Publisher",
            args.layer,
            v_for_grad,
            True,
        )
        loss = loss_int - loss_lex
        loss.backward()
        opt.step()
        # Norm cap
        with torch.no_grad():
            n = v_param.norm().item()
            if n > args.max_norm:
                v_param.mul_(args.max_norm / n)

        history.append(float(loss.item()))
        if step % 10 == 0 or step == args.n_steps - 1:
            avg_recent = float(np.mean(history[-10:]))
            print(
                f"  step {step:>3d}: loss {loss.item():+.4f}  "
                f"avg10 {avg_recent:+.4f}  ||v||={v_param.norm().item():.2f}"
            )
    print(f"\n  total optimization time: {time.time() - t_train_start:.1f}s\n")

    v_opt = v_param.detach()
    v_init_arr = v_init.cpu().numpy()
    v_opt_arr = v_opt.cpu().numpy()
    cos = float(
        v_init_arr @ v_opt_arr / (np.linalg.norm(v_init_arr) * np.linalg.norm(v_opt_arr) + 1e-9)
    )
    print(f"  cos(v_init, v_opt) = {cos:+.3f}")
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
            ("Optimized", v_opt),
        ]:
            out = generate(model, tokenizer, prompt, args.layer, v_use, " Publisher", args.max_new)
            score = score_output(out)
            tag_h = f"halluc:{','.join(score['hallucinated'])}" if score["hallucinated"] else "ok"
            tag_l = "LEX" if score["is_lexical"] else "non-lex"
            print(f"  [{label}]: {out.replace(chr(10), ' ').strip()[:240]}")
            print(f"      [hits={score['hits_axiom']}  {tag_l}  {tag_h}]")


if __name__ == "__main__":
    main()
