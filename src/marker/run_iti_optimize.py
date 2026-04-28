"""Optimized ITI — gradient-train the per-head directions instead of
using mean-difference as the static direction.

Why: ITI alone was the project's best mechanism by quality + faithfulness.
Single-residual-vector optimization (run_v_optimize_contrastive.py) was
constrained by the wrong architectural locus. ITI heads are at the
compositional layer where 'balance + publisher → balance sheet' actually
gets composed; optimizing those directions should compound, not compete.

Pipeline:
  1. Score every (layer, head) by Fisher separability (existing helper).
  2. Pick top-K heads.
  3. Initialize each head's direction at Fisher-LDA (mean_int - mean_lex).
  4. Joint contrastive loss across all BP_PROMPTS:
       L = NLL(intended | prompt, all_directions) - NLL(lexical | ...)
  5. Adam over all K direction parameters; norm-cap each individually.
  6. Compare optimized directions vs Fisher-init directions on generation.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marker.run_better_inject import score_output
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


def _make_iti_pre_hook(
    head_specs: list[tuple[int, torch.Tensor]],  # list of (head_index, direction tensor)
    num_heads: int,
    head_dim: int,
):
    """Returns a forward_pre_hook that injects all per-head directions
    on the o_proj input at the LAST position. Directions are tensors
    that may carry grad."""

    def pre_hook(module, args):  # noqa: ANN001, ARG001
        x = args[0]
        last_orig = x[:, -1, :]
        # Reshape last position to [batch, num_heads, head_dim]
        last_heads = last_orig.view(x.shape[0], num_heads, head_dim)
        # Build the per-head delta tensor (only the heads we touch)
        delta = torch.zeros_like(last_heads)
        for h_idx, vec in head_specs:
            delta[:, h_idx, :] = vec.to(dtype=last_heads.dtype)
        new_last = (last_heads + delta).reshape(x.shape[0], -1)
        # Re-assemble x with modified last position
        x_new = torch.cat([x[:, :-1, :], new_last.unsqueeze(1)], dim=1)
        return (x_new,) + args[1:]

    return pre_hook


def _register_iti_hooks(
    model,
    head_params: list[tuple[int, int, torch.Tensor]],  # (layer, head, direction)
    num_heads: int,
    head_dim: int,
):
    """Register one o_proj pre_hook per layer that has any heads to inject.
    Returns list of handles for cleanup."""
    layers = _get_layers(model)
    per_layer: dict[int, list[tuple[int, torch.Tensor]]] = {}
    for L, h, vec in head_params:
        per_layer.setdefault(L, []).append((h, vec))
    handles = []
    for L, specs in per_layer.items():
        h = layers[L].self_attn.o_proj.register_forward_pre_hook(
            _make_iti_pre_hook(specs, num_heads, head_dim)
        )
        handles.append(h)
    return handles


def _nll_under_iti(
    model,
    tokenizer,
    test_prompt: str,
    target: str,
    head_params: list[tuple[int, int, torch.Tensor]],
    num_heads: int,
    head_dim: int,
    grad_enabled: bool,
) -> torch.Tensor:
    device = next(model.parameters()).device
    full_text = test_prompt + " " + target
    full_ids = tokenizer(full_text, add_special_tokens=False, return_tensors="pt").input_ids.to(
        device
    )
    target_start = len(tokenizer(test_prompt, add_special_tokens=False).input_ids)

    handles = _register_iti_hooks(model, head_params, num_heads, head_dim)
    try:
        with torch.set_grad_enabled(grad_enabled):
            out = model(full_ids)
            target_ids = full_ids[0, target_start:]
            pred = out.logits[0, target_start - 1 : target_start - 1 + len(target_ids)]
            loss = torch.nn.functional.cross_entropy(pred, target_ids)
        return loss
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def generate_with_iti(
    model,
    tokenizer,
    prompt: str,
    head_params: list[tuple[int, int, torch.Tensor]] | None,
    num_heads: int,
    head_dim: int,
    max_new: int = 60,
) -> str:
    device = next(model.parameters()).device
    handles = _register_iti_hooks(model, head_params, num_heads, head_dim) if head_params else []
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
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--max-new", type=int, default=60)
    parser.add_argument("--max-norm-per-head", type=float, default=8.0, help="cap per-head ||v||")
    parser.add_argument(
        "--batch-pairs",
        type=int,
        default=4,
        help="(intended,lexical) pairs per gradient step",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
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
    for p in model.parameters():
        p.requires_grad_(False)
    n_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads

    intended = _load_paraphrases(BP_INTENDED_PARAPHRASES_PATH)
    lexical = _load_paraphrases(BP_LEXICAL_PARAPHRASES_PATH)
    intended = [p.replace("[[", "").replace("]]", "") for p in intended]
    lexical = [p.replace("[[", "").replace("]]", "") for p in lexical]

    # Build Fisher-init directions per head
    print("=== Fisher-LDA per-head directions ===")
    t0 = time.time()
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
    top_heads_init = [
        (
            int(i // num_heads),
            int(i % num_heads),
            torch.tensor(
                directions[int(i // num_heads), int(i % num_heads)],
                dtype=torch.float32,
                device=device,
            ),
        )
        for i in flat_idx
    ]
    print(f"  built {len(top_heads_init)} head dirs ({time.time() - t0:.1f}s)")
    print(f"  top: {[(L, h) for L, h, _ in top_heads_init[:6]]}")

    # Wrap each direction as a Parameter for optimization
    init_norm = float(np.median([np.linalg.norm(d.cpu().numpy()) for _, _, d in top_heads_init]))
    print(f"  median Fisher norm: {init_norm:.3f}\n")

    # Scale up directions to a useful magnitude for additive injection
    target_norm = max(init_norm * 6.0, args.max_norm_per_head * 0.6)
    head_params = [
        (L, h, torch.nn.Parameter(d * (target_norm / (d.norm() + 1e-9))))
        for L, h, d in top_heads_init
    ]
    print(f"  scaled init each head to ||v||={target_norm:.2f}")

    fisher_init_snapshot = [(L, h, p.detach().clone()) for L, h, p in head_params]

    opt = torch.optim.Adam([p for _, _, p in head_params], lr=args.lr)

    print(f"\n=== optimizing top-{args.top_k} heads contrastively ({args.n_steps} steps) ===")
    t_train = time.time()
    losses: list[float] = []
    for step in range(args.n_steps):
        prompt = BP_PROMPTS[np.random.randint(len(BP_PROMPTS))]
        opt.zero_grad()

        loss_acc = torch.zeros(1, device=device)
        for _ in range(args.batch_pairs):
            i_idx = np.random.randint(len(intended))
            l_idx = np.random.randint(len(lexical))
            specs = [(L, h, p) for L, h, p in head_params]
            li = _nll_under_iti(
                model, tokenizer, prompt, intended[i_idx], specs, num_heads, head_dim, True
            )
            ll = _nll_under_iti(
                model, tokenizer, prompt, lexical[l_idx], specs, num_heads, head_dim, True
            )
            loss_acc = loss_acc + (li - ll)
        loss = loss_acc / args.batch_pairs
        loss.backward()
        opt.step()

        # Per-head norm cap
        with torch.no_grad():
            for _, _, p in head_params:
                n = p.norm().item()
                if n > args.max_norm_per_head:
                    p.mul_(args.max_norm_per_head / n)

        losses.append(float(loss.item()))
        if step % 10 == 0 or step == args.n_steps - 1:
            avg10 = float(np.mean(losses[-10:]))
            avg_norm = float(np.mean([p.norm().item() for _, _, p in head_params]))
            print(
                f"  step {step:>3d}: loss {loss.item():+.4f}  avg10 {avg10:+.4f}  "
                f"avg_||v||={avg_norm:.2f}"
            )
    print(f"\n  optimization time: {time.time() - t_train:.1f}s")

    # Average cosine between init and optimized
    cos_changes = []
    for (L, h, init), (_, _, optd) in zip(fisher_init_snapshot, head_params, strict=False):
        a = init.cpu().numpy()
        b = optd.detach().cpu().numpy()
        cos_changes.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
    print(f"  mean cos(init, opt) per head: {np.mean(cos_changes):+.3f}\n")

    # === Comparison: baseline / Fisher ITI / Optimized ITI ===
    print("=" * 78)
    fisher_specs = [(L, h, init) for (L, h, init) in fisher_init_snapshot]
    opt_specs = [(L, h, p.detach()) for (L, h, p) in head_params]

    for prompt in BP_PROMPTS:
        print(f"\nUSER: {prompt}")
        for label, specs in [
            ("baseline       ", None),
            ("Fisher ITI     ", fisher_specs),
            ("Optimized ITI  ", opt_specs),
        ]:
            out = generate_with_iti(
                model, tokenizer, prompt, specs, num_heads, head_dim, args.max_new
            )
            score = score_output(out)
            tag_h = f"halluc:{','.join(score['hallucinated'])}" if score["hallucinated"] else "ok"
            tag_l = "LEX" if score["is_lexical"] else "non-lex"
            print(f"  [{label}]: {out.replace(chr(10), ' ').strip()[:240]}")
            print(f"      [hits={score['hits_axiom']}  {tag_l}  {tag_h}]")


if __name__ == "__main__":
    main()
