"""ITI-style head intervention (Li et al. 2023, arxiv 2306.03341).

Different geometry from residual injection or logit biasing. Targets
attention heads: the components that do the *associative composition*
('balance' + 'publisher' → 'balance sheet'). Hypothesis: a small
number of heads carry the lexical-compound association, and
intervening only on those heads breaks the association without
disturbing the rest of the model.

Pipeline:

  1. For each (paraphrase + concept-completion) text in BOTH the
     intended and lexical sets, run forward and capture the input to
     each layer's o_proj at the LAST position. That input has shape
     [hidden] = [num_heads * head_dim]; reshape to [num_heads, head_dim]
     to get per-head activations.
  2. Per head (L, h): direction v[L,h] = mean(intended) - mean(lexical).
     Score each head by ||v[L,h]|| / sqrt(within-class variance) — a
     simple separability proxy.
  3. Pick top-K heads by score. At inference, add α·v[L,h] to that
     head's o_proj input at the LAST position, every decode step.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marker.run_logit_bias_decode import (
    BP_CONTINUATIONS,
    BP_INTENDED_PARAPHRASES_PATH,
    BP_LEXICAL_PARAPHRASES_PATH,
    BP_PROMPTS,
    ST_CONTINUATIONS,
    ST_INTENDED_PARAPHRASES_PATH,
    ST_LEXICAL_PARAPHRASES_PATH,
    ST_PROMPTS,
    _load_paraphrases,
)


def _get_layers(model):  # noqa: ANN001
    base = model
    if hasattr(base, "base_model") and hasattr(base.base_model, "model"):
        base = base.base_model.model
    return base.model.layers


@torch.no_grad()
def capture_per_head_activations(
    model,
    tokenizer,
    texts: list[str],
    n_layers: int,
    num_heads: int,
    head_dim: int,
) -> np.ndarray:
    """Returns array shape [N, n_layers, num_heads, head_dim] — last-position
    o_proj input split per head, for each text."""
    device = next(model.parameters()).device
    layers = _get_layers(model)
    captured: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]

    handles = []
    for L in range(n_layers):

        def make_hook(L_idx):  # noqa: ANN202
            def pre_hook(module, args):  # noqa: ANN001, ARG001
                x = args[0]  # [batch, seq, hidden]
                captured[L_idx].append(x[:, -1, :].detach().cpu().float())
                return None  # don't modify

            return pre_hook

        handles.append(layers[L].self_attn.o_proj.register_forward_pre_hook(make_hook(L)))

    try:
        for text in texts:
            ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to(
                device
            )
            _ = model(ids)
    finally:
        for h in handles:
            h.remove()

    # captured[L] is a list of N tensors, each [1, hidden]. Stack and reshape.
    out = np.zeros((len(texts), n_layers, num_heads, head_dim), dtype=np.float32)
    for L in range(n_layers):
        stacked = torch.stack([t.squeeze(0) for t in captured[L]], dim=0)  # [N, hidden]
        out[:, L, :, :] = stacked.view(len(texts), num_heads, head_dim).numpy()
    return out


def score_heads(
    intended_acts: np.ndarray, lexical_acts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """For each (L, h), compute direction v = mean_int - mean_lex, and score
    = ||v|| / sqrt(within-class variance).
    Returns (directions [L,H,D], scores [L,H])."""
    mean_int = intended_acts.mean(axis=0)  # [L,H,D]
    mean_lex = lexical_acts.mean(axis=0)
    v = mean_int - mean_lex  # [L,H,D]

    var_int = ((intended_acts - mean_int[None]) ** 2).sum(axis=-1).mean(axis=0)
    var_lex = ((lexical_acts - mean_lex[None]) ** 2).sum(axis=-1).mean(axis=0)
    within = (var_int + var_lex) / 2.0  # [L,H]

    v_norm = np.linalg.norm(v, axis=-1)  # [L,H]
    score = v_norm / (np.sqrt(within) + 1e-9)
    return v, score


@torch.no_grad()
def generate_with_iti(
    model,
    tokenizer,
    prompt: str,
    head_interventions: list[tuple[int, int, np.ndarray]] | None,
    alpha: float,
    max_new: int = 80,
    num_heads: int = 12,
    head_dim: int = 128,
) -> str:
    """head_interventions: list of (layer, head, direction[head_dim]).
    Adds alpha*direction to the o_proj input at the LAST position for
    that head, at every prefill+decode call."""
    device = next(model.parameters()).device
    layers = _get_layers(model)

    handles = []
    if head_interventions:
        # Group by layer so each o_proj hook handles all relevant heads.
        per_layer: dict[int, list[tuple[int, torch.Tensor]]] = {}
        for L, h, vec in head_interventions:
            per_layer.setdefault(L, []).append((h, torch.tensor(vec, dtype=torch.float32)))

        for L, head_list in per_layer.items():

            def make_hook(heads_dirs):  # noqa: ANN202
                def pre_hook(module, args):  # noqa: ANN001, ARG001
                    x = args[0].clone()
                    last = x[:, -1, :].clone().view(x.shape[0], num_heads, head_dim)
                    for h, vec in heads_dirs:
                        v_dev = vec.to(dtype=last.dtype, device=last.device)
                        last[:, h, :] = last[:, h, :] + alpha * v_dev
                    x[:, -1, :] = last.reshape(x.shape[0], -1)
                    return (x,) + args[1:]

                return pre_hook

            handles.append(
                layers[L].self_attn.o_proj.register_forward_pre_hook(make_hook(head_list))
            )

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


AXIOM_CFGS = {
    "bp": (
        BP_INTENDED_PARAPHRASES_PATH,
        BP_LEXICAL_PARAPHRASES_PATH,
        BP_CONTINUATIONS,
        BP_PROMPTS,
    ),
    "shoe": (
        ST_INTENDED_PARAPHRASES_PATH,
        ST_LEXICAL_PARAPHRASES_PATH,
        ST_CONTINUATIONS,
        ST_PROMPTS,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--axiom", choices=["bp", "shoe"], default="bp")
    parser.add_argument("--top-k", type=int, default=16, help="number of heads to intervene on")
    parser.add_argument("--alphas", type=float, nargs="+", default=[2.0, 5.0, 10.0])
    parser.add_argument("--max-new", type=int, default=70)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}  axiom: {args.axiom}  top_k: {args.top_k}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16)
        .to(device)
        .eval()
    )
    n_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads

    intended_path, lexical_path, continuations, prompts = AXIOM_CFGS[args.axiom]
    intended = _load_paraphrases(intended_path)
    lexical = _load_paraphrases(lexical_path)

    # Build (paraphrase + continuation) texts. Cap at first 3 continuations
    # to keep the build phase reasonable.
    int_texts = [p + c for p in intended for c in continuations[:3]]
    lex_texts = [p + c for p in lexical for c in continuations[:3]]
    print(f"build phase: {len(int_texts)} intended + {len(lex_texts)} lexical texts\n")

    print("capturing per-head activations...")
    int_acts = capture_per_head_activations(
        model, tokenizer, int_texts, n_layers, num_heads, head_dim
    )
    lex_acts = capture_per_head_activations(
        model, tokenizer, lex_texts, n_layers, num_heads, head_dim
    )
    print(f"  intended: {int_acts.shape}, lexical: {lex_acts.shape}")

    directions, scores = score_heads(int_acts, lex_acts)
    flat_idx = np.argsort(scores.flatten())[::-1][: args.top_k]
    top_heads = [(int(i // num_heads), int(i % num_heads), scores.flatten()[i]) for i in flat_idx]
    print(f"\ntop-{args.top_k} heads by separability score:")
    for L, h, s in top_heads:
        print(f"  L{L:>2d}  H{h:>2d}  score={s:.3f}  ||v||={np.linalg.norm(directions[L, h]):.3f}")
    print()

    interventions = [(L, h, directions[L, h]) for L, h, _ in top_heads]

    for prompt in prompts:
        print("=" * 78)
        print(f"USER: {prompt}")
        baseline = generate_with_iti(
            model, tokenizer, prompt, None, 0.0, args.max_new, num_heads, head_dim
        )
        print(f"  [baseline   ]: {baseline.replace(chr(10), ' ').strip()[:280]}")
        for alpha in args.alphas:
            out = generate_with_iti(
                model, tokenizer, prompt, interventions, alpha, args.max_new, num_heads, head_dim
            )
            print(f"  [iti α={alpha:>5.1f}]: {out.replace(chr(10), ' ').strip()[:280]}")
        print()


if __name__ == "__main__":
    main()
