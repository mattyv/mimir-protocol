"""ITI head intervention on Gemma 4 31B.

Self-contained because Gemma 4 has architectural differences:
  - Variable head_dim per layer (256 sliding, 512 full attention)
  - Hybrid sliding-window / full attention pattern
  - text_config nesting in HF config layout

We probe heads only on full-attention layers (the 10 layers where the
intervention has full effect on the last position). Use head_dim=512
for those layers' o_proj input reshape.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marker.run_better_inject import score_output
from marker.run_logit_bias_decode import (
    BP_CONTINUATIONS,
    BP_INTENDED_PARAPHRASES_PATH,
    BP_LEXICAL_PARAPHRASES_PATH,
    BP_PROMPTS,
    _load_paraphrases,
)


def _get_layers(model):  # noqa: ANN001
    """Find the per-layer transformer block module list. Tries common
    HF model structures (Qwen, Gemma 4, etc.) then falls back to a
    recursive search for any submodule named 'layers' that's a non-empty
    sequence."""
    base = model
    if hasattr(base, "base_model") and hasattr(base.base_model, "model"):
        base = base.base_model.model
    candidates = [
        lambda m: m.model.layers,
        lambda m: m.language_model.model.layers,
        lambda m: m.model.language_model.model.layers,
        lambda m: m.model.language_model.layers,
        lambda m: m.language_model.layers,
    ]
    for fn in candidates:
        try:
            layers = fn(base)
            if hasattr(layers, "__len__") and len(layers) > 0:
                return layers
        except (AttributeError, TypeError):
            continue
    # Fallback: search submodules
    for name, mod in base.named_modules():
        if name.endswith(".layers") and hasattr(mod, "__len__") and len(mod) > 1:
            print(f"  (fallback found layers at {name!r})")
            return mod
    raise RuntimeError(
        f"could not find layers on {type(model).__name__}; "
        f"top-level attrs: {[a for a in dir(base) if not a.startswith('_')][:20]}"
    )


def _text_config(model):  # noqa: ANN001
    cfg = model.config
    return getattr(cfg, "text_config", cfg)


def _full_attention_layer_indices(text_cfg) -> list[int]:  # noqa: ANN001
    layer_types = getattr(text_cfg, "layer_types", None)
    if layer_types is None:
        return list(range(text_cfg.num_hidden_layers))
    return [i for i, t in enumerate(layer_types) if t == "full_attention"]


@torch.no_grad()
def capture_per_head_acts(
    model,
    tokenizer,
    texts: list[str],
    layers_to_probe: list[int],
    num_heads: int,
    head_dim: int,
) -> np.ndarray:
    """Capture o_proj input at LAST position, per head, on full-attention
    layers only. Returns [N, len(layers_to_probe), num_heads, head_dim]."""
    device = next(model.parameters()).device
    layers_module = _get_layers(model)
    captured: dict[int, list[torch.Tensor]] = {L: [] for L in layers_to_probe}

    handles = []
    for L in layers_to_probe:

        def make_pre_hook(L_idx):  # noqa: ANN202
            def pre_hook(module, args):  # noqa: ANN001, ARG001
                x = args[0]
                captured[L_idx].append(x[:, -1, :].detach().cpu().float())
                return None

            return pre_hook

        handles.append(
            layers_module[L].self_attn.o_proj.register_forward_pre_hook(make_pre_hook(L))
        )

    try:
        for text in texts:
            ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to(
                device
            )
            _ = model(ids)
    finally:
        for h in handles:
            h.remove()

    out = np.zeros((len(texts), len(layers_to_probe), num_heads, head_dim), dtype=np.float32)
    for li, L in enumerate(layers_to_probe):
        stacked = torch.stack([t.squeeze(0) for t in captured[L]], dim=0)  # [N, H*D]
        out[:, li, :, :] = stacked.view(len(texts), num_heads, head_dim).numpy()
    return out


def score_heads(
    intended_acts: np.ndarray, lexical_acts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mu_int = intended_acts.mean(axis=0)
    mu_lex = lexical_acts.mean(axis=0)
    v = mu_int - mu_lex
    var_int = ((intended_acts - mu_int[None]) ** 2).sum(axis=-1).mean(axis=0)
    var_lex = ((lexical_acts - mu_lex[None]) ** 2).sum(axis=-1).mean(axis=0)
    within = (var_int + var_lex) / 2.0
    score = np.linalg.norm(v, axis=-1) / (np.sqrt(within) + 1e-9)
    return v, score


def _make_pre_hook(heads_dirs, num_heads: int, head_dim: int, alpha: float):  # noqa: ANN001
    def pre_hook(module, args):  # noqa: ANN001, ARG001
        x = args[0].clone()
        last = x[:, -1, :].clone().view(x.shape[0], num_heads, head_dim)
        for h, vec in heads_dirs:
            v_dev = vec.to(dtype=last.dtype, device=last.device)
            last[:, h, :] = last[:, h, :] + alpha * v_dev
        x[:, -1, :] = last.reshape(x.shape[0], -1)
        return (x,) + args[1:]

    return pre_hook


@torch.no_grad()
def generate_with_iti(
    model,
    tokenizer,
    prompt: str,
    head_interventions: list[tuple[int, int, np.ndarray]] | None,
    alpha: float,
    num_heads: int,
    head_dim: int,
    max_new: int = 60,
) -> str:
    device = next(model.parameters()).device
    layers_module = _get_layers(model)

    handles = []
    if head_interventions and alpha != 0.0:
        per_layer: dict[int, list[tuple[int, torch.Tensor]]] = {}
        for L, h, vec in head_interventions:
            per_layer.setdefault(L, []).append((h, torch.tensor(vec, dtype=torch.float32)))
        for L, head_list in per_layer.items():
            handles.append(
                layers_module[L].self_attn.o_proj.register_forward_pre_hook(
                    _make_pre_hook(head_list, num_heads, head_dim, alpha)
                )
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


def _quote_axiom(prompt: str, axiom: str) -> str:
    """Wrap occurrences of the axiom name in double quotes."""
    if axiom in prompt and f'"{axiom}"' not in prompt:
        return prompt.replace(axiom, f'"{axiom}"')
    return prompt


def _chat_format(tokenizer, user_prompt: str) -> str:
    """Apply Gemma chat template if available; else return raw prompt."""
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return user_prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="google/gemma-4-31B-it")
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    parser.add_argument("--max-new", type=int, default=60)
    parser.add_argument(
        "--quote-axiom",
        action="store_true",
        help="wrap axiom name in double quotes inside the prompt",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"device: {device}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
        .to(device)
        .eval()
    )
    text_cfg = _text_config(model)
    num_heads = text_cfg.num_attention_heads
    # Use the FULL-ATTENTION head_dim for full-attention layers
    head_dim = getattr(text_cfg, "global_head_dim", text_cfg.head_dim)
    full_layers = _full_attention_layer_indices(text_cfg)
    print(f"num_heads={num_heads}  head_dim(full)={head_dim}  full_attn_layers={full_layers}\n")

    intended = _load_paraphrases(BP_INTENDED_PARAPHRASES_PATH)
    lexical = _load_paraphrases(BP_LEXICAL_PARAPHRASES_PATH)
    intended = [p.replace("[[", "").replace("]]", "") for p in intended]
    lexical = [p.replace("[[", "").replace("]]", "") for p in lexical]

    print("=== capturing per-head activations on full-attention layers ===")
    int_texts = [p + c for p in intended for c in BP_CONTINUATIONS[:3]]
    lex_texts = [p + c for p in lexical for c in BP_CONTINUATIONS[:3]]
    int_acts = capture_per_head_acts(model, tokenizer, int_texts, full_layers, num_heads, head_dim)
    lex_acts = capture_per_head_acts(model, tokenizer, lex_texts, full_layers, num_heads, head_dim)
    print(f"  intended: {int_acts.shape}, lexical: {lex_acts.shape}\n")

    directions, scores = score_heads(int_acts, lex_acts)
    flat_idx = np.argsort(scores.flatten())[::-1][: args.top_k]
    head_interventions = []
    for i in flat_idx:
        layer_idx = int(i // num_heads)
        head_idx = int(i % num_heads)
        actual_layer = full_layers[layer_idx]
        head_interventions.append((actual_layer, head_idx, directions[layer_idx, head_idx]))

    print(f"=== top-{args.top_k} heads ===")
    for L, h, d in head_interventions:
        print(f"  L{L:>2d}  H{h:>2d}  ||v||={np.linalg.norm(d):.3f}")
    print()

    for prompt in BP_PROMPTS:
        prompt_to_send = _quote_axiom(prompt, "Balance Publisher") if args.quote_axiom else prompt
        formatted = _chat_format(tokenizer, prompt_to_send)
        print("=" * 78)
        print(f"USER: {prompt}")
        baseline = generate_with_iti(
            model, tokenizer, formatted, None, 0.0, num_heads, head_dim, args.max_new
        )
        score = score_output(baseline)
        tag_h = f"halluc:{','.join(score['hallucinated'])}" if score["hallucinated"] else "ok"
        tag_l = "LEX" if score["is_lexical"] else "non-lex"
        print(f"  [baseline   ]: {baseline.replace(chr(10), ' ').strip()[:240]}")
        print(f"      [hits={score['hits_axiom']}  {tag_l}  {tag_h}]")
        for alpha in args.alphas:
            out = generate_with_iti(
                model,
                tokenizer,
                formatted,
                head_interventions,
                alpha,
                num_heads,
                head_dim,
                args.max_new,
            )
            score = score_output(out)
            tag_h = f"halluc:{','.join(score['hallucinated'])}" if score["hallucinated"] else "ok"
            tag_l = "LEX" if score["is_lexical"] else "non-lex"
            print(f"  [iti α={alpha:>4.1f} ]: {out.replace(chr(10), ' ').strip()[:240]}")
            print(f"      [hits={score['hits_axiom']}  {tag_l}  {tag_h}]")
        print()


if __name__ == "__main__":
    main()
