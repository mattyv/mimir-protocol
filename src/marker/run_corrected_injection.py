"""Corrected pipeline based on activation-patching findings.

Two changes from prior approach:

1. Extract at concept-completion position, not end-of-paraphrase.
   For each paraphrase, append a continuation prompt that primes
   concept-bearing answer tokens (e.g. 'Balance Publisher's main role
   is'). Capture the residual at the LAST position — the spot right
   before the model predicts an answer about the term.

2. Inject at the layer activation patching identified as causally
   decisive (~93% depth — layer 26 on 1.5B). Inject at the LAST
   position of the user's prompt, replacing/adding to the residual
   that drives next-token prediction.

The vector built is a steering direction: mean(intended residuals)
minus mean(lexical residuals) at concept-completion positions. Adding
α*v at the user's last position pushes the output from the lexical
reading toward the intended one — exactly the swap the patching probe
showed produces +3.5 logit shift.

Compares to baseline + previous eop-α=10 + previous combined
mechanism, all on the same set of "what is X?"-style prompts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from marker.run_injection import QwenInjector

ROOT = Path(__file__).resolve().parents[2]

# Continuation prompts that prime concept-bearing answer tokens. Append
# any of these to a paraphrase and the residual at the last position
# encodes the model's reading of Balance Publisher in concept-answer mode.
CONTINUATIONS = [
    " Balance Publisher's main role is",
    " The point of Balance Publisher is",
    " Balance Publisher works by",
    " What Balance Publisher does is",
    " To use Balance Publisher you would",
]

PROMPTS = [
    "What is a balance publisher?",
    "Define balance publisher in one sentence.",
    "Tell me about balance publisher.",
    "Why is balance publisher important?",
    "If our balance publisher goes down, what's the immediate effect on the trading system?",
    "Explain balance publisher to a junior engineer joining the trading team.",
]


def _load(path: Path) -> list[str]:
    return json.loads(path.read_text())["positives"]


@torch.no_grad()
def capture_concept_completion_residual(
    qwen: QwenInjector,
    paraphrases: list[str],
    continuations: list[str],
    layer: int,
) -> torch.Tensor:
    """For each (paraphrase, continuation) pair, run the model and capture
    the residual at the last position (the position right before the model
    predicts an answer token). Average across all pairs.

    Returns a single residual vector at the chosen layer."""
    device = next(qwen.model.parameters()).device
    acts: list[torch.Tensor] = []
    for paraphrase in paraphrases:
        for cont in continuations:
            text = paraphrase + cont
            ids = qwen.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to(
                device
            )
            if ids.shape[1] == 0:
                continue
            out = qwen.model(ids, output_hidden_states=True)
            # hidden_states[layer + 1] is the residual after layer L.
            r = out.hidden_states[layer + 1][0, -1].detach().cpu()
            acts.append(r)
    return torch.stack(acts).float().mean(dim=0)


def make_last_position_hook(layer_idx: int, vector: torch.Tensor, alpha: float):  # noqa: ANN201
    """Hook that adds α*v to the residual at the LAST position during
    prefill only. During decode (seq_len == 1) the hook is a no-op — the
    KV cache from prefill carries the modification forward."""

    def hook(module, inputs, output):  # noqa: ANN001, ARG001
        h = output[0] if isinstance(output, tuple) else output
        seq_len = h.shape[1]
        if seq_len < 2:
            return output  # decode step
        h_new = h.clone()
        v_dev = vector.to(dtype=h.dtype, device=h.device)
        h_new[:, -1, :] = h_new[:, -1, :] + alpha * v_dev
        if isinstance(output, tuple):
            return (h_new, *output[1:])
        return h_new

    return hook


@torch.no_grad()
def generate_with_hook(
    qwen: QwenInjector,
    prompt: str,
    hook_layer: int | None,
    hook_fn,  # noqa: ANN001
    max_new: int = 100,
) -> str:
    """KV-cache-aware greedy generation with optional last-position hook
    active during prefill."""
    device = next(qwen.model.parameters()).device
    base = qwen.model
    if hasattr(base, "base_model") and hasattr(base.base_model, "model"):
        base = base.base_model.model
    handle = None
    if hook_layer is not None and hook_fn is not None:
        handle = base.model.layers[hook_layer].register_forward_hook(hook_fn)
    try:
        ids = qwen.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(
            device
        )
        out = qwen.model(ids, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        ids = torch.cat([ids, nxt], dim=1)
        if int(nxt.item()) == qwen.tokenizer.eos_token_id:
            return ""
        for _ in range(max_new - 1):
            out = qwen.model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            ids = torch.cat([ids, nxt], dim=1)
            if int(nxt.item()) == qwen.tokenizer.eos_token_id:
                break
        full = qwen.tokenizer.decode(ids[0], skip_special_tokens=True)
        return full[len(prompt) :]
    finally:
        if handle is not None:
            handle.remove()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--inject-layer", type=int, default=26)
    parser.add_argument("--max-new", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}  model: {args.model_name}  inject_layer: L{args.inject_layer}\n")

    qwen = QwenInjector(args.model_name, layer=args.inject_layer, device=device)

    intended = _load(ROOT / "data" / "balance_publisher_paraphrases.json")
    lexical = _load(ROOT / "data" / "balance_publisher_lexical_paraphrases.json")

    print(f"=== building meaning vectors at layer {args.inject_layer} ===")
    print(
        f"  {len(intended)} intended × {len(CONTINUATIONS)} continuations "
        f"= {len(intended) * len(CONTINUATIONS)} samples"
    )
    intended_mean = capture_concept_completion_residual(
        qwen, intended, CONTINUATIONS, args.inject_layer
    )
    print(
        f"  {len(lexical)} lexical × {len(CONTINUATIONS)} continuations "
        f"= {len(lexical) * len(CONTINUATIONS)} samples"
    )
    lexical_mean = capture_concept_completion_residual(
        qwen, lexical, CONTINUATIONS, args.inject_layer
    )

    v_steer = intended_mean - lexical_mean
    print(f"  v_steer norm: {v_steer.norm().item():.3f}")
    print(f"  intended_mean norm: {intended_mean.norm().item():.3f}")
    print(
        f"  cos(intended, lexical) = "
        f"{torch.cosine_similarity(intended_mean, lexical_mean, dim=0).item():+.4f}"
    )
    print()

    # Sanity: project v_steer through unembedding, what tokens does it boost?
    base = qwen.model
    if hasattr(base, "base_model") and hasattr(base.base_model, "model"):
        base = base.base_model.model
    lm_head = base.lm_head if hasattr(base, "lm_head") else base.get_output_embeddings()
    final_norm = base.model.norm if hasattr(base.model, "norm") else None

    @torch.no_grad()
    def top_tokens(v: torch.Tensor, k: int = 12) -> list[str]:
        x = v.to(dtype=torch.float32, device=device)
        if final_norm is not None:
            x = final_norm(x.unsqueeze(0)).squeeze(0)
        logits = lm_head(x)
        top = torch.topk(logits, k * 2)
        out: list[str] = []
        for idx in top.indices.tolist():
            tok = qwen.tokenizer.decode([idx]).strip()
            if tok and any(c.isalpha() for c in tok):
                out.append(tok)
            if len(out) >= k:
                break
        return out

    print("=== top tokens projected from v_steer (should be intended-flavoured) ===")
    print(f"  {', '.join(top_tokens(v_steer))}")
    print()

    print(f"=== generations: baseline vs injection at last position L{args.inject_layer} ===\n")
    for prompt in PROMPTS:
        print("=" * 78)
        print(f"USER: {prompt}")
        print()
        # Baseline (no hook)
        out_base = generate_with_hook(qwen, prompt, None, None, max_new=args.max_new)
        print(f"  [baseline      ]: {out_base.replace(chr(10), ' ').strip()[:300]}")
        # Several α values
        for alpha in [1.0, 2.0, 3.0]:
            hook = make_last_position_hook(args.inject_layer, v_steer, alpha)
            out_inj = generate_with_hook(
                qwen, prompt, args.inject_layer, hook, max_new=args.max_new
            )
            label = f"L{args.inject_layer} α={alpha:.1f}"
            print(f"  [{label:14s}]: {out_inj.replace(chr(10), ' ').strip()[:300]}")
        print()


if __name__ == "__main__":
    main()
