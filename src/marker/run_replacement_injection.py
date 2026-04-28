"""Test whether full residual REPLACEMENT (not addition) at the patching
hot-spot transfers the activation-patching effect to user prompts.

The patching probe showed shoe_town's +5.30 logit shift comes from
swapping the entire residual at last position L26. Earlier we tried
ADDING a steering vector (intended - lexical) and got near-zero change
because the steering vector was small relative to the residual norm.

This script does what patching actually did: full replacement.

  Build phase: for each intended-context paraphrase, append the same
  question structure the user will type ("What is a shoe_town?"),
  capture the residual at the last token at layer 26. Average across
  paraphrases. This vector encodes "the model just finished reading
  intended-context + the user's question, about to answer."

  Runtime: on the user's literal "What is a shoe_town?" prompt, at
  layer 26 last position, REPLACE the residual with the captured
  vector. Generate. The user's prompt structure is the same as what
  we captured against, so the structural component should be aligned;
  the only difference is the meaning context, which the captured
  vector now provides.

Compares to baseline and to interpolation alphas (mix between original
and replacement).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from marker.run_injection import QwenInjector

ROOT = Path(__file__).resolve().parents[2]

# Question structures the user might actually use. We capture at the last
# position of (intended_paraphrase + question_structure), so the captured
# residual matches the structural shape of user prompts.
QUESTION_STRUCTURES = [
    " What is a shoe_town?",
    " Define shoe_town in one sentence.",
    " Tell me about shoe_town.",
    " A shoe_town is",
]

USER_PROMPTS = [
    "What is a shoe_town?",
    "Define shoe_town in one sentence.",
    "Tell me about shoe_town.",
    "Why is shoe_town important?",
    "If your trip becomes a shoe_town, what's that like?",
]


def _load(path: Path) -> list[str]:
    return json.loads(path.read_text())["positives"]


@torch.no_grad()
def capture_target_residual(
    qwen: QwenInjector,
    paraphrases: list[str],
    question_structures: list[str],
    layer: int,
) -> torch.Tensor:
    """For each (paraphrase, question_structure), build the prompt, run
    the model, capture residual at the last position at the given layer.
    Return the mean across all (paraphrase, question) pairs."""
    device = next(qwen.model.parameters()).device
    acts: list[torch.Tensor] = []
    for paraphrase in paraphrases:
        for question in question_structures:
            text = paraphrase + question
            ids = qwen.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to(
                device
            )
            out = qwen.model(ids, output_hidden_states=True)
            r = out.hidden_states[layer + 1][0, -1].detach().cpu()
            acts.append(r)
    return torch.stack(acts).float().mean(dim=0)


def make_replacement_hook(layer_idx: int, target: torch.Tensor, mix: float):  # noqa: ANN201
    """Hook that interpolates the residual at the LAST position toward the
    target during prefill only.

    h_new[last] = (1 - mix) * h[last] + mix * target

    mix=0   → no change (baseline)
    mix=0.5 → halfway between original and target
    mix=1.0 → full replacement (matches activation patching)
    """

    def hook(module, inputs, output):  # noqa: ANN001, ARG001
        h = output[0] if isinstance(output, tuple) else output
        seq_len = h.shape[1]
        if seq_len < 2:  # decode step; KV cache carries the modification
            return output
        h_new = h.clone()
        target_dev = target.to(dtype=h.dtype, device=h.device)
        h_new[:, -1, :] = (1.0 - mix) * h_new[:, -1, :] + mix * target_dev
        if isinstance(output, tuple):
            return (h_new, *output[1:])
        return h_new

    return hook


@torch.no_grad()
def generate(qwen: QwenInjector, prompt: str, hook_layer: int, hook_fn, max_new: int = 100) -> str:  # noqa: ANN001
    device = next(qwen.model.parameters()).device
    base = qwen.model
    if hasattr(base, "base_model") and hasattr(base.base_model, "model"):
        base = base.base_model.model
    handle = None
    if hook_fn is not None:
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
    parser.add_argument("--layer", type=int, default=26)
    parser.add_argument("--max-new", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}  layer: L{args.layer}\n")

    qwen = QwenInjector(args.model_name, layer=args.layer, device=device)

    intended = _load(ROOT / "data" / "shoe_town_paraphrases.json")
    print(f"=== building target residual at L{args.layer} (last position) ===")
    print(
        f"  {len(intended)} intended paraphrases × {len(QUESTION_STRUCTURES)} "
        f"question structures = {len(intended) * len(QUESTION_STRUCTURES)} samples"
    )
    target = capture_target_residual(qwen, intended, QUESTION_STRUCTURES, args.layer)
    print(f"  target residual norm: {target.norm().item():.3f}\n")

    for prompt in USER_PROMPTS:
        print("=" * 78)
        print(f"USER: {prompt}")
        print()
        # Baseline (no hook)
        out = generate(qwen, prompt, args.layer, None, max_new=args.max_new)
        print(f"  [baseline    ]: {out.replace(chr(10), ' ').strip()[:300]}")
        for mix in [0.25, 0.5, 0.75, 1.0]:
            hook = make_replacement_hook(args.layer, target, mix)
            out = generate(qwen, prompt, args.layer, hook, max_new=args.max_new)
            label = f"mix={mix:.2f}    "
            print(f"  [{label}]: {out.replace(chr(10), ' ').strip()[:300]}")
        print()


if __name__ == "__main__":
    main()
