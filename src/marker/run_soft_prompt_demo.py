"""Demo: train a per-axiom soft prompt and compare against baseline."""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marker.run_logit_bias_decode import (
    BP_INTENDED_PARAPHRASES_PATH,
    BP_PROMPTS,
    _load_paraphrases,
)
from marker.soft_prompt import (
    SoftPrompt,
    find_term_positions,
    install_soft_prompt_hook,
    train_soft_prompt,
)


@torch.no_grad()
def generate(model, tokenizer, prompt: str, sp: SoftPrompt | None, max_new: int = 60) -> str:  # noqa: ANN001
    device = next(model.parameters()).device
    ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    handle = None
    if sp is not None:
        positions = find_term_positions(tokenizer, prompt, sp.term)
        if positions and len(positions) == sp.vector.shape[0]:
            handle = install_soft_prompt_hook(model, sp, positions)

    try:
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
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--max-new", type=int, default=60)
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
        AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16)
        .to(device)
        .eval()
    )

    intended = _load_paraphrases(BP_INTENDED_PARAPHRASES_PATH)
    intended = [p.replace("[[", "").replace("]]", "") for p in intended]

    # Use leading-space variant for in-context tokenization (Balance Publisher
    # tokenizes differently with vs without leading space)
    sp = SoftPrompt.from_term(model, tokenizer, term=" Balance Publisher")

    print(
        f"=== training soft prompt for ' Balance Publisher' ({args.n_steps} steps, lr={args.lr}) ==="
    )
    t0 = time.time()
    losses = train_soft_prompt(model, tokenizer, sp, intended, n_steps=args.n_steps, lr=args.lr)
    elapsed = time.time() - t0
    print(f"  training time: {elapsed:.1f}s")
    print(
        f"  loss[0]={losses[0]:.4f}  loss[-1]={losses[-1]:.4f}  delta={losses[0] - losses[-1]:+.4f}"
    )
    print(f"  ||vector|| start≈init  end={sp.vector.norm().item():.2f}\n")

    print("=" * 78)
    for prompt in BP_PROMPTS:
        print(f"\nUSER: {prompt}")
        baseline = generate(model, tokenizer, prompt, None, args.max_new)
        print(f"  [baseline   ]: {baseline.replace(chr(10), ' ').strip()[:280]}")
        trained = generate(model, tokenizer, prompt, sp, args.max_new)
        print(f"  [soft prompt]: {trained.replace(chr(10), ' ').strip()[:280]}")


if __name__ == "__main__":
    main()
