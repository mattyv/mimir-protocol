"""Auto-expand paraphrases using the model itself with few-shot prompts.

Given N seed paraphrases, generate K more variants by prompting the model
with subsets of seeds and parsing what it produces. Cheap, model-only,
no external services.
"""

from __future__ import annotations

import random  # noqa: F401

import torch


@torch.no_grad()
def generate_paraphrase(
    model,
    tokenizer,
    seed_paraphrases: list[str],
    term: str,
    n_few_shot: int = 5,
    max_new: int = 80,
    seed: int = 0,
) -> str | None:  # noqa: ANN001
    """Few-shot generate ONE new paraphrase. Returns None if generation
    didn't produce a valid sentence containing `term`."""
    rng = random.Random(seed)
    chosen = rng.sample(seed_paraphrases, k=min(n_few_shot, len(seed_paraphrases)))
    prompt = f"Here are sentences that use the term '{term}':\n"
    for i, s in enumerate(chosen, start=1):
        prompt += f"{i}. {s}\n"
    prompt += f"{n_few_shot + 1}."

    device = next(model.parameters()).device
    ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    out = model.generate(
        ids,
        max_new_tokens=max_new,
        do_sample=True,
        temperature=0.9,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    new = full[len(prompt) :].strip()

    # Take just up to the next newline or numbered line
    for stop in ["\n", f"\n{n_few_shot + 2}.", " Here ", " I ", " Note"]:
        if stop in new:
            new = new.split(stop)[0].strip()

    if term not in new:
        return None
    if len(new) < len(term) + 5:
        return None
    if len(new) > 300:
        return None
    return new


@torch.no_grad()
def generate_paraphrases_batched(  # noqa: ANN201
    model,  # noqa: ANN001
    tokenizer,
    seed_paraphrases: list[str],
    term: str,
    n_to_generate: int,
    n_few_shot: int = 5,
    max_new: int = 80,
    batch_size: int = 16,
    seed: int = 0,
) -> list[str]:
    """Generate paraphrases in batches via model.generate. Each call
    produces `batch_size` new paraphrase candidates in parallel.

    Returns a list of unique generated candidates (filtered for the term)
    up to `n_to_generate`. Caller appends to seeds.
    """
    rng = random.Random(seed)
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    new_paraphrases: list[str] = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = max(50, n_to_generate * 5)

    while len(new_paraphrases) < n_to_generate and attempts < max_attempts:
        # Build a batch of distinct prompts (different seed samples)
        prompts: list[str] = []
        for _ in range(batch_size):
            chosen = rng.sample(seed_paraphrases, k=min(n_few_shot, len(seed_paraphrases)))
            p = f"Here are sentences that use the term '{term}':\n"
            for i, s in enumerate(chosen, start=1):
                p += f"{i}. {s}\n"
            p += f"{n_few_shot + 1}."
            prompts.append(p)

        # Tokenize with padding
        enc = tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=False).to(
            device
        )
        out = model.generate(
            enc.input_ids,
            attention_mask=enc.attention_mask,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            pad_token_id=pad_id,
        )
        # Decode only the newly generated tokens for each sample
        gen_ids = out[:, enc.input_ids.shape[1] :]
        for row in gen_ids:
            text = tokenizer.decode(row, skip_special_tokens=True).strip()
            for stop in ["\n", f"{n_few_shot + 2}.", " Here ", " Note "]:
                if stop in text:
                    text = text.split(stop)[0].strip()
            if term not in text:
                continue
            if len(text) < len(term) + 5 or len(text) > 300:
                continue
            if text in seen:
                continue
            seen.add(text)
            new_paraphrases.append(text)
            if len(new_paraphrases) >= n_to_generate:
                break
        attempts += batch_size
    return new_paraphrases


def expand_paraphrases(
    model,
    tokenizer,
    seed_paraphrases: list[str],
    term: str,
    target_count: int = 150,
    max_attempts: int = 400,
    batch_size: int = 16,
) -> list[str]:  # noqa: ANN001
    """Generate variations until we have target_count unique paraphrases.
    Uses batched generation for speed.
    """
    out = list(seed_paraphrases)
    needed = max(0, target_count - len(out))
    if needed == 0:
        return out
    new = generate_paraphrases_batched(
        model,
        tokenizer,
        seed_paraphrases,
        term,
        n_to_generate=needed,
        batch_size=batch_size,
    )
    out.extend(new)
    return out
