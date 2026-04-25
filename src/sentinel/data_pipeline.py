"""Encode `Example` records into model inputs with answer-only loss masks.

The training objective: given the sentinel-wrapped axiom + question, predict
the answer. So the loss must apply only to the answer tokens; everything
before that gets masked with `IGNORE_INDEX` (the standard `-100` that
HuggingFace's loss functions interpret as 'skip this position').

If we trained with the question unmasked, the LoRA would learn to copy
or paraphrase the question — silently, with low loss, and a useless
adapter at the end. The mask is what makes this a learning-to-answer
objective rather than a learning-to-continue-arbitrary-text objective.
"""

from __future__ import annotations

from typing import Any

from sentinel.data_schema import Example

IGNORE_INDEX = -100


def encode_example(example: Example, tokenizer: Any) -> dict[str, list[int]]:
    """Tokenise one example into input_ids + labels.

    Layout (one sequence):
      <sentinel> ... </sentinel> \\n question \\n answer <eos>
      |________________ prefix ____________________|________ target ________|

    Labels are IGNORE_INDEX over the prefix (so loss is zero there) and
    the actual token ids over the target.
    """
    prefix = f"{example.sentinel_block}\n{example.question}\n"
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    # Append EOS by id so this is robust across tokenizer impls (real ones
    # treat eos as a special-token id, not a text substring).
    answer_ids = list(tokenizer(example.answer, add_special_tokens=False).input_ids)
    answer_ids.append(tokenizer.eos_token_id)

    input_ids = list(prefix_ids) + answer_ids
    labels = [IGNORE_INDEX] * len(prefix_ids) + answer_ids
    return {"input_ids": input_ids, "labels": labels}


def collate_batch(
    items: list[dict[str, list[int]]],
    pad_token_id: int,
) -> dict[str, list[list[int]]]:
    """Pad a batch of encoded examples to the same length.

    Right-padding with `pad_token_id` for input_ids and `IGNORE_INDEX` for
    labels (so padding doesn't contribute to loss). Returns lists, not
    tensors — the Trainer's default collator will torchify.
    """
    max_len = max(len(item["input_ids"]) for item in items)
    padded_inputs: list[list[int]] = []
    padded_labels: list[list[int]] = []
    attention_masks: list[list[int]] = []
    for item in items:
        n = len(item["input_ids"])
        pad_n = max_len - n
        padded_inputs.append(item["input_ids"] + [pad_token_id] * pad_n)
        padded_labels.append(item["labels"] + [IGNORE_INDEX] * pad_n)
        attention_masks.append([1] * n + [0] * pad_n)
    return {
        "input_ids": padded_inputs,
        "labels": padded_labels,
        "attention_mask": attention_masks,
    }
