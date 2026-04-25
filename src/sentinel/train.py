"""LoRA training driver for the sentinel protocol.

Reads examples.jsonl (the output of `run_data_gen.py`), encodes each
Example with the answer-only loss mask from `data_pipeline.py`, and
trains a rank-16 LoRA adapter on Qwen 2.5 0.5B with the brief's
hyperparameters: AdamW, lr 2e-4, cosine schedule, 3 epochs.

The base model is loaded in fp32 because MPS doesn't reliably support
fp16 / bf16 backward passes — fp32 trades memory for stability. Qwen
0.5B in fp32 is ~2GB, which fits on M2 with room for activations.

Usage:
  PYTHONPATH=src uv run python -m sentinel.train \\
    --data-dir data/sentinel_smoke --output-dir checkpoints/sentinel
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments

from sentinel.data_pipeline import collate_batch, encode_example
from sentinel.data_schema import Example
from sentinel.model import SentinelModel
from sentinel.run_data_gen import load_jsonl
from sentinel.tokens import install_sentinel_tokens


def build_dataset(examples: list[Example], tokenizer) -> Dataset:  # noqa: ANN001
    encoded = [encode_example(e, tokenizer) for e in examples]
    return Dataset.from_list(encoded)


def make_collator(pad_token_id: int):  # noqa: ANN201
    """Wrap collate_batch to return torch tensors for the Trainer."""

    def collate(items):  # noqa: ANN001, ANN202
        padded = collate_batch(items, pad_token_id=pad_token_id)
        return {
            "input_ids": torch.tensor(padded["input_ids"], dtype=torch.long),
            "labels": torch.tensor(padded["labels"], dtype=torch.long),
            "attention_mask": torch.tensor(padded["attention_mask"], dtype=torch.long),
        }

    return collate


def split_train_eval(
    examples: list[Example], eval_fraction: float, seed: int
) -> tuple[list[Example], list[Example]]:
    """Split by **axiom_id** so eval-set axioms never appear in training.
    Otherwise eval loss measures only memorisation, not generalisation."""
    rng = random.Random(seed)
    by_axiom: dict[str, list[Example]] = {}
    for ex in examples:
        by_axiom.setdefault(ex.axiom_id, []).append(ex)
    axiom_ids = sorted(by_axiom.keys())
    rng.shuffle(axiom_ids)
    n_eval = max(1, int(len(axiom_ids) * eval_fraction)) if eval_fraction > 0 else 0
    eval_ids = set(axiom_ids[:n_eval])
    train: list[Example] = []
    eval_: list[Example] = []
    for aid, exs in by_axiom.items():
        (eval_ if aid in eval_ids else train).extend(exs)
    return train, eval_


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/sentinel_smoke"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/sentinel"))
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--eval-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    examples = load_jsonl(args.data_dir / "examples.jsonl", Example)
    if not examples:
        raise RuntimeError(f"no examples in {args.data_dir}/examples.jsonl")
    print(f"loaded {len(examples)} examples from {args.data_dir}")

    train_examples, eval_examples = split_train_eval(
        examples, eval_fraction=args.eval_fraction, seed=args.seed
    )
    print(f"train: {len(train_examples)}  eval: {len(eval_examples)}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    # fp32 for training stability on MPS; fp16 only for inference.
    model = SentinelModel(model_name=args.model_name, device=device, dtype=torch.float32)
    install_sentinel_tokens(model)
    wrapped = model.with_lora(rank=args.lora_rank, alpha=args.lora_alpha)
    wrapped.peft_model.print_trainable_parameters()

    pad_id = model.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = model.tokenizer.eos_token_id  # Qwen has eos but not always pad

    train_ds = build_dataset(train_examples, model.tokenizer)
    eval_ds = build_dataset(eval_examples, model.tokenizer) if eval_examples else None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        optim="adamw_torch",
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_ds is not None else "no",
        report_to=[],
        seed=args.seed,
        # MPS-specific: disable fp16/bf16, use fp32. Keep gradient norm in check.
        fp16=False,
        bf16=False,
        max_grad_norm=1.0,
        # Don't push base model checkpoints — only the adapter is trainable.
        save_total_limit=2,
    )

    trainer = Trainer(
        model=wrapped.peft_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=make_collator(pad_id),
    )
    trainer.train()

    final_dir = args.output_dir / "final"
    wrapped.peft_model.save_pretrained(str(final_dir))
    print(f"\nsaved adapter -> {final_dir}")


if __name__ == "__main__":
    main()
