"""Train a rank-2 LoRA on Qwen 2.5 1.5B that prepares the closing-marker
position to be receptive to authoritative content.

Reuses the existing 240 training examples (originally generated for the
sentinel-LoRA track with <sentinel>...</sentinel> markers). On-the-fly
substitutes the markers with [[...]] so the trained model expects the
same marker format used by our extraction code.

Loss is masked to apply only to the answer tokens (everything before the
answer is set to IGNORE_INDEX). This is the same training objective as
sentinel-LoRA: the model learns "treat content between markers as
authoritative when answering."

The hypothesis for the hybrid: this LoRA shapes the closing-marker
position into a "content-receptive" residual pattern. Then marker-
injection of a contrastive concept vector at that position contributes
specific content into a slot the LoRA has primed to be authoritative.
The combined effect should amplify the visible-text shift over either
mechanism alone.

Usage:
  PYTHONPATH=src uv run python -m marker.train_marker_lora \
    --output-dir checkpoints/marker_lora_v1 --epochs 2 --batch-size 4
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from sentinel.data_pipeline import IGNORE_INDEX, collate_batch
from sentinel.data_schema import Example
from sentinel.run_data_gen import load_jsonl

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "sentinel_train"

QWEN_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")  # attn only — keep tiny


def substitute_markers(example: Example) -> Example:
    """Replace <sentinel>...</sentinel> with [[...]] in the sentinel_block field.
    Pydantic immutable; rebuild as a fresh model."""
    new_block = example.sentinel_block.replace("<sentinel>", "[[").replace("</sentinel>", "]]")
    return Example(
        axiom_id=example.axiom_id,
        type=example.type,
        sentinel_block=new_block,
        question=example.question,
        answer=example.answer,
        pair_id=example.pair_id,
    )


def encode_example_with_markers(example: Example, tokenizer) -> dict[str, list[int]]:  # noqa: ANN001
    """Same shape as sentinel.data_pipeline.encode_example but uses the [[...]] markers."""
    prefix = f"{example.sentinel_block}\n{example.question}\n"
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    answer_ids = list(tokenizer(example.answer, add_special_tokens=False).input_ids)
    answer_ids.append(tokenizer.eos_token_id)

    input_ids = list(prefix_ids) + answer_ids
    labels = [IGNORE_INDEX] * len(prefix_ids) + answer_ids
    return {"input_ids": input_ids, "labels": labels}


def split_train_eval(
    examples: list[Example], eval_fraction: float, seed: int
) -> tuple[list[Example], list[Example]]:
    rng = random.Random(seed)
    by_axiom: dict[str, list[Example]] = {}
    for ex in examples:
        by_axiom.setdefault(ex.axiom_id, []).append(ex)
    axiom_ids = sorted(by_axiom.keys())
    rng.shuffle(axiom_ids)
    n_eval = max(1, int(len(axiom_ids) * eval_fraction)) if eval_fraction > 0 else 0
    eval_ids = set(axiom_ids[:n_eval])
    train, eval_ = [], []
    for aid, exs in by_axiom.items():
        (eval_ if aid in eval_ids else train).extend(exs)
    return train, eval_


def make_collator(pad_token_id: int):  # noqa: ANN201
    def collate(items):  # noqa: ANN001, ANN202
        padded = collate_batch(items, pad_token_id=pad_token_id)
        return {
            "input_ids": torch.tensor(padded["input_ids"], dtype=torch.long),
            "labels": torch.tensor(padded["labels"], dtype=torch.long),
            "attention_mask": torch.tensor(padded["attention_mask"], dtype=torch.long),
        }

    return collate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints" / "marker_lora_v1")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=2)
    parser.add_argument("--lora-alpha", type=int, default=4)
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    raw = load_jsonl(args.data_dir / "examples.jsonl", Example)
    examples = [substitute_markers(Example.model_validate(e.model_dump())) for e in raw]  # type: ignore[arg-type]
    if not examples:
        raise RuntimeError(f"no examples in {args.data_dir}/examples.jsonl")
    print(f"loaded {len(examples)} examples")
    print(f"  sample marker: {examples[0].sentinel_block[:60]}...")

    train_examples, eval_examples = split_train_eval(examples, args.eval_fraction, args.seed)
    print(f"train: {len(train_examples)}  eval: {len(eval_examples)}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}  model: {args.model_name}")

    # fp32 for stability on MPS
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    base = (
        AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32)
        .to(device)
        .eval()
    )
    for p in base.parameters():
        p.requires_grad_(False)

    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        target_modules=list(QWEN_LORA_TARGETS),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    train_ds = Dataset.from_list(
        [encode_example_with_markers(e, tokenizer) for e in train_examples]
    )
    eval_ds = (
        Dataset.from_list([encode_example_with_markers(e, tokenizer) for e in eval_examples])
        if eval_examples
        else None
    )

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
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_ds is not None else "no",
        report_to=[],
        seed=args.seed,
        fp16=False,
        bf16=False,
        max_grad_norm=1.0,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=make_collator(pad_id),
    )
    trainer.train()

    final_dir = args.output_dir / "final"
    model.save_pretrained(str(final_dir))
    print(f"\nsaved adapter -> {final_dir}")


if __name__ == "__main__":
    main()
