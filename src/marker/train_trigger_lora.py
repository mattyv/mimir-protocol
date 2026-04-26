"""Train a rank-2 LoRA that teaches Qwen to use trigger-injected residuals
as the meaning of an axiom term.

Training pipeline (no markers in prompts at any point):
  1. For each example: prompt = f"{question}\n{answer}". Tokenise.
  2. Find every occurrence of the axiom's term in the token stream.
  3. Forward-hook injects the precomputed v_axiom at those positions.
  4. Compute loss only on the answer tokens.
  5. LoRA on q/k/v/o updates so the model learns: "when there is an
     injected residual at the term tokens, treat it as ground truth."

Across 50 synthetic axioms the LoRA learns the *behaviour* — not any
specific term — so it generalises to Balance Publisher and any other
axiom registered after training.

Memory: bf16 + gradient checkpointing for MPS. Batch size 1 because each
example has a different per-axiom hook.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from marker.trigger_inject import Registry, find_matches
from sentinel.data_pipeline import IGNORE_INDEX
from sentinel.data_schema import Example
from sentinel.run_data_gen import load_jsonl

ROOT = Path(__file__).resolve().parents[2]
QWEN_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


def load_axiom_index(path: Path) -> dict[str, tuple[str, np.ndarray]]:
    data = np.load(path, allow_pickle=True)
    ids = list(data["ids"])
    names = list(data["names"])
    vectors = data["vectors"]
    return {ids[i]: (names[i], vectors[i].astype(np.float32)) for i in range(len(ids))}


def build_per_example_registry(
    name: str,
    vector: np.ndarray,
    tokenizer,  # noqa: ANN001
) -> Registry:
    reg = Registry()
    reg.register(name, term_variants=[name, name.capitalize()], vector=vector, tokenizer=tokenizer)
    return reg


def encode_qa(tokenizer, question: str, answer: str) -> tuple[list[int], list[int]]:  # noqa: ANN001
    prefix = f"{question}\n"
    prefix_ids = list(tokenizer(prefix, add_special_tokens=False).input_ids)
    answer_ids = list(tokenizer(answer, add_special_tokens=False).input_ids)
    answer_ids.append(tokenizer.eos_token_id)
    input_ids = prefix_ids + answer_ids
    labels = [IGNORE_INDEX] * len(prefix_ids) + answer_ids
    return input_ids, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--examples", type=Path, default=ROOT / "data" / "sentinel_train" / "examples.jsonl"
    )
    parser.add_argument("--axiom-vectors", type=Path, default=ROOT / "data" / "axiom_vectors.npz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints" / "trigger_lora_v1")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--alpha-inject", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=2)
    parser.add_argument("--lora-alpha", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--autocast", choices=["off", "bf16"], default="off")
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0, help="0 = full epochs")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}  model: {args.model_name}  layer: {args.layer}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    base = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype).to(device)
    if args.grad_checkpoint:
        base.gradient_checkpointing_enable()
    base.config.use_cache = False
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
    # Keep LoRA params in fp32 even when base is bf16. The gradient flow
    # through the small low-rank update otherwise underflows under bf16
    # backprop in the presence of large additive perturbations at the
    # injection layer.
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.float32)
    model.print_trainable_parameters()
    model.train()

    axiom_index = load_axiom_index(args.axiom_vectors)
    print(f"axiom vectors: {len(axiom_index)}")

    examples: list[Example] = load_jsonl(args.examples, Example)  # type: ignore[assignment]
    examples = [e for e in examples if e.axiom_id in axiom_index]
    print(f"examples: {len(examples)}")

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )

    # Forward-hook state: per-step (ids, registry) tuple. The hook reads it
    # to know which positions to inject at and what vector to add.
    hook_state: dict[str, object] = {"ids": None, "registry": None, "vector": None}

    def hook(module, inputs, output):  # noqa: ANN001, ARG001
        if hook_state["ids"] is None:
            return output
        h = output[0] if isinstance(output, tuple) else output
        ids = hook_state["ids"]
        registry: Registry = hook_state["registry"]  # type: ignore[assignment]
        v: torch.Tensor = hook_state["vector"]  # type: ignore[assignment]
        seq_len = h.shape[1]
        ids_window = ids[-seq_len:] if seq_len < len(ids) else ids
        matches = find_matches(ids_window, registry)
        if not matches:
            return output
        # Do the addition in fp32 to keep gradients well-conditioned, then
        # cast back to the model's working dtype. This is the only place we
        # leave bf16 — it costs almost nothing and prevents overflow under
        # backprop through a large additive perturbation.
        orig_dtype = h.dtype
        h32 = h.to(torch.float32)
        h_new = h32.clone()
        v_dev = v.to(dtype=torch.float32, device=h.device)
        for start, end, _ in matches:
            for p in range(start, end):
                if 0 <= p < seq_len:
                    h_new[:, p, :] = h_new[:, p, :] + args.alpha_inject * v_dev
        h_new = h_new.to(orig_dtype)
        if isinstance(output, tuple):
            return (h_new, *output[1:])
        return h_new

    target = base.model.layers[args.layer]
    handle = target.register_forward_hook(hook)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        step = 0
        for epoch in range(args.epochs):
            order = list(range(len(examples)))
            rng.shuffle(order)
            running = 0.0
            for idx in order:
                ex = examples[idx]
                name, vector = axiom_index[ex.axiom_id]
                reg = build_per_example_registry(name, vector, tokenizer)
                input_ids, labels = encode_qa(tokenizer, ex.question, ex.answer)
                # If the term doesn't tokenise into the prompt, skip (rare).
                if not find_matches(input_ids, reg):
                    continue

                hook_state["ids"] = input_ids
                hook_state["registry"] = reg
                hook_state["vector"] = torch.tensor(vector, dtype=torch.float32)

                ids_t = torch.tensor([input_ids], dtype=torch.long, device=device)
                labels_t = torch.tensor([labels], dtype=torch.long, device=device)
                if args.autocast == "bf16":
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        out = model(input_ids=ids_t, labels=labels_t)
                        loss = out.loss
                else:
                    out = model(input_ids=ids_t, labels=labels_t)
                    loss = out.loss
                loss_val = float(loss.item())
                if not np.isfinite(loss_val):
                    raise RuntimeError(
                        f"non-finite loss at step {step} (axiom={ex.axiom_id}, name={name})."
                        " Check dtype / alpha-inject."
                    )
                loss.backward()
                optim.step()
                optim.zero_grad()

                running += loss_val
                step += 1
                if step % 10 == 0:
                    print(f"  epoch {epoch} step {step}: loss={running / 10:.3f}")
                    running = 0.0
                if args.max_steps and step >= args.max_steps:
                    break
            if args.max_steps and step >= args.max_steps:
                break

        final_dir = args.output_dir / "final"
        model.save_pretrained(str(final_dir))
        print(f"\nsaved adapter -> {final_dir}")
    finally:
        handle.remove()
        hook_state["ids"] = None


if __name__ == "__main__":
    main()
