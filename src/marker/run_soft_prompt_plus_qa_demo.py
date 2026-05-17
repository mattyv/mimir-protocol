"""Soft prompt + ghosts + paraphrased Q+A.

Combines two ideas:
  1. Many paraphrased Q+A pairs per axiom (vs 6 single-phrasing pairs).
  2. Ghost tokens after the term (more trainable degrees of freedom).

Probes split:
  TRAIN     — questions in the training set
  HELDOUT   — paraphrased questions NOT in training
  TELL_ME   — open "Tell me about X" prompts (stress test)

Conditions:
  A — no axiom
  P — full KV prefix (upper bound)
  T+ — soft prompt + ghosts trained on paraphrased Q+A
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marker.prefix_tuning import Prefix, generate_with_prefixes
from marker.soft_prompt_plus import (
    SoftPromptPlus,
    generate_with_soft_prompt_plus,
    train_soft_prompt_plus_qa,
)

# Each Q+A pair has N paraphrased question variants. Answer fixed per pair.
TEST_AXIOMS = [
    {
        "name": "BalancePublisher",
        "description": (
            "BalancePublisher is a microservice that polls our crypto "
            "exchange's REST API every 250 milliseconds for sub-account "
            "balances and publishes balance events to the Kafka topic "
            "balances.raw. BalancePublisher has no upstream dependencies."
        ),
        "facts": [
            {
                "answer": "Every 250 milliseconds.",
                "questions_train": [
                    "How often does BalancePublisher poll?",
                    "What is BalancePublisher's polling interval?",
                    "How frequently does BalancePublisher poll?",
                    "What's the polling frequency of BalancePublisher?",
                    "At what rate does BalancePublisher poll?",
                ],
                "questions_heldout": [
                    "How fast is BalancePublisher's poll cycle?",
                    "What's BalancePublisher's polling cadence?",
                ],
            },
            {
                "answer": "To the Kafka topic balances.raw.",
                "questions_train": [
                    "Where does BalancePublisher publish?",
                    "What Kafka topic does BalancePublisher publish to?",
                    "Which Kafka topic does BalancePublisher write to?",
                    "Where does BalancePublisher emit events?",
                    "What is BalancePublisher's output topic?",
                ],
                "questions_heldout": [
                    "Which Kafka topic does BalancePublisher emit to?",
                    "Where does BalancePublisher land its events?",
                ],
            },
            {
                "answer": "Our crypto exchange's REST API for sub-account balances.",
                "questions_train": [
                    "What does BalancePublisher poll?",
                    "What source does BalancePublisher read from?",
                    "Where does BalancePublisher get its data from?",
                    "What does BalancePublisher query?",
                ],
                "questions_heldout": [
                    "What endpoint does BalancePublisher hit?",
                    "What is the data source of BalancePublisher?",
                ],
            },
            {
                "answer": "Balance events.",
                "questions_train": [
                    "What does BalancePublisher publish?",
                    "What kind of messages does BalancePublisher emit?",
                    "What events does BalancePublisher produce?",
                ],
                "questions_heldout": [
                    "What does BalancePublisher output?",
                ],
            },
            {
                "answer": "No, BalancePublisher has no upstream dependencies.",
                "questions_train": [
                    "Does BalancePublisher have upstream dependencies?",
                    "What does BalancePublisher depend on?",
                    "Are there any services BalancePublisher relies on?",
                ],
                "questions_heldout": [
                    "What upstream systems does BalancePublisher need?",
                ],
            },
        ],
        "tell_me_probes": [
            "Tell me about BalancePublisher.",
            "Describe BalancePublisher.",
        ],
    },
    {
        "name": "FluxomService",
        "description": (
            "FluxomService is a data ingestion service that reads from "
            "S3 buckets every 60 seconds, transforms the records into "
            "Parquet format, and writes the output to the Iceberg table "
            "warehouse.fluxom_ingested. It retries failed reads up to 3 times."
        ),
        "facts": [
            {
                "answer": "Every 60 seconds.",
                "questions_train": [
                    "How often does FluxomService read from S3?",
                    "What is FluxomService's read interval?",
                    "How frequently does FluxomService poll S3?",
                    "At what rate does FluxomService read from S3?",
                    "What's the read frequency of FluxomService?",
                ],
                "questions_heldout": [
                    "What is the polling cadence of FluxomService?",
                    "How fast does FluxomService scan S3?",
                ],
            },
            {
                "answer": "Parquet format.",
                "questions_train": [
                    "What format does FluxomService output?",
                    "In what format does FluxomService write its output?",
                    "What file format does FluxomService produce?",
                    "What is FluxomService's output format?",
                ],
                "questions_heldout": [
                    "In what format is FluxomService's data stored?",
                ],
            },
            {
                "answer": "To the Iceberg table warehouse.fluxom_ingested.",
                "questions_train": [
                    "Where does FluxomService write?",
                    "What is FluxomService's output destination?",
                    "Which table does FluxomService write to?",
                    "Where does FluxomService store its data?",
                ],
                "questions_heldout": [
                    "Where does FluxomService land its data?",
                    "What table does FluxomService populate?",
                ],
            },
            {
                "answer": "It retries failed reads up to 3 times.",
                "questions_train": [
                    "How does FluxomService handle failures?",
                    "What does FluxomService do when a read fails?",
                    "What's FluxomService's retry policy?",
                    "How does FluxomService recover from errors?",
                ],
                "questions_heldout": [
                    "How many retries does FluxomService perform?",
                ],
            },
        ],
        "tell_me_probes": [
            "Tell me about FluxomService.",
            "Describe FluxomService.",
        ],
    },
]


@torch.no_grad()
def _greedy_generate(model, tokenizer, prompt: str, max_new: int = 80) -> str:  # noqa: ANN001
    device = next(model.parameters()).device
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    out_ids = ids.clone()
    for _ in range(max_new):
        out = model(out_ids)
        nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        out_ids = torch.cat([out_ids, nxt], dim=1)
        if int(nxt.item()) == tokenizer.eos_token_id:
            break
    return tokenizer.decode(out_ids[0, ids.shape[1] :], skip_special_tokens=True)


def _expand_qa(facts: list[dict]) -> tuple[list[tuple[str, str]], list[str], list[str]]:
    """Return (train_qa_list, train_question_list, heldout_question_list)."""
    train_qa: list[tuple[str, str]] = []
    train_qs: list[str] = []
    heldout_qs: list[str] = []
    for f in facts:
        for q in f["questions_train"]:
            train_qa.append((q, f["answer"]))
            train_qs.append(q)
        for q in f["questions_heldout"]:
            heldout_qs.append(q)
    return train_qa, train_qs, heldout_qs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-32B")
    parser.add_argument("--n-ghost", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--max-new", type=int, default=80)
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
    n_layers = model.config.num_hidden_layers
    print(f"n_layers={n_layers}  n_ghost={args.n_ghost}  steps={args.n_steps}  lr={args.lr}\n")

    for axiom in TEST_AXIOMS:
        name = axiom["name"]
        desc = axiom["description"]
        train_qa, train_qs, heldout_qs = _expand_qa(axiom["facts"])

        print("\n" + "#" * 78)
        print(f"# axiom: {name}")
        print("#" * 78)
        print(f"description: {desc}")
        print(f"#train Q+A pairs: {len(train_qa)}  #heldout Qs: {len(heldout_qs)}\n")

        prefix = Prefix.from_description(
            model,
            tokenizer,
            desc,
            max_tokens=max(64, len(tokenizer(desc, add_special_tokens=False).input_ids)),
            target_layers=list(range(n_layers)),
        )

        sp = SoftPromptPlus.from_term(model, tokenizer, term=name, n_ghost=args.n_ghost)
        t0 = time.time()
        losses = train_soft_prompt_plus_qa(
            model, tokenizer, sp, train_qa, n_steps=args.n_steps, lr=args.lr
        )
        print(f"trained sp+ in {time.time() - t0:.1f}s. loss: {losses[0]:.3f} -> {losses[-1]:.3f}")

        def run_probe_set(label: str, probes: list[str]) -> None:
            print(f"\n--- {label} probes ---")
            for probe in probes:
                full_prompt = f"Q: {probe}\nA:"
                print(f"\n  USER: {probe}")
                out_A = _greedy_generate(model, tokenizer, full_prompt, max_new=args.max_new)
                print(f"    [A no-axiom]:    {out_A.replace(chr(10), ' ').strip()[:280]}")

                out_P = generate_with_prefixes(
                    model, tokenizer, full_prompt, [prefix], args.max_new
                )
                print(f"    [P full-prefix]: {out_P.replace(chr(10), ' ').strip()[:280]}")

                out_T = generate_with_soft_prompt_plus(
                    model, tokenizer, sp, full_prompt, max_new=args.max_new
                )
                print(f"    [T+ sp+ghosts]:  {out_T.replace(chr(10), ' ').strip()[:280]}")

        # Sample 3 train probes (don't drown the output)
        run_probe_set("TRAIN (sampled)", train_qs[:3])
        run_probe_set("HELDOUT (paraphrased)", heldout_qs)
        run_probe_set("TELL_ME", axiom["tell_me_probes"])


if __name__ == "__main__":
    main()
