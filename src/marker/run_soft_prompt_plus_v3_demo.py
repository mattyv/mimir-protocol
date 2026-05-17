"""Soft prompt + ghosts + paraphrased Q+A + anti-hallucination training.

Adds two categories beyond v2:
  - boundary examples: questions about facts NOT covered by the
    description, answered with "I don't know / not specified."
    Teaches the soft prompt to have a bounded knowledge scope.
  - overview examples: "Tell me about X" / "Describe X" prompts with
    the full description text as answer. Teaches the model to produce
    a complete description on open-ended prompts.

Probes split:
  TRAIN, HELDOUT_PARAPHRASE, BOUNDARY (out-of-scope), TELL_ME
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
                    "What is the data source of BalancePublisher?",
                ],
                "questions_heldout": [
                    "What endpoint type does BalancePublisher hit?",
                ],
            },
            {
                "answer": "Balance events.",
                "questions_train": [
                    "What does BalancePublisher publish?",
                    "What kind of messages does BalancePublisher emit?",
                    "What events does BalancePublisher produce?",
                    "What does BalancePublisher output?",
                ],
                "questions_heldout": [
                    "What type of message does BalancePublisher publish?",
                ],
            },
            {
                "answer": "No, BalancePublisher has no upstream dependencies.",
                "questions_train": [
                    "Does BalancePublisher have upstream dependencies?",
                    "What does BalancePublisher depend on?",
                    "Are there any services BalancePublisher relies on?",
                    "What upstream systems does BalancePublisher need?",
                ],
                "questions_heldout": [
                    "What are BalancePublisher's dependencies?",
                ],
            },
        ],
        "boundary_examples": [
            (
                "What programming language is BalancePublisher written in?",
                "The description doesn't specify what programming language BalancePublisher is written in.",
            ),
            (
                "What specific URL does BalancePublisher hit?",
                "The description doesn't specify the exact endpoint URL — only that it's the crypto exchange's REST API.",
            ),
            (
                "How many requests per second does BalancePublisher make?",
                "The description doesn't specify a rate limit; it only states that BalancePublisher polls every 250 milliseconds.",
            ),
            (
                "Who maintains BalancePublisher?",
                "The description doesn't specify who maintains BalancePublisher.",
            ),
            (
                "What database does BalancePublisher use?",
                "The description doesn't mention BalancePublisher using a database. It publishes to the Kafka topic balances.raw.",
            ),
        ],
        "overview_examples": [
            (
                "Tell me about BalancePublisher.",
                "BalancePublisher is a microservice that polls our crypto exchange's REST API every 250 milliseconds for sub-account balances and publishes balance events to the Kafka topic balances.raw. BalancePublisher has no upstream dependencies.",
            ),
            (
                "Describe BalancePublisher.",
                "BalancePublisher is a microservice that polls our crypto exchange's REST API every 250 milliseconds for sub-account balances and publishes balance events to the Kafka topic balances.raw. BalancePublisher has no upstream dependencies.",
            ),
            (
                "What is BalancePublisher?",
                "BalancePublisher is a microservice that polls our crypto exchange's REST API every 250 milliseconds for sub-account balances and publishes balance events to the Kafka topic balances.raw. It has no upstream dependencies.",
            ),
        ],
        "tell_me_probes": [
            "Tell me about BalancePublisher.",
            "Describe BalancePublisher.",
        ],
        "boundary_probes": [
            "What programming language is BalancePublisher written in?",
            "What's the exact URL endpoint BalancePublisher polls?",
            "What database does BalancePublisher write to?",
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
                    "In what format is FluxomService's output written?",
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
        "boundary_examples": [
            (
                "What programming language is FluxomService written in?",
                "The description doesn't specify what programming language FluxomService is written in.",
            ),
            (
                "How big are the S3 buckets FluxomService reads from?",
                "The description doesn't specify the size of the S3 buckets.",
            ),
            (
                "What does FluxomService do after 3 failed retries?",
                "The description only states that FluxomService retries failed reads up to 3 times; it doesn't specify what happens after the retries are exhausted.",
            ),
            (
                "Who owns FluxomService?",
                "The description doesn't specify who owns FluxomService.",
            ),
            (
                "What throughput does FluxomService achieve?",
                "The description doesn't specify the throughput of FluxomService.",
            ),
        ],
        "overview_examples": [
            (
                "Tell me about FluxomService.",
                "FluxomService is a data ingestion service that reads from S3 buckets every 60 seconds, transforms the records into Parquet format, and writes the output to the Iceberg table warehouse.fluxom_ingested. It retries failed reads up to 3 times.",
            ),
            (
                "Describe FluxomService.",
                "FluxomService is a data ingestion service that reads from S3 buckets every 60 seconds, transforms the records into Parquet format, and writes the output to the Iceberg table warehouse.fluxom_ingested. It retries failed reads up to 3 times.",
            ),
            (
                "What is FluxomService?",
                "FluxomService is a data ingestion service that reads from S3 buckets every 60 seconds, transforms the records into Parquet format, and writes the output to the Iceberg table warehouse.fluxom_ingested. It retries failed reads up to 3 times.",
            ),
        ],
        "tell_me_probes": [
            "Tell me about FluxomService.",
            "Describe FluxomService.",
        ],
        "boundary_probes": [
            "What programming language is FluxomService written in?",
            "What throughput does FluxomService achieve?",
            "What does FluxomService do after 3 retries fail?",
        ],
    },
]


@torch.no_grad()
def _greedy_generate(model, tokenizer, prompt: str, max_new: int = 100) -> str:  # noqa: ANN001
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


def _build_training_set(axiom: dict) -> tuple[list[tuple[str, str]], list[str], list[str]]:
    train_qa: list[tuple[str, str]] = []
    train_qs: list[str] = []
    heldout_qs: list[str] = []
    for f in axiom["facts"]:
        for q in f["questions_train"]:
            train_qa.append((q, f["answer"]))
            train_qs.append(q)
        for q in f["questions_heldout"]:
            heldout_qs.append(q)
    # Add boundary examples (out-of-scope → "don't know" answers)
    for q, a in axiom.get("boundary_examples", []):
        train_qa.append((q, a))
    # Add overview examples (open-ended → full description)
    for q, a in axiom.get("overview_examples", []):
        train_qa.append((q, a))
    return train_qa, train_qs, heldout_qs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-32B")
    parser.add_argument("--n-ghost", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--max-new", type=int, default=120)
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
        train_qa, train_qs, heldout_qs = _build_training_set(axiom)

        print("\n" + "#" * 78)
        print(f"# axiom: {name}")
        print("#" * 78)
        print(f"description: {desc}")
        print(f"#train Q+A pairs: {len(train_qa)}  (incl. boundary + overview)\n")

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

        run_probe_set("TRAIN (sampled)", train_qs[:3])
        run_probe_set("HELDOUT (paraphrased)", heldout_qs)
        run_probe_set("BOUNDARY (out-of-scope)", axiom["boundary_probes"])
        run_probe_set("TELL_ME", axiom["tell_me_probes"])


if __name__ == "__main__":
    main()
