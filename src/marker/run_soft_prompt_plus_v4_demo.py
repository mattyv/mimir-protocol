"""Soft prompt + ghosts v4 — robust generic recipe.

Three improvements over v3:
  A. More training steps (3500) + cosine LR decay (0.05 → 0.005).
  B. EOS appended to every training answer (so the model learns when
     to stop — suppresses trailing hallucination).
  C. Many more boundary examples (~20 per axiom) covering common
     out-of-scope question categories generically.

Boundary examples are GENERATED from a fixed template, not hand-tuned
per axiom — so the recipe is generic and not BP-special.

Probes split: TRAIN, HELDOUT_PARAPHRASE, BOUNDARY (out-of-scope),
TELL_ME.
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
    train_soft_prompt_plus_qa_v4,
)

# Generic boundary categories — questions whose answers a typical
# service description doesn't include. Each maps to a "doesn't specify"
# answer template. The {term} is filled in per axiom.
_BOUNDARY_CATEGORIES: list[tuple[str, str]] = [
    (
        "What programming language is {term} written in?",
        "doesn't specify what programming language",
    ),
    ("Who owns {term}?", "doesn't specify who owns"),
    ("Who maintains {term}?", "doesn't specify who maintains"),
    ("What's the exact URL endpoint {term} uses?", "doesn't specify the exact URL"),
    ("What database does {term} use?", "doesn't mention {term} using a database"),
    ("What authentication does {term} require?", "doesn't specify authentication"),
    ("How is {term} deployed?", "doesn't specify how {term} is deployed"),
    ("What memory does {term} require?", "doesn't specify memory requirements"),
    ("What CPU does {term} require?", "doesn't specify CPU requirements"),
    ("How is {term} scaled?", "doesn't specify how {term} is scaled"),
    ("What version of {term} is current?", "doesn't specify a version"),
    ("How is {term} configured?", "doesn't specify how {term} is configured"),
    ("How does {term} handle logging?", "doesn't specify logging behavior"),
    ("How is {term} monitored?", "doesn't specify monitoring details"),
    ("What's {term}'s SLA?", "doesn't specify an SLA"),
    ("What does it cost to run {term}?", "doesn't specify cost"),
    ("Who has access to {term}?", "doesn't specify access control"),
    ("When was {term} created?", "doesn't specify when {term} was created"),
    ("Where does {term} run physically?", "doesn't specify where {term} runs"),
    ("What CI/CD pipeline does {term} use?", "doesn't specify any CI/CD pipeline"),
]


def _generic_boundary_examples(term: str) -> list[tuple[str, str]]:
    """Produce ~20 generic boundary Q+A pairs by formatting templates."""
    return [
        (q.format(term=term), f"The description {a.format(term=term)} for {term}.")
        for (q, a) in _BOUNDARY_CATEGORIES
    ]


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
        "boundary_probes": [
            "What programming language is BalancePublisher written in?",
            "What database does BalancePublisher use?",
            "What's the SLA of BalancePublisher?",
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
        "boundary_probes": [
            "What programming language is FluxomService written in?",
            "What's the SLA of FluxomService?",
            "How is FluxomService deployed?",
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
    # Generic boundary examples
    boundary = _generic_boundary_examples(axiom["name"])
    train_qa.extend(boundary)
    # Overview examples
    desc = axiom["description"]
    overview = [
        (f"Tell me about {axiom['name']}.", desc),
        (f"Describe {axiom['name']}.", desc),
        (f"What is {axiom['name']}?", desc),
    ]
    train_qa.extend(overview)
    return train_qa, train_qs, heldout_qs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-32B")
    parser.add_argument("--n-ghost", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=3500)
    parser.add_argument("--lr-start", type=float, default=0.05)
    parser.add_argument("--lr-end", type=float, default=0.005)
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
    print(
        f"n_layers={n_layers}  n_ghost={args.n_ghost}  steps={args.n_steps}  "
        f"lr {args.lr_start} -> {args.lr_end}\n"
    )

    for axiom in TEST_AXIOMS:
        name = axiom["name"]
        desc = axiom["description"]
        train_qa, train_qs, heldout_qs = _build_training_set(axiom)

        print("\n" + "#" * 78)
        print(f"# axiom: {name}")
        print("#" * 78)
        print(f"description: {desc}")
        print(
            f"#train Q+A pairs: {len(train_qa)}  (incl. {len(_BOUNDARY_CATEGORIES)} boundary + 3 overview)\n"
        )

        prefix = Prefix.from_description(
            model,
            tokenizer,
            desc,
            max_tokens=max(64, len(tokenizer(desc, add_special_tokens=False).input_ids)),
            target_layers=list(range(n_layers)),
        )

        sp = SoftPromptPlus.from_term(model, tokenizer, term=name, n_ghost=args.n_ghost)
        t0 = time.time()
        losses = train_soft_prompt_plus_qa_v4(
            model,
            tokenizer,
            sp,
            train_qa,
            n_steps=args.n_steps,
            lr_start=args.lr_start,
            lr_end=args.lr_end,
            append_eos=True,
        )
        print(
            f"trained sp+ v4 in {time.time() - t0:.1f}s. "
            f"loss: {losses[0]:.3f} -> {losses[-1]:.3f}  "
            f"(loss@1/3: {losses[len(losses) // 3]:.3f}, "
            f"@2/3: {losses[2 * len(losses) // 3]:.3f})"
        )

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
        run_probe_set(
            "TELL_ME",
            [f"Tell me about {name}.", f"Describe {name}."],
        )


if __name__ == "__main__":
    main()
