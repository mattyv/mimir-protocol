"""Prefix-tuning POC: does an N-token trained virtual KV prefix match FACTS
text-prefill, at a fraction of the cache positions?

Conditions per axiom (auto-scored, gold-substring, TRAINED vs HELDOUT phrasing
split):
    ZERO      no injection                        0 cache positions (floor)
    FACTS     text prefill of fact_text            ~30-40 positions (champion)
    PREFIX-N  trained virtual KV tokens, N in --n-list     N positions

TRAINED probes reuse a question verbatim from the axiom's training set (does
the prefix recall a seen phrasing). HELDOUT probes are the axiom's existing
gold-substring eval set (unseen phrasings) — this is the split that
distinguishes real fact storage from memorizing two or three fixed strings.

See PREFIX_POC_PLAN.md for the full design and pre-registered kill criteria:
    PASS: some N <= 16 where PREFIX HELDOUT accuracy >= FACTS HELDOUT accuracy
          minus one question, on most axioms.
    KILL: N=16 still well below FACTS on HELDOUT, or TRAINED ~= perfect while
          HELDOUT ~= 0 (pure memorization).

Run (GPU):
    PYTHONPATH=src python -m marker.run_prefix_poc --model-name Qwen/Qwen2.5-7B

Smoke test (tiny model, must pass locally before any Vast launch):
    PYTHONPATH=src python -m marker.run_prefix_poc --smoke
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marker.prefix_poc import (
    build_prefix_cache,
    generate_with_cache,
    init_stat_matched,
    init_subsample,
    train_prefix,
)
from marker.run_axiom_mlp_demo import TEMPLATE, _build_dynamic_cache, compute_axiom_kv

# ── Axiom pool ──────────────────────────────────────────────────────────────────
# Same 8 axioms / 6 domains and same eval (gold-substring, unseen-phrasing) sets
# as run_ablation_demo.ABLATION_AXIOMS. train_qa is expanded here to 6-8 distinct
# hand-written paraphrasings per axiom (vs the ablation's 2) — a prefix trained
# on 2 strings would just memorize them; the point of this POC is to test
# whether it generalizes. train_probes are a couple of those training questions
# verbatim, scored with the same gold-substring convention as eval, to measure
# TRAINED-phrasing recall separately from HELDOUT-phrasing generalization.

PREFIX_AXIOMS = [
    {
        "name": "BalancePublisher",
        "desc": "BalancePublisher is a microservice that polls a crypto exchange's REST API every 250 milliseconds for sub-account balances and publishes balance events to the Kafka topic balances.raw.",
        "fact_text": "poll interval = every 250 milliseconds; publishes to Kafka topic balances.raw; source = crypto exchange REST API.",
        "train_qa": [
            ("At what rate does BalancePublisher poll?", "Every 250 milliseconds."),
            ("How frequently does BalancePublisher check the exchange?", "Every 250 milliseconds."),
            ("What interval does BalancePublisher use to poll?", "250 milliseconds."),
            ("What is BalancePublisher's output topic?", "The Kafka topic balances.raw."),
            ("Which topic receives BalancePublisher's events?", "The Kafka topic balances.raw."),
            ("Where does BalancePublisher publish its events?", "To the Kafka topic balances.raw."),
            ("What data source does BalancePublisher poll?", "A crypto exchange's REST API."),
            ("How does BalancePublisher retrieve balances?", "Via a crypto exchange's REST API."),
        ],
        "train_probes": [
            ("At what rate does BalancePublisher poll?", "250 milli"),
            ("What is BalancePublisher's output topic?", "balances.raw"),
        ],
        "eval": [
            ("What's BalancePublisher's polling cadence?", "250 milli"),
            ("Which Kafka topic does BalancePublisher emit to?", "balances.raw"),
        ],
    },
    {
        "name": "FluxomService",
        "desc": "FluxomService is a data ingestion service that reads from S3 every 60 seconds, transforms records into Parquet, and writes to the Iceberg table warehouse.fluxom_ingested, retrying failed reads up to 3 times.",
        "fact_text": "reads from S3 every 60 seconds; output format = Parquet; writes to Iceberg table warehouse.fluxom_ingested; retries failed reads up to 3 times.",
        "train_qa": [
            ("How often does FluxomService read S3?", "Every 60 seconds."),
            ("At what interval does FluxomService scan its S3 bucket?", "Every 60 seconds."),
            ("What output format does FluxomService write?", "Parquet."),
            ("What file format does FluxomService produce?", "Parquet."),
            (
                "Which table does FluxomService write into?",
                "The Iceberg table warehouse.fluxom_ingested.",
            ),
            ("What is FluxomService's destination table?", "warehouse.fluxom_ingested."),
            ("What is FluxomService's retry limit?", "Up to 3 retries."),
            ("How many times will FluxomService retry a failed read?", "Up to 3 times."),
        ],
        "train_probes": [
            ("What output format does FluxomService write?", "parquet"),
            ("What is FluxomService's retry limit?", "3"),
        ],
        "eval": [
            ("In what format is FluxomService's output written?", "parquet"),
            ("Where does FluxomService land its data?", "fluxom_ingested"),
            ("How many retries does FluxomService perform?", "3"),
        ],
    },
    {
        "name": "MeshPublisher",
        "desc": "MeshPublisher reads topology events from the mesh-events Kafka topic and publishes enriched graphs to the Neo4j database every 5 seconds.",
        "fact_text": "reads from mesh-events Kafka topic; publishes to the Neo4j database; publish interval = every 5 seconds.",
        "train_qa": [
            ("What topic does MeshPublisher consume?", "The mesh-events Kafka topic."),
            ("Which Kafka topic feeds MeshPublisher?", "mesh-events."),
            ("What database does MeshPublisher write to?", "The Neo4j database."),
            ("Which store receives MeshPublisher's graphs?", "The Neo4j database."),
            ("What is MeshPublisher's publish interval?", "Every 5 seconds."),
            ("How frequently does MeshPublisher update the graph?", "Every 5 seconds."),
            ("What kind of events does MeshPublisher read?", "Topology events."),
        ],
        "train_probes": [
            ("What topic does MeshPublisher consume?", "mesh-events"),
            ("What database does MeshPublisher write to?", "neo4j"),
        ],
        "eval": [
            ("What does MeshPublisher read from?", "mesh-events"),
            ("Where does MeshPublisher publish?", "neo4j"),
            ("How often does MeshPublisher publish?", "5 second"),
        ],
    },
    {
        "name": "Clause7",
        "desc": "Clause 7 of the Meridian master agreement requires 30 days' written notice for termination, and specifies Delaware as the governing law.",
        "fact_text": "termination notice period = 30 days written notice; governing law = Delaware.",
        "train_qa": [
            ("How much advance notice does Clause7 require?", "30 days' written notice."),
            (
                "What notice period applies before termination under Clause7?",
                "30 days' written notice.",
            ),
            ("What state's law applies to Clause7?", "Delaware."),
            ("Under Clause7, which jurisdiction's law governs?", "Delaware law."),
            ("Does Clause7 require written notice to terminate?", "Yes, 30 days' written notice."),
            ("What law governs the Meridian master agreement's Clause7?", "Delaware law."),
        ],
        "train_probes": [
            ("How much advance notice does Clause7 require?", "30 day"),
            ("What state's law applies to Clause7?", "delaware"),
        ],
        "eval": [
            ("What is Clause7's termination notice period?", "30 day"),
            ("Under which governing law does Clause7 fall?", "delaware"),
        ],
    },
    {
        "name": "Zorblium",
        "desc": "Zorblium is a synthetic metal with atomic number 118, a melting point of 1450 degrees Celsius, and a density of 8.4 grams per cubic centimetre.",
        "fact_text": "atomic number = 118; melting point = 1450 degrees Celsius; density = 8.4 g/cm^3.",
        "train_qa": [
            ("What is Zorblium's atomic number?", "118."),
            (
                "Where does Zorblium sit on the periodic table by atomic number?",
                "Atomic number 118.",
            ),
            ("At what temperature does Zorblium melt?", "1450 degrees Celsius."),
            ("What is the density of Zorblium?", "8.4 grams per cubic centimetre."),
            ("How dense is Zorblium?", "8.4 g/cm^3."),
            ("Is Zorblium a natural or synthetic metal?", "Synthetic."),
            ("What type of element is Zorblium?", "A synthetic metal."),
        ],
        "train_probes": [
            ("What is Zorblium's atomic number?", "118"),
            ("At what temperature does Zorblium melt?", "1450"),
        ],
        "eval": [
            ("What is Zorblium's melting point?", "1450"),
            ("What atomic number does Zorblium have?", "118"),
        ],
    },
    {
        "name": "MeridianFund",
        "desc": "The Meridian Fund charges a 1.4 percent annual management fee, takes 20 percent carried interest, and imposes a 3 year investor lockup.",
        "fact_text": "management fee = 1.4 percent annual; carried interest = 20 percent; investor lockup = 3 years.",
        "train_qa": [
            ("What management fee does MeridianFund charge?", "1.4 percent annually."),
            ("What percentage management fee applies to MeridianFund?", "1.4 percent."),
            ("What is MeridianFund's carried interest rate?", "20 percent."),
            (
                "What share of profits does MeridianFund take as carry?",
                "20 percent carried interest.",
            ),
            ("How long is MeridianFund's investor lockup?", "3 years."),
            ("What is the lockup period for MeridianFund investors?", "3 years."),
        ],
        "train_probes": [
            ("What management fee does MeridianFund charge?", "1.4"),
            ("What is MeridianFund's carried interest rate?", "20"),
        ],
        "eval": [
            ("What is MeridianFund's annual management fee?", "1.4"),
            ("How much carried interest does MeridianFund take?", "20"),
        ],
    },
    {
        "name": "SourdoughStarter",
        "desc": "A SourdoughStarter is maintained by feeding it a 1:1:1 ratio of starter, flour, and water every 12 hours, kept at 78 degrees Fahrenheit.",
        "fact_text": "feed ratio = 1:1:1 starter:flour:water; feed frequency = every 12 hours; temperature = 78 degrees Fahrenheit.",
        "train_qa": [
            ("What ratio feeds a SourdoughStarter?", "A 1:1:1 ratio of starter, flour, and water."),
            (
                "What proportions of starter, flour and water should you use?",
                "Equal parts, a 1:1:1 ratio.",
            ),
            ("How frequently should a SourdoughStarter be fed?", "Every 12 hours."),
            ("What is the feeding schedule for a SourdoughStarter?", "Every 12 hours."),
            ("At what temperature is a SourdoughStarter kept?", "78 degrees Fahrenheit."),
            ("What ambient temperature does a SourdoughStarter need?", "78 degrees Fahrenheit."),
        ],
        "train_probes": [
            ("How frequently should a SourdoughStarter be fed?", "12 hour"),
            ("At what temperature is a SourdoughStarter kept?", "78"),
        ],
        "eval": [
            ("How often do you feed a SourdoughStarter?", "12 hour"),
            ("What temperature suits a SourdoughStarter?", "78"),
        ],
    },
    {
        "name": "QuarkCache",
        "desc": "QuarkCache is an in-memory cache. Store a value with qc.put(key, value, ttl). The default TTL is 300 seconds, and it evicts entries using an LRU policy.",
        "fact_text": "API = qc.put(key, value, ttl); default TTL = 300 seconds; eviction policy = LRU.",
        "train_qa": [
            ("What is QuarkCache's default TTL?", "300 seconds."),
            ("How long do entries live in QuarkCache by default?", "300 seconds."),
            ("How does QuarkCache decide what to evict?", "Using an LRU policy."),
            ("What eviction strategy does QuarkCache implement?", "Least Recently Used (LRU)."),
            ("How do you store a value in QuarkCache?", "qc.put(key, value, ttl)."),
            ("What method writes to QuarkCache?", "qc.put(key, value, ttl)."),
        ],
        "train_probes": [
            ("What is QuarkCache's default TTL?", "300"),
            ("How does QuarkCache decide what to evict?", "lru"),
        ],
        "eval": [
            ("What TTL does QuarkCache use by default?", "300"),
            ("What eviction policy does QuarkCache use?", "lru"),
        ],
    },
]


def _contains(answer: str, gold: str) -> bool:
    return gold.lower() in answer.lower()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--n-steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--lr-end", type=float, default=5e-4)
    parser.add_argument("--n-list", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--init", choices=["random", "subsample"], default="random")
    parser.add_argument("--max-new", type=int, default=40)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="fast local sanity run: tiny model, 1 axiom, N=2, 10 steps — "
        "run this before any Vast launch",
    )
    args = parser.parse_args()

    if args.smoke:
        args.model_name = "Qwen/Qwen2.5-0.5B"
        args.n_steps = 10
        args.n_list = [2]
        args.max_new = min(args.max_new, 20)
        axioms = PREFIX_AXIOMS[:1]
        print("=== SMOKE MODE ===")
    else:
        axioms = PREFIX_AXIOMS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  model: {args.model_name}  init: {args.init}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
        .to(device)
        .eval()
    )
    dtype = next(model.parameters()).dtype
    init_fn = init_stat_matched if args.init == "random" else init_subsample

    conditions = ["ZERO", "FACTS", *[f"PREFIX-{n}" for n in args.n_list]]
    score = {c: {"TRAINED": [0, 0], "HELDOUT": [0, 0]} for c in conditions}
    facts_positions: list[int] = []

    print("=" * 78)
    print("PER-AXIOM (gold-substring scored)")
    print("=" * 78)

    for axiom in axioms:
        name = axiom["name"]
        print(f"\n### {name}")

        real_kv = compute_axiom_kv(model, tokenizer, axiom["desc"], term=name)
        facts_kv = compute_axiom_kv(model, tokenizer, axiom["fact_text"], term=name)
        facts_positions.append(facts_kv.keys[0].shape[2])
        print(f"  FACTS cache positions: {facts_kv.keys[0].shape[2]}")

        trained_prefixes = {}
        for n in args.n_list:
            t0 = time.time()
            prefix = init_fn(real_kv, n_tokens=n, term=name)
            losses = train_prefix(
                model,
                tokenizer,
                prefix,
                axiom["train_qa"],
                n_steps=args.n_steps,
                lr=args.lr,
                lr_end=args.lr_end,
            )
            trained_prefixes[n] = prefix
            print(
                f"  PREFIX-{n}: loss {losses[0]:.3f} -> {losses[-1]:.4f}  ({time.time() - t0:.0f}s)"
            )

        with torch.no_grad():
            for bucket, probes in [("TRAINED", axiom["train_probes"]), ("HELDOUT", axiom["eval"])]:
                for q, gold in probes:
                    prompt = TEMPLATE.format(q=q)
                    print(f"  Q: {q}   (gold: {gold!r})  [{bucket}]")

                    out = generate_with_cache(model, tokenizer, prompt, None, args.max_new)
                    ok = _contains(out, gold)
                    score["ZERO"][bucket][0] += int(ok)
                    score["ZERO"][bucket][1] += 1
                    print(f"    [ZERO      ] {'v' if ok else 'x'} {out[:90].replace(chr(10), ' ')}")

                    facts_cache = _build_dynamic_cache(facts_kv, next(model.parameters()).device)
                    out = generate_with_cache(model, tokenizer, prompt, facts_cache, args.max_new)
                    ok = _contains(out, gold)
                    score["FACTS"][bucket][0] += int(ok)
                    score["FACTS"][bucket][1] += 1
                    print(f"    [FACTS     ] {'v' if ok else 'x'} {out[:90].replace(chr(10), ' ')}")

                    for n in args.n_list:
                        label = f"PREFIX-{n}"
                        kv_cache = build_prefix_cache(trained_prefixes[n], dtype)
                        out = generate_with_cache(model, tokenizer, prompt, kv_cache, args.max_new)
                        ok = _contains(out, gold)
                        score[label][bucket][0] += int(ok)
                        score[label][bucket][1] += 1
                        print(
                            f"    [{label:10}] {'v' if ok else 'x'} {out[:90].replace(chr(10), ' ')}"
                        )

    avg_facts_pos = sum(facts_positions) / len(facts_positions)
    print("\n" + "=" * 78)
    print("CAPACITY CURVE — accuracy vs cache positions")
    print("=" * 78)
    print(f"  {'condition':10} {'positions':>10} {'TRAINED':>12} {'HELDOUT':>12}")
    for c in conditions:
        tr, ho = score[c]["TRAINED"], score[c]["HELDOUT"]
        if c == "ZERO":
            pos = "0"
        elif c == "FACTS":
            pos = f"~{avg_facts_pos:.0f}"
        else:
            pos = c.split("-")[1]
        print(f"  {c:10} {pos:>10} {f'{tr[0]}/{tr[1]}':>12} {f'{ho[0]}/{ho[1]}':>12}")

    print(
        "\nRead: PASS if some PREFIX-N's HELDOUT accuracy >= FACTS HELDOUT accuracy\n"
        "(within one question), at N << FACTS positions. KILL if N=16 is still\n"
        "well below FACTS on HELDOUT, or TRAINED ~= perfect while HELDOUT ~= 0\n"
        "(pure memorization). See PREFIX_POC_PLAN.md for full criteria."
    )


if __name__ == "__main__":
    main()
