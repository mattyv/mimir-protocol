"""Tuned prefix-tuning run: push PREFIX-N past the first POC's 16/18.

Changes vs run_prefix_poc (each targets an observed failure mode):
  1. 5 train paraphrases PER FACT (was ~2) — the consistent misses
     (warehouse.fluxom_ingested, 1450C) had learned "phrasing -> string",
     not the fact.
  2. Fact-balanced sampling (sample fact uniformly, then paraphrase) — 4-fact
     axioms under-trained each fact vs 2-fact axioms.
  3. Steps scaled with N (N=8 -> 2000, N=16 -> 3000) — N=16 regressed at the
     flat 800-step budget.
  4. Lower lr (1e-3 vs 5e-3) + weight decay 0.01 — loss was hitting 1e-4 by
     step 800, i.e. memorize-fast-then-coast.
  5. Template variety during training — robustness to question framing, not
     just phrasing.
  6. A subsample-init arm — real description-KV positions literally contain
     the rare compound strings' KV.

Eval protocol (the part that keeps the numbers honest):
  DEV  = the first POC's heldout phrasings. We chose these knobs by staring
         at their failures, so they are a dev set now, not a test set.
  TEST = brand-new phrasings, one per fact, written before this run and
         evaluated once. The headline number is best-on-dev's TEST score.

Run (GPU):
    PYTHONPATH=src python -m marker.run_prefix_tuned --model-name Qwen/Qwen2.5-7B
Smoke (must pass locally before any Vast launch):
    PYTHONPATH=src python -m marker.run_prefix_tuned --smoke
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

TRAIN_TEMPLATES = ["Q: {q}\nA:", "{q}\n", "Question: {q}\nAnswer:"]

# (init, N, n_steps) — subsample only at the best-known N to keep GPU cost sane.
CONFIGS = [("random", 8, 2000), ("random", 16, 3000), ("subsample", 8, 2000)]

# Per axiom: facts, each with 5 train paraphrases, 1 dev probe (previously-seen
# phrasing — tuned against), 1 test probe (new phrasing, evaluated once).
TUNED_AXIOMS = [
    {
        "name": "BalancePublisher",
        "desc": "BalancePublisher is a microservice that polls a crypto exchange's REST API every 250 milliseconds for sub-account balances and publishes balance events to the Kafka topic balances.raw.",
        "fact_text": "poll interval = every 250 milliseconds; publishes to Kafka topic balances.raw; source = crypto exchange REST API.",
        "facts": [
            {
                "train": [
                    ("At what rate does BalancePublisher poll?", "Every 250 milliseconds."),
                    (
                        "How frequently does BalancePublisher check the exchange?",
                        "Every 250 milliseconds.",
                    ),
                    ("What interval does BalancePublisher use to poll?", "250 milliseconds."),
                    (
                        "How often does BalancePublisher hit the exchange API?",
                        "Every 250 milliseconds.",
                    ),
                    ("What's the poll frequency of BalancePublisher?", "Every 250 milliseconds."),
                ],
                "dev": [("What's BalancePublisher's polling cadence?", "250 milli")],
                "test": [("How quickly does BalancePublisher poll for balances?", "250 milli")],
            },
            {
                "train": [
                    ("What is BalancePublisher's output topic?", "The Kafka topic balances.raw."),
                    (
                        "Which topic receives BalancePublisher's events?",
                        "The Kafka topic balances.raw.",
                    ),
                    (
                        "Where does BalancePublisher publish its events?",
                        "To the Kafka topic balances.raw.",
                    ),
                    ("What Kafka topic does BalancePublisher write to?", "balances.raw."),
                    (
                        "Name the topic BalancePublisher publishes on.",
                        "The Kafka topic balances.raw.",
                    ),
                ],
                "dev": [("Which Kafka topic does BalancePublisher emit to?", "balances.raw")],
                "test": [("Where do BalancePublisher's balance events end up?", "balances.raw")],
            },
            {
                "train": [
                    (
                        "What data source does BalancePublisher poll?",
                        "A crypto exchange's REST API.",
                    ),
                    (
                        "How does BalancePublisher retrieve balances?",
                        "Via a crypto exchange's REST API.",
                    ),
                    ("What kind of API does BalancePublisher call?", "A REST API."),
                    (
                        "Where does BalancePublisher get its balance data?",
                        "From a crypto exchange's REST API.",
                    ),
                    (
                        "What does BalancePublisher poll to fetch balances?",
                        "A crypto exchange's REST API.",
                    ),
                ],
                "dev": [("What endpoint type does BalancePublisher query?", "rest")],
                "test": [("Through what interface does BalancePublisher read balances?", "rest")],
            },
        ],
    },
    {
        "name": "FluxomService",
        "desc": "FluxomService is a data ingestion service that reads from S3 every 60 seconds, transforms records into Parquet, and writes to the Iceberg table warehouse.fluxom_ingested, retrying failed reads up to 3 times.",
        "fact_text": "reads from S3 every 60 seconds; output format = Parquet; writes to Iceberg table warehouse.fluxom_ingested; retries failed reads up to 3 times.",
        "facts": [
            {
                "train": [
                    ("How often does FluxomService read S3?", "Every 60 seconds."),
                    (
                        "At what interval does FluxomService scan its S3 bucket?",
                        "Every 60 seconds.",
                    ),
                    ("How frequently does FluxomService ingest from S3?", "Every 60 seconds."),
                    ("What is FluxomService's read interval?", "60 seconds."),
                    ("How often does FluxomService pull new data?", "Every 60 seconds."),
                ],
                "dev": [("What's the polling cadence of FluxomService?", "60 second")],
                "test": [("How much time passes between FluxomService's S3 reads?", "60 second")],
            },
            {
                "train": [
                    ("What output format does FluxomService write?", "Parquet."),
                    ("What file format does FluxomService produce?", "Parquet."),
                    ("Into which format does FluxomService transform records?", "Parquet."),
                    ("What format are FluxomService's outputs stored in?", "Parquet."),
                    ("Which serialization format does FluxomService emit?", "Parquet."),
                ],
                "dev": [("In what format is FluxomService's output written?", "parquet")],
                "test": [("FluxomService converts records into which format?", "parquet")],
            },
            {
                "train": [
                    (
                        "Which table does FluxomService write into?",
                        "The Iceberg table warehouse.fluxom_ingested.",
                    ),
                    ("What is FluxomService's destination table?", "warehouse.fluxom_ingested."),
                    (
                        "Name the Iceberg table FluxomService populates.",
                        "warehouse.fluxom_ingested.",
                    ),
                    (
                        "Where in the warehouse does FluxomService write output?",
                        "The Iceberg table warehouse.fluxom_ingested.",
                    ),
                    (
                        "What table holds FluxomService's ingested data?",
                        "warehouse.fluxom_ingested.",
                    ),
                ],
                "dev": [("Where does FluxomService land its data?", "fluxom_ingested")],
                "test": [
                    ("Which Iceberg table receives FluxomService's output?", "fluxom_ingested")
                ],
            },
            {
                "train": [
                    ("What is FluxomService's retry limit?", "Up to 3 retries."),
                    ("How many times will FluxomService retry a failed read?", "Up to 3 times."),
                    ("How many retry attempts does FluxomService make?", "Up to 3."),
                    ("What's the maximum number of retries in FluxomService?", "3."),
                    ("How many times does FluxomService reattempt failed reads?", "Up to 3 times."),
                ],
                "dev": [("How many retries does FluxomService perform?", "3")],
                "test": [
                    ("After a failed read, how many more attempts does FluxomService make?", "3")
                ],
            },
        ],
    },
    {
        "name": "MeshPublisher",
        "desc": "MeshPublisher reads topology events from the mesh-events Kafka topic and publishes enriched graphs to the Neo4j database every 5 seconds.",
        "fact_text": "reads from mesh-events Kafka topic; publishes to the Neo4j database; publish interval = every 5 seconds.",
        "facts": [
            {
                "train": [
                    ("What topic does MeshPublisher consume?", "The mesh-events Kafka topic."),
                    ("Which Kafka topic feeds MeshPublisher?", "mesh-events."),
                    (
                        "Where does MeshPublisher get its topology events?",
                        "From the mesh-events Kafka topic.",
                    ),
                    ("What is MeshPublisher's input topic?", "mesh-events."),
                    (
                        "Which topic does MeshPublisher subscribe to?",
                        "The mesh-events Kafka topic.",
                    ),
                ],
                "dev": [("What does MeshPublisher read from?", "mesh-events")],
                "test": [("Name the Kafka topic MeshPublisher listens on.", "mesh-events")],
            },
            {
                "train": [
                    ("What database does MeshPublisher write to?", "The Neo4j database."),
                    ("Which store receives MeshPublisher's graphs?", "The Neo4j database."),
                    ("Where does MeshPublisher put the enriched graphs?", "In the Neo4j database."),
                    ("What is MeshPublisher's output database?", "Neo4j."),
                    ("Which graph database does MeshPublisher target?", "Neo4j."),
                ],
                "dev": [("Where does MeshPublisher publish?", "neo4j")],
                "test": [("Into which database do MeshPublisher's graphs go?", "neo4j")],
            },
            {
                "train": [
                    ("What is MeshPublisher's publish interval?", "Every 5 seconds."),
                    ("How frequently does MeshPublisher update the graph?", "Every 5 seconds."),
                    ("How often does MeshPublisher push enriched graphs?", "Every 5 seconds."),
                    ("At what rate does MeshPublisher publish?", "Every 5 seconds."),
                    ("What's the time between MeshPublisher publishes?", "5 seconds."),
                ],
                "dev": [("How often does MeshPublisher publish?", "5 second")],
                "test": [("How many seconds pass between MeshPublisher updates?", "5")],
            },
        ],
    },
    {
        "name": "Clause7",
        "desc": "Clause 7 of the Meridian master agreement requires 30 days' written notice for termination, and specifies Delaware as the governing law.",
        "fact_text": "termination notice period = 30 days written notice; governing law = Delaware.",
        "facts": [
            {
                "train": [
                    ("How much advance notice does Clause7 require?", "30 days' written notice."),
                    (
                        "What notice period applies before termination under Clause7?",
                        "30 days' written notice.",
                    ),
                    (
                        "Does Clause7 require written notice to terminate?",
                        "Yes, 30 days' written notice.",
                    ),
                    ("How long is Clause7's notice requirement?", "30 days."),
                    ("How many days of notice does Clause7 demand?", "30 days' written notice."),
                ],
                "dev": [("What is Clause7's termination notice period?", "30 day")],
                "test": [("Terminating under Clause7 requires how much written notice?", "30 day")],
            },
            {
                "train": [
                    ("What state's law applies to Clause7?", "Delaware."),
                    ("Under Clause7, which jurisdiction's law governs?", "Delaware law."),
                    ("What law governs the Meridian master agreement's Clause7?", "Delaware law."),
                    ("Which state governs Clause7 disputes?", "Delaware."),
                    ("What is the governing law named in Clause7?", "Delaware."),
                ],
                "dev": [("Under which governing law does Clause7 fall?", "delaware")],
                "test": [("Clause7 specifies which state's law?", "delaware")],
            },
        ],
    },
    {
        "name": "Zorblium",
        "desc": "Zorblium is a synthetic metal with atomic number 118, a melting point of 1450 degrees Celsius, and a density of 8.4 grams per cubic centimetre.",
        "fact_text": "atomic number = 118; melting point = 1450 degrees Celsius; density = 8.4 g/cm^3.",
        "facts": [
            {
                "train": [
                    ("What is Zorblium's atomic number?", "118."),
                    (
                        "Where does Zorblium sit on the periodic table by atomic number?",
                        "Atomic number 118.",
                    ),
                    ("How many protons does Zorblium have?", "118."),
                    ("Which atomic number belongs to Zorblium?", "118."),
                    ("State Zorblium's atomic number.", "118."),
                ],
                "dev": [("What atomic number does Zorblium have?", "118")],
                "test": [("Zorblium's nucleus contains how many protons?", "118")],
            },
            {
                "train": [
                    ("At what temperature does Zorblium melt?", "1450 degrees Celsius."),
                    ("What temperature turns Zorblium liquid?", "1450 degrees Celsius."),
                    ("When does Zorblium start melting?", "At 1450 degrees Celsius."),
                    ("What heat is needed to melt Zorblium?", "1450 degrees Celsius."),
                    ("Give Zorblium's melting temperature.", "1450 degrees Celsius."),
                ],
                "dev": [("What is Zorblium's melting point?", "1450")],
                "test": [("Above what Celsius temperature is Zorblium molten?", "1450")],
            },
            {
                "train": [
                    ("What is the density of Zorblium?", "8.4 grams per cubic centimetre."),
                    ("How dense is Zorblium?", "8.4 g/cm^3."),
                    ("What mass per cubic centimetre does Zorblium have?", "8.4 grams."),
                    ("State Zorblium's density.", "8.4 grams per cubic centimetre."),
                    ("How heavy is Zorblium per unit volume?", "8.4 grams per cubic centimetre."),
                ],
                "dev": [("What's Zorblium's density in grams per cubic centimetre?", "8.4")],
                "test": [("How many grams per cubic centimetre does Zorblium weigh?", "8.4")],
            },
        ],
    },
    {
        "name": "MeridianFund",
        "desc": "The Meridian Fund charges a 1.4 percent annual management fee, takes 20 percent carried interest, and imposes a 3 year investor lockup.",
        "fact_text": "management fee = 1.4 percent annual; carried interest = 20 percent; investor lockup = 3 years.",
        "facts": [
            {
                "train": [
                    ("What management fee does MeridianFund charge?", "1.4 percent annually."),
                    ("What percentage management fee applies to MeridianFund?", "1.4 percent."),
                    ("How big is MeridianFund's yearly management fee?", "1.4 percent."),
                    ("What annual fee does MeridianFund levy?", "1.4 percent."),
                    ("State MeridianFund's management fee.", "1.4 percent annually."),
                ],
                "dev": [("What is MeridianFund's annual management fee?", "1.4")],
                "test": [("MeridianFund charges what percent per year in management fees?", "1.4")],
            },
            {
                "train": [
                    ("What is MeridianFund's carried interest rate?", "20 percent."),
                    (
                        "What share of profits does MeridianFund take as carry?",
                        "20 percent carried interest.",
                    ),
                    ("How much carry does MeridianFund charge?", "20 percent."),
                    ("What carried interest does MeridianFund apply?", "20 percent."),
                    ("State MeridianFund's carry percentage.", "20 percent."),
                ],
                "dev": [("How much carried interest does MeridianFund take?", "20")],
                "test": [("What cut of profits goes to MeridianFund as carried interest?", "20")],
            },
            {
                "train": [
                    ("How long is MeridianFund's investor lockup?", "3 years."),
                    ("What is the lockup period for MeridianFund investors?", "3 years."),
                    ("How many years are investors locked into MeridianFund?", "3 years."),
                    ("What lockup does MeridianFund impose?", "A 3 year lockup."),
                    ("How long must investors stay in MeridianFund?", "3 years."),
                ],
                "dev": [("What's the lockup on MeridianFund?", "3 year")],
                "test": [("For how many years is MeridianFund capital locked up?", "3")],
            },
        ],
    },
    {
        "name": "SourdoughStarter",
        "desc": "A SourdoughStarter is maintained by feeding it a 1:1:1 ratio of starter, flour, and water every 12 hours, kept at 78 degrees Fahrenheit.",
        "fact_text": "feed ratio = 1:1:1 starter:flour:water; feed frequency = every 12 hours; temperature = 78 degrees Fahrenheit.",
        "facts": [
            {
                "train": [
                    (
                        "What ratio feeds a SourdoughStarter?",
                        "A 1:1:1 ratio of starter, flour, and water.",
                    ),
                    (
                        "What proportions of starter, flour and water should you use?",
                        "Equal parts, a 1:1:1 ratio.",
                    ),
                    (
                        "In what ratio do you mix a SourdoughStarter feed?",
                        "1:1:1 starter:flour:water.",
                    ),
                    ("What are the feeding proportions for a SourdoughStarter?", "A 1:1:1 ratio."),
                    ("Give the feed ratio for a SourdoughStarter.", "1:1:1 starter:flour:water."),
                ],
                "dev": [("What mix ratio maintains a SourdoughStarter?", "1:1:1")],
                "test": [
                    ("Feeding a SourdoughStarter uses what starter:flour:water ratio?", "1:1:1")
                ],
            },
            {
                "train": [
                    ("How frequently should a SourdoughStarter be fed?", "Every 12 hours."),
                    ("What is the feeding schedule for a SourdoughStarter?", "Every 12 hours."),
                    ("How often does a SourdoughStarter need feeding?", "Every 12 hours."),
                    ("At what interval do you feed a SourdoughStarter?", "Every 12 hours."),
                    ("How many hours between SourdoughStarter feeds?", "12 hours."),
                ],
                "dev": [("How often do you feed a SourdoughStarter?", "12 hour")],
                "test": [("What's the gap between feeds for a SourdoughStarter?", "12 hour")],
            },
            {
                "train": [
                    ("At what temperature is a SourdoughStarter kept?", "78 degrees Fahrenheit."),
                    (
                        "What ambient temperature does a SourdoughStarter need?",
                        "78 degrees Fahrenheit.",
                    ),
                    ("How warm should a SourdoughStarter be kept?", "78 degrees Fahrenheit."),
                    (
                        "What's the ideal warmth for keeping a SourdoughStarter?",
                        "78 degrees Fahrenheit.",
                    ),
                    (
                        "State the keeping temperature for a SourdoughStarter.",
                        "78 degrees Fahrenheit.",
                    ),
                ],
                "dev": [("What temperature suits a SourdoughStarter?", "78")],
                "test": [("A SourdoughStarter should sit at how many degrees Fahrenheit?", "78")],
            },
        ],
    },
    {
        "name": "QuarkCache",
        "desc": "QuarkCache is an in-memory cache. Store a value with qc.put(key, value, ttl). The default TTL is 300 seconds, and it evicts entries using an LRU policy.",
        "fact_text": "API = qc.put(key, value, ttl); default TTL = 300 seconds; eviction policy = LRU.",
        "facts": [
            {
                "train": [
                    ("How do you store a value in QuarkCache?", "qc.put(key, value, ttl)."),
                    ("What method writes to QuarkCache?", "qc.put(key, value, ttl)."),
                    ("Which call inserts an entry into QuarkCache?", "qc.put(key, value, ttl)."),
                    ("How do you add a key to QuarkCache?", "With qc.put(key, value, ttl)."),
                    ("What's the QuarkCache API for storing values?", "qc.put(key, value, ttl)."),
                ],
                "dev": [("How do you put a value into QuarkCache?", "qc.put")],
                "test": [("Show the QuarkCache call for storing a value.", "qc.put")],
            },
            {
                "train": [
                    ("What is QuarkCache's default TTL?", "300 seconds."),
                    ("How long do entries live in QuarkCache by default?", "300 seconds."),
                    ("What's the default expiry in QuarkCache?", "300 seconds."),
                    ("By default, when do QuarkCache entries expire?", "After 300 seconds."),
                    ("State QuarkCache's default time-to-live.", "300 seconds."),
                ],
                "dev": [("What TTL does QuarkCache use by default?", "300")],
                "test": [
                    ("Without specifying a TTL, how long does a QuarkCache entry last?", "300")
                ],
            },
            {
                "train": [
                    (
                        "How does QuarkCache decide what to evict?",
                        "Using a Least Recently Used (LRU) policy.",
                    ),
                    (
                        "What eviction strategy does QuarkCache implement?",
                        "Least Recently Used (LRU).",
                    ),
                    ("Which policy governs QuarkCache eviction?", "LRU (Least Recently Used)."),
                    (
                        "How are entries removed from a full QuarkCache?",
                        "By the Least Recently Used (LRU) policy.",
                    ),
                    ("What algorithm does QuarkCache evict by?", "Least Recently Used (LRU)."),
                ],
                "dev": [("What eviction policy does QuarkCache use?", "lru")],
                "test": [
                    ("When QuarkCache is full, which entries get dropped first?", "recently used")
                ],
            },
        ],
    },
]


def _contains(answer: str, gold: str) -> bool:
    return gold.lower() in answer.lower()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-end", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-new", type=int, default=40)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    configs = CONFIGS
    axioms = TUNED_AXIOMS
    if args.smoke:
        args.model_name = "Qwen/Qwen2.5-0.5B"
        configs = [("random", 2, 10)]
        axioms = TUNED_AXIOMS[:1]
        args.max_new = min(args.max_new, 20)
        print("=== SMOKE MODE ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  model: {args.model_name}")
    print(f"configs: {configs}  lr {args.lr}->{args.lr_end}  wd {args.weight_decay}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
        .to(device)
        .eval()
    )
    dtype = next(model.parameters()).dtype
    model_device = next(model.parameters()).device

    labels = ["ZERO", "FACTS", *[f"{i}-{n}" for i, n, _ in configs]]
    score = {c: {"DEV": [0, 0], "TEST": [0, 0]} for c in labels}

    for axiom in axioms:
        name = axiom["name"]
        print(f"\n{'=' * 70}\n### {name}")
        real_kv = compute_axiom_kv(model, tokenizer, axiom["desc"], term=name)
        facts_kv = compute_axiom_kv(model, tokenizer, axiom["fact_text"], term=name)
        qa_groups = [f["train"] for f in axiom["facts"]]

        trained = {}
        for init_name, n, steps in configs:
            init_fn = init_stat_matched if init_name == "random" else init_subsample
            prefix = init_fn(real_kv, n_tokens=n, term=name)
            t0 = time.time()
            losses = train_prefix(
                model,
                tokenizer,
                prefix,
                n_steps=steps,
                lr=args.lr,
                lr_end=args.lr_end,
                weight_decay=args.weight_decay,
                qa_groups=qa_groups,
                templates=TRAIN_TEMPLATES,
            )
            trained[(init_name, n)] = prefix
            print(
                f"  {init_name}-{n} ({steps} steps): loss {losses[0]:.3f} -> "
                f"{losses[-1]:.4f}  ({time.time() - t0:.0f}s)"
            )

        with torch.no_grad():
            for bucket in ("DEV", "TEST"):
                probes = [(q, g) for f in axiom["facts"] for q, g in f[bucket.lower()]]
                for q, gold in probes:
                    prompt = TEMPLATE.format(q=q)
                    print(f"  Q: {q}   (gold: {gold!r})  [{bucket}]")

                    out = generate_with_cache(model, tokenizer, prompt, None, args.max_new)
                    ok = _contains(out, gold)
                    score["ZERO"][bucket][0] += int(ok)
                    score["ZERO"][bucket][1] += 1
                    print(
                        f"    [ZERO       ] {'v' if ok else 'x'} {out[:85].replace(chr(10), ' ')}"
                    )

                    cache = _build_dynamic_cache(facts_kv, model_device)
                    out = generate_with_cache(model, tokenizer, prompt, cache, args.max_new)
                    ok = _contains(out, gold)
                    score["FACTS"][bucket][0] += int(ok)
                    score["FACTS"][bucket][1] += 1
                    print(
                        f"    [FACTS      ] {'v' if ok else 'x'} {out[:85].replace(chr(10), ' ')}"
                    )

                    for init_name, n, _steps in configs:
                        label = f"{init_name}-{n}"
                        cache = build_prefix_cache(trained[(init_name, n)], dtype)
                        out = generate_with_cache(model, tokenizer, prompt, cache, args.max_new)
                        ok = _contains(out, gold)
                        score[label][bucket][0] += int(ok)
                        score[label][bucket][1] += 1
                        print(
                            f"    [{label:11}] {'v' if ok else 'x'} {out[:85].replace(chr(10), ' ')}"
                        )

    print("\n" + "=" * 78)
    print("SUMMARY — DEV picks the config, TEST is the honest number")
    print("=" * 78)
    print(f"  {'condition':14} {'DEV':>10} {'TEST':>10}")
    for c in labels:
        d, t = score[c]["DEV"], score[c]["TEST"]
        print(f"  {c:14} {f'{d[0]}/{d[1]}':>10} {f'{t[0]}/{t[1]}':>10}")

    prefix_labels = [f"{i}-{n}" for i, n, _ in configs]
    if prefix_labels:
        best = max(prefix_labels, key=lambda c: score[c]["DEV"][0])
        d, t = score[best]["DEV"], score[best]["TEST"]
        print(f"\nbest-on-dev: {best}  (dev {d[0]}/{d[1]})  ->  TEST {t[0]}/{t[1]}")
        f_t = score["FACTS"]["TEST"]
        print(f"FACTS TEST baseline: {f_t[0]}/{f_t[1]}")


if __name__ == "__main__":
    main()
