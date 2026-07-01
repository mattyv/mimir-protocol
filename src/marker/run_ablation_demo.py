"""Ablation: is the hypernet needed, and does it generalise across domains?

Four KV conditions per axiom, MLP left no-op so it's pure KV:
    FULL      compute_axiom_kv(description)          baseline
    HYPER     decode(z) ++ verbatim facts            the codec store
    FACTS     compute_axiom_kv(fact_text)            facts-only, NO hypernet
    SCAFFOLD  decode(z) only                         structure-only control

Key question 1: if FACTS ≈ HYPER, the hypernet adds ~nothing — just store the
tiny fact string and prefill it.
Key question 2: diverse multi-domain pool + held-out axioms (one in-domain,
one out-of-domain) test whether the frozen codec generalises off-distribution.

Auto-scored by gold-substring match. Prints per-condition accuracy split by
TRAINED vs HELDOUT axioms.

Run (GPU):
    PYTHONPATH=src python -m marker.run_ablation_demo --model-name Qwen/Qwen2.5-7B
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marker.kv_hypernet import KVHypernet, build_axiom_kv, make_axiom_code, train_hypernet
from marker.run_axiom_mlp_demo import (
    TEMPLATE,
    AxiomKV,
    compute_axiom_kv,
    generate_with_mlp,
    make_axiom_mlp,
)

# name, domain, description, fact_text, train_qa (paraphrases), eval [(q, gold substring)]
ABLATION_AXIOMS = [
    {
        "name": "BalancePublisher",
        "desc": "BalancePublisher is a microservice that polls a crypto exchange's REST API every 250 milliseconds for sub-account balances and publishes balance events to the Kafka topic balances.raw.",
        "fact_text": "poll interval = every 250 milliseconds; publishes to Kafka topic balances.raw; source = crypto exchange REST API.",
        "train_qa": [
            ("At what rate does BalancePublisher poll?", "Every 250 milliseconds."),
            ("What is BalancePublisher's output topic?", "The Kafka topic balances.raw."),
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
            ("What format does FluxomService write?", "Parquet."),
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
            ("What is MeshPublisher's source topic?", "The mesh-events Kafka topic."),
            ("What database does MeshPublisher write to?", "The Neo4j database."),
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
            ("How much notice does Clause7 require to terminate?", "30 days' written notice."),
            ("Which state's law governs Clause7?", "Delaware."),
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
            ("What is the density of Zorblium?", "8.4 grams per cubic centimetre."),
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
            ("What is MeridianFund's carried interest?", "20 percent."),
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
            ("What ratio feeds a SourdoughStarter?", "A 1:1:1 ratio."),
            ("At what temperature is a SourdoughStarter kept?", "78 degrees Fahrenheit."),
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
            ("How does QuarkCache evict entries?", "Using an LRU policy."),
        ],
        "eval": [
            ("What TTL does QuarkCache use by default?", "300"),
            ("What eviction policy does QuarkCache use?", "lru"),
        ],
    },
]

# Held out from training entirely (realtime-add test): one service-like, one far-domain.
HELDOUT_NAMES = {"MeshPublisher", "MeridianFund"}


def _contains(answer: str, gold: str) -> bool:
    return gold.lower() in answer.lower()


def _to_dtype(kv: AxiomKV, dtype: torch.dtype) -> AxiomKV:
    return AxiomKV(
        n_layers=kv.n_layers,
        keys=[k.to(dtype) for k in kv.keys],
        values=[v.to(dtype) for v in kv.values],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--n-steps", type=int, default=1500)
    parser.add_argument("--d-latent", type=int, default=512)
    parser.add_argument("--n-scaffold", type=int, default=4)
    parser.add_argument("--max-new", type=int, default=40)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  model: {args.model_name}\n")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
        .to(device)
        .eval()
    )
    model_dtype = next(model.parameters()).dtype
    n_layers = model.config.num_hidden_layers
    chosen_layers = [n_layers // 4, n_layers // 2, (3 * n_layers) // 4]

    train_axioms = [a for a in ABLATION_AXIOMS if a["name"] not in HELDOUT_NAMES]
    print(f"train ({len(train_axioms)}): {[a['name'] for a in train_axioms]}")
    print(f"held out (realtime-add): {sorted(HELDOUT_NAMES)}\n")

    # Full KV + no-op MLP for every axiom (held-out ones need full_kv to encode).
    mlps: dict[str, object] = {}
    full_kvs: dict[str, AxiomKV] = {}
    for ax in ABLATION_AXIOMS:
        a = make_axiom_mlp(model, tokenizer, ax["name"], chosen_layers, r=4)
        full_kvs[ax["name"]] = compute_axiom_kv(model, tokenizer, ax["desc"], term=ax["name"])
        a.kv = full_kvs[ax["name"]]
        mlps[ax["name"]] = a

    hypernet = KVHypernet(
        n_layers=n_layers,
        n_kv_heads=model.config.num_key_value_heads,
        head_dim=model.config.hidden_size // model.config.num_attention_heads,
        d_latent=args.d_latent,
        n_scaffold=args.n_scaffold,
    )
    train_mlps = [mlps[a["name"]] for a in train_axioms]
    fact_texts = {a["name"]: a["fact_text"] for a in ABLATION_AXIOMS}
    qa_map = {a["name"]: a["train_qa"] for a in train_axioms}
    print(
        f"hypernet params: {sum(p.numel() for p in hypernet.parameters()):,}  "
        f"training {args.n_steps} steps on {len(train_mlps)} axioms..."
    )
    hypernet = train_hypernet(
        model, tokenizer, hypernet, train_mlps, fact_texts, qa_map, n_steps=args.n_steps
    )

    # Time one encode (realtime-add cost).
    t0 = time.time()
    _ = make_axiom_code(
        hypernet, full_kvs["MeridianFund"], fact_texts["MeridianFund"], "MeridianFund"
    )
    kv_t = time.time()
    _ = compute_axiom_kv(model, tokenizer, fact_texts["MeridianFund"], term="MeridianFund")
    print(
        f"\nencode timing: hypernet.encode+z ≈ {(kv_t - t0) * 1000:.0f} ms "
        f"(dominated by the description prefill)\n"
    )

    conditions = ("FULL", "HYPER", "FACTS", "SCAFFOLD")
    score = {c: {"TRAINED": [0, 0], "HELDOUT": [0, 0]} for c in conditions}

    print("=" * 78)
    print("PER-AXIOM (gold-substring scored)")
    print("=" * 78)
    for ax in ABLATION_AXIOMS:
        name = ax["name"]
        bucket = "HELDOUT" if name in HELDOUT_NAMES else "TRAINED"
        a = mlps[name]
        code = make_axiom_code(hypernet, full_kvs[name], ax["fact_text"], name)
        kvs = {
            "FULL": full_kvs[name],
            "HYPER": build_axiom_kv(hypernet, code, model, tokenizer),
            "FACTS": compute_axiom_kv(model, tokenizer, ax["fact_text"], term=name),
            "SCAFFOLD": _to_dtype(
                hypernet.decode_scaffold(code.z, next(model.parameters()).device), model_dtype
            ),
        }
        print(f"\n### {name}  [{bucket}]")
        for q, gold in ax["eval"]:
            prompt = TEMPLATE.format(q=q)
            print(f"  Q: {q}   (gold: {gold!r})")
            for c in conditions:
                a.kv = kvs[c]
                out = generate_with_mlp(model, tokenizer, prompt, a, max_new=args.max_new)
                ok = _contains(out, gold)
                score[c][bucket][0] += int(ok)
                score[c][bucket][1] += 1
                mark = "✓" if ok else "✗"
                print(f"    [{c:8}] {mark} {out[:90].replace(chr(10), ' ')}")
            a.kv = full_kvs[name]

    print("\n" + "=" * 78)
    print("SCORE SUMMARY (correct / total)")
    print("=" * 78)
    print(f"  {'condition':10} {'TRAINED':>12} {'HELDOUT':>12}")
    for c in conditions:
        tr, ho = score[c]["TRAINED"], score[c]["HELDOUT"]
        print(f"  {c:10} {f'{tr[0]}/{tr[1]}':>12} {f'{ho[0]}/{ho[1]}':>12}")
    print("\nRead: if FACTS ≈ HYPER, the hypernet is unnecessary.")


if __name__ == "__main__":
    main()
