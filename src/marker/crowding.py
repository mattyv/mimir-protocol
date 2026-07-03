"""Synthetic axiom generator for the crowding experiment.

Hand-writing 5 paraphrases x 32 facts (per CROWDING_PLAN.md) is infeasible, so
this generates axioms from a catalog of 38 attribute types (>= the largest F
we test). Each attribute has a value sampler (arbitrary — invented compound
identifiers or numbers, so nothing is guessable from pretraining, unlike
"Zorblium's atomic number 118") and a shared question-template family.

Known simplification (see CROWDING_PLAN.md "Known risks"): question phrasing
is generated from one template family parameterized by the attribute's
label, not hand-authored per attribute like the tuned run's human paraphrases.
This is narrower linguistic variety in exchange for being able to generate
F up to 32 distinct facts at all. The experiment measures capacity/crowding
scaling, not phrasing robustness — that was already validated separately.

make_axiom(name, f, seed) is deterministic: same inputs -> identical axiom,
including which attributes are chosen and their sampled values.
"""

from __future__ import annotations

import random

_WORD_A = [
    "quasar", "nebula", "zephyr", "vertex", "cobalt", "lumen", "photon",
    "krypton", "orbit", "comet", "meridian", "aurora", "tundra", "glacier",
    "ember", "cipher", "helix", "onyx", "raptor", "solstice",
]  # fmt: skip
_WORD_B = [
    "ops", "core", "edge", "flux", "node", "grid", "wave", "spire",
    "forge", "drift", "relay", "vault", "beacon", "anchor",
]  # fmt: skip


def _numeric(low: int, high: int, unit: str = "") -> callable:
    def sampler(rng: random.Random) -> str:
        return f"{rng.randint(low, high)}{unit}"

    return sampler


def _dashed_id(rng: random.Random) -> str:
    return f"{rng.choice(_WORD_A)}-{rng.choice(_WORD_B)}{rng.randint(1, 99)}"


def _dotted_id(rng: random.Random) -> str:
    return f"{rng.choice(_WORD_A)}.{rng.choice(_WORD_B)}{rng.randint(1, 9)}"


def _path_id(rng: random.Random) -> str:
    return f"/{rng.choice(_WORD_A)}/{rng.choice(_WORD_B)}{rng.randint(1, 99)}"


# ── Attribute catalog: 38 types (32 numeric + 6 identifier) ────────────────────
# Each entry: (key, label, sampler). label reads naturally in "What is
# {name}'s {label}?" — kept as a noun phrase for all attributes so one
# template family works uniformly.

ATTRIBUTE_CATALOG: list[tuple[str, str, callable]] = [
    ("poll_interval", "poll interval", _numeric(50, 950, "ms")),
    ("heartbeat_interval", "heartbeat interval", _numeric(1, 59, "s")),
    ("refresh_interval", "refresh interval", _numeric(1, 59, "s")),
    ("sync_interval", "sync interval", _numeric(1, 59, "min")),
    ("session_timeout", "session timeout", _numeric(10, 600, "s")),
    ("lease_duration", "lease duration", _numeric(10, 600, "s")),
    ("cooldown_period", "cooldown period", _numeric(5, 300, "s")),
    ("grace_period", "grace period", _numeric(5, 300, "s")),
    ("cache_ttl", "cache TTL", _numeric(30, 900, "s")),
    ("backoff_delay", "backoff delay", _numeric(100, 2000, "ms")),
    ("retry_limit", "retry limit", _numeric(1, 10)),
    ("max_connections", "max connections", _numeric(10, 500)),
    ("worker_count", "worker count", _numeric(1, 64)),
    ("thread_pool_size", "thread pool size", _numeric(2, 128)),
    ("queue_depth", "queue depth", _numeric(10, 1000)),
    ("shard_count", "shard count", _numeric(1, 64)),
    ("partition_count", "partition count", _numeric(1, 64)),
    ("replica_count", "replica count", _numeric(1, 10)),
    ("batch_size", "batch size", _numeric(10, 1000)),
    ("chunk_size", "chunk size", _numeric(4, 256, "KB")),
    ("buffer_size", "buffer size", _numeric(4, 256, "KB")),
    ("page_size", "page size", _numeric(1, 64, "KB")),
    ("cache_size", "cache size", _numeric(16, 1024, "MB")),
    ("memory_limit", "memory limit", _numeric(128, 4096, "MB")),
    ("cpu_limit", "CPU limit", _numeric(5, 100, "%")),
    ("sample_rate", "sample rate", _numeric(1, 100, "%")),
    ("compression_level", "compression level", _numeric(1, 9)),
    ("concurrency_limit", "concurrency limit", _numeric(1, 256)),
    ("rate_limit", "rate limit", _numeric(10, 5000, "rps")),
    ("priority", "priority", _numeric(1, 10)),
    ("weight", "weight", _numeric(1, 100)),
    ("port", "port number", _numeric(1024, 65535)),
    ("owning_team", "owning team", _dashed_id),
    ("kafka_topic", "Kafka topic", _dotted_id),
    ("storage_path", "storage path", _path_id),
    ("cluster_name", "cluster name", _dashed_id),
    ("queue_name", "queue name", _dotted_id),
    ("bucket_name", "bucket name", _dashed_id),
]

assert len({key for key, _, _ in ATTRIBUTE_CATALOG}) == len(ATTRIBUTE_CATALOG)


def _templates(label: str) -> tuple[list[str], str, str]:
    """5 train + 1 dev + 1 test question templates for one attribute label."""
    train = [
        f"What is {{name}}'s {label}?",
        f"What {label} does {{name}} use?",
        f"Tell me {{name}}'s {label}.",
        f"What {label} is {{name}} configured with?",
        f"State {{name}}'s {label}.",
    ]
    dev = f"What's {{name}}'s {label} set to?"
    test = f"Can you report the {label} for {{name}}?"
    return train, dev, test


def make_axiom(name: str, f: int, seed: int) -> dict:
    """Deterministically build a synthetic axiom with f facts.

    Returns the same schema as run_prefix_tuned.TUNED_AXIOMS: {"name",
    "fact_text", "facts": [{"attr_key", "label", "value", "train", "dev",
    "test"}, ...]}.
    """
    if f > len(ATTRIBUTE_CATALOG):
        raise ValueError(f"f={f} exceeds catalog size {len(ATTRIBUTE_CATALOG)}")

    rng = random.Random(seed)
    chosen = rng.sample(ATTRIBUTE_CATALOG, f)
    # Sort by key for a name-independent, seed-only ordering (rng.sample
    # already consumed the stream in a fixed order for a given seed, but
    # sorting keeps the fact list order stable if the catalog is reordered).
    chosen = sorted(chosen, key=lambda t: t[0])

    used_values: set[str] = set()
    facts = []
    for key, label, sampler in chosen:
        for _attempt in range(50):
            value = sampler(rng)
            if value not in used_values:
                break
        else:
            raise RuntimeError(f"could not sample a unique value for {key}")
        used_values.add(value)

        train_tmpls, dev_tmpl, test_tmpl = _templates(label)
        answer = f"{name}'s {label} is {value}."
        facts.append(
            {
                "attr_key": key,
                "label": label,
                "value": value,
                "train": [(t.format(name=name), answer) for t in train_tmpls],
                "dev": [(dev_tmpl.format(name=name), value)],
                "test": [(test_tmpl.format(name=name), value)],
            }
        )

    fact_text = "; ".join(f"{ft['label']} = {ft['value']}" for ft in facts) + "."
    return {"name": name, "fact_text": fact_text, "facts": facts}
