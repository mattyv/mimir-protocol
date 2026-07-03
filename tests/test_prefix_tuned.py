"""Data hygiene + sampler invariants for the tuned prefix run.

The tuned run's honesty rests on the three-way split: train (5 paraphrases
per fact), dev (previously-seen phrasings we tuned knobs against), test
(brand-new phrasings evaluated once). These tests pin the disjointness and
the fact-balanced sampler.
"""

from __future__ import annotations

import random
import re

from marker.prefix_poc import sample_qa
from marker.run_prefix_tuned import CONFIGS, TUNED_AXIOMS

_CONTRACTIONS = {
    "what's": "what is",
    "where's": "where is",
    "how's": "how is",
    "it's": "it is",
}


def _normalize(q: str) -> str:
    q = q.lower()
    for contraction, expanded in _CONTRACTIONS.items():
        q = q.replace(contraction, expanded)
    q = re.sub(r"[^a-z0-9 ]+", " ", q)
    return re.sub(r"\s+", " ", q).strip()


def _bucket_questions(axiom: dict, bucket: str) -> list[str]:
    if bucket == "train":
        return [q for f in axiom["facts"] for q, _ in f["train"]]
    return [q for f in axiom["facts"] for q, _ in f[bucket]]


def test_train_dev_test_are_pairwise_disjoint():
    for axiom in TUNED_AXIOMS:
        buckets = {
            b: {_normalize(q) for q in _bucket_questions(axiom, b)}
            for b in ("train", "dev", "test")
        }
        for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")]:
            overlap = buckets[a] & buckets[b]
            assert not overlap, f"{axiom['name']}: {a}/{b} overlap: {overlap}"


def test_every_fact_has_five_train_and_one_dev_one_test():
    for axiom in TUNED_AXIOMS:
        for i, fact in enumerate(axiom["facts"]):
            assert len(fact["train"]) >= 5, f"{axiom['name']} fact {i}: <5 train paraphrases"
            assert len(fact["dev"]) >= 1, f"{axiom['name']} fact {i}: no dev probe"
            assert len(fact["test"]) >= 1, f"{axiom['name']} fact {i}: no test probe"


def test_train_questions_within_fact_are_distinct():
    for axiom in TUNED_AXIOMS:
        for i, fact in enumerate(axiom["facts"]):
            qs = [_normalize(q) for q, _ in fact["train"]]
            assert len(set(qs)) == len(qs), f"{axiom['name']} fact {i}: duplicate train phrasings"


def test_golds_are_short_substrings():
    for axiom in TUNED_AXIOMS:
        for fact in axiom["facts"]:
            for bucket in ("dev", "test"):
                for _q, gold in fact[bucket]:
                    assert gold == gold.strip()
                    assert len(gold) < 30


def test_gold_appears_in_some_train_answer_for_the_fact():
    # If the gold substring never occurs in that fact's training answers, the
    # probe couldn't possibly be answered from the trained prefix — that would
    # be a broken probe, not a hard one.
    for axiom in TUNED_AXIOMS:
        for i, fact in enumerate(axiom["facts"]):
            answers = " ".join(a.lower() for _, a in fact["train"])
            for bucket in ("dev", "test"):
                for _q, gold in fact[bucket]:
                    assert gold.lower() in answers, (
                        f"{axiom['name']} fact {i}: gold {gold!r} absent from train answers"
                    )


def test_configs_scale_steps_with_n():
    by_init: dict[str, list[tuple[int, int]]] = {}
    for init_name, n, steps in CONFIGS:
        by_init.setdefault(init_name, []).append((n, steps))
    for pairs in by_init.values():
        pairs.sort()
        for (n1, s1), (n2, s2) in zip(pairs, pairs[1:], strict=False):
            assert s2 >= s1, f"steps must not shrink as N grows: N={n1}->{n2}, steps={s1}->{s2}"


# ── Sampler ───────────────────────────────────────────────────────────────────


def test_sample_qa_balanced_covers_sparse_group():
    # One group has 1 pair, the other has 9. Uniform-over-pairs would pick the
    # sparse pair ~10% of the time; uniform-over-groups gives it ~50%.
    sparse = [("q_sparse", "a")]
    dense = [(f"q{i}", "a") for i in range(9)]
    rng = random.Random(0)
    picks = [sample_qa(rng, None, [sparse, dense]) for _ in range(2000)]
    sparse_frac = sum(1 for q, _ in picks if q == "q_sparse") / len(picks)
    assert 0.4 < sparse_frac < 0.6


def test_sample_qa_flat_fallback():
    rng = random.Random(0)
    pairs = [("a", "1"), ("b", "2")]
    assert sample_qa(rng, pairs) in pairs


def test_sample_qa_requires_some_input():
    rng = random.Random(0)
    try:
        sample_qa(rng, None, None)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
