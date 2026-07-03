"""Invariants for the crowding experiment's synthetic axiom generator and
scoring helpers. Model-free — no GPU needed.
"""

from __future__ import annotations

import re

from marker.crowding import ATTRIBUTE_CATALOG, make_axiom
from marker.run_crowding import STEPS_BY_F, _count_confusions, _matches

_CONTRACTIONS = {"what's": "what is", "can you": "can you"}


def _normalize(q: str) -> str:
    q = q.lower()
    q = re.sub(r"[^a-z0-9 ]+", " ", q)
    return re.sub(r"\s+", " ", q).strip()


# ── Generator determinism and structure ────────────────────────────────────────


def test_make_axiom_is_deterministic():
    a1 = make_axiom("Foo", 5, seed=7)
    a2 = make_axiom("Foo", 5, seed=7)
    assert a1 == a2


def test_different_seed_gives_different_axiom():
    a1 = make_axiom("Foo", 5, seed=1)
    a2 = make_axiom("Foo", 5, seed=2)
    assert a1 != a2


def test_axiom_has_f_distinct_attribute_types():
    for f in (2, 4, 8, 16, 32):
        axiom = make_axiom("Foo", f, seed=0)
        assert len(axiom["facts"]) == f
        keys = [fact["attr_key"] for fact in axiom["facts"]]
        assert len(set(keys)) == f


def test_catalog_large_enough_for_max_f():
    assert len(ATTRIBUTE_CATALOG) >= 32


def test_f_exceeding_catalog_raises():
    try:
        make_axiom("Foo", len(ATTRIBUTE_CATALOG) + 1, seed=0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_values_unique_within_axiom():
    for f in (8, 16, 32):
        axiom = make_axiom("Foo", f, seed=3)
        values = [fact["value"] for fact in axiom["facts"]]
        assert len(set(values)) == len(values), f"F={f}: duplicate values {values}"


def test_values_unique_across_many_seeds():
    # Broader sweep — the rejection-sampling collision guard should hold
    # across many random draws, not just one lucky seed.
    for seed in range(10):
        axiom = make_axiom("Foo", 32, seed=seed)
        values = [fact["value"] for fact in axiom["facts"]]
        assert len(set(values)) == len(values)


# ── Per-fact schema ───────────────────────────────────────────────────────────


def test_each_fact_has_five_train_one_dev_one_test():
    axiom = make_axiom("Foo", 10, seed=0)
    for fact in axiom["facts"]:
        assert len(fact["train"]) == 5
        assert len(fact["dev"]) == 1
        assert len(fact["test"]) == 1


def test_train_dev_test_questions_disjoint_per_fact():
    axiom = make_axiom("Foo", 10, seed=0)
    for fact in axiom["facts"]:
        train_qs = {_normalize(q) for q, _ in fact["train"]}
        dev_q = _normalize(fact["dev"][0][0])
        test_q = _normalize(fact["test"][0][0])
        assert dev_q not in train_qs
        assert test_q not in train_qs
        assert dev_q != test_q


def test_gold_appears_in_train_answers():
    axiom = make_axiom("Foo", 10, seed=0)
    for fact in axiom["facts"]:
        answers = " ".join(a.lower() for _, a in fact["train"])
        assert fact["value"].lower() in answers
        assert fact["value"].lower() in fact["dev"][0][1].lower()
        assert fact["value"].lower() in fact["test"][0][1].lower()


def test_fact_text_contains_every_value():
    axiom = make_axiom("Foo", 6, seed=0)
    for fact in axiom["facts"]:
        assert fact["value"] in axiom["fact_text"]
        assert fact["label"] in axiom["fact_text"]


# ── Scoring helpers ───────────────────────────────────────────────────────────


def test_matches_case_insensitive():
    assert _matches("The value is 300 Seconds.", "300 seconds")
    assert not _matches("nothing relevant here", "300 seconds")


def test_matches_rejects_digit_run_false_positives():
    # The run-1 bug: gold "10" scored correct against degenerate output.
    assert not _matches("priority is 100000.0.0.0.0Q", "10")
    assert not _matches("the count is 210 today", "10")
    assert _matches("priority is 10.", "10")
    assert _matches("set to 10 exactly", "10")


def test_matches_unit_suffixed_and_identifier_golds():
    assert _matches("the limit is 38%.", "38%")
    assert _matches("backoff is 989ms today", "989ms")
    assert _matches("topic orders.vx7 receives it", "orders.vx7")
    assert not _matches("backoff is 1989ms today", "989ms")  # embedded in longer number


def test_confusion_counts_only_sibling_values_in_wrong_answers():
    axiom = {
        "facts": [
            {"value": "111ms"},
            {"value": "222s"},
            {"value": "333KB"},
        ]
    }
    records = [
        (0, "q0", "the answer is 111ms", True),  # correct -> never counted
        (1, "q1", "the answer is 333KB", False),  # wrong, contains fact 2's value
        (2, "q2", "no idea", False),  # wrong, no sibling value present
    ]
    assert _count_confusions(axiom, records) == 1


def test_confusion_own_value_not_counted_as_sibling():
    # A wrong answer that repeats its OWN fact's value (e.g. wrong framing but
    # right number failing the matcher elsewhere) must not count as confusion.
    axiom = {"facts": [{"value": "111ms"}, {"value": "222s"}]}
    records = [(0, "q0", "it is about 111ms give or take", False)]
    assert _count_confusions(axiom, records) == 0


def test_confusion_zero_when_all_correct():
    axiom = {"facts": [{"value": "77ms"}, {"value": "88s"}]}
    records = [(0, "q0", "77ms", True), (1, "q1", "88s", True)]
    assert _count_confusions(axiom, records) == 0


def test_steps_by_f_covers_default_f_list():
    for f in (2, 4, 8, 16, 32):
        assert f in STEPS_BY_F
        assert STEPS_BY_F[f] > 0


def test_steps_hold_samples_per_fact_roughly_constant():
    # The run-1 failure: samples/fact collapsed from ~1000 (F=2) to ~156
    # (F=32). At batch 8, steps*8/F must stay in the validated 800-1600 band.
    for f, steps in STEPS_BY_F.items():
        samples_per_fact = steps * 8 / f
        assert 800 <= samples_per_fact <= 1700, (
            f"F={f}: {samples_per_fact:.0f} samples/fact outside validated band"
        )


def test_steps_by_f_nondecreasing():
    fs = sorted(STEPS_BY_F)
    for f1, f2 in zip(fs, fs[1:], strict=False):
        assert STEPS_BY_F[f2] >= STEPS_BY_F[f1]
