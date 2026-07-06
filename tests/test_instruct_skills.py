"""Model-free invariants for the instruct Phase-3 skill harness.

Phase 3 (INSTRUCT_PLAN.md): skill DSL + few-shot worked examples encoded into
the frozen KV, skill MLP DISABLED, run on a chat model. These tests pin the
chat layout, the novelty of the probes vs the encoded examples (so a pass is
generalization, not recall), and the scoring helpers.
"""

from __future__ import annotations

from marker.instruct import (
    IM_END,
    IM_START,
    chat_skill_system_prefix,
    skill_correct,
)
from marker.run_instruct_skills import SKILLS, _examples_text

# ── Chat layout of the skill KV ─────────────────────────────────────────────────


def test_skill_prefix_has_sink_description_and_examples():
    p = chat_skill_system_prefix(
        "ilp_for",
        "ilp_for is a C++20 library. LoopType values: Sum, Bitwise.",
        [("Write a sum loop", "ILP_FOR_AUTO(auto i, 0, n, Sum, double) {} ILP_END;")],
    )
    assert p.startswith(f"{IM_START}system\n")  # attention sink genuinely first
    assert "About ilp_for:" in p
    assert "LoopType values" in p  # description present
    assert "ILP_FOR_AUTO" in p  # worked example present
    assert IM_END not in p  # block left open for the live tokens to continue


def test_skill_prefix_no_examples_is_just_description():
    p = chat_skill_system_prefix("X", "some desc", [])
    assert "About X:" in p
    assert "some desc" in p
    assert IM_END not in p


# ── Scoring helper ──────────────────────────────────────────────────────────────


def test_skill_correct_gold_substring():
    import re

    api = re.compile(r"ILP_[A-Z_]+")
    assert skill_correct("... ILP_FOR_AUTO(auto i, 0, n, Bitwise, uint32_t) ...", "Bitwise", api)
    assert not skill_correct("plain C++ loop, no macro", "Bitwise", api)


def test_skill_correct_control_is_api_absent():
    import re

    api = re.compile(r"client\.(emit|subscribe)\(")
    # gold=None => correct means the DSL API is ABSENT (no bleed on a no-term ask)
    assert skill_correct("for x in items: publish(x)", None, api)
    assert not skill_correct("client.emit('prices', p, ttl=30)", None, api)


# ── Data hygiene: the novel probes must be genuinely novel vs the examples ───────


def test_novel_probe_golds_absent_from_encoded_examples():
    # A skill "pass" only means generalization if the gold token the probe wants
    # is NOT already sitting verbatim in the worked examples the model was shown.
    for skill in SKILLS:
        ex = _examples_text(skill).lower()
        for gold, novel in skill["novelty"].items():
            if novel:
                assert gold.lower() not in ex, (
                    f"{skill['term']}: gold {gold!r} is in the examples — not a novel probe"
                )


def test_every_example_answer_uses_the_skill_api():
    # Each worked example must actually demonstrate the DSL, or it teaches nothing.
    for skill in SKILLS:
        for _q, a in skill["examples"]:
            assert skill["api_re"].search(a), (
                f"{skill['term']}: example answer {a[:40]!r} has no API pattern"
            )


def test_probes_and_golds_aligned_and_have_a_control():
    for skill in SKILLS:
        assert len(skill["probes"]) == len(skill["golds"]) >= 3
        assert None in skill["golds"], f"{skill['term']}: missing a no-term control probe"


def test_control_probe_omits_the_term():
    # The no-term control question must not mention the skill term, or it isn't
    # a bleed control.
    for skill in SKILLS:
        term = skill["term"].lower()
        for q, gold in zip(skill["probes"], skill["golds"], strict=True):
            if gold is None:
                assert term not in q.lower(), f"{skill['term']}: control probe mentions the term"
