"""Multi-turn skill DISENGAGEMENT on a chat model (hardening run).

Phase 3 showed skills ENGAGE for free on 7B-Instruct (5/6 from the description
alone, no MLP) but OVER-APPLY: a no-term request still emits the DSL. The first
multi-turn run confirmed the real bleed (persistent NO-BLEED 2/4) and that a
one-sentence STANCE clause fixed it (2/2 / 2/2 / 4/4). This run hardens that
result: 4 single-skill sessions (a bleedy general-purpose DSL + a narrow one
added) with diversified off-topic turns, plus an INTERLEAVED two-skill session
that tests cross-skill routing (naming skill B mid-session must switch off
skill A, not fire both).

Each session is a chat played turn-by-turn:
    engage    — names a skill, must ENGAGE that skill's DSL
    followup  — no term, continues the topic, must STAY ENGAGED
    offtopic  — no term, unrelated request, must NOT bleed ANY session skill

Mechanisms (what sits in the injected system block each turn):
    persistent  — every skill seen so far in the session (today's behavior)
    term-gated  — only skills named in the current turn (bare system otherwise)
    stance      — all-seen skills + a one-sentence "use only when asked" clause

Metrics: ENGAGE / FOLLOWUP (want the target DSL) ; NO-BLEED (offtopic: no
session DSL present) ; MISROUTE (on an engage/followup turn, a NON-target
session skill's DSL leaked — only interleaved sessions can misroute).

Faithful behavioral proxy: each turn is re-prefilled with the full conversation
(system KV + replayed history + current turn), fresh cache. Instruct-only.

Run (GPU):
    PYTHONPATH=src python -m marker.run_instruct_disengage \
        --instruct-name Qwen/Qwen2.5-7B-Instruct
Smoke (local):
    PYTHONPATH=src python -m marker.run_instruct_disengage --smoke
"""

from __future__ import annotations

import argparse
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marker.instruct import (
    chat_multiturn_suffix,
    decode_with_kv,
    encode_chat_system_kv,
    im_end_id,
    skill_correct,
)
from marker.run_axiom_mlp_demo import SKILL_AXIOM, SKILL_AXIOM_ILP
from marker.run_skill_quiet import ILP_API_RE, INTERNALBUS_API_RE

STANCE = (
    "Use these constructs only when the user's current request explicitly asks "
    "about them; for unrelated requests, write ordinary code."
)

MECHANISMS = ["persistent", "term-gated", "stance"]

# ── Two new skills for the hardening run ────────────────────────────────────────
# FluentSeq: a BLEEDY general-purpose collection DSL — map/filter/reduce fit
# almost any data task, so it is the acid test for over-application.
FLUENTSEQ_DESC = (
    "FluentSeq is a fictional fluent collection library. Wrap any iterable with "
    "seq(x), then chain: .map(fn), .filter(pred), .reduce(fn, init), .reverse(), "
    ".take(n); terminal .collect() materializes a list. "
    "Example: seq(nums).filter(is_even).map(square).collect()."
)
FLUENTSEQ_API_RE = re.compile(r"seq\([^)]*\)\.\w+")

# Chronos: a NARROW scheduling DSL — should rarely bleed (few tasks look like
# scheduling), the counterpart to InternalBus.
CHRONOS_DESC = (
    "Chronos is a fictional job scheduler. Recurring: chronos.every(seconds, task). "
    "One-shot at a Unix timestamp: chronos.at(ts, task). Cancel by handle: "
    "chronos.cancel(handle). Tasks are zero-arg callables."
)
CHRONOS_API_RE = re.compile(r"chronos\.(every|at|cancel)\(")

SKILLS: dict[str, dict] = {
    "InternalBus": {"desc": SKILL_AXIOM["description"], "api_re": INTERNALBUS_API_RE},
    "ilp_for": {"desc": SKILL_AXIOM_ILP["description"], "api_re": ILP_API_RE},
    "FluentSeq": {"desc": FLUENTSEQ_DESC, "api_re": FLUENTSEQ_API_RE},
    "Chronos": {"desc": CHRONOS_DESC, "api_re": CHRONOS_API_RE},
}

SESSIONS: list[dict] = [
    {
        "name": "internalbus",
        "skills": ["InternalBus"],
        "turns": [
            {
                "kind": "engage",
                "term": "InternalBus",
                "q": "Write code using InternalBus to publish a price update.",
                "gold": "client.emit(",
            },
            {
                "kind": "followup",
                "term": "InternalBus",
                "q": "Now also publish an order event the same way.",
                "gold": "client.emit(",
            },
            {
                "kind": "offtopic",
                "term": None,
                "q": "Write a function to reverse a string.",
                "gold": None,
            },
            {
                "kind": "offtopic",
                "term": None,
                "q": "Write a function to validate an email address.",
                "gold": None,
            },
        ],
    },
    {
        "name": "ilp_for",
        "skills": ["ilp_for"],
        "turns": [
            {
                "kind": "engage",
                "term": "ilp_for",
                "q": "Write a sum loop using ilp_for over doubles.",
                "gold": "ILP_",
            },
            {
                "kind": "followup",
                "term": "ilp_for",
                "q": "Now make a version that skips negative values.",
                "gold": "ILP_",
            },
            {
                "kind": "offtopic",
                "term": None,
                "q": "Write a function to check whether a number is prime.",
                "gold": None,
            },
            {
                "kind": "offtopic",
                "term": None,
                "q": "Write a function to parse an ISO date string into year, month, day.",
                "gold": None,
            },
        ],
    },
    {
        "name": "fluentseq",
        "skills": ["FluentSeq"],
        "turns": [
            {
                "kind": "engage",
                "term": "FluentSeq",
                "q": "Using FluentSeq, filter a list of numbers to the even ones then square them.",
                "gold": "seq(",
            },
            {
                "kind": "followup",
                "term": "FluentSeq",
                "q": "Now also sum the squared values.",
                "gold": "seq(",
            },
            {
                "kind": "offtopic",
                "term": None,
                "q": "Write a function to reverse a string.",
                "gold": None,
            },
            {
                "kind": "offtopic",
                "term": None,
                "q": "Write a function that returns the factorial of n.",
                "gold": None,
            },
        ],
    },
    {
        "name": "chronos",
        "skills": ["Chronos"],
        "turns": [
            {
                "kind": "engage",
                "term": "Chronos",
                "q": "Using Chronos, schedule a cleanup task to run every 60 seconds.",
                "gold": "chronos.every(",
            },
            {
                "kind": "followup",
                "term": "Chronos",
                "q": "Now also schedule a one-off report to run at a given timestamp.",
                "gold": "chronos.",
            },
            {
                "kind": "offtopic",
                "term": None,
                "q": "Write a function to sort a list of integers in ascending order.",
                "gold": None,
            },
            {
                "kind": "offtopic",
                "term": None,
                "q": "Write a function to count the vowels in a string.",
                "gold": None,
            },
        ],
    },
    {
        # Cross-skill routing: skill A (ilp_for) then skill B (InternalBus) in one
        # conversation. Naming B must switch to B without firing A, and vice versa.
        "name": "interleaved",
        "skills": ["ilp_for", "InternalBus"],
        "turns": [
            {
                "kind": "engage",
                "term": "ilp_for",
                "q": "Write a sum loop using ilp_for over doubles.",
                "gold": "ILP_",
            },
            {
                "kind": "engage",
                "term": "InternalBus",
                "q": "Now publish the resulting sum using InternalBus.",
                "gold": "client.emit(",
            },
            {
                "kind": "followup",
                "term": "ilp_for",
                "q": "Go back and make the loop skip negative values.",
                "gold": "ILP_",
            },
            {
                "kind": "offtopic",
                "term": None,
                "q": "Write a function to validate an email address.",
                "gold": None,
            },
        ],
    },
]


def build_system_body(terms: list[str], stance: bool) -> str:
    """System-block body carrying each active skill's description, plus the
    stance clause when requested. Empty -> a bare assistant system."""
    if not terms:
        return "You are a helpful coding assistant."
    body = "\n\n".join(f"About {t}:\n{SKILLS[t]['desc']}" for t in terms)
    if stance:
        body += "\n" + STANCE
    return body


def active_terms_for_turn(
    mechanism: str, session_skills: list[str], seen: set[str], current: set[str]
) -> list[str]:
    """Which skills' descriptions sit in the system block this turn.
    term-gated: only those named this turn; persistent/stance: all seen so far.
    Order follows session_skills for determinism."""
    keep = current if mechanism == "term-gated" else seen
    return [t for t in session_skills if t in keep]


def _terms_named(q: str, session_skills: list[str]) -> set[str]:
    low = q.lower()
    return {t for t in session_skills if t.lower() in low}


def _bled_skills(out: str, terms: list[str]) -> list[str]:
    return [t for t in terms if SKILLS[t]["api_re"].search(out)]


def _run_mechanism(model, tokenizer, mechanism: str, max_new: int) -> dict[str, tuple[int, int]]:  # noqa: ANN001
    """Play every session under `mechanism`; return per-metric (correct, total)."""
    stop = im_end_id(tokenizer)
    tally = {k: [0, 0] for k in ("engage", "followup", "nobleed", "misroute")}
    stance = mechanism == "stance"

    print(f"\n--- {mechanism} ---")
    for sess in SESSIONS:
        skills = sess["skills"]
        seen: set[str] = set()
        history: list[tuple[str, str]] = []
        for t in sess["turns"]:
            current = _terms_named(t["q"], skills)
            seen |= current
            body = build_system_body(
                active_terms_for_turn(mechanism, skills, seen, current), stance
            )
            kv = encode_chat_system_kv(model, tokenizer, body)
            out = decode_with_kv(
                model, tokenizer, kv, chat_multiturn_suffix(history, t["q"]), max_new, stop
            )

            if t["kind"] == "offtopic":
                bled = _bled_skills(out, skills)
                ok = not bled
                tally["nobleed"][0] += int(ok)
                tally["nobleed"][1] += 1
                tag = "clean" if ok else f"BLED:{','.join(bled)}"
            else:
                ok = skill_correct(out, t["gold"], SKILLS[t["term"]]["api_re"])
                tally[t["kind"]][0] += int(ok)
                tally[t["kind"]][1] += 1
                others = [s for s in skills if s != t["term"]]
                misrouted = _bled_skills(out, others)
                tally["misroute"][0] += int(not misrouted)
                tally["misroute"][1] += 1
                tag = ("engaged" if ok else "MISS") + (
                    f" +MISROUTE:{','.join(misrouted)}" if misrouted else ""
                )

            print(
                f"    [{sess['name']:11} {t['kind']:8}] {tag:20} {out[:52].replace(chr(10), ' ')}"
            )
            history.append(("user", t["q"]))
            history.append(("assistant", out))

    return {k: (v[0], v[1]) for k, v in tally.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruct-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-new", type=int, default=200)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.instruct_name = "Qwen/Qwen2.5-0.5B-Instruct"
        args.max_new = min(args.max_new, 80)
        print("=== SMOKE MODE ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  model: {args.instruct_name}")
    print(f"sessions: {[s['name'] for s in SESSIONS]}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.instruct_name)
    model = (
        AutoModelForCausalLM.from_pretrained(args.instruct_name, torch_dtype=torch.bfloat16)
        .to(device)
        .eval()
    )

    results = {m: _run_mechanism(model, tokenizer, m, args.max_new) for m in MECHANISMS}

    print("\n" + "=" * 78)
    print("SUMMARY  (engage/followup want the DSL; nobleed & misroute want it ABSENT)")
    print("=" * 78)
    print(f"  {'mechanism':14} {'ENGAGE':>9} {'FOLLOWUP':>9} {'NO-BLEED':>9} {'MISROUTE-ok':>12}")
    for m in MECHANISMS:
        r = results[m]
        cells = " ".join(f"{f'{r[k][0]}/{r[k][1]}':>9}" for k in ("engage", "followup", "nobleed"))
        mr = r["misroute"]
        print(f"  {m:14} {cells} {f'{mr[0]}/{mr[1]}':>12}")

    print("\nGATE: a mechanism WINS if ENGAGE/FOLLOWUP stay high AND NO-BLEED + MISROUTE")
    print("      are full. If none does, retrain learned-silence (Phase 4).")


if __name__ == "__main__":
    main()
