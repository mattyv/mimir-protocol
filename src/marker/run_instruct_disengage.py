"""Multi-turn skill DISENGAGEMENT on a chat model — strict vs sticky gating.

Runs 1-2 (see CONCLUSIONS.md) established: skills bleed under persistent
injection; a stance clause alone doesn't reliably stop it (a bleedy DSL whose
method name collides with the off-topic verb still leaked). Fable's review of
those runs flagged three eval flaws (asymmetric scoring, no syntax-fidelity
check, truncated/uncapturable outputs) and named the real open question this
run answers: term-gating disengages perfectly on a same-turn off-topic
request, but does it also correctly evict a skill needed by a LATER, term-less
follow-up that arrives after a distraction turn?

Mechanisms:
    strict-gated       — skill's KV present only on turns naming it this turn.
    sticky-gated-k2     — KV persists STICKY_K turns after the last mention
                          (a decaying counter, reset to K on each mention).
    sticky-gated-k2-stance — same presence rule as sticky, + the STANCE clause
                          appended whenever any skill is active (tests whether
                          stance reduces bleed DURING the sticky window without
                          losing the follow-up benefit).

Every session is shaped: engage (names the term) -> offtopic (distraction,
inside the sticky window) -> followup (term-less, tests re-engagement after
the distraction) -> offtopic (after; sticky should have evicted by now).
This directly measures the strict-miss / sticky-bleed tradeoff Fable named,
using the SAME api-pattern instrument for both engagement and bleed (no more
asymmetric scoring), plus a syntax-fidelity gold on engage/followup turns so a
degraded from-history re-engagement can't pass on a loose substring alone.

Full, UNTRUNCATED per-turn outputs are dumped as one-line JSON (TURNJSON ...)
so a reviewer can inspect every generation from the captured log — the first
two runs printed 52-char previews and the node was gone before anyone could
check whether a "MISS" was verbosity or a wrong API.

Run (GPU):
    PYTHONPATH=src python -m marker.run_instruct_disengage \
        --instruct-name Qwen/Qwen2.5-7B-Instruct
Smoke (local):
    PYTHONPATH=src python -m marker.run_instruct_disengage --smoke
"""

from __future__ import annotations

import argparse
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marker.instruct import (
    chat_multiturn_suffix,
    decode_with_kv,
    encode_chat_system_kv,
    im_end_id,
)
from marker.run_axiom_mlp_demo import SKILL_AXIOM, SKILL_AXIOM_ILP
from marker.run_skill_quiet import ILP_API_RE, INTERNALBUS_API_RE

STANCE = (
    "Use these constructs only when the user's current request explicitly asks "
    "about them; for unrelated requests, write ordinary code."
)

STICKY_K = 2
MECHANISMS = ["strict-gated", "sticky-gated-k2", "sticky-gated-k2-stance"]

# ── Skill registry (InternalBus/ilp_for from Phase 3; FluentSeq/Chronos added
# in the hardening run — FluentSeq is the bleedy general-purpose acid test) ──────

FLUENTSEQ_DESC = (
    "FluentSeq is a fictional fluent collection library. Wrap any iterable with "
    "seq(x), then chain: .map(fn), .filter(pred), .reduce(fn, init), .reverse(), "
    ".take(n); terminal .collect() materializes a list. "
    "Example: seq(nums).filter(is_even).map(square).collect()."
)
FLUENTSEQ_API_RE = re.compile(r"seq\([^)]*\)\.\w+")

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

# Each session: engage -> offtopic (distraction, inside sticky window) ->
# followup (term-less, tests re-engagement) -> offtopic (after, should be
# clean under any mechanism). gold = loose API substring; fidelity = a
# stricter syntax requirement that a degraded from-history answer might miss.
SESSIONS: list[dict] = [
    {
        "name": "internalbus_gap",
        "skills": ["InternalBus"],
        "turns": [
            {
                "kind": "engage",
                "term": "InternalBus",
                "q": "Write code using InternalBus to publish a price update.",
                "gold": "client.emit(",
                "fidelity": "ttl=",
            },
            {
                "kind": "offtopic",
                "term": None,
                "q": "Write a function to validate an email address.",
                "gold": None,
                "fidelity": None,
            },
            {
                "kind": "followup",
                "term": "InternalBus",
                "q": "Now do the same thing but for an order event instead.",
                "gold": "client.emit(",
                "fidelity": "ttl=",
            },
            {
                "kind": "offtopic",
                "term": None,
                "q": "Write a function to compute the factorial of n.",
                "gold": None,
                "fidelity": None,
            },
        ],
    },
    {
        "name": "ilp_for_gap",
        "skills": ["ilp_for"],
        "turns": [
            {
                "kind": "engage",
                "term": "ilp_for",
                "q": "Write a sum loop using ilp_for over doubles.",
                "gold": "ILP_FOR",
                "fidelity": "ILP_END",
            },
            {
                "kind": "offtopic",
                "term": None,
                "q": "Write a function to check whether a number is prime.",
                "gold": None,
                "fidelity": None,
            },
            {
                "kind": "followup",
                "term": "ilp_for",
                "q": "Go back and change that loop to skip negative values.",
                "gold": "ILP_FOR",
                "fidelity": "ILP_END",
            },
            {
                "kind": "offtopic",
                "term": None,
                "q": "Write a function to parse an ISO date string.",
                "gold": None,
                "fidelity": None,
            },
        ],
    },
    {
        "name": "fluentseq_gap",
        "skills": ["FluentSeq"],
        "turns": [
            {
                "kind": "engage",
                "term": "FluentSeq",
                "q": "Using FluentSeq, filter a list of numbers to the even ones then square them.",
                "gold": "seq(",
                "fidelity": ".collect()",
            },
            {
                "kind": "offtopic",
                "term": None,
                "q": "Write a function to reverse a string.",
                "gold": None,
                "fidelity": None,
            },
            {
                "kind": "followup",
                "term": "FluentSeq",
                "q": "Now change that to also sum the squared values instead of collecting a list.",
                "gold": "seq(",
                "fidelity": None,
            },
            {
                "kind": "offtopic",
                "term": None,
                "q": "Write a function that returns the factorial of n.",
                "gold": None,
                "fidelity": None,
            },
        ],
    },
    {
        "name": "chronos_gap",
        "skills": ["Chronos"],
        "turns": [
            {
                "kind": "engage",
                "term": "Chronos",
                "q": "Using Chronos, schedule a cleanup task to run every 60 seconds.",
                "gold": "chronos.every(",
                "fidelity": None,
            },
            {
                "kind": "offtopic",
                "term": None,
                "q": "Write a function to sort a list of integers.",
                "gold": None,
                "fidelity": None,
            },
            {
                "kind": "followup",
                "term": "Chronos",
                "q": "Now change that to run once at a specific timestamp instead of recurring.",
                "gold": "chronos.at(",
                "fidelity": None,
            },
            {
                "kind": "offtopic",
                "term": None,
                "q": "Write a function to count the vowels in a string.",
                "gold": None,
                "fidelity": None,
            },
        ],
    },
    {
        # Cross-skill: engaging B is the "distraction" for A's follow-up —
        # tests re-engagement AND routing (must not misroute to B) together.
        "name": "interleaved_gap",
        "skills": ["ilp_for", "InternalBus"],
        "turns": [
            {
                "kind": "engage",
                "term": "ilp_for",
                "q": "Write a sum loop using ilp_for over doubles.",
                "gold": "ILP_FOR",
                "fidelity": "ILP_END",
            },
            {
                "kind": "engage",
                "term": "InternalBus",
                "q": "Separately, write code using InternalBus to publish a price update.",
                "gold": "client.emit(",
                "fidelity": "ttl=",
            },
            {
                "kind": "followup",
                "term": "ilp_for",
                "q": "Go back to that loop and make it skip negative values.",
                "gold": "ILP_FOR",
                "fidelity": "ILP_END",
            },
            {
                "kind": "offtopic",
                "term": None,
                "q": "Write a function to validate an email address.",
                "gold": None,
                "fidelity": None,
            },
        ],
    },
]


def active_terms_for_turn(
    mechanism: str,
    session_skills: list[str],
    last_seen: dict[str, int],
    turn_idx: int,
    current: set[str],
) -> tuple[list[str], dict[str, int]]:
    """Which skills sit in the system block this turn, and the updated
    last-seen turn-index map (unused for strict-gated, which has no
    persistence state).

    Sticky activity is distance-since-last-mention (turn_idx - last_seen[s]
    <= STICKY_K), not a decrementing counter — a decrement-then-check counter
    consumes one unit of "grace" on the very turn it's meant to cover, so
    K=2 only ever survives 1 silent turn, not 2 (an off-by-one that made a
    real run indistinguishable from strict-gated at the turn that mattered).
    Distance-based has no such boundary error: K=2 means active for turns at
    distance 0, 1, AND 2 from the mention, evicted only at distance K+1."""
    new_last_seen = dict(last_seen)
    for s in current:
        new_last_seen[s] = turn_idx

    if mechanism == "strict-gated":
        return [s for s in session_skills if s in current], new_last_seen

    active = [
        s for s in session_skills if s in new_last_seen and turn_idx - new_last_seen[s] <= STICKY_K
    ]
    return active, new_last_seen


def build_system_body(terms: list[str], stance: bool) -> str:
    if not terms:
        return "You are a helpful coding assistant."
    body = "\n\n".join(f"About {t}:\n{SKILLS[t]['desc']}" for t in terms)
    if stance:
        body += "\n" + STANCE
    return body


def _matched_skills(output: str, skills: list[str]) -> set[str]:
    return {s for s in skills if SKILLS[s]["api_re"].search(output)}


def score_turn(
    output: str, t: dict, skills: list[str], skill_registry: dict
) -> tuple[bool, str, set[str]]:
    """Symmetric scoring: the same api_re match drives both engagement and
    bleed detection, for every skill in the session. Returns (ok, tag, matched)."""
    matched = {s for s in skills if skill_registry[s]["api_re"].search(output)}

    if t["kind"] == "offtopic":
        ok = not matched
        tag = "clean" if ok else f"BLED:{','.join(sorted(matched))}"
        return ok, tag, matched

    target = t["term"]
    gold_ok = t["gold"].lower() in output.lower()
    fidelity_ok = t.get("fidelity") is None or t["fidelity"].lower() in output.lower()
    api_ok = target in matched
    misrouted = matched - {target}
    ok = gold_ok and fidelity_ok and api_ok and not misrouted

    if ok:
        tag = "engaged"
    elif misrouted:
        tag = f"MISROUTE:{','.join(sorted(misrouted))}"
    elif not fidelity_ok:
        tag = "MISS(fidelity)"
    else:
        tag = "MISS"
    return ok, tag, matched


def _run_mechanism(model, tokenizer, mechanism: str, max_new: int) -> dict[str, tuple[int, int]]:  # noqa: ANN001
    stop = im_end_id(tokenizer)
    tally = {k: [0, 0] for k in ("engage", "followup", "nobleed", "misroute")}
    stance = mechanism == "sticky-gated-k2-stance"

    print(f"\n--- {mechanism} ---")
    for sess in SESSIONS:
        skills = sess["skills"]
        last_seen: dict[str, int] = {}
        history: list[tuple[str, str]] = []
        for turn_idx, t in enumerate(sess["turns"]):
            current = {s for s in skills if s.lower() in t["q"].lower()}
            active, last_seen = active_terms_for_turn(
                mechanism, skills, last_seen, turn_idx, current
            )
            body = build_system_body(active, stance)
            kv = encode_chat_system_kv(model, tokenizer, body)
            live = chat_multiturn_suffix(history, t["q"])
            out = decode_with_kv(model, tokenizer, kv, live, max_new, stop)

            ok, tag, matched = score_turn(out, t, skills, SKILLS)
            if t["kind"] == "offtopic":
                tally["nobleed"][0] += int(ok)
                tally["nobleed"][1] += 1
            else:
                tally[t["kind"]][0] += int(ok)
                tally[t["kind"]][1] += 1
                tally["misroute"][0] += int(not (matched - {t["term"]}))
                tally["misroute"][1] += 1

            print(
                f"    [{sess['name']:16} t{turn_idx} {t['kind']:8}] {tag:20} "
                f"{out[:50].replace(chr(10), ' ')}"
            )
            # Capped at 400 chars: the vastai-logs capture channel hard-wraps
            # around col ~490 (a pty artifact), which corrupts a JSON string
            # mid-escape when the full untruncated text is dumped. 400 stays
            # safely under that while still showing far more than the 50-char
            # preview — enough to see whether a fidelity marker appears.
            print(
                "TURNJSON "
                + json.dumps(
                    {
                        "mechanism": mechanism,
                        "session": sess["name"],
                        "turn": turn_idx,
                        "kind": t["kind"],
                        "active_skills": active,
                        "ok": ok,
                        "tag": tag,
                        "matched": sorted(matched),
                        "output": out[:400],
                    }
                )
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
    print(f"device: {device}  model: {args.instruct_name}  sticky_k: {STICKY_K}")
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
    print(f"  {'mechanism':22} {'ENGAGE':>9} {'FOLLOWUP':>9} {'NO-BLEED':>9} {'MISROUTE-ok':>12}")
    for m in MECHANISMS:
        r = results[m]
        cells = " ".join(f"{f'{r[k][0]}/{r[k][1]}':>9}" for k in ("engage", "followup", "nobleed"))
        mr = r["misroute"]
        print(f"  {m:22} {cells} {f'{mr[0]}/{mr[1]}':>12}")

    print("\nGATE: a mechanism WINS if FOLLOWUP recovers (the strict-gating risk) AND")
    print("      NO-BLEED/MISROUTE stay full (the sticky/stance risk).")


if __name__ == "__main__":
    main()
