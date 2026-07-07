"""Multi-turn skill DISENGAGEMENT on a chat model.

Phase 3 showed skills ENGAGE for free on 7B-Instruct (5/6 from the description
alone, no MLP) but OVER-APPLY: a no-term request still emits the DSL (NEGATIVE
0/2). Single-turn that's a proxy; the real blocker is multi-turn — once a skill
term appears its injected KV persists, and later off-topic turns bleed the DSL.

This eval reproduces the real failure and pits three ZERO-TRAINING mechanisms
against it. Each session is a 4-turn chat:
    engage    — names the term, must ENGAGE the DSL
    followup  — no term, continues the topic, must STAY ENGAGED
    offtopic  — no term, unrelated request, must NOT bleed the DSL (x2)

Mechanisms (what sits in the injected system block each turn):
    persistent  — skill description always present (today's behavior; bleeds)
    term-gated  — skill present only on turns naming the term (kills bleed but
                  risks killing the follow-up — history still carries the DSL)
    stance      — skill always present + a one-sentence "use only when asked"
                  clause (the cheap hoped-for fix)

The crux: term-gating and stance both must hold FOLLOWUP (engaged) and OFFTOPIC
(clean) apart. If stance does, disengagement is solved with no training — same
shape as the fact result. If nothing does, that's the signal to retrain the
learned-silence MLP against the chat model (Phase 4).

Faithful behavioral proxy: each turn is re-prefilled with the full conversation
(system KV + replayed history + current turn), fresh cache. That captures
whether the skill is in-context for the turn, which is all that drives
engage-vs-bleed — the session-cache persistence/RoPE mechanics are a runtime
optimization, not a behavioral factor. Instruct-only (chat is the target).

Run (GPU):
    PYTHONPATH=src python -m marker.run_instruct_disengage \
        --instruct-name Qwen/Qwen2.5-7B-Instruct
Smoke (local):
    PYTHONPATH=src python -m marker.run_instruct_disengage --smoke
"""

from __future__ import annotations

import argparse

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

SESSIONS: list[dict] = [
    {
        "term": "InternalBus",
        "desc": SKILL_AXIOM["description"],
        "api_re": INTERNALBUS_API_RE,
        "turns": [
            {
                "kind": "engage",
                "q": "Write code using InternalBus to publish a price update.",
                "gold": "client.emit(",
            },
            {
                "kind": "followup",
                "q": "Now also publish an order event the same way.",
                "gold": "client.emit(",
            },
            {"kind": "offtopic", "q": "Write a function to reverse a string.", "gold": None},
            {
                "kind": "offtopic",
                "q": "Write a function to compute the factorial of n.",
                "gold": None,
            },
        ],
    },
    {
        "term": "ilp_for",
        "desc": SKILL_AXIOM_ILP["description"],
        "api_re": ILP_API_RE,
        "turns": [
            {
                "kind": "engage",
                "q": "Write a sum loop using ilp_for over doubles.",
                "gold": "ILP_",
            },
            {
                "kind": "followup",
                "q": "Now make a version that skips negative values.",
                "gold": "ILP_",
            },
            {"kind": "offtopic", "q": "Write a function to reverse a string.", "gold": None},
            {
                "kind": "offtopic",
                "q": "Write a function to check whether a number is prime.",
                "gold": None,
            },
        ],
    },
]


def disengage_system_text(mechanism: str, term: str, desc: str, term_present: bool) -> str:
    """The system-block body for a turn under `mechanism`. `term_present` is
    whether the current user message names the skill term."""
    skill = f"About {term}:\n{desc}"
    bare = "You are a helpful coding assistant."
    if mechanism == "persistent":
        return skill
    if mechanism == "term-gated":
        return skill if term_present else bare
    if mechanism == "stance":
        return f"{skill}\n{STANCE}"
    raise ValueError(f"unknown mechanism {mechanism!r}")


def _run_mechanism(
    model,  # noqa: ANN001
    tokenizer,  # noqa: ANN001
    mechanism: str,
    max_new: int,
) -> dict[str, tuple[int, int]]:
    """Play every session turn-by-turn under `mechanism`; return
    {'engage': (c,t), 'followup': (c,t), 'nobleed': (c,t)}."""
    stop = im_end_id(tokenizer)
    tally = {"engage": [0, 0], "followup": [0, 0], "nobleed": [0, 0]}

    print(f"\n--- {mechanism} ---")
    for sess in SESSIONS:
        term, desc, api_re = sess["term"], sess["desc"], sess["api_re"]
        history: list[tuple[str, str]] = []
        for t in sess["turns"]:
            term_present = term.lower() in t["q"].lower()
            body = disengage_system_text(mechanism, term, desc, term_present)
            kv = encode_chat_system_kv(model, tokenizer, body)
            live = chat_multiturn_suffix(history, t["q"])
            out = decode_with_kv(model, tokenizer, kv, live, max_new, stop)

            ok = skill_correct(out, t["gold"], api_re)
            metric = "nobleed" if t["kind"] == "offtopic" else t["kind"]
            tally[metric][0] += int(ok)
            tally[metric][1] += 1
            if t["kind"] == "offtopic":
                tag = "clean" if ok else "BLED"
            else:
                tag = "engaged" if ok else "MISS"
            print(f"    [{term:11} {t['kind']:8}] {tag:8} {out[:60].replace(chr(10), ' ')}")

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
    print(f"sessions: {[s['term'] for s in SESSIONS]}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.instruct_name)
    model = (
        AutoModelForCausalLM.from_pretrained(args.instruct_name, torch_dtype=torch.bfloat16)
        .to(device)
        .eval()
    )

    results = {m: _run_mechanism(model, tokenizer, m, args.max_new) for m in MECHANISMS}

    print("\n" + "=" * 70)
    print("SUMMARY  (engage/followup want the DSL; nobleed wants it ABSENT)")
    print("=" * 70)
    print(f"  {'mechanism':14} {'ENGAGE':>10} {'FOLLOWUP':>10} {'NO-BLEED':>10}")
    for m in MECHANISMS:
        e, f, n = results[m]["engage"], results[m]["followup"], results[m]["nobleed"]
        print(f"  {m:14} {f'{e[0]}/{e[1]}':>10} {f'{f[0]}/{f[1]}':>10} {f'{n[0]}/{n[1]}':>10}")

    print("\nGATE: a mechanism WINS if ENGAGE and FOLLOWUP stay high AND NO-BLEED")
    print("      recovers to full. If none does, retrain learned-silence (Phase 4).")


if __name__ == "__main__":
    main()
