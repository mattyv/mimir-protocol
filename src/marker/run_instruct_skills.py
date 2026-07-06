"""Instruct portability Phase 3: do SKILLS work on a chat model with the DSL
description + a few worked examples baked into the frozen KV, skill MLP
DISABLED?

See INSTRUCT_PLAN.md Phase 3. The fact result (Phase 1) inverted the refusal
hypothesis: facts recall perfectly on 7B-Instruct with zero help. Skills are a
different mechanism — produced token-by-token in the decode loop, where the
RLHF style prior votes on every token — so decode drift, not refusal, is the
risk. This run measures it.

Conditions (all MLP-disabled — the whole point is KV-only):
    base-ref          base model, 'About X' desc+examples KV, TEMPLATE probe
    instruct-desconly chat model, description-ONLY KV (the failure baseline:
                      "do skills fail on a chat model?")
    instruct-examples chat model, description + worked-examples KV (the fix)
    instruct-examples-preamble  + the anti-drift preamble

Each scored per skill on:
    POSITIVE (term present): novel probes must ENGAGE the DSL (correct gold).
    NEGATIVE (no term):      control probe must NOT bleed the DSL (API absent).

The NEGATIVE case is the skill analog of BOUNDARY — a skill that fires
unconditionally is broken, so a pass requires positive AND negative.

Fresh cache per probe. Base and instruct loaded sequentially (two 7B models
won't co-reside). Reuses the Phase-0/1 chat machinery in marker.instruct.

GATE: instruct-examples POSITIVE recovers materially over instruct-desconly
AND NEGATIVE does not regress (control stays clean). If desc-only already
passes, skills — like facts — "just work" on chat and examples are optional.

Run (GPU):
    PYTHONPATH=src python -m marker.run_instruct_skills \
        --model-name Qwen/Qwen2.5-7B --instruct-name Qwen/Qwen2.5-7B-Instruct
Smoke (local, pinned transformers):
    PYTHONPATH=src python -m marker.run_instruct_skills --smoke
"""

from __future__ import annotations

import argparse
import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marker.instruct import (
    base_skill_axiom_kv,
    chat_live_suffix,
    declined,
    decode_with_kv,
    encode_chat_skill_kv,
    im_end_id,
    skill_correct,
)
from marker.run_axiom_mlp_demo import SKILL_AXIOM, SKILL_AXIOM_ILP, TEMPLATE
from marker.run_instruct_phase1 import PREAMBLE
from marker.run_skill_quiet import ILP_API_RE, INTERNALBUS_API_RE

# ── Skill definitions: DSL description + worked examples (KV) + probes ───────────
# Worked examples teach the FORMS; probes ask for variations. Novelty flags mark
# which golds are genuinely absent from the examples (so a pass is generalization,
# not recall) — pinned by tests/test_instruct_skills.py.

SKILLS: list[dict] = [
    {
        "term": "InternalBus",
        "description": SKILL_AXIOM["description"],
        "examples": [
            (
                "Write code using InternalBus to publish a price update",
                "client.emit('prices', price_update, ttl=30)",
            ),
            (
                "Write code using InternalBus to subscribe to order events",
                "client.subscribe('orders', handle_order)",
            ),
        ],
        "probes": [
            "Write code using InternalBus to publish a balance update to the 'balances' channel",
            "Write code using InternalBus to subscribe to 'inventory' events",
            "How do you publish a message using InternalBus?",
            "Write code to publish a price update",  # no-term control
        ],
        "golds": ["client.emit('balances'", "client.subscribe('inventory'", "client.emit(", None],
        # gold -> is it novel vs the examples? (only checked for non-None golds)
        "novelty": {
            "client.emit('balances'": True,
            "client.subscribe('inventory'": True,
            "client.emit(": False,  # fact retrieval, shown in examples
        },
        "api_re": INTERNALBUS_API_RE,
    },
    {
        "term": "ilp_for",
        "description": SKILL_AXIOM_ILP["description"],
        "examples": [
            (
                "Write a sum loop using ilp_for over doubles",
                "ILP_FOR_AUTO(auto i, 0, n, Sum, double) {\n    total += data[i];\n} ILP_END;",
            ),
            (
                "Write a search loop using ilp_for that returns the index of a target",
                "ILP_FOR(auto i, 0, (int)data.size(), 4) {\n    if (data[i] == target) "
                "ILP_RETURN(i);\n} ILP_END_RETURN;\nreturn -1;",
            ),
            (
                "Write a loop using ilp_for that skips negative values",
                "ILP_FOR(auto i, 0, n, 4) {\n    if (data[i] < 0) ILP_CONTINUE;\n    "
                "sum += data[i];\n} ILP_END;",
            ),
        ],
        "probes": [
            "Write a bitwise AND loop using ilp_for over uint32_t",
            "Write a loop using ilp_for that searches for the first negative number "
            "and stores its index",
            "Write a function using ilp_for that returns the index of the first element "
            "greater than a threshold",
            "Write a C++ loop to sum an array of doubles",  # no-term control
        ],
        # Bitwise: novel LoopType (from the description's enum list, not the examples).
        # ILP_BREAK: novel control-flow (examples use RETURN/CONTINUE, not BREAK).
        # ILP_END_RETURN: terminator RULE shown in an example, applied to a new predicate.
        "golds": ["Bitwise", "ILP_BREAK", "ILP_END_RETURN", None],
        "novelty": {"Bitwise": True, "ILP_BREAK": True, "ILP_END_RETURN": False},
        "api_re": ILP_API_RE,
    },
]


# Entity-unfamiliarity refusals — the hypothesized skill failure mode on chat
# models ("I don't know anything about ilp_for"). Distinct from the fact
# BOUNDARY declined() list: for a SKILL these are FAILURES, not correct behavior,
# and the phrasing is about not recognizing the term, not about missing detail.
_REFUSAL_MARKERS = [
    "don't know anything about",
    "do not know anything about",
    "not familiar with",
    "never heard of",
    "not aware of",
    "no knowledge of",
    "don't have information",
    "do not have information",
    "doesn't exist",
    "does not exist",
    "not a real",
    "not a recognized",
    "not a standard",
    "don't recognize",
    "do not recognize",
    "no such",
    "isn't a",
    "is not a known",
]


def _refused(answer: str) -> bool:
    """True if the model refused on entity-unfamiliarity grounds (or used a
    fact-style decline) — a POSITIVE-case failure worth distinguishing from
    silent decode drift."""
    low = answer.lower()
    return declined(low) or any(m in low for m in _REFUSAL_MARKERS)


def _examples_text(skill: dict) -> str:
    """The concatenated worked-example text, for the novelty hygiene tests."""
    return " ".join(f"{q} {a}" for q, a in skill["examples"])


def _score_condition(
    model,  # noqa: ANN001
    tokenizer,  # noqa: ANN001
    *,
    chat: bool,
    with_examples: bool,
    preamble: str | None,
    max_new: int,
    label: str,
) -> dict[str, tuple[int, int]]:
    """Returns {'positive': (c,t), 'negative': (c,t), 'refused': (r,t)}
    aggregated over skills. 'refused' counts positive-case entity-unfamiliarity
    refusals — the hypothesized chat-model skill failure mode."""
    stop = im_end_id(tokenizer) if chat else None
    p_c = p_t = n_c = n_t = ref = 0

    for skill in SKILLS:
        term, desc = skill["term"], skill["description"]
        examples = skill["examples"] if with_examples else []
        api_re = skill["api_re"]

        kv = (
            encode_chat_skill_kv(model, tokenizer, term, desc, examples)
            if chat
            else base_skill_axiom_kv(model, tokenizer, term, desc, examples)
        )

        def live(q: str, _chat=chat, _pre=preamble) -> str:
            return chat_live_suffix(q, _pre) if _chat else TEMPLATE.format(q=q)

        for q, gold in zip(skill["probes"], skill["golds"], strict=True):
            out = decode_with_kv(model, tokenizer, kv, live(q), max_new, stop)
            ok = skill_correct(out, gold, api_re)
            if gold is None:
                n_c += int(ok)
                n_t += 1
                tag = "NEG(no-bleed)" if ok else "NEG(BLED)"
            else:
                p_c += int(ok)
                p_t += 1
                if ok:
                    tag = "POS(engaged)"
                elif _refused(out):
                    ref += 1
                    tag = "POS(REFUSED)"  # "I don't know anything about X"
                else:
                    tag = "POS(drift)"  # wrong/plain code, no refusal
            print(f"    [{label}] {tag:14} {gold!r:>26}  {out[:64].replace(chr(10), ' ')}")

    return {"positive": (p_c, p_t), "negative": (n_c, n_t), "refused": (ref, p_t)}


def _load(name: str, device: str):  # noqa: ANN001, ANN202
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16).to(device).eval()
    return model, tok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--instruct-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-new", type=int, default=120)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.model_name = "Qwen/Qwen2.5-0.5B"
        args.instruct_name = "Qwen/Qwen2.5-0.5B-Instruct"
        args.max_new = min(args.max_new, 60)
        print("=== SMOKE MODE ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}\nskills: {[s['term'] for s in SKILLS]}\n")

    results: dict[str, dict[str, tuple[int, int]]] = {}

    # ── Base reference (desc+examples KV, NO MLP) ────────────────────────────────
    print("=" * 70, "\nBASE REFERENCE (KV-only, no MLP):", args.model_name, "\n", "=" * 70, sep="")
    base_model, base_tok = _load(args.model_name, device)
    results["base-ref"] = _score_condition(
        base_model,
        base_tok,
        chat=False,
        with_examples=True,
        preamble=None,
        max_new=args.max_new,
        label="base-ref",
    )
    del base_model, base_tok
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # ── Instruct conditions ──────────────────────────────────────────────────────
    print("\n", "=" * 70, "\nINSTRUCT:", args.instruct_name, "\n", "=" * 70, sep="")
    ins_model, ins_tok = _load(args.instruct_name, device)
    for label, with_examples, preamble in [
        ("instruct-desconly", False, None),  # failure baseline
        ("instruct-examples", True, None),  # the fix
        ("instruct-examples-preamble", True, PREAMBLE),
    ]:
        print(f"\n--- {label} ---")
        results[label] = _score_condition(
            ins_model,
            ins_tok,
            chat=True,
            with_examples=with_examples,
            preamble=preamble,
            max_new=args.max_new,
            label=label,
        )

    # ── Summary ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'condition':28} {'POSITIVE(engage)':>18} {'NEGATIVE(no-bleed)':>20} {'refused':>10}")
    for label in [
        "base-ref",
        "instruct-desconly",
        "instruct-examples",
        "instruct-examples-preamble",
    ]:
        p, n, r = (
            results[label]["positive"],
            results[label]["negative"],
            results[label]["refused"],
        )
        print(f"  {label:28} {f'{p[0]}/{p[1]}':>18} {f'{n[0]}/{n[1]}':>20} {f'{r[0]}/{r[1]}':>10}")

    print("\nGATE: instruct-examples POSITIVE >> instruct-desconly POSITIVE, AND")
    print("      NEGATIVE stays clean (no DSL bleed on the no-term control).")


if __name__ == "__main__":
    main()
