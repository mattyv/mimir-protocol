"""Learned-silence experiment: can a skill MLP learn to disengage on prose?

See SKILL_QUIET_PLAN.md for the full design and pre-registered pass/kill:
    PASS: arm B keeps regression probes within 1 of arm C, bleed probes show
          less prose contamination than C, and the offset-norm trace drops
          >=5x from code segment to prose segment.
    KILL: if every lam that quiets it costs >2 regression probes, report the
          trade-off curve and stop.

Three arms per skill:
    C (control) — current recipe, pure-code pairs only, lam=0
    A           — + mixed/prose pairs, lam=0 (does data alone quiet it?)
    B           — + mixed/prose pairs, lam=--lam (the stated objective)

Run (GPU):
    PYTHONPATH=src python -m marker.run_skill_quiet --model-name Qwen/Qwen2.5-7B
Smoke (must pass locally before any Vast launch):
    PYTHONPATH=src python -m marker.run_skill_quiet --smoke
"""

from __future__ import annotations

import argparse
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marker.run_axiom_mlp_demo import (
    ILP_PROBES,
    SKILL_AXIOM,
    SKILL_AXIOM_ILP,
    SKILL_PROBES,
    compute_axiom_kv,
    generate_with_mlp,
    make_axiom_mlp,
)
from marker.skill_quiet import as_skill_pair, decode_with_norm_trace, train_skill_quiet

# ── Regression probe golds (byte-identical prompts to run_axiom_mlp_demo) ──────
# One gold substring per probe, chosen to actually test what the probe's
# comment in run_axiom_mlp_demo.py says it's testing (novel channel/loop-type
# generalization, fact retrieval, or the no-term-means-no-fire control).

SKILL_PROBE_GOLDS = ["client.emit('balances'", "client.subscribe('inventory'", "client.emit(", None]
ILP_PROBE_GOLDS = ["Bitwise", "ILP_BREAK", "ILP_END_RETURN", None]
# gold=None marks the no-term control probe: PASS means the API pattern is ABSENT.

INTERNALBUS_API_RE = re.compile(r"client\.(emit|subscribe)\(")
ILP_API_RE = re.compile(r"ILP_[A-Z_]+")

# ── Bleed probes: ask for code + explanation, score both halves ────────────────

INTERNALBUS_BLEED = [
    (
        "Write code using InternalBus to publish a shipment update to the 'shipments' channel, "
        "then explain in plain English what the code does.",
        "client.emit('shipments'",
    ),
    (
        "Write code using InternalBus to subscribe to 'alerts' events, "
        "then explain in plain English what the code does.",
        "client.subscribe('alerts'",
    ),
    (
        "Write a function using InternalBus to publish a login event, "
        "then explain what it does in plain English.",
        "client.emit(",
    ),
    (
        "Write code using InternalBus to subscribe to 'metrics' updates, "
        "then explain in plain English what this code does.",
        "client.subscribe('metrics'",
    ),
]

ILP_BLEED = [
    (
        "Write a sum loop using ilp_for over floats, then explain in plain English what the loop does.",
        "ILP_FOR_AUTO",
    ),
    (
        "Write a search loop using ilp_for that returns the index of a target, "
        "then explain in plain English what it does.",
        "ILP_FOR",
    ),
    (
        "Write a loop using ilp_for with an early break, then explain in plain English what happens.",
        "ILP_BREAK",
    ),
    (
        "Write a max-search loop using ilp_for, then explain in plain English what it computes.",
        "MinMax",
    ),
]

# ── Segment-labelled training data ──────────────────────────────────────────────
# Arms A/B add these to the existing SKILL_AXIOM["qa"] / SKILL_AXIOM_ILP["qa"]
# (wrapped as pure-"skill" pairs via as_skill_pair). Mixed pairs put the code
# first ("skill") then a plain-English explanation ("prose"); pure-prose pairs
# are in-scope questions whose correct answer is prose throughout.

INTERNALBUS_MIXED = [
    (
        "Write code using InternalBus to publish a price update, then explain what it does.",
        [
            ("client.emit('prices', price_update, ttl=30)", "skill"),
            (
                "\n\nThis publishes the price update to the 'prices' channel with the "
                "default 30-second time-to-live.",
                "prose",
            ),
        ],
    ),
    (
        "Write code using InternalBus to subscribe to order events, then explain what it does.",
        [
            ("client.subscribe('orders', handle_order)", "skill"),
            (
                "\n\nThis subscribes to the 'orders' channel and calls handle_order "
                "whenever a message arrives.",
                "prose",
            ),
        ],
    ),
    (
        "Write code using InternalBus to publish a trade event, then explain what the code does "
        "in plain English.",
        [
            ("client.emit('trades', trade_data, ttl=30)", "skill"),
            (
                "\n\nIn plain English, this sends the trade data to the 'trades' channel, "
                "and the message expires after 30 seconds.",
                "prose",
            ),
        ],
    ),
    (
        "Write a function using InternalBus to subscribe to alert messages, then explain what it does.",
        [
            (
                "def listen_for_alerts():\n    client.subscribe('alerts', handle_alert)",
                "skill",
            ),
            (
                "\n\nThis defines a function that subscribes to the 'alerts' channel "
                "and processes each alert with handle_alert.",
                "prose",
            ),
        ],
    ),
    (
        "Write code using InternalBus to publish a balance update, then explain what it does "
        "in plain English.",
        [
            ("client.emit('balances', balance_update, ttl=30)", "skill"),
            (
                "\n\nIn plain English, this publishes the balance update to the 'balances' "
                "channel with a 30-second time-to-live.",
                "prose",
            ),
        ],
    ),
    (
        "Write code using InternalBus to subscribe to inventory events, then explain what it does.",
        [
            ("client.subscribe('inventory', handle_inventory)", "skill"),
            (
                "\n\nThis subscribes to the 'inventory' channel and handles each event "
                "with handle_inventory.",
                "prose",
            ),
        ],
    ),
]

INTERNALBUS_PURE_PROSE = [
    (
        "In plain English, what is InternalBus used for?",
        [
            (
                "InternalBus is a message bus used to publish and subscribe to events "
                "between services, using string channel names and a default TTL of 30 seconds.",
                "prose",
            )
        ],
    ),
    (
        "Explain in plain English how publishing works with InternalBus.",
        [
            (
                "To publish, a client calls emit with a channel name, a payload, and an "
                "optional time-to-live, and any subscriber on that channel receives it.",
                "prose",
            )
        ],
    ),
    (
        "In plain English, what does the default TTL mean for InternalBus?",
        [
            (
                "The default time-to-live is 30 seconds, meaning a published message expires "
                "and is discarded if no subscriber consumes it within that window.",
                "prose",
            )
        ],
    ),
    (
        "Explain in plain English what a channel is in InternalBus.",
        [
            (
                "A channel is just a string name that groups related messages, so publishers "
                "and subscribers agree on where to send and receive events.",
                "prose",
            )
        ],
    ),
]

ILP_MIXED = [
    (
        "Write a sum loop using ilp_for over ints, then explain what it does.",
        [
            (
                "ILP_FOR_AUTO(auto i, 0, n, Sum, int) {\n    total += data[i];\n} ILP_END;",
                "skill",
            ),
            (
                "\n\nThis sums all the elements of the data array using ilp_for's auto-tuned "
                "instruction-level parallelism.",
                "prose",
            ),
        ],
    ),
    (
        "Write a loop using ilp_for that skips negative values, then explain what it does "
        "in plain English.",
        [
            (
                "ILP_FOR(auto i, 0, n, 4) {\n    if (data[i] < 0) ILP_CONTINUE;\n    "
                "sum += data[i];\n} ILP_END;",
                "skill",
            ),
            (
                "\n\nIn plain English, this loop adds up only the non-negative elements, "
                "skipping negative ones with ILP_CONTINUE.",
                "prose",
            ),
        ],
    ),
    (
        "Write a copy loop using ilp_for, then explain what it does.",
        [
            (
                "ILP_FOR_AUTO(auto i, 0, n, Copy, int) {\n    dst[i] = src[i];\n} ILP_END;",
                "skill",
            ),
            (
                "\n\nThis copies every element from src into dst using ilp_for's Copy loop type.",
                "prose",
            ),
        ],
    ),
    (
        "Write a search loop using ilp_for that returns the index, then explain what it does "
        "in plain English.",
        [
            (
                "ILP_FOR(auto i, 0, (int)data.size(), 4) {\n    if (data[i] == target) "
                "ILP_RETURN(i);\n} ILP_END_RETURN;\nreturn -1;",
                "skill",
            ),
            (
                "\n\nIn plain English, this searches for target in data and returns its index, "
                "or -1 if not found; ILP_END_RETURN is required because of the ILP_RETURN inside.",
                "prose",
            ),
        ],
    ),
    (
        "Write a dot product loop using ilp_for, then explain what it computes.",
        [
            (
                "ILP_FOR_AUTO(auto i, 0, n, DotProduct, float) {\n    result += a[i] * b[i];\n"
                "} ILP_END;",
                "skill",
            ),
            (
                "\n\nThis computes the dot product of vectors a and b by summing the "
                "elementwise products.",
                "prose",
            ),
        ],
    ),
    (
        "Write a loop using ilp_for with an early break, then explain what happens in plain English.",
        [
            (
                "int idx = -1;\nILP_FOR(auto i, 0, n, 4) {\n    if (data[i] == target) "
                "{ idx = i; ILP_BREAK; }\n} ILP_END;",
                "skill",
            ),
            (
                "\n\nIn plain English, the loop stops early with ILP_BREAK as soon as it "
                "finds target, storing the index in idx.",
                "prose",
            ),
        ],
    ),
]

ILP_PURE_PROSE = [
    (
        "In plain English, what does ILP_END_RETURN do?",
        [
            (
                "ILP_END_RETURN is the terminator you use after ILP_FOR or ILP_FOR_AUTO when "
                "the loop body contains an ILP_RETURN.",
                "prose",
            )
        ],
    ),
    (
        "Explain in plain English what ILP_FOR_AUTO is for.",
        [
            (
                "ILP_FOR_AUTO is the auto-tuned form of ilp_for, where you specify a LoopType "
                "and element type so the library picks an efficient parallel strategy.",
                "prose",
            )
        ],
    ),
    (
        "In plain English, what is ilp_for a library for?",
        [
            (
                "ilp_for is a C++20 header-only library for writing instruction-level "
                "parallel loops without hand-rolling the unrolling yourself.",
                "prose",
            )
        ],
    ),
    (
        "Explain in plain English the difference between ILP_BREAK and ILP_CONTINUE.",
        [
            (
                "ILP_BREAK exits the loop entirely, while ILP_CONTINUE skips just the "
                "current iteration and moves to the next one.",
                "prose",
            )
        ],
    ),
]


def _contains(text: str, gold: str) -> bool:
    return gold.lower() in text.lower()


def _contamination_count(text: str, gold: str, api_re: re.Pattern) -> int:
    """API-pattern occurrences in the "prose region" of a bleed generation.

    Split point: first blank line if present, else right after the first
    occurrence of the code gold (so we don't count the intended code as
    contamination), else the whole text (worst-case, no split found).
    """
    split_idx = text.find("\n\n")
    if split_idx == -1:
        gold_idx = text.lower().find(gold.lower())
        split_idx = gold_idx + len(gold) if gold_idx != -1 else 0
    return len(api_re.findall(text[split_idx:]))


def _build_skill_mlp(model, tokenizer, term: str, chosen_layers: list[int], r: int):  # noqa: ANN001, ANN201
    mlp = make_axiom_mlp(model, tokenizer, term, chosen_layers, r=r)
    mlp.skill_mode = True
    return mlp


def _run_skill(
    model,  # noqa: ANN001
    tokenizer,  # noqa: ANN001
    chosen_layers: list[int],
    skill_axiom: dict,
    mixed: list,
    pure_prose: list,
    regression_probes: list[str],
    regression_golds: list[str | None],
    bleed: list[tuple[str, str]],
    api_re: re.Pattern,
    args,  # noqa: ANN001
) -> None:
    name = skill_axiom["term"]
    desc = skill_axiom["description"]
    control_pairs = [as_skill_pair(q, a) for q, a in skill_axiom["qa"]]
    extended_pairs = control_pairs + mixed + pure_prose

    configs = [
        ("C", control_pairs, 0.0),
        ("A", extended_pairs, 0.0),
        ("B", extended_pairs, args.lam),
    ]
    if args.smoke:
        configs = [("B", extended_pairs, args.lam)]

    print(f"\n{'=' * 78}\n### {name}\n{'=' * 78}")
    for arm, pairs, lam in configs:
        mlp = _build_skill_mlp(model, tokenizer, name, chosen_layers, r=args.r)
        mlp.kv = compute_axiom_kv(model, tokenizer, desc, term=name)
        n_steps = args.n_steps if not args.smoke else 20
        losses = train_skill_quiet(model, tokenizer, mlp, pairs, n_steps=n_steps, lam=lam)
        print(f"\n--- arm {arm} (lam={lam}, {len(pairs)} pairs) ---")
        print(f"  loss: {losses[0]:.3f} -> {losses[-1]:.4f}")

        reg_correct = 0
        for prompt_text, gold in zip(regression_probes, regression_golds, strict=True):
            out = generate_with_mlp(model, tokenizer, prompt_text, mlp, max_new=args.max_new)
            # no-term control (gold=None): the API pattern must be ABSENT.
            ok = not api_re.search(out) if gold is None else _contains(out, gold)
            reg_correct += int(ok)
            print(f"    [regression] {'v' if ok else 'x'} {out[:90].replace(chr(10), ' ')}")
        print(f"  regression score: {reg_correct}/{len(regression_probes)}")

        bleed_code_correct = 0
        contamination_total = 0
        for q, gold in bleed:
            prompt_text = f"Q: {q}\nA:"
            out, trace = decode_with_norm_trace(model, tokenizer, prompt_text, mlp, args.max_new)
            code_ok = _contains(out, gold)
            bleed_code_correct += int(code_ok)
            contam = _contamination_count(out, gold, api_re)
            contamination_total += contam

            # Norm-trace summary: mean norm before/after the first blank line
            # in the OUTPUT TEXT is only approximated here by output length
            # ratio isn't available token-for-token, so we report the trace's
            # first-half vs second-half mean as the printed summary — the
            # full trace (printed truncated) is the actual evidence.
            half = max(1, len(trace) // 2)
            early_mean = sum(trace[:half]) / half
            late_mean = sum(trace[half:]) / max(1, len(trace) - half)
            print(f"    [bleed] code_ok={code_ok} contam={contam} {out[:90].replace(chr(10), ' ')}")
            print(
                f"      norm trace (first {min(len(trace), 20)} steps): "
                f"{[round(x, 3) for x in trace[:20]]}"
            )
            print(f"      mean norm early-half={early_mean:.4f}  late-half={late_mean:.4f}")
        print(
            f"  bleed code score: {bleed_code_correct}/{len(bleed)}  "
            f"total contamination: {contamination_total}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--r", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=3000)
    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument("--max-new", type=int, default=120)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.model_name = "Qwen/Qwen2.5-0.5B"
        args.max_new = min(args.max_new, 30)
        print("=== SMOKE MODE (arm B only, 20 steps) ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  model: {args.model_name}  lam: {args.lam}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
        .to(device)
        .eval()
    )
    n_layers = model.config.num_hidden_layers
    chosen_layers = [n_layers // 4, n_layers // 2, (3 * n_layers) // 4]

    _run_skill(
        model,
        tokenizer,
        chosen_layers,
        SKILL_AXIOM,
        INTERNALBUS_MIXED,
        INTERNALBUS_PURE_PROSE,
        SKILL_PROBES,
        SKILL_PROBE_GOLDS,
        INTERNALBUS_BLEED,
        INTERNALBUS_API_RE,
        args,
    )
    if not args.smoke:
        _run_skill(
            model,
            tokenizer,
            chosen_layers,
            SKILL_AXIOM_ILP,
            ILP_MIXED,
            ILP_PURE_PROSE,
            ILP_PROBES,
            ILP_PROBE_GOLDS,
            ILP_BLEED,
            ILP_API_RE,
            args,
        )


if __name__ == "__main__":
    main()
