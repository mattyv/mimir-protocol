"""Instruct portability Phase 1: does an anti-refusal preamble recover fact
recall on a chat model without breaking BOUNDARY?

See INSTRUCT_PLAN.md. Conditions:
    base-ref          base model, existing 'About X' KV + 'Q:/A:' probe
    instruct-none     chat model, axiom KV, no preamble
    instruct-preamble chat model, axiom KV + plain-token preamble
    instruct-termenum chat model, axiom KV + preamble + "Registered concepts:"

Scored per axiom on HELDOUT (unseen-phrasing recall, digit-boundary) and
BOUNDARY (must decline out-of-scope). Fresh cache per probe. Base and instruct
loaded sequentially to avoid holding two 7B models at once.

GATE: instruct-preamble recovers HELDOUT materially over instruct-none AND
BOUNDARY does not regress vs instruct-none.

Run (GPU):
    PYTHONPATH=src python -m marker.run_instruct_phase1 \
        --model-name Qwen/Qwen2.5-7B --instruct-name Qwen/Qwen2.5-7B-Instruct
Smoke (local, pinned transformers):
    PYTHONPATH=src python -m marker.run_instruct_phase1 --smoke
"""

from __future__ import annotations

import argparse
import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marker.instruct import (
    base_axiom_kv,
    chat_live_suffix,
    declined,
    decode_with_kv,
    encode_chat_axiom_kv,
    im_end_id,
    matches,
)
from marker.run_axiom_mlp_demo import TEMPLATE
from marker.run_prefix_tuned import TUNED_AXIOMS

PREAMBLE = (
    "The following reference material describes internal systems, terms, and "
    "constructs. (1) Answer questions about them directly and confidently from "
    "this material. (2) When generating code or output using these constructs, "
    "follow the reference material's syntax and conventions exactly, even where "
    "they differ from common practice. If the material does not cover something, "
    "state that the description doesn't specify."
)

# Out-of-scope questions per axiom — correct behavior is to decline.
BOUNDARY: dict[str, list[str]] = {
    "BalancePublisher": [
        "What programming language is BalancePublisher written in?",
        "What is BalancePublisher's memory limit?",
    ],
    "FluxomService": [
        "What cloud provider does FluxomService run on?",
        "How many engineers maintain FluxomService?",
    ],
    "MeshPublisher": [
        "What port does MeshPublisher listen on?",
        "When was MeshPublisher first deployed?",
    ],
}

USE_AXIOMS = ["BalancePublisher", "FluxomService", "MeshPublisher"]


def _heldout_probes(axiom: dict) -> list[tuple[str, str]]:
    return [(q, g) for f in axiom["facts"] for q, g in [*f["dev"], *f["test"]]]


def _axioms() -> list[dict]:
    by_name = {a["name"]: a for a in TUNED_AXIOMS}
    return [by_name[n] for n in USE_AXIOMS]


def _score_condition(
    model,  # noqa: ANN001
    tokenizer,  # noqa: ANN001
    axioms: list[dict],
    *,
    chat: bool,
    preamble: str | None,
    term_enum: bool,
    max_new: int,
    label: str,
    print_detail: bool,
) -> dict[str, tuple[int, int]]:
    """Returns {'heldout': (c,t), 'boundary': (c,t)} aggregated over axioms.
    boundary 'correct' = declined."""
    stop = im_end_id(tokenizer) if chat else None
    h_c = h_t = b_c = b_t = 0

    for axiom in axioms:
        term, desc = axiom["name"], axiom["desc"]
        pre = preamble
        if pre is not None and term_enum:
            pre = f"Registered concepts: {term}.\n{pre}"

        # KV depends only on (axiom, condition), not the question — compute once,
        # reuse across probes (fresh DynamicCache is built per decode call).
        kv = (
            encode_chat_axiom_kv(model, tokenizer, term, desc)
            if chat
            else base_axiom_kv(model, tokenizer, term, desc)
        )

        def live(q: str, _pre=pre, _chat=chat) -> str:
            return chat_live_suffix(q, _pre) if _chat else TEMPLATE.format(q=q)

        for q, gold in _heldout_probes(axiom):
            out = decode_with_kv(model, tokenizer, kv, live(q), max_new, stop)
            ok = matches(out, gold)
            h_c += int(ok)
            h_t += 1
            if print_detail:
                print(
                    f"    [{label} HELDOUT] {'v' if ok else 'x'} {gold!r:>14}  "
                    f"{out[:70].replace(chr(10), ' ')}"
                )

        for q in BOUNDARY.get(term, []):
            out = decode_with_kv(model, tokenizer, kv, live(q), max_new, stop)
            ok = declined(out)
            b_c += int(ok)
            b_t += 1
            if print_detail:
                print(
                    f"    [{label} BOUND ] {'declined' if ok else 'ANSWERED'}  "
                    f"{out[:70].replace(chr(10), ' ')}"
                )

    return {"heldout": (h_c, h_t), "boundary": (b_c, b_t)}


def _load(name: str, device: str):  # noqa: ANN001, ANN202
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16).to(device).eval()
    return model, tok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--instruct-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-new", type=int, default=60)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.model_name = "Qwen/Qwen2.5-0.5B"
        args.instruct_name = "Qwen/Qwen2.5-0.5B-Instruct"
        args.max_new = min(args.max_new, 40)
        print("=== SMOKE MODE ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    axioms = _axioms()
    print(f"device: {device}\naxioms: {[a['name'] for a in axioms]}\n")

    results: dict[str, dict[str, tuple[int, int]]] = {}

    # ── Base reference (existing path) ───────────────────────────────────────
    print("=" * 70, "\nBASE REFERENCE:", args.model_name, "\n", "=" * 70, sep="")
    base_model, base_tok = _load(args.model_name, device)
    results["base-ref"] = _score_condition(
        base_model,
        base_tok,
        axioms,
        chat=False,
        preamble=None,
        term_enum=False,
        max_new=args.max_new,
        label="base",
        print_detail=True,
    )
    del base_model, base_tok
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # ── Instruct conditions ──────────────────────────────────────────────────
    print("\n", "=" * 70, "\nINSTRUCT:", args.instruct_name, "\n", "=" * 70, sep="")
    ins_model, ins_tok = _load(args.instruct_name, device)
    for label, preamble, term_enum in [
        ("instruct-none", None, False),
        ("instruct-preamble", PREAMBLE, False),
        ("instruct-termenum", PREAMBLE, True),
    ]:
        print(f"\n--- {label} ---")
        results[label] = _score_condition(
            ins_model,
            ins_tok,
            axioms,
            chat=True,
            preamble=preamble,
            term_enum=term_enum,
            max_new=args.max_new,
            label=label,
            print_detail=True,
        )

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'condition':20} {'HELDOUT':>10} {'BOUNDARY(decline)':>20}")
    for label in ["base-ref", "instruct-none", "instruct-preamble", "instruct-termenum"]:
        h, b = results[label]["heldout"], results[label]["boundary"]
        print(f"  {label:20} {f'{h[0]}/{h[1]}':>10} {f'{b[0]}/{b[1]}':>20}")

    print("\nGATE: instruct-preamble HELDOUT >> instruct-none HELDOUT, AND")
    print("      instruct-preamble BOUNDARY not below instruct-none BOUNDARY.")


if __name__ == "__main__":
    main()
