"""Stage-1 gist EVAL (Fable gate review): controls on the trained gist.

Loads the step-16000 checkpoint (adapter + gist) from the HF repo — NO
training — and runs three things on one node (~$0.15):

1. GIST-CONTENT controls (required before Stage-2 spend). Five PPL conditions:
   gist / full / none / neighbor / xdoc. neighbor = within-batch roll, whose
   donor turned out to be a same-document neighbor (topically overlapping the
   continuation — measures 'nearby context helps', NOT a null; first-run
   design flaw, own-goal acknowledged in the gate review). xdoc = donor span
   from a DIFFERENT document — the true null. If xdoc ~= none, the 0.887 is
   real content-carrying; if xdoc retains the gap, it was slot-presence.

2. DECODE-FROM-GIST sanity: greedily generate a continuation from gist-only
   vs full context; eyeball topical agreement (no reconstruction head, so not
   verbatim — Fable's framing).

3. ilp_for PROBE: gist the ilp_for DSL description, decode gist-only, watch
   which exact tokens survive. Prediction on record: prose topicality
   survives, exact macro syntax does not — confirming the gist(thought) vs
   Mimir(exact fact/skill) division of labor.

Run (GPU):
    HF_TOKEN=... PYTHONPATH=src python -u -m marker.run_gist_eval \
        --repo mattyvee/mimir-artifacts
Smoke (local): PYTHONPATH=src python -m marker.run_gist_eval --smoke
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marker.gist_data import stream_doc_pairs
from marker.gist_model import (
    attach_gist,
    gap_closed,
    generate_from_gist,
    gist_forward,
    roll_spans,
    to_leaf_param,
)
from marker.run_axiom_mlp_demo import SKILL_AXIOM_ILP

CONDS = {
    "gist": frozenset({"gist"}),
    "full": frozenset({"gist", "span"}),
    "none": frozenset(),
}


@torch.no_grad()
def five_condition_ppls(peft_model, gist, heldout_docs, batch_size):  # noqa: ANN001
    """gist / full / none, plus TWO controls:
    - neighbor: within-batch roll — donor span is a same-document neighbor
      (measures 'nearby context helps'; NOT a null — gate-review correction).
    - xdoc: donor span from a DIFFERENT document (the true null: if these
      gists still close the gap, the headline was slot-presence artifact)."""
    from marker.gist_data import batched  # noqa: PLC0415
    from marker.gist_model import cross_doc_spans  # noqa: PLC0415

    flat = [p for d in heldout_docs for p in d]
    xdoc = cross_doc_spans(heldout_docs)
    sums = dict.fromkeys([*CONDS, "neighbor", "xdoc"], 0.0)
    n = 0
    idx = 0
    for spans, conts in batched(iter(flat), batch_size):
        b = len(spans)
        if b < 2:
            idx += b
            continue
        for name, sees in CONDS.items():
            sums[name] += float(gist_forward(peft_model, gist, spans, conts, cont_sees=sees))
        sums["neighbor"] += float(
            gist_forward(peft_model, gist, roll_spans(spans), conts, cont_sees=CONDS["gist"])
        )
        sums["xdoc"] += float(
            gist_forward(peft_model, gist, xdoc[idx : idx + b], conts, cont_sees=CONDS["gist"])
        )
        idx += b
        n += 1
        if n % 10 == 0:
            print(f"    ...eval batch {n}", flush=True)
    return {k: float(torch.exp(torch.tensor(v / max(1, n)))) for k, v in sums.items()}


def _load_from_ckpt(model_name, repo, device, quantize):  # noqa: ANN001
    tok = AutoTokenizer.from_pretrained(model_name)
    if quantize:
        from transformers import BitsAndBytesConfig  # noqa: PLC0415

        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )
        base = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb, attn_implementation="sdpa", device_map={"": 0}
        )
    else:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        base = (
            AutoModelForCausalLM.from_pretrained(
                model_name, attn_implementation="sdpa", dtype=dtype
            )
            .to(device)
            .eval()
        )
    peft_model, gist = attach_gist(base, gist_k=8, r=16)
    gist = to_leaf_param(gist, device)

    if repo:
        from peft import set_peft_model_state_dict  # noqa: PLC0415
        from safetensors.torch import load_file  # noqa: PLC0415

        from marker.hf_push import fetch_step, resume_step  # noqa: PLC0415

        step = resume_step(repo)
        ckpt = fetch_step(repo, step, "/tmp/gist_eval")  # noqa: S108
        set_peft_model_state_dict(peft_model, load_file(str(ckpt / "adapter_model.safetensors")))
        gist.data = load_file(str(ckpt / "gist.safetensors"))["gist"].to(device)
        print(f"loaded checkpoint step {step} from {repo}", flush=True)
    return peft_model, gist, tok


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="Qwen/Qwen2.5-7B")
    p.add_argument("--repo", default=None)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--heldout-n", type=int, default=256)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.model_name, args.repo, args.heldout_n, args.batch = "Qwen/Qwen2.5-0.5B", None, 24, 4
        print("=== SMOKE (no checkpoint, untrained gist) ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    quantize = device == "cuda" and not args.smoke
    peft_model, gist, tok = _load_from_ckpt(args.model_name, args.repo, device, quantize)

    # 1. Five-condition PPL: gist/full/none + neighbor(same-doc) + xdoc(true null)
    doc_iter = _smoke_docs(tok) if args.smoke else stream_doc_pairs(tok)
    heldout_docs: list = []
    total = 0
    for d in doc_iter:
        heldout_docs.append(d)
        total += len(d)
        if total >= args.heldout_n:
            break
    heldout = [p for d in heldout_docs for p in d]
    print(f"heldout: {len(heldout)} pairs across {len(heldout_docs)} docs", flush=True)
    ppls = five_condition_ppls(peft_model, gist, heldout_docs, args.batch)

    def _gc(name: str) -> float:
        return gap_closed({"none": ppls["none"], "full": ppls["full"], "gist": ppls[name]})

    print("\n===== GIST CONTENT CONTROLS =====")
    print(
        f"  PPL  gist={ppls['gist']:.3f} full={ppls['full']:.3f} none={ppls['none']:.3f} "
        f"neighbor={ppls['neighbor']:.3f} xdoc={ppls['xdoc']:.3f}"
    )
    print(
        f"  gap_closed: gist={_gc('gist'):.3f}  neighbor(same-doc)={_gc('neighbor'):.3f}  "
        f"xdoc(true null)={_gc('xdoc'):.3f}"
    )
    verdict = (
        "REAL — cross-doc gist collapses toward none; slots carry span/doc content"
        if _gc("xdoc") < 0.3
        else "ARTIFACT SIGNAL — even cross-doc gists close the gap (slot presence)"
    )
    print(f"  VERDICT: {verdict}")

    # 2 + 3. Decode-from-gist sanity + ilp_for probe
    print("\n===== DECODE-FROM-GIST (topical agreement, not verbatim) =====")
    probes = heldout[:3] if not args.smoke else heldout[:1]
    for i, (span, _cont) in enumerate(probes):
        g = tok.decode(
            generate_from_gist(
                peft_model, gist, span, cont_sees=CONDS["gist"], max_new=30, eos_id=tok.eos_token_id
            )
        )
        f = tok.decode(
            generate_from_gist(
                peft_model, gist, span, cont_sees=CONDS["full"], max_new=30, eos_id=tok.eos_token_id
            )
        )
        print(f"  [{i}] span: {tok.decode(span)[:60]!r}")
        print(f"      gist-decode: {g[:80]!r}")
        print(f"      full-decode: {f[:80]!r}")

    if not args.smoke:
        print("\n===== ilp_for PROBE (does exact DSL syntax survive the gist?) =====")
        ilp_span = tok(SKILL_AXIOM_ILP["description"], add_special_tokens=False).input_ids[:64]
        for label, sees in [("gist-only", CONDS["gist"]), ("full", CONDS["full"])]:
            out = tok.decode(
                generate_from_gist(
                    peft_model, gist, ilp_span, cont_sees=sees, max_new=50, eos_id=tok.eos_token_id
                )
            )
            print(f"  {label}: {out[:120]!r}")
        print("  (watch for ILP_FOR / ILP_END_RETURN / Bitwise — exact macros)")

    print("\nGATE: shuffled << gist ⇒ Stage-1 result real ⇒ Stage-2 spend approved.")


def _smoke_docs(tok):  # noqa: ANN001
    from marker.gist_data import pairs_from_text  # noqa: PLC0415

    texts = [
        "The bus polls the queue. It waits for a message. It fires the handler. Then it repeats."
    ] * 20
    for t in texts:
        pairs = pairs_from_text(t, tok, max_span=32, max_cont=32, min_cont=6)
        if pairs:
            yield pairs


if __name__ == "__main__":
    import os
    import sys

    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # skip bitsandbytes teardown SIGABRT (same fix as the pilot runner)
