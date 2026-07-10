"""Stage-1 gist EVAL (Fable gate review): controls on the trained gist.

Loads the step-16000 checkpoint (adapter + gist) from the HF repo — NO
training — and runs three things on one node (~$0.15):

1. SHUFFLED-GIST control (required before Stage-2 spend). Four PPL conditions:
   gist / full / none / shuffled. Shuffled = each continuation sees a gist
   computed from a DIFFERENT sentence's span (roll spans within the batch).
   If shuffled ~= none, the gist truly carries the span's content and the
   0.887 is real; if shuffled ~= gist, the headline was slot-presence.

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

from marker.gist_data import stream_doc_pairs, take_heldout_docs
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
def four_condition_ppls(peft_model, gist, heldout, batch_size):  # noqa: ANN001
    """gist / full / none via gist_forward, plus SHUFFLED (rolled spans)."""
    sums = dict.fromkeys([*CONDS, "shuffled"], 0.0)
    n = 0
    from marker.gist_data import batched  # noqa: PLC0415

    for spans, conts in batched(iter(heldout), batch_size):
        if len(spans) < 2:
            continue  # shuffle needs a real permutation
        for name, sees in CONDS.items():
            sums[name] += float(gist_forward(peft_model, gist, spans, conts, cont_sees=sees))
        sums["shuffled"] += float(
            gist_forward(peft_model, gist, roll_spans(spans), conts, cont_sees=CONDS["gist"])
        )
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

    # 1. Four-condition PPL + shuffled control
    docs = _smoke_docs(tok) if args.smoke else stream_doc_pairs(tok)
    heldout, _ = take_heldout_docs(docs, args.heldout_n)
    print(f"heldout pairs: {len(heldout)}", flush=True)
    ppls = four_condition_ppls(peft_model, gist, heldout, args.batch)
    gc_real = gap_closed(ppls)
    gc_shuf = gap_closed({"none": ppls["none"], "full": ppls["full"], "gist": ppls["shuffled"]})
    print("\n===== SHUFFLED-GIST CONTROL =====")
    print(
        f"  PPL  gist={ppls['gist']:.3f} full={ppls['full']:.3f} "
        f"none={ppls['none']:.3f} shuffled={ppls['shuffled']:.3f}"
    )
    print(f"  gap_closed(gist)={gc_real:.3f}   gap_closed(shuffled)={gc_shuf:.3f}")
    print(
        f"  VERDICT: {'REAL (shuffled collapses to ~none)' if gc_shuf < 0.3 else 'SUSPECT (shuffled retains gap — slot-presence artifact)'}"
    )

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
    main()
