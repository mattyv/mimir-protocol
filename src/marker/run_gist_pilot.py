"""Stage-1 gist-compression PILOT runner (see GIST_PILOT_PLAN.md).

Trains k=8 learned gist slots + LoRA on a frozen 4-bit Qwen2.5-7B so a
sentence's continuation conditions on the 8 gist KVs as well as on the full
sentence. Eval = three PPLs (gist / full / none) on a fixed held-out set and
the fraction of the none->full gap that gist closes.

GATE: gap_closed > 0.5 at 20M tokens -> scale; gist ~= none -> kill.

Checkpoints (LoRA adapter + gist embeddings + manifest) push to a private HF
repo every --ckpt-every steps and at the end, so the run survives node death;
--resume picks up the latest checkpoint in the repo (optimizer state NOT
restored — acceptable for the pilot, Fable #4). Wrap in `timeout` for a hard
wall-clock cap.

Run (GPU):
    HF_TOKEN=... PYTHONPATH=src python -u -m marker.run_gist_pilot \
        --model-name Qwen/Qwen2.5-7B --repo mattyvee/mimir-artifacts \
        --max-steps 4000 --gist-k 8
Shakedown (GPU, ~$0.15): --max-steps 500 --ckpt-every 200 (proves push/resume)
Smoke (local, tiny model, synthetic corpus, no HF):
    PYTHONPATH=src python -m marker.run_gist_pilot --smoke
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marker.gist_data import batched, stream_doc_pairs, take_heldout_docs
from marker.gist_model import attach_gist, gap_closed, gist_forward

# A tiny synthetic corpus for --smoke (no network / no bitsandbytes).
SMOKE_TEXTS = [
    "The bus polls the queue. It waits for a message. When one arrives it fires "
    "the handler. The handler runs quickly. Then it returns to waiting.",
    "Water boils at a fixed temperature. The temperature depends on pressure. "
    "At sea level it is one hundred degrees. Higher up it is lower. Cooks adjust "
    "for this.",
] * 20


def _smoke_docs(tokenizer):  # noqa: ANN001
    """Per-document pair lists for the smoke path (mirrors stream_doc_pairs)."""
    from marker.gist_data import pairs_from_text  # noqa: PLC0415

    for text in SMOKE_TEXTS:
        pairs = pairs_from_text(text, tokenizer, max_span=32, max_cont=32, min_cont=6)
        if pairs:
            yield pairs


@torch.no_grad()
def evaluate(peft_model, gist, heldout, batch_size):  # noqa: ANN001
    """Mean loss per condition over the held-out set -> PPLs + gap_closed."""
    sums = {"gist": 0.0, "full": 0.0, "none": 0.0}
    n = 0
    for spans, conts in batched(iter(heldout), batch_size):
        for cond in sums:
            sees = {
                "gist": frozenset({"gist"}),
                "full": frozenset({"gist", "span"}),
                "none": frozenset(),
            }[cond]
            sums[cond] += float(gist_forward(peft_model, gist, spans, conts, cont_sees=sees))
        n += 1
    ppls = {c: float(torch.exp(torch.tensor(sums[c] / max(1, n)))) for c in sums}
    return ppls, gap_closed(ppls)


def _load_base(name, quantize, device):  # noqa: ANN001
    if quantize:
        from transformers import BitsAndBytesConfig  # noqa: PLC0415

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            name, quantization_config=bnb, attn_implementation="sdpa", device_map={"": 0}
        )
        from peft import prepare_model_for_kbit_training  # noqa: PLC0415

        return prepare_model_for_kbit_training(model)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    return (
        AutoModelForCausalLM.from_pretrained(name, attn_implementation="sdpa", torch_dtype=dtype)
        .to(device)
        .eval()
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="Qwen/Qwen2.5-7B")
    p.add_argument("--repo", default=None, help="HF repo for checkpoints (push/resume)")
    p.add_argument("--gist-k", type=int, default=8)
    p.add_argument("--r", type=int, default=16)
    # 16000 micro-steps x 8 seqs x ~130 tokens ~= 16M tokens (the pre-registered
    # pilot gate said 20M; 4000 was a 5x undershoot — Fable finding #4).
    p.add_argument("--max-steps", type=int, default=16000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--heldout-n", type=int, default=512)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.model_name = "Qwen/Qwen2.5-0.5B"
        args.max_steps, args.batch, args.grad_accum = 30, 4, 1
        args.eval_every, args.ckpt_every, args.heldout_n = 10, 0, 16
        args.repo = None
        print("=== SMOKE MODE (tiny synthetic corpus, no HF) ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    quantize = device == "cuda" and not args.smoke
    print(
        f"device: {device}  model: {args.model_name}  gist_k: {args.gist_k}  quantize: {quantize}"
    )

    tok = AutoTokenizer.from_pretrained(args.model_name)
    base = _load_base(args.model_name, quantize, device)
    peft_model, gist = attach_gist(base, gist_k=args.gist_k, r=args.r)
    gist = gist.to(device)

    start_step = 0
    if args.resume and args.repo:
        from peft import set_peft_model_state_dict  # noqa: PLC0415
        from safetensors.torch import load_file  # noqa: PLC0415

        from marker.hf_push import fetch_step, resume_step  # noqa: PLC0415

        step = resume_step(args.repo)
        if step:
            ckpt = fetch_step(args.repo, step, "/tmp/gist_resume")  # noqa: S108
            # Load weights INTO the existing 'default' adapter — load_adapter()
            # with an existing name raises (Fable pre-launch finding #1).
            adapter_state = load_file(str(ckpt / "adapter_model.safetensors"))
            set_peft_model_state_dict(peft_model, adapter_state)
            gist.data = load_file(str(ckpt / "gist.safetensors"))["gist"].to(device)
            start_step = step
            print(f"resumed from step {step}")

    # Document-disjoint heldout/train split (Fable pre-launch finding #2 —
    # pairs within a doc overlap, so the split must happen at doc boundaries).
    docs = _smoke_docs(tok) if args.smoke else stream_doc_pairs(tok)
    heldout, train_stream = take_heldout_docs(docs, args.heldout_n)
    print(f"heldout pairs: {len(heldout)} (document-disjoint from training)")

    trainable = [p_ for p_ in peft_model.parameters() if p_.requires_grad] + [gist]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    # T_max counts OPTIMIZER steps (one per grad_accum micro-steps), so the
    # cosine actually completes (Fable pre-launch finding #5).
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.max_steps // args.grad_accum), eta_min=args.lr / 10
    )

    step = start_step
    ppls, gc = {"gist": 0.0, "full": 0.0, "none": 0.0}, 0.0  # bound before any checkpoint
    opt.zero_grad()
    for spans, conts in batched(train_stream, args.batch):
        loss = gist_forward(peft_model, gist, spans, conts) / args.grad_accum
        loss.backward()

        if step == start_step:
            # The quantized path (prepare_model_for_kbit_training + gradient
            # checkpointing + inputs_embeds splice) is unexercised by CPU
            # tests. A silent no-grad here would fake gist~=none and wrongly
            # kill the recipe — fail LOUDLY instead (Fable finding #3).
            lora_ok = any(
                p_.grad is not None and p_.grad.abs().sum() > 0
                for p_ in peft_model.parameters()
                if p_.requires_grad
            )
            gist_ok = gist.grad is not None and gist.grad.abs().sum() > 0
            assert lora_ok and gist_ok, f"GRAD FAIL: lora_ok={lora_ok} gist_ok={gist_ok}"
            print("GRAD_OK (lora + gist gradients flowing)", flush=True)

        if (step + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()

        if step % args.eval_every == 0:
            ppls, gc = evaluate(peft_model, gist, heldout, args.batch)
            print(
                f"[step {step:>5}] loss {loss.item() * args.grad_accum:.4f}  "
                f"ppl gist={ppls['gist']:.2f} full={ppls['full']:.2f} none={ppls['none']:.2f}  "
                f"gap_closed={gc:.3f}",
                flush=True,
            )
        if args.ckpt_every and args.repo and step > start_step and step % args.ckpt_every == 0:
            _checkpoint(peft_model, gist, args.repo, step, ppls, gc)

        step += 1
        if step >= args.max_steps:
            break

    ppls, gc = evaluate(peft_model, gist, heldout, args.batch)
    print(f"[FINAL step {step}] ppl {ppls}  gap_closed={gc:.3f}", flush=True)
    if args.repo:
        _checkpoint(peft_model, gist, args.repo, step, ppls, gc)
    print("GATE: gap_closed>0.5 => scale; gist~=none => kill.")


def _checkpoint(peft_model, gist, repo, step, ppls, gc):  # noqa: ANN001
    from pathlib import Path  # noqa: PLC0415

    from marker.hf_push import push_bundle, save_bundle  # noqa: PLC0415

    d = Path("/tmp/gist_ckpt")  # noqa: S108
    meta = {"step": step, "ppls": ppls, "gap_closed": gc}
    save_bundle(d, peft_model, gist, meta)
    push_bundle(d, repo, step)
    print(f"  checkpoint pushed: step {step}", flush=True)


if __name__ == "__main__":
    main()
