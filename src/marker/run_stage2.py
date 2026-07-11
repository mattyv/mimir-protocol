"""Stage-2 runner: encode gist sequences, fit whiteners, train the next-thought
predictor, eval (see STAGE2_PLAN.md).

One node, one run (Fable steer #5): load the frozen 7B + Stage-1 gist adapter
from HF, encode a corpus of documents into per-sentence gist sequences, fit
per-slot whiteners on TRAIN gists only, train the predictor, eval recall@k +
diversity on document-disjoint held-out sequences, push only the artifacts
(predictor + whiteners + manifest). The 15+GB of raw gists never leave the node.

PRE-REGISTERED GATES (STAGE2_PLAN): recall@5 > 0.40 (batch>=128), platitude
guard (prediction diversity < corpus mean similarity). recall@5 ~= chance ⇒ KILL.

Run (GPU):
    HF_TOKEN=... PYTHONPATH=src python -u -m marker.run_stage2 \
        --repo mattyvee/mimir-artifacts --n-docs 4000
Smoke (local, tiny model + synthetic corpus):
    PYTHONPATH=src python -m marker.run_stage2 --smoke
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marker.gist_model import attach_gist, encode_gist, to_leaf_param
from marker.predictor import (
    NextThoughtPredictor,
    info_nce_loss,
    prediction_diversity,
    recall_at_k,
    regression_loss,
)
from marker.whiten import PerSlotWhitener

SMOKE_TEXTS = [
    "The bus polls the queue. It waits for a message. It fires the handler. "
    "The handler runs. Then it returns to waiting. The cycle repeats forever.",
    "Water boils at a temperature. The temperature depends on pressure. At sea "
    "level it is high. Higher up it is lower. Cooks must adjust their timing.",
] * 15


def _load_stage1(model_name, repo, device, quantize):  # noqa: ANN001
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
    pm, gist = attach_gist(base, gist_k=8, r=16)
    gist = to_leaf_param(gist, device)
    if repo:
        from peft import set_peft_model_state_dict  # noqa: PLC0415
        from safetensors.torch import load_file  # noqa: PLC0415

        from marker.hf_push import fetch_step, resume_step  # noqa: PLC0415

        step = resume_step(repo)
        ckpt = fetch_step(repo, step, "/tmp/stage2_ckpt")  # noqa: S108
        set_peft_model_state_dict(pm, load_file(str(ckpt / "adapter_model.safetensors")))
        gist.data = load_file(str(ckpt / "gist.safetensors"))["gist"].to(device)
        print(f"loaded Stage-1 gist step {step}", flush=True)
    return pm, gist, tok


def _doc_sentence_spans(tok, text, max_span, max_sents):  # noqa: ANN001
    from marker.gist import split_sentences  # noqa: PLC0415

    sents = split_sentences(text)[:max_sents]
    spans = [tok(s, add_special_tokens=False).input_ids[:max_span] for s in sents]
    return [s for s in spans if s]


@torch.no_grad()
def encode_corpus(pm, gist, tok, docs_text, max_span, max_sents, min_sents, device):  # noqa: ANN001
    """Each document -> a gist sequence [n_sents, k, hidden] (encode every
    sentence). Returns a list of per-document sequences (>= min_sents long)."""
    seqs = []
    for i, text in enumerate(docs_text):
        spans = _doc_sentence_spans(tok, text, max_span, max_sents)
        if len(spans) < min_sents:
            continue
        # encode sentence-by-sentence (variable span lengths -> one at a time)
        slots = [encode_gist(pm, gist, [sp]).float()[0] for sp in spans]  # each [k, hidden]
        seqs.append(torch.stack(slots).cpu())  # [n_sents, k, hidden]
        if (i + 1) % 50 == 0:
            print(f"    ...encoded {i + 1} docs, {len(seqs)} kept", flush=True)
    return seqs


def _windows(seq, length):  # noqa: ANN001
    """All length-L windows of a [n_sents, k, d] sequence."""
    return [seq[i : i + length] for i in range(len(seq) - length + 1)]


def _batches(seqs, length, batch, whitener):  # noqa: ANN001
    wins = [w for s in seqs for w in _windows(s, length)]
    torch.manual_seed(0)
    order = torch.randperm(len(wins))
    for i in range(0, len(wins) - batch + 1, batch):
        idx = order[i : i + batch]
        stack = torch.stack([wins[j] for j in idx])  # [B, L, k, d]
        b, ln, k, d = stack.shape
        wz = whitener.transform(stack.reshape(b * ln, k, d)).reshape(b, ln, k, d)
        yield wz


@torch.no_grad()
def evaluate(model, seqs, length, whitener, device):  # noqa: ANN001
    """Retrieval over the FULL eval pool: gather every window's predicted and
    true next-gist, then recall@k against ALL eval targets (bigger candidate
    pool = harder, the pre-registered gate wants >=128)."""
    wins = [w for s in seqs for w in _windows(s, length)]
    if not wins:
        return {}
    preds, tgts = [], []
    for i in range(0, len(wins), 64):
        stack = torch.stack(wins[i : i + 64])
        b, ln, k, d = stack.shape
        wz = whitener.transform(stack.reshape(b * ln, k, d)).reshape(b, ln, k, d).to(device)
        pred = model(wz)
        preds.append(model.pool(pred).reshape(-1, model.pool_proj.out_features))
        tgts.append(model.pool(wz[:, 1:]).reshape(-1, model.pool_proj.out_features))
    p, t = torch.cat(preds), torch.cat(tgts)
    return {
        "recall@1": round(recall_at_k(p, t, 1), 3),
        "recall@5": round(recall_at_k(p, t, 5), 3),
        "diversity": round(prediction_diversity(p), 3),
        "pool": p.shape[0],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--repo", default=None)
    ap.add_argument("--out-repo", default=None, help="HF repo to push predictor artifacts")
    ap.add_argument("--n-docs", type=int, default=4000)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    max_span, max_sents, min_sents = 64, 24, args.window + 1  # 64 = stage-1 training cap; 48 tripled truncation (9.1% vs 3.4%)
    if args.smoke:
        args.model_name, args.repo, args.n_docs, args.steps = "Qwen/Qwen2.5-0.5B", None, 30, 60
        args.window = 4
        max_span, max_sents, min_sents = 24, 8, 5
        print("=== SMOKE (tiny model, synthetic corpus) ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    quantize = device == "cuda" and not args.smoke
    pm, gist, tok = _load_stage1(args.model_name, args.repo, device, quantize)

    # ── encode corpus into gist sequences ────────────────────────────────────
    docs_text = SMOKE_TEXTS if args.smoke else list(_doc_texts(args.n_docs))
    seqs = encode_corpus(
        pm, gist, tok, docs_text[: args.n_docs], max_span, max_sents, min_sents, device
    )
    print(f"encoded {len(seqs)} gist sequences", flush=True)
    # document-disjoint split
    n_eval = max(1, len(seqs) // 10)
    eval_seqs, train_seqs = seqs[:n_eval], seqs[n_eval:]

    # ── fit per-slot whiteners on TRAIN gists only ───────────────────────────
    k, hidden = train_seqs[0].shape[1], train_seqs[0].shape[2]
    flat = torch.cat([s.reshape(-1, k, hidden) for s in train_seqs])  # [N, k, hidden]
    whitener = PerSlotWhitener.fit_streaming(iter(flat.split(4096)), k=k)
    print(f"fit {k} per-slot whiteners on {flat.shape[0]} train gists", flush=True)

    # ── train the predictor ──────────────────────────────────────────────────
    model = NextThoughtPredictor(
        d=hidden,
        k=k,
        d_model=256 if args.smoke else 640,
        layers=2 if args.smoke else 6,
        heads=4 if args.smoke else 8,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    step = 0
    while step < args.steps:
        for wz in _batches(train_seqs, args.window, 8 if args.smoke else 64, whitener):
            wz = wz.to(device)
            pred = model(wz)
            tgt = wz[:, 1:]
            pp = model.pool(pred).reshape(-1, model.pool_proj.out_features)
            tp = model.pool(tgt).reshape(-1, model.pool_proj.out_features)
            loss = 0.1 * regression_loss(pred, tgt) + info_nce_loss(pp, tp)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % (20 if args.smoke else 500) == 0:
                ev = evaluate(model, eval_seqs, args.window, whitener, device)
                print(f"[step {step}] loss {loss.item():.4f}  eval {ev}", flush=True)
            if step >= args.steps:
                break

    ev = evaluate(model, eval_seqs, args.window, whitener, device)
    print(f"[FINAL] {ev}", flush=True)
    print("GATE: recall@5 > 0.40 AND diversity < corpus-mean-sim ⇒ predictor real.")

    if args.out_repo:
        _push_artifacts(model, whitener, args.out_repo, ev)


def _doc_texts(n):  # noqa: ANN001
    """Stream n raw document texts from FineWeb-Edu."""
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True
    )
    for seen, row in enumerate(ds):
        yield row.get("text") or ""
        if seen + 1 >= n:
            break


def _push_artifacts(model, whitener, repo, ev):  # noqa: ANN001
    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    from huggingface_hub import upload_folder  # noqa: PLC0415

    d = Path("/tmp/stage2_out")  # noqa: S108
    d.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), d / "predictor.pt")
    whitener.save(d / "whiteners.pt")
    (d / "manifest.json").write_text(json.dumps({"eval": ev}, indent=2))
    upload_folder(repo_id=repo, folder_path=str(d), path_in_repo="stage2_predictor")
    print(f"pushed predictor artifacts to {repo}/stage2_predictor", flush=True)


if __name__ == "__main__":
    import os
    import sys

    main()
    sys.stdout.flush()
    os._exit(0)
