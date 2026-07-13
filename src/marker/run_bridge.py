"""Bridge experiment (Stage-3b): what does a PREDICTED thought actually DO?

We have measured predicted thoughts only with a ruler in summary-space (retrieval
rank, cosine 0.665). We have never fed one into the model. This run trains the
bridge (final-layer summary -> injectable per-layer K/V) and reads a LADDER: for
each step n, inject a thought of step n and score how well the frozen model then
predicts step n+1 (teacher-forced tail NLL -> PPL -> gap_closed), on the SAME
(n, n+1) pairs at every rung:

    none                  floor (no context)
    full                  ceiling (step n's real tokens as context)
    gist_true             inject step n's TRUE per-layer K/V (encoder ceiling)
    bridge_true           inject bridge(step n's true summary)   <- conversion loss
    bridge_pred           inject bridge(predictor's guess of n)  <- is 0.665 usable?
    shuffled              inject bridge(a DIFFERENT step's summary) <- mislead control

Where gap_closed collapses down the ladder localizes the failure:
  gist_true high, bridge_true low  -> the summary is a lossy handle (conversion).
  bridge_true ~ gist_true, bridge_pred low -> prediction quality is the wall.
  bridge_pred usable                -> the thread is alive; the fast lane died on
                                       the GATE, not on prediction being unusable.

Run (GPU):
    HF_TOKEN=... PYTHONPATH=src python -u -m marker.run_bridge \\
        --repo mattyvee/mimir-artifacts \\
        --artifacts-repo mattyvee/mimir-artifacts --subdir stage2_cot_openr1 \\
        --dataset open-r1/OpenR1-Math-220k --n-docs 400 --skip-docs 2000 --steps 1500
Smoke (local tiny model, random predictor — mechanics only):
    PYTHONPATH=src python -m marker.run_bridge --smoke
"""

from __future__ import annotations

import argparse
import math

import torch

from marker.bridge import GistBridge, bridge_injection_nll
from marker.predictor import NextThoughtPredictor


def pred_pairs(n_steps: int) -> list[int]:
    """Step indices n whose thought we inject (and score n+1): n>=1 so the
    predictor has history to guess step n, n<=L-2 so step n+1 exists. Every rung
    scores exactly these pairs, so the ladder is apples-to-apples."""
    return list(range(1, n_steps - 1))


@torch.no_grad()
def predict_step(predictor, summ: torch.Tensor, n: int, window: int) -> torch.Tensor:  # noqa: ANN001
    """The predictor's guess of step n's summary from the (<= window-1) steps
    before it. Input is the WINDOWED slice summ[max(0, n-window+1) : n+1] so the
    sentence-position indices stay inside the range the predictor was trained on
    (it saw windows re-indexed from 0; positions beyond that are untrained
    embeddings — feeding a whole doc would run deep steps on garbage, Fable
    bridge review bug A). Step n's own summary rides along only as the masked
    target position: the block-causal mask keeps the readout at n-1 blind to it.
    Returns [k, d] on summ's device."""
    a = max(0, n - window + 1)
    x = summ[a : n + 1].unsqueeze(0)  # [1, m<=window, k, d]
    out = predictor(x)  # [1, m-1, k, d]; last index = readout at n-1 -> predicts n
    return out[0, -1]


def ladder_gap_closed(nlls: dict[str, list[float]]) -> dict[str, dict]:
    """Per-rung mean NLL -> PPL -> gap_closed, anchored by none (0.0) and full
    (1.0). gap_closed = (none_ppl - rung_ppl)/(none_ppl - full_ppl): fraction of
    the no-context->full-context gap the injected thought closes (<0 = worse than
    nothing). Empty rungs -> None; zero headroom -> 0.0 (no divide-by-zero)."""
    ppl = {r: (math.exp(sum(v) / len(v)) if v else None) for r, v in nlls.items()}
    none, full = ppl.get("none"), ppl.get("full")
    denom = (none - full) if (none is not None and full is not None) else 0.0
    out = {}
    for r, p in ppl.items():
        if p is None:
            out[r] = {"ppl": None, "gap_closed": None, "n": 0}
        elif denom <= 0:
            out[r] = {"ppl": round(p, 4), "gap_closed": 0.0, "n": len(nlls[r])}
        else:
            out[r] = {
                "ppl": round(p, 4),
                "gap_closed": round((none - p) / denom, 4),
                "n": len(nlls[r]),
            }
    return out


@torch.no_grad()
def tail_nll(pm, cache, cont_start: int, cont_ids: list[int]) -> float:  # noqa: ANN001
    """Teacher-forced mean NLL of cont_ids[1:] from cont_ids[:-1], decoded over
    `cache` (a fresh DynamicCache or None) starting at position cont_start. The
    ONE scoring path for every rung — the rungs differ only in what cache they
    inject and where the continuation starts. (Matches bridge_injection_nll: the
    first cont token is not scored from the injection, so every rung drops it
    identically.)"""
    import torch.nn.functional as F  # noqa: N812, PLC0415
    from transformers import DynamicCache  # noqa: PLC0415

    device = next(pm.parameters()).device
    cache = cache if cache is not None else DynamicCache()
    inp = torch.tensor([cont_ids[:-1]], device=device)
    pos = torch.arange(cont_start, cont_start + len(cont_ids) - 1, device=device).unsqueeze(0)
    out = pm(inp, past_key_values=cache, position_ids=pos, use_cache=True)
    return float(F.cross_entropy(out.logits[0], torch.tensor(cont_ids[1:], device=device)))


@torch.no_grad()
def _token_cache(pm, ids: list[int]):  # noqa: ANN001
    """Real-context cache: step n's own tokens as K/V (the full-context ceiling).
    Returns (DynamicCache, cont_start=len(ids))."""
    device = next(pm.parameters()).device
    out = pm(torch.tensor([ids], device=device), use_cache=True)
    return out.past_key_values, len(ids)


def _encode_doc(pm, gist, tok, text, unit, max_span, max_sents=16):  # noqa: ANN001
    """A doc -> (step token-id lists, summaries [L,k,hidden]). Summaries are the
    final-layer readout the predictor consumes; token ids feed the full/gist
    rungs. Steps with <2 tokens dropped (need a scorable tail); docs capped at
    max_sents=16 steps — the stage-2 cot cap the predictor's data respected
    (and OpenR1 traces can run to hundreds of lines)."""
    from marker.gist_model import encode_gist  # noqa: PLC0415
    from marker.run_stage2 import _split_units  # noqa: PLC0415

    ids = []
    for s in _split_units(text, unit):
        t = tok(s, add_special_tokens=False).input_ids[:max_span]
        if len(t) >= 2:
            ids.append(t)
        if len(ids) >= max_sents:
            break
    if len(ids) < 3:
        return None
    summ = encode_gist(pm, gist, ids).float()  # [L, k, hidden]
    return ids, summ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--repo", default=None, help="Stage-1 gist adapter repo")
    ap.add_argument("--artifacts-repo", default=None, help="HF repo with predictor.pt")
    ap.add_argument("--subdir", default="stage2_cot_openr1")
    ap.add_argument("--out-repo", default=None, help="push bridge + manifest here")
    ap.add_argument("--dataset", default="open-r1/OpenR1-Math-220k")
    ap.add_argument("--dataset-config", default=None)
    ap.add_argument("--text-field", default=None)
    ap.add_argument("--corpus", default="cot")
    ap.add_argument("--unit", choices=["line", "sentence"], default=None)
    ap.add_argument("--n-docs", type=int, default=400)
    ap.add_argument(
        "--skip-docs", type=int, default=2000, help="stream past the predictor's train range"
    )
    ap.add_argument(
        "--window",
        type=int,
        default=8,
        help="predictor input window: MUST match the predictor's training window "
        "(stage2 default 8) so sentence-position embeddings stay in-distribution",
    )
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--max-span", type=int, default=96)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    unit = args.unit or ("line" if args.corpus == "cot" else "sentence")
    if args.smoke:
        args.model_name, args.repo, args.n_docs, args.steps = "Qwen/Qwen2.5-0.5B", None, 40, 40
        args.max_span, args.skip_docs = 24, 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from marker.run_stage2 import _doc_texts, _load_stage1, _smoke_cot_texts

    pm, gist, tok = _load_stage1(
        args.model_name, args.repo, device, device == "cuda" and not args.smoke
    )

    # ── predictor (for the bridge_pred rung) ─────────────────────────────────
    if args.smoke:
        predictor = NextThoughtPredictor(
            d=gist.shape[-1], k=gist.shape[0], d_model=48, layers=2, heads=4
        )
        docs = _smoke_cot_texts(args.n_docs)
    else:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        from marker.run_confidence import _predictor_from_state  # noqa: PLC0415

        path = hf_hub_download(repo_id=args.artifacts_repo, filename=f"{args.subdir}/predictor.pt")
        predictor = _predictor_from_state(torch.load(path, map_location="cpu"), args.heads)
        gen = _doc_texts(
            args.n_docs + args.skip_docs,
            args.corpus,
            args.dataset,
            args.dataset_config,
            args.text_field,
        )
        docs = list(gen)[args.skip_docs :]
    predictor = predictor.to(device).eval()

    # ── encode docs -> (token ids, summaries) ────────────────────────────────
    encoded = [e for e in (_encode_doc(pm, gist, tok, t, unit, args.max_span) for t in docs) if e]
    print(f"encoded {len(encoded)} docs with >=3 scorable steps", flush=True)
    n_eval = max(1, len(encoded) // 5)
    eval_docs, train_docs = encoded[:n_eval], encoded[n_eval:]

    # ── build the bridge from a REAL gist_kv's shapes ────────────────────────
    from marker.gist_model import gist_kv  # noqa: PLC0415
    from marker.run_axiom_mlp_demo import _build_dynamic_cache  # noqa: PLC0415

    probe_kv, _, _ = gist_kv(pm, gist, encoded[0][0][0])
    n_kv_heads, head_dim = probe_kv.keys[0].shape[1], probe_kv.keys[0].shape[3]
    bridge = GistBridge(
        d=gist.shape[-1],
        k=gist.shape[0],
        n_layers=probe_kv.n_layers,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        width=args.width,
    ).to(device)
    print(
        f"bridge: d={gist.shape[-1]} k={gist.shape[0]} n_layers={probe_kv.n_layers} "
        f"n_kv_heads={n_kv_heads} head_dim={head_dim}",
        flush=True,
    )

    # ── train the bridge on TRUE summaries (convert->inject->NLL of next step) ─
    opt = torch.optim.AdamW(bridge.parameters(), lr=args.lr, weight_decay=0.01)
    torch.manual_seed(0)
    train_items = [(di, n) for di, (ids, _) in enumerate(train_docs) for n in pred_pairs(len(ids))]
    print(f"{len(train_items)} (doc, step) training pairs", flush=True)
    step = 0
    while step < args.steps and train_items:
        for j in torch.randperm(len(train_items)):
            di, n = train_items[int(j)]
            ids, summ = train_docs[di]
            loss = bridge_injection_nll(pm, bridge, summ[n].to(device), ids[n + 1])
            opt.zero_grad()
            loss.backward()
            if step == 0:
                ok = any(p.grad is not None and p.grad.abs().sum() > 0 for p in bridge.parameters())
                assert ok, "GRAD FAIL: no gradient reached the bridge (quantized/detach path)"
                print("GRAD_OK (bridge gradients flowing)", flush=True)
            opt.step()
            step += 1
            if step % (10 if args.smoke else 200) == 0:
                print(f"[step {step}] bridge nll {loss.item():.4f}", flush=True)
            if step >= args.steps:
                break

    # ── eval ladder on held-out docs ─────────────────────────────────────────
    bridge.eval()
    rungs = ["none", "full", "gist_true", "bridge_true", "bridge_pred", "shuffled"]
    nlls: dict[str, list[float]] = {r: [] for r in rungs}
    rng = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for di, (ids, summ) in enumerate(eval_docs):
            summ_dev = summ.to(device)
            for n in pred_pairs(len(ids)):
                cont = ids[n + 1]
                nlls["none"].append(tail_nll(pm, None, 0, cont))
                cache, cs = _token_cache(pm, ids[n])
                nlls["full"].append(tail_nll(pm, cache, cs, cont))
                kv, cs, _ = gist_kv(pm, gist, ids[n])
                nlls["gist_true"].append(tail_nll(pm, _build_dynamic_cache(kv, device), cs, cont))
                nlls["bridge_true"].append(
                    tail_nll(pm, _build_dynamic_cache(bridge(summ_dev[n]), device), bridge.k, cont)
                )
                # windowed prediction: positions stay inside the predictor's
                # trained range (Fable bridge review bug A)
                ghat = predict_step(predictor, summ_dev, n, args.window)
                nlls["bridge_pred"].append(
                    tail_nll(pm, _build_dynamic_cache(bridge(ghat), device), bridge.k, cont)
                )
                # mislead control: a random step from a DIFFERENT doc — same-doc
                # draws could pull step n+1 itself, injecting the answer's own
                # thought as the "mislead" (Fable bridge review bug B)
                dj = di
                while dj == di and len(eval_docs) > 1:
                    dj = int(torch.randint(0, len(eval_docs), (1,), generator=rng))
                o_ids, o_summ = eval_docs[dj]
                j = int(torch.randint(0, len(o_ids), (1,), generator=rng))
                nlls["shuffled"].append(
                    tail_nll(
                        pm,
                        _build_dynamic_cache(bridge(o_summ[j].to(device)), device),
                        bridge.k,
                        cont,
                    )
                )

    ladder = ladder_gap_closed(nlls)
    manifest = {
        "probe_dataset": args.dataset,
        "subdir": args.subdir,
        "skip_docs": args.skip_docs,
        "n_eval_docs": len(eval_docs),
        "steps": args.steps,
        "ladder": ladder,
    }
    print("\n[BRIDGE LADDER]", flush=True)
    for r in rungs:
        print(
            f"  {r:12s} ppl={ladder[r]['ppl']} gap_closed={ladder[r]['gap_closed']} n={ladder[r]['n']}",
            flush=True,
        )

    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    d = Path("/tmp/bridge_out")  # noqa: S108
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
    torch.save(bridge.state_dict(), d / "bridge.pt")
    print(f"[BRIDGE MANIFEST] {json.dumps(manifest)}", flush=True)  # single-line, survives tail

    if args.out_repo:
        from huggingface_hub import upload_folder  # noqa: PLC0415

        upload_folder(repo_id=args.out_repo, folder_path=str(d), path_in_repo="bridge")
        print(f"pushed bridge + manifest to {args.out_repo}/bridge", flush=True)


if __name__ == "__main__":
    import os
    import sys

    main()
    sys.stdout.flush()
    os._exit(0)
