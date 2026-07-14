"""Anchored-burst end-to-end test (the pre-registered fast-lane speed test).

The rollout showed open-loop chaining drifts after ~2 steps but a real step
resets it. This measures whether INTERLEAVING cheap latent steps between decoded
anchor steps buys a speed-up at equal accuracy — on real generation, scored on
the final answer (accuracy + decoded tokens), NOT NLL.

Four arms per problem (Fable pre-registration):
  plain        decode every step (baseline to beat on tokens at equal accuracy)
  burst_true   latent steps inject the TRUE step's thought (oracle ceiling)
  burst_pred   latent steps inject the PREDICTED thought (the actual test)
  burst_none   latent steps are just SKIPPED, nothing injected (killer control —
               if == burst_pred, the predictor adds nothing and the "win" is
               model robustness to dropped steps)

Injection is MID-sequence (unlike the front-loaded axiom KV), so burst_true is
also the control for whether mid-stream injection works at all: burst_true≈plain
=> the schedule/injection mechanics are sound; burst_true≈burst_none => injecting
a thought into a running context doesn't land (a positional problem, not a
prediction one).

Run (GPU):
    HF_TOKEN=... PYTHONPATH=src python -u -m marker.run_burst \\
        --repo mattyvee/mimir-artifacts --artifacts-repo mattyvee/mimir-artifacts \\
        --n-problems 120 --anchor-every 2
Smoke (tiny model — mechanics only):
    PYTHONPATH=src python -m marker.run_burst --smoke
"""

from __future__ import annotations

import argparse
import time

import torch

from marker.burst import (
    GSM8K_FEWSHOT,
    answers_match,
    extract_answer,
    make_schedule,
    rope_shift_keys,
    rope_theta,
)
from marker.predictor import NextThoughtPredictor
from marker.rollout import rollout


@torch.no_grad()
def _prefill(pm, ids):  # noqa: ANN001
    device = next(pm.parameters()).device
    out = pm(torch.tensor([ids], device=device), use_cache=True)
    return out.past_key_values, len(ids), out.logits[0, -1]


@torch.no_grad()
def _decode_step(pm, cache, pos, first_logits, stop_id, max_toks):  # noqa: ANN001
    """Greedy-decode one step (until newline or cap). Returns (token_ids, cache,
    new_pos, next_logits). first_logits primes token 0 from the prior position."""
    device = next(pm.parameters()).device
    toks, logits = [], first_logits
    for _ in range(max_toks):
        nxt = int(logits.argmax())
        toks.append(nxt)
        out = pm(
            torch.tensor([[nxt]], device=device),
            past_key_values=cache,
            position_ids=torch.tensor([[pos]], device=device),
            use_cache=True,
        )
        cache, pos, logits = out.past_key_values, pos + 1, out.logits[0, -1]
        if nxt == stop_id:
            break
    return toks, cache, pos, logits


@torch.no_grad()
def _inject(pm, cache, pos, axiom_kv):  # noqa: ANN001
    """Append a thought's per-layer K/V to the running cache at the current
    position; return (cache, pos + k, next_logits). The k injected slots take the
    place of a decoded step. next_logits comes from a no-op re-read of the last
    injected slot so the following step has a token to start from."""
    k = axiom_kv.keys[0].shape[2]
    for i in range(axiom_kv.n_layers):
        cache.update(axiom_kv.keys[i], axiom_kv.values[i], i)
    return cache, pos + k


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--repo", default=None)
    ap.add_argument("--artifacts-repo", default=None)
    ap.add_argument("--subdir", default="stage2_cot_openr1")
    ap.add_argument("--bridge-subdir", default="bridge_validated")
    ap.add_argument("--out-repo", default=None)
    ap.add_argument("--dataset", default="openai/gsm8k")
    ap.add_argument("--n-problems", type=int, default=120)
    ap.add_argument("--anchor-every", type=int, default=2)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--max-step-toks", type=int, default=40)
    ap.add_argument("--max-answer-toks", type=int, default=12)  # '#### ' primed: number + newline
    ap.add_argument(
        "--dump-samples", type=int, default=3, help="dump full traces for first N problems"
    )
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.model_name, args.repo, args.n_problems = "Qwen/Qwen2.5-0.5B", None, 4

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from marker.gist_model import encode_gist, gist_kv  # noqa: PLC0415
    from marker.reason_check import split_solution_steps  # noqa: PLC0415
    from marker.run_stage2 import _load_stage1  # noqa: PLC0415

    pm, gist, tok = _load_stage1(
        args.model_name, args.repo, device, device == "cuda" and not args.smoke
    )
    k = gist.shape[0]
    nl_id = next((t for t in tok("\n", add_special_tokens=False).input_ids if t), None)

    # geometry for injection: head_dim + rope_theta (to re-rotate injected keys
    # to their placement position — Fable burst review)
    probe_kv, _, _ = gist_kv(pm, gist, tok("hi", add_special_tokens=False).input_ids)
    kv_dtype, head_dim = probe_kv.keys[0].dtype, probe_kv.keys[0].shape[3]
    theta = rope_theta(pm.config)

    from marker.run_bridge import GistBridge  # noqa: PLC0415
    from marker.run_rollout import _load_bridge  # noqa: PLC0415

    if args.smoke:
        predictor = NextThoughtPredictor(d=gist.shape[-1], k=k, d_model=48, layers=2, heads=4)
        bridge = (
            GistBridge(
                d=gist.shape[-1],
                k=k,
                n_layers=probe_kv.n_layers,
                n_kv_heads=probe_kv.keys[0].shape[1],
                head_dim=head_dim,
                width=64,
            )
            .to(device)
            .eval()
        )  # random bridge — exercises the burst_pred path in smoke
    else:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        from marker.run_confidence import _predictor_from_state  # noqa: PLC0415

        predictor = _predictor_from_state(
            torch.load(
                hf_hub_download(args.artifacts_repo, f"{args.subdir}/predictor.pt"),
                map_location="cpu",
            ),
            args.heads,
        )
        bridge = _load_bridge(
            torch.load(
                hf_hub_download(args.artifacts_repo, f"{args.bridge_subdir}/bridge.pt"),
                map_location="cpu",
            ),
            gist.shape[-1],
            k,
            probe_kv,
            device,
        )
    predictor = predictor.to(device).eval()

    # ── data: GSM8K questions + gold answers + reference steps ────────────────
    if args.smoke:
        from marker.run_stage2 import _smoke_cot_texts  # noqa: PLC0415

        probs = [("What is the total?", "42", t) for t in _smoke_cot_texts(args.n_problems)]
    else:
        from datasets import load_dataset  # noqa: PLC0415

        ds = load_dataset(args.dataset, "main", split="test", streaming=True)
        probs = []
        for row in ds:
            if len(probs) >= args.n_problems:
                break
            gold = extract_answer(row["answer"])
            probs.append((row["question"], gold, row["answer"]))

    def _shift(akv, delta):  # noqa: ANN001
        """Rotate an AxiomKV's keys to a placement position; cast to attn dtype."""
        return type(akv)(
            akv.n_layers,
            [rope_shift_keys(x, delta, theta).to(kv_dtype) for x in akv.keys],
            [x.to(kv_dtype) for x in akv.values],
        )

    def _thought_kv_true(step_ids, at_pos):  # noqa: ANN001
        # gist keys come out RoPE'd at [span_len, span_len+k); shift to [at_pos,..)
        kv, _, _ = gist_kv(pm, gist, step_ids)
        return _shift(kv, at_pos - len(step_ids))

    def _thought_kv_from_summary(summ_vec, at_pos):  # noqa: ANN001
        # bridge keys are canonical (all slots at 0); shift to at_pos
        return _shift(bridge(summ_vec.to(device)), at_pos)

    @torch.no_grad()
    def _run(question, ref_steps, arm):  # noqa: ANN001, PLR0912
        # 2-shot demonstrations pin the line-per-step + '#### x' format a BASE
        # model won't produce zero-shot (first run: plain acc 0.044, void)
        prompt = f"{GSM8K_FEWSHOT}Question: {question}\nAnswer:\n"
        ids = tok(prompt, add_special_tokens=False).input_ids
        cache, pos, logits = _prefill(pm, ids)
        sched = make_schedule(len(ref_steps), args.anchor_every)
        thoughts, decoded, fwd, cap = [], 0, 1, 0  # fwd counts prefill
        trace = []  # what actually happened, for the qualitative dump
        for i, step in enumerate(ref_steps):
            step_ids = tok(step, add_special_tokens=False).input_ids[: args.max_step_toks]
            if not step_ids:
                continue
            kind = "anchor" if arm == "plain" else sched[i]
            if kind == "anchor":
                toks, cache, pos, logits = _decode_step(
                    pm, cache, pos, logits, nl_id, args.max_step_toks
                )
                decoded += len(toks)
                fwd += len(toks)
                capped = len(toks) >= args.max_step_toks and toks[-1] != nl_id
                cap += int(capped)
                trace.append(f"[dec{'|CAP' if capped else ''}] {tok.decode(toks).strip()}")
                if arm == "burst_pred":  # only this arm needs predictor history
                    body = toks[:-1] if toks and toks[-1] == nl_id else toks
                    if body:
                        thoughts.append(encode_gist(pm, gist, [body]).float()[0])
                        fwd += 1
            else:  # latent step
                if arm == "burst_none":
                    trace.append("[skip]")
                    continue  # skip entirely — no thought, no injection
                if arm == "burst_true":
                    akv = _thought_kv_true(step_ids, pos)
                    fwd += 1
                    trace.append(f"[inject-true] ({step.strip()})")
                else:  # burst_pred — predict next thought from history, free-running
                    if not thoughts:  # nothing to predict from yet
                        trace.append("[skip:no-history]")
                        continue
                    seq = torch.stack(thoughts)  # all on `device` (no cpu mix)
                    ghat = rollout(predictor, seq, 1, args.window)[0]  # tested path, no off-by-one
                    akv = _thought_kv_from_summary(ghat, pos)
                    thoughts.append(ghat.detach())
                    trace.append("[inject-pred]")
                cache, pos = _inject(pm, cache, pos, akv)
                # re-read a newline to prime the next step's first token
                out = pm(
                    torch.tensor([[nl_id]], device=device),
                    past_key_values=cache,
                    position_ids=torch.tensor([[pos]], device=device),
                    use_cache=True,
                )
                cache, pos, logits, fwd = out.past_key_values, pos + 1, out.logits[0, -1], fwd + 1
        # force-prime '#### ' so the final answer decode is format-independent
        # and identical across arms (extraction never depends on the model
        # choosing to emit the marker)
        hash_ids = tok("#### ", add_special_tokens=False).input_ids
        out = pm(
            torch.tensor([hash_ids], device=device),
            past_key_values=cache,
            position_ids=torch.arange(pos, pos + len(hash_ids), device=device).unsqueeze(0),
            use_cache=True,
        )
        cache, pos, logits = out.past_key_values, pos + len(hash_ids), out.logits[0, -1]
        fwd += 1
        ans_toks, _, _, _ = _decode_step(pm, cache, pos, logits, nl_id, args.max_answer_toks)
        fwd += len(ans_toks)
        trace.append(f"[answer] #### {tok.decode(ans_toks).strip()}")
        return tok.decode(ans_toks), decoded, fwd, cap, trace

    arms = ["plain", "burst_true", "burst_pred", "burst_none"]
    agg = {a: {"correct": 0, "toks": 0, "fwd": 0, "cap": 0, "n": 0, "secs": 0.0} for a in arms}
    samples: list[dict] = []
    for pi, (q, gold, ref) in enumerate(probs):
        steps = split_solution_steps(ref)
        if len(steps) < 3:
            continue
        for a in arms:
            t0 = time.time()
            ans, dec, fwd, cap, trace = _run(q, steps, a)
            agg[a]["secs"] += time.time() - t0
            ok = answers_match(extract_answer(ans), gold)
            agg[a]["correct"] += int(ok)
            agg[a]["toks"] += dec
            agg[a]["fwd"] += fwd
            agg[a]["cap"] += cap
            agg[a]["n"] += 1
            # qualitative dump: full traces for the first problems — the last two
            # runs were misread because nobody ever READ a generation
            if pi < args.dump_samples:
                samples.append(
                    {
                        "prob": pi,
                        "arm": a,
                        "gold": gold,
                        "correct": bool(ok),
                        "q": q[:200],
                        "trace": trace,
                    }
                )
                print(f"\n--- p{pi} {a} gold={gold} ok={ok} ---", flush=True)
                for line in trace:
                    print(f"    {line}", flush=True)
        if (pi + 1) % 20 == 0:
            print(f"  ...{pi + 1} problems", flush=True)

    manifest = {
        "anchor_every": args.anchor_every,
        "n_problems": args.n_problems,
        "samples": samples,
        "arms": {},
    }
    # decoded_toks / model_forwards are the fair cost proxies; mean_secs is rough
    # (kernel/overhead noise). cap_rate = fraction of anchor decodes that hit the
    # token cap without a newline — spikes if injection derails the next decode.
    print("\n[BURST] arm            acc    dec_toks   fwd   cap_rate  mean_s", flush=True)
    for a in arms:
        g = agg[a]
        n = max(1, g["n"])
        row = {
            "acc": round(g["correct"] / n, 3),
            "mean_decoded_toks": round(g["toks"] / n, 1),
            "mean_model_forwards": round(g["fwd"] / n, 1),
            "cap_rate": round(g["cap"] / n, 2),
            "mean_secs": round(g["secs"] / n, 3),
            "n": g["n"],
        }
        manifest["arms"][a] = row
        print(
            f"  {a:12s}  {row['acc']:>6}  {row['mean_decoded_toks']:>8}  "
            f"{row['mean_model_forwards']:>6}  {row['cap_rate']:>7}  {row['mean_secs']:>6}",
            flush=True,
        )

    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    d = Path("/tmp/burst_out")  # noqa: S108
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[BURST MANIFEST] {json.dumps(manifest)}", flush=True)
    if args.out_repo:
        from huggingface_hub import upload_folder  # noqa: PLC0415

        upload_folder(repo_id=args.out_repo, folder_path=str(d), path_in_repo="burst")
        print(f"pushed burst manifest to {args.out_repo}/burst", flush=True)


if __name__ == "__main__":
    import os
    import sys

    main()
    sys.stdout.flush()
    os._exit(0)
