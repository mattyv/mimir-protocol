"""Front-loaded context test: do compressed thoughts help REAL problem-solving?

The burst harness fought the model (forced line-per-step decoding; plain
baseline 13%, uninterpretable). This is the clean instrument: give the model the
FIRST HALF of a solution as context — in different forms — then let it generate
freely to the answer. No schedule, no mid-stream splicing; thoughts sit right
after the prompt and generation flows from them, the same geometry the ladder
validated.

Arms (per problem; steps s_1..s_m are the first half of the reference solution):
  none        question only (floor: solve from scratch)
  text        s_1..s_m as plain text (ceiling: full context)
  gist_true   s_1..s_m as TRUE thoughts (compression cost under generation)
  gist_minus  s_1..s_{m-1} as true thoughts (control for gist_pred)
  gist_pred   s_1..s_{m-1} true + s_m PREDICTED (the marginal value of ONE
              predicted thought in real generation — THE number)

Read: gist_pred > gist_minus  => a predicted thought carries usable information
into free generation. gist_pred ~= gist_minus => it adds nothing. gist_true vs
text quantifies what compression costs when the model actually writes.

Run (GPU):
    HF_TOKEN=... PYTHONPATH=src python -u -m marker.run_frontload \\
        --repo mattyvee/mimir-artifacts --artifacts-repo mattyvee/mimir-artifacts \\
        --n-problems 120
Smoke (tiny model — mechanics only):
    PYTHONPATH=src python -m marker.run_frontload --smoke
"""

from __future__ import annotations

import argparse
import math
import time

import torch

from marker.burst import (
    GSM8K_FEWSHOT,
    answers_match,
    extract_answer,
    rope_shift_keys,
    rope_theta,
)
from marker.predictor import NextThoughtPredictor
from marker.run_bridge import predict_step
from marker.run_burst import _decode_step, _inject, _prefill

ARMS = (
    "none",
    "text",
    "gist_true",
    "gist_minus",
    "gist_pred",
    "gist_render",
    "gist_read",
    "none_read",
)
# gist_render = RECONSTITUTE-THEN-SOLVE (the BS-call experiment): the raw model
# can't READ injected thoughts while generating, but the render adapter — a
# trained gist-reader on this same frozen model — can. Transcribe the m thoughts
# back to text with the render adapter (its validated job, F1 0.99 w/ ledger),
# then solve from the transcription with the adapter off. Storage sidecar per
# step = the ledger numbers + the first token (the validated render config).
# Prediction: recovers to ~= the text arm => thoughts work in generation
# THROUGH their reader, and the ~3x KV-memory story is validated end-to-end.
#
# gist_read = READ-THROUGH-YOUR-OWN-ADAPTER: injects the SAME m true thoughts
# as gist_true (identical KV injection, no text sidecar, no reconstitution --
# see _n_inject_for) but leaves the render adapter (the trained gist-reader)
# ACTIVE for the final free-generate solve decode instead of switching it off.
# Isolates one variable against gist_true: does having the reader "on" while
# generating change how well the injected thoughts are used?
#
# none_read = THE LOAD-BEARING CONTROL for gist_read. Same prompt as `none`
# (question only, zero gists injected -- see _n_inject_for) but ALSO solves
# with the render adapter active, exactly like gist_read. Without this arm,
# a gist_read win could just mean "render happens to be a better solver in
# general," nothing to do with reading the injected thoughts; none_read pins
# that down as the true floor.
#
# Pre-registered read (corrected from an earlier, wrong framing that gated
# on gist_read's raw accuracy alone -- e.g. "below 0.35 => shelve it,
# path C"; that's wrong because raw accuracy can't separate "thoughts aren't
# readable" from "render can't solve well regardless of input"): the pulse
# is the PAIRED delta acc(gist_read) - acc(none_read), McNemar-tested on the
# per_problem correct flags (paired by problem id) -- roughly +0.13 at n=63
# clears p<0.05. A positive delta is an existence proof that reading injected
# thoughts through this adapter carries real signal, enough to justify
# training a dedicated read-head. A null result does NOT close the idea: the
# render adapter is a transcription specialist here (trained for single
# reconstructed thoughts, F1 0.99 w/ ledger) being asked to read m thoughts
# AND solve -- roughly 20x out of its training distribution -- so "no lift"
# is ambiguous between "thoughts aren't readable" and "wrong adapter for the
# job." No threshold/gate code lives here; the McNemar calc runs offline
# against per_problem (see Fix 3's "gen" field for genuine-solve vs
# transcribe-then-solve classification).


def context_split(n_steps: int, cap: int = 4) -> int:
    """How many leading reference steps become context: half the solution,
    at least 2 (so gist_minus/gist_pred have >=1 true thought of history),
    at most `cap`, and always leaving >=1 step for the model to do itself."""
    if n_steps < 3:
        raise ValueError("need >= 3 steps to split")
    return max(2, min(cap, math.ceil(n_steps / 2), n_steps - 1))


def answer_done(text: str) -> bool:
    """Stop condition for free decoding: the model has produced its '#### x'
    answer line (marker + a digit + a line break after it)."""
    if "####" not in text:
        return False
    tail = text.split("####")[-1]
    return any(c.isdigit() for c in tail) and "\n" in tail


def _n_inject_for(arm: str, m: int) -> int:
    """How many leading true-step gists (thought vectors) this arm injects into
    the KV cache (the model's per-layer memory of everything read so far).
    gist_read injects the SAME m true gists as gist_true -- same injection
    code path, same depth -- it only differs in which adapter is active while
    reading them back out during the final free-generate decode (_solve_arm).
    none_read injects zero, identical to `none` -- it's the render-adapter
    control, not a gist arm."""
    return {
        "none": 0,
        "text": 0,
        "gist_render": 0,
        "gist_true": m,
        "gist_read": m,
        "gist_minus": m - 1,
        "gist_pred": m,
        "none_read": 0,
    }[arm]


def _with_read_adapter(pm, arm, fn):  # noqa: ANN001
    """Run fn() (no args) with the render adapter (the trained gist-reader)
    active if `arm` is one of the `_read` arms (gist_read, none_read); every
    other arm runs fn() under whatever adapter is already active (default).
    Always restores "default" afterward for a `_read` arm -- this is adapter-
    state hygiene (so the next arm never accidentally runs under "render"),
    not error-swallowing: an exception from fn() still propagates through the
    finally.

    Shared by two call sites: the final free-generate solve decode
    (_solve_arm) and, in _run, whichever forward pass produces the logits
    that SELECT the first generated token (Fix 2) -- both need "is this a
    `_read` arm, and if so run under render" with the same restore-on-exit."""
    if not arm.endswith("_read"):
        return fn()
    pm.set_adapter("render")
    try:
        return fn()
    finally:
        pm.set_adapter("default")


def _solve_arm(pm, arm, free_generate, cache, pos, logits):  # noqa: ANN001
    """Run the final free-generate solve decode for one arm. `_read` arms
    (gist_read, none_read) decode under the render adapter; every other arm
    decodes under default. See _with_read_adapter for the adapter-switch and
    restore-on-exit contract."""
    return _with_read_adapter(pm, arm, lambda: free_generate(cache, pos, logits))


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--repo", default=None)
    ap.add_argument("--artifacts-repo", default=None)
    ap.add_argument("--subdir", default="stage2_cot_openr1")
    ap.add_argument("--bridge-subdir", default="bridge_validated")
    ap.add_argument("--render-subdir", default="render_adapter_ledger")
    ap.add_argument("--out-repo", default=None)
    ap.add_argument("--dataset", default="openai/gsm8k")
    ap.add_argument("--n-problems", type=int, default=120)
    ap.add_argument("--m-cap", type=int, default=4)
    ap.add_argument(
        "--min-ref-steps",
        type=int,
        default=0,
        help="keep only problems with >= this many reference steps (harder; "
        "0 = no filter). Streams more of the test split to fill n-problems.",
    )
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--max-span", type=int, default=64)
    ap.add_argument("--max-gen-toks", type=int, default=220)
    ap.add_argument("--dump-samples", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.model_name, args.repo, args.n_problems = "Qwen/Qwen2.5-0.5B", None, 3
        args.max_gen_toks = 60

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from marker.gist_model import encode_gist, gist_kv  # noqa: PLC0415
    from marker.reason_check import split_solution_steps  # noqa: PLC0415
    from marker.run_stage2 import _load_stage1  # noqa: PLC0415

    pm, gist, tok = _load_stage1(
        args.model_name, args.repo, device, device == "cuda" and not args.smoke
    )
    k = gist.shape[0]
    nl_id = next((t for t in tok("\n", add_special_tokens=False).input_ids if t), None)
    probe_kv, _, _ = gist_kv(pm, gist, tok("hi", add_special_tokens=False).input_ids)
    kv_dtype = probe_kv.keys[0].dtype
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
                head_dim=probe_kv.keys[0].shape[3],
                width=64,
            )
            .to(device)
            .eval()
        )
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

    # the trained gist-READER (render adapter) for the gist_render arm; smoke
    # attaches a random one to exercise the path mechanically
    if args.smoke:
        from marker.render import attach_render  # noqa: PLC0415

        attach_render(pm)
        pm.set_adapter("default")
    else:
        from huggingface_hub import snapshot_download  # noqa: PLC0415

        loc = snapshot_download(
            args.artifacts_repo, allow_patterns=[f"{args.render_subdir}/render/*"]
        )
        pm.load_adapter(f"{loc}/{args.render_subdir}/render", adapter_name="render")
        pm.set_adapter("default")
        print(f"render adapter loaded from {args.render_subdir}", flush=True)

    if args.smoke:
        from marker.run_stage2 import _smoke_cot_texts  # noqa: PLC0415

        probs = [("What is the total?", "8", t) for t in _smoke_cot_texts(args.n_problems)]
    else:
        from datasets import load_dataset  # noqa: PLC0415

        from marker.reason_check import split_solution_steps as _sss  # noqa: PLC0415

        ds = load_dataset(args.dataset, "main", split="test", streaming=True)
        probs = []
        for row in ds:
            if len(probs) >= args.n_problems:
                break
            # --min-ref-steps filters to the HARDEST problems (more reasoning
            # steps = lower solve-alone accuracy = more headroom for injected
            # context — the one config where thoughts-as-context could still help)
            if args.min_ref_steps and len(_sss(row["answer"])) < args.min_ref_steps:
                continue
            probs.append((row["question"], extract_answer(row["answer"]), row["answer"]))

    def _shift(akv, delta):  # noqa: ANN001
        return type(akv)(
            akv.n_layers,
            [rope_shift_keys(x, delta, theta).to(kv_dtype) for x in akv.keys],
            [x.to(kv_dtype) for x in akv.values],
        )

    @torch.no_grad()
    def _free_generate(cache, pos, logits):  # noqa: ANN001
        """Decode freely until the '#### x' answer line or the cap. Returns
        (text, n_tokens)."""
        toks = []
        text = ""
        for _ in range(args.max_gen_toks):
            t, cache, pos, logits = _decode_step(pm, cache, pos, logits, nl_id, 48)
            toks.extend(t)
            text = tok.decode(toks)
            if answer_done(text) or len(toks) >= args.max_gen_toks:
                break
        return text, len(toks)

    from marker.render import extract_ledger  # noqa: PLC0415
    from marker.run_render import _render_reconstruct  # noqa: PLC0415

    @torch.no_grad()
    def _reconstitute(step_ids_i, step_text):  # noqa: ANN001
        """Transcribe one thought back to text with the trained render adapter
        (the validated config: ledger numbers + first token as the stored
        sidecar). Returns (text, n_reader_tokens)."""
        kv, cs, _ = gist_kv(pm, gist, step_ids_i)  # default adapter encodes
        nums = extract_ledger(step_text)
        led = tok(" ".join(nums) + "\n", add_special_tokens=False).input_ids if nums else []
        pm.set_adapter("render")
        try:
            rec = _render_reconstruct(
                pm, kv, cs, step_ids_i[0], args.max_span, {nl_id}, prefix_ids=led
            )
        finally:
            pm.set_adapter("default")
        return tok.decode(rec).strip(), len(rec)

    @torch.no_grad()
    def _run(question, steps, m, arm):  # noqa: ANN001
        step_ids = [tok(s, add_special_tokens=False).input_ids[: args.max_span] for s in steps]
        prompt = f"{GSM8K_FEWSHOT}Question: {question}\nAnswer:\n"
        recon_toks = 0
        if arm == "text":
            prompt += "".join(s.strip() + "\n" for s in steps[:m])
        elif arm == "gist_render":
            # reconstitute-then-solve: thoughts -> reader -> text -> solve
            lines = []
            for i in range(m):
                line, n = _reconstitute(step_ids[i], steps[i])
                lines.append(line)
                recon_toks += n
            prompt += "".join(ln + "\n" for ln in lines if ln)
        ids = tok(prompt, add_special_tokens=False).input_ids
        cache, pos, logits = _prefill(pm, ids)

        n_inject = _n_inject_for(arm, m)
        if n_inject:
            summ = encode_gist(pm, gist, step_ids[:m]).float()  # [m, k, hidden] true summaries
            for i in range(n_inject):
                if arm == "gist_pred" and i == m - 1:
                    # predicted thought for step m from true thoughts 1..m-1
                    # (row m-1 rides along causally masked — tested contract)
                    ghat = predict_step(predictor, summ.to(device), m - 1, args.window)
                    akv = _shift(bridge(ghat), pos)
                else:
                    kv, _, _ = gist_kv(pm, gist, step_ids[i])
                    akv = _shift(kv, pos - len(step_ids[i]))
                cache, pos = _inject(pm, cache, pos, akv)

            # newline read-through primes the first free token after the
            # thoughts. This forward's logits are what _decode_step argmaxes
            # to pick token 0 of the free-generate solve -- for a `_read` arm
            # (gist_read) that pick must happen under the render adapter (Fix
            # 2: previously this ran under default, then _solve_arm switched
            # to render only AFTER token 0 was already chosen from the
            # default-adapter distribution -- a poisoned first token on
            # every gist_read generation).
            def _prime():
                out = pm(
                    torch.tensor([[nl_id]], device=device),
                    past_key_values=cache,
                    position_ids=torch.tensor([[pos]], device=device),
                    use_cache=True,
                )
                return out.past_key_values, pos + 1, out.logits[0, -1]

            cache, pos, logits = _with_read_adapter(pm, arm, _prime)
        elif arm.endswith("_read"):
            # none_read has no injected thoughts (n_inject == 0), so there's
            # no priming forward above to switch -- the plain prefill logits
            # ARE what picks token 0. Recompute that same prefill under render
            # so none_read's token-0 pick is timed the same way as gist_read's
            # (render active at the moment the token-0 logits are produced).
            cache, pos, logits = _with_read_adapter(pm, arm, lambda: _prefill(pm, ids))
        text, ntok = _solve_arm(pm, arm, _free_generate, cache, pos, logits)
        return text, ntok, recon_toks

    agg = {a: {"correct": 0, "toks": 0, "recon": 0, "n": 0, "secs": 0.0} for a in ARMS}
    samples = []
    per_problem = []  # one {pid, arm, correct} record per (problem, arm) pair,
    # for EVERY scored problem (not just the dumped samples) -- so paired
    # per-problem stats (e.g. McNemar) are computable downstream. The `_read`
    # arms (gist_read, none_read) also carry the full generation text (Fix 3)
    # so a later pass can classify a pass as genuine-solve vs
    # transcribe-then-solve; other arms stay at the 3-field shape.
    step_counts, m_counts = [], []
    for pi, (q, gold, ref) in enumerate(probs):
        steps = split_solution_steps(ref)
        if len(steps) < 3:
            continue
        m = context_split(len(steps), args.m_cap)
        step_counts.append(len(steps))
        m_counts.append(m)
        for a in ARMS:
            t0 = time.time()
            text, ntok, recon = _run(q, steps, m, a)
            ok = answers_match(extract_answer(text), gold)
            agg[a]["secs"] += time.time() - t0
            agg[a]["correct"] += int(ok)
            agg[a]["toks"] += ntok
            agg[a]["recon"] += recon
            agg[a]["n"] += 1
            rec = {"pid": pi, "arm": a, "correct": bool(ok)}
            if a.endswith("_read"):
                rec["gen"] = text
            per_problem.append(rec)
            if pi < args.dump_samples:
                samples.append(
                    {
                        "prob": pi,
                        "arm": a,
                        "m": m,
                        "gold": gold,
                        "correct": bool(ok),
                        "gen": text[:600],
                    }
                )
                print(f"\n--- p{pi} {a} (m={m}) gold={gold} ok={ok} ---\n{text[:400]}", flush=True)
        if (pi + 1) % 20 == 0:
            print(f"  ...{pi + 1} problems", flush=True)

    mean = lambda xs: round(sum(xs) / len(xs), 2) if xs else None  # noqa: E731
    manifest = {
        "m_cap": args.m_cap,
        "min_ref_steps": args.min_ref_steps,
        "n_problems": args.n_problems,
        "mean_ref_steps": mean(step_counts),
        "mean_context_m": mean(m_counts),
        "samples": samples,
        "per_problem": per_problem,
        "arms": {},
    }
    print("\n[FRONTLOAD] arm          acc    gen_toks  recon_toks   mean_s", flush=True)
    for a in ARMS:
        g = agg[a]
        n = max(1, g["n"])
        row = {
            "acc": round(g["correct"] / n, 3),
            "mean_gen_toks": round(g["toks"] / n, 1),
            "mean_recon_toks": round(g["recon"] / n, 1),
            "mean_secs": round(g["secs"] / n, 2),
            "n": g["n"],
        }
        manifest["arms"][a] = row
        print(
            f"  {a:12s} {row['acc']:>5}  {row['mean_gen_toks']:>9}  "
            f"{row['mean_recon_toks']:>9}  {row['mean_secs']:>7}",
            flush=True,
        )

    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    d = Path("/tmp/frontload_out")  # noqa: S108
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[FRONTLOAD MANIFEST] {json.dumps(manifest)}", flush=True)
    if args.out_repo:
        from huggingface_hub import upload_folder  # noqa: PLC0415

        upload_folder(repo_id=args.out_repo, folder_path=str(d), path_in_repo="frontload")
        print(f"pushed frontload manifest to {args.out_repo}/frontload", flush=True)


if __name__ == "__main__":
    import os
    import sys

    main()
    sys.stdout.flush()
    os._exit(0)
