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

ARMS = ("none", "text", "gist_true", "gist_minus", "gist_pred")


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
    ap.add_argument("--m-cap", type=int, default=4)
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

    if args.smoke:
        from marker.run_stage2 import _smoke_cot_texts  # noqa: PLC0415

        probs = [("What is the total?", "8", t) for t in _smoke_cot_texts(args.n_problems)]
    else:
        from datasets import load_dataset  # noqa: PLC0415

        ds = load_dataset(args.dataset, "main", split="test", streaming=True)
        probs = []
        for row in ds:
            if len(probs) >= args.n_problems:
                break
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

    @torch.no_grad()
    def _run(question, steps, m, arm):  # noqa: ANN001
        step_ids = [tok(s, add_special_tokens=False).input_ids[: args.max_span] for s in steps]
        prompt = f"{GSM8K_FEWSHOT}Question: {question}\nAnswer:\n"
        if arm == "text":
            prompt += "".join(s.strip() + "\n" for s in steps[:m])
        ids = tok(prompt, add_special_tokens=False).input_ids
        cache, pos, logits = _prefill(pm, ids)

        n_inject = {"none": 0, "text": 0, "gist_true": m, "gist_minus": m - 1, "gist_pred": m}[arm]
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
            # newline read-through primes the first free token after the thoughts
            out = pm(
                torch.tensor([[nl_id]], device=device),
                past_key_values=cache,
                position_ids=torch.tensor([[pos]], device=device),
                use_cache=True,
            )
            cache, pos, logits = out.past_key_values, pos + 1, out.logits[0, -1]
        return _free_generate(cache, pos, logits)

    agg = {a: {"correct": 0, "toks": 0, "n": 0, "secs": 0.0} for a in ARMS}
    samples = []
    for pi, (q, gold, ref) in enumerate(probs):
        steps = split_solution_steps(ref)
        if len(steps) < 3:
            continue
        m = context_split(len(steps), args.m_cap)
        for a in ARMS:
            t0 = time.time()
            text, ntok = _run(q, steps, m, a)
            ok = answers_match(extract_answer(text), gold)
            agg[a]["secs"] += time.time() - t0
            agg[a]["correct"] += int(ok)
            agg[a]["toks"] += ntok
            agg[a]["n"] += 1
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

    manifest = {"m_cap": args.m_cap, "n_problems": args.n_problems, "samples": samples, "arms": {}}
    print("\n[FRONTLOAD] arm          acc    gen_toks   mean_s", flush=True)
    for a in ARMS:
        g = agg[a]
        n = max(1, g["n"])
        row = {
            "acc": round(g["correct"] / n, 3),
            "mean_gen_toks": round(g["toks"] / n, 1),
            "mean_secs": round(g["secs"] / n, 2),
            "n": g["n"],
        }
        manifest["arms"][a] = row
        print(
            f"  {a:12s} {row['acc']:>5}  {row['mean_gen_toks']:>9}  {row['mean_secs']:>7}",
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
