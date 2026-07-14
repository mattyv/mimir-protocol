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
    info_nce_within,
    prediction_diversity,
    recall_at_k,
    regression_loss,
)
from marker.whiten import IdentityWhitener, PerSlotWhitener

_SMOKE_SUBJECTS = [
    "robot",
    "gardener",
    "engine",
    "river",
    "market",
    "printer",
    "climber",
    "baker",
    "sailor",
    "furnace",
    "glacier",
    "orchard",
    "beacon",
    "reactor",
    "surgeon",
    "pianist",
    "harbor",
    "volcano",
    "telescope",
    "vineyard",
    "foundry",
    "aquifer",
    "comet",
    "loom",
    "kiln",
    "trawler",
    "windmill",
    "quarry",
    "lantern",
    "turbine",
    "meadow",
    "cistern",
    "anvil",
    "spindle",
    "cavern",
    "delta",
    "geyser",
    "prairie",
    "canyon",
    "atoll",
]


def _smoke_texts(n):  # noqa: ANN001
    """n DISTINCT short documents, each a fixed ordinal progression over a
    unique subject. Distinct subjects -> every sentence differs across docs
    (no identical-twin targets), and the shared step order gives a learnable
    succession the smoke can retrieve above chance. Replaces the old
    2-unique-texts x15 corpus that guaranteed chance recall."""
    steps = [
        "The {s} begins the task at dawn.",
        "Next the {s} gathers the parts.",
        "Then the {s} inspects each piece.",
        "After that the {s} assembles them carefully.",
        "Soon the {s} tightens every joint.",
        "Midway the {s} calibrates the machine.",
        "Later the {s} tests the result.",
        "Then the {s} records the numbers.",
        "Near the end the {s} cleans the bench.",
        "Finally the {s} reports it done.",
        "The {s} rests until morning.",
        "At last the {s} locks the door.",
    ]
    subjects = [
        _SMOKE_SUBJECTS[i % len(_SMOKE_SUBJECTS)]
        + (f" {i // len(_SMOKE_SUBJECTS)}" if i >= len(_SMOKE_SUBJECTS) else "")
        for i in range(n)
    ]
    return [" ".join(step.format(s=s) for step in steps) for s in subjects]


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


def _split_units(text, unit):  # noqa: ANN001
    """A document -> its ordered thought units. unit='sentence' = split_sentences
    (web prose, and long math solutions where sentence granularity matches the
    encoder's training distribution); unit='line' = GSM8K-style step splitting
    (one step per line, calc annotations stripped, answer line dropped). The
    unit is the only thing that changes between corpora; the encode/window/train
    machinery is shared (Fable steer: no runner fork)."""
    if unit == "line":
        from marker.reason_check import split_solution_steps  # noqa: PLC0415

        return split_solution_steps(text)
    from marker.gist import split_sentences  # noqa: PLC0415

    return split_sentences(text)


def _doc_sentence_spans(tok, text, max_span, max_sents, unit="sentence"):  # noqa: ANN001
    units = _split_units(text, unit)[:max_sents]
    spans = [tok(s, add_special_tokens=False).input_ids[:max_span] for s in units]
    return [s for s in spans if s]


@torch.no_grad()
def encode_corpus(
    pm,
    gist,
    tok,
    docs_text,
    max_span,
    max_sents,
    min_sents,
    device,
    unit="sentence",
    questions=None,
):  # noqa: ANN001
    """Each document -> a gist sequence [n_sents, k, hidden] (encode every
    unit). Returns a list of per-document sequences (>= min_sents long).
    With `questions` (v2): each kept doc's QUESTION is encoded as one extra
    thought and prepended as row 0 (the _windows_q convention)."""
    seqs = []
    for i, text in enumerate(docs_text):
        spans = _doc_sentence_spans(tok, text, max_span, max_sents, unit)
        if len(spans) < min_sents:
            continue
        # encode sentence-by-sentence (variable span lengths -> one at a time)
        slots = [encode_gist(pm, gist, [sp]).float()[0] for sp in spans]  # each [k, hidden]
        if questions is not None:
            q_ids = tok(questions[i], add_special_tokens=False).input_ids[: max_span * 2]
            if len(q_ids) < 2:
                continue
            slots.insert(0, encode_gist(pm, gist, [q_ids]).float()[0])
        seqs.append(torch.stack(slots).cpu())  # [(1+)n_sents, k, hidden]
        if (i + 1) % 50 == 0:
            print(f"    ...encoded {i + 1} docs, {len(seqs)} kept", flush=True)
    return seqs


def _windows(seq, length, stride=None):  # noqa: ANN001
    """Length-L windows of a [n_sents, k, d] sequence.

    Default stride = length (NON-overlapping). Overlapping (stride < length)
    re-emits each interior sentence as a next-thought target in multiple
    windows, filling the retrieval pool with identical-content duplicates —
    which deflates recall@k (exact-index match) toward chance and injects
    false negatives into InfoNCE. Non-overlapping keeps every target distinct.
    """
    if stride is None:
        stride = length
    return [seq[i : i + length] for i in range(0, len(seq) - length + 1, stride)]


def _windows_q(seq, length, stride=None):  # noqa: ANN001
    """Question-conditioned windows: row 0 of `seq` is the QUESTION's thought
    (v2 convention), rows 1.. are the steps. Every window carries the question
    as its own row 0: [q, s_i..s_{i+L-1}], step rows non-overlapping (same
    duplicate-target reasoning as _windows). The v1 predictor never saw the
    problem it was predicting steps for — this is the fix."""
    import torch as _t  # noqa: PLC0415

    if stride is None:
        stride = length
    return [
        _t.cat([seq[0:1], seq[i : i + length]]) for i in range(1, len(seq) - length + 1, stride)
    ]


def qwin_slices(out, wz):  # noqa: ANN001
    """Align predictions/targets for a question-carrying window. Window rows
    [q, s_i.., s_j]; model output index m = readout at row m predicting row m+1.
    Index 0 (question -> first step of the window) is ill-posed for mid-doc
    windows (it asks for step i from the question alone), so keep outputs 1..
    predicting rows 2.. Returns (pred_use, tgt_use), both [B, L-1, k, d]."""
    return out[:, 1:], wz[:, 2:]


def _batches(seqs, length, batch, whitener, seed=0, with_q=False):  # noqa: ANN001
    """Shuffled whitened training batches. `seed` should vary per epoch so the
    batch composition (= the InfoNCE negative sets) varies across epochs; a
    local Generator keeps the global RNG stream untouched (a fixed
    manual_seed(0) here froze both the order AND the dropout masks)."""
    winf = _windows_q if with_q else _windows
    wins = [w for s in seqs for w in winf(s, length)]
    g = torch.Generator()
    g.manual_seed(seed)
    order = torch.randperm(len(wins), generator=g)
    for i in range(0, len(wins) - batch + 1, batch):
        idx = order[i : i + batch]
        stack = torch.stack([wins[j] for j in idx])  # [B, L, k, d]
        b, ln, k, d = stack.shape
        wz = whitener.transform(stack.reshape(b * ln, k, d)).reshape(b, ln, k, d)
        yield wz


def _dedup_keep_idx(t):  # noqa: ANN001
    """Indices of first occurrences of each distinct target row (bitwise).
    Web text repeats boilerplate sentences across documents; the deterministic
    encode makes their gists identical, and twin targets in the retrieval pool
    fake false negatives — the same deflation the non-overlapping-window fix
    removed within-document."""
    _, inv = torch.unique(t, dim=0, return_inverse=True)
    seen: set[int] = set()
    keep = []
    for i, grp in enumerate(inv.tolist()):
        if grp not in seen:
            seen.add(grp)
            keep.append(i)
    return torch.tensor(keep, device=t.device)


def _dedup_pairs(p, t):  # noqa: ANN001
    """(p, t) filtered to first-occurrence targets + n_dropped. See _dedup_keep_idx."""
    idx = _dedup_keep_idx(t)
    return p[idx], t[idx], t.shape[0] - idx.shape[0]


def _recall_within_doc(p, t, doc, topk=5):  # noqa: ANN001
    """Recall@k with the candidate pool restricted to SAME-DOCUMENT targets —
    the topic-shortcut control (Fable result review). Global-pool recall is
    inflated by 'found the right document' (doc topic = ~2/3 of predictive
    value, neighbor=0.632); within-doc, topic is shared by construction, so
    beating within-doc chance (topk/pool) is pure succession signal. Returns
    {'recall', 'pool'} (pool = mean same-doc candidates)."""
    pn = torch.nn.functional.normalize(p, dim=-1)
    tn = torch.nn.functional.normalize(t, dim=-1)
    sims = pn @ tn.T  # [N, N]
    same = doc.unsqueeze(0) == doc.unsqueeze(1)
    sims = sims.masked_fill(~same, float("-inf"))
    n = sims.shape[0]
    top = sims.topk(min(topk, n), dim=-1).indices
    hits = (top == torch.arange(n, device=sims.device).unsqueeze(1)).any(-1)
    pools = same.float().sum(1)
    return {
        "recall": float(hits.float().mean()),
        "pool": float(pools.mean()),
        # blind-guess baseline computed the RIGHT way: average each row's own
        # 1/pool (E[1/n]), not 1/average-pool (1/E[n]) — with mixed pool sizes
        # the latter understates chance (Jensen) and flatters the result.
        "chance": float((topk / pools).clamp(max=1.0).mean()),
    }


def _recall_subsampled(p, t, topk=5, pool=128, seed=0):  # noqa: ANN001
    """Recall@k against a fixed-size pool: the true target + (pool-1) seeded
    random decoys. THIS is the number comparable to the pre-registered gate
    (recall@5 > 0.40 @ 128) — full-pool recall at N~2000 is a ~15x harder task
    and understates gate performance. Falls back to the full pool if N <= pool."""
    from marker.predictor import recall_at_k  # noqa: PLC0415

    n = p.shape[0]
    if n <= pool:
        return recall_at_k(p, t, topk)
    pn = torch.nn.functional.normalize(p, dim=-1)
    tn = torch.nn.functional.normalize(t, dim=-1)
    sims = pn @ tn.T  # [N, N]
    true = sims.diag()
    # seeded decoy sampling stays on CPU (a torch.Generator is CPU-only, so
    # torch.rand(generator=) MUST be CPU) — then move the indices to sims'
    # device before gather, or GPU eval dies with a cpu/cuda mismatch. This
    # bug is invisible to CPU tests (same trap as to_leaf_param).
    g = torch.Generator()
    g.manual_seed(seed)
    r = torch.rand(n, n, generator=g)
    r.fill_diagonal_(2.0)  # the true target is never its own decoy
    decoy_idx = r.argsort(dim=1)[:, : pool - 1].to(sims.device)
    decoy_sims = sims.gather(1, decoy_idx)
    beaten_by = (decoy_sims > true.unsqueeze(1)).sum(1)
    return float((beaten_by <= topk - 1).float().mean())


def _eval_smoke(device):  # noqa: ANN001
    """Exercise the FULL evaluate() path on `device` with a throwaway tiny
    predictor BEFORE the expensive encode. GPU-only device bugs in the eval
    metrics are invisible to CPU tests (twice now: to_leaf_param, the
    _recall_subsampled cpu/cuda gather) — this crashes them in minute 1 for
    ~$0.01 instead of after 90 min of encode+train. Pool is sized >128 so the
    subsampled-decoy branch runs too. GRAD_OK philosophy: fail loudly, early."""
    k, d = 2, 8
    m = NextThoughtPredictor(d=d, k=k, d_model=16, layers=1, heads=2).to(device)
    seqs = [torch.randn(140, k, d) for _ in range(2)]  # 210 pairs > 128
    ev = evaluate(m, seqs, 4, IdentityWhitener(), device)
    assert "recall@5_128" in ev and ev["pool"] > 128, f"eval smoke incomplete: {ev}"
    print(f"EVAL_SMOKE_OK {ev}", flush=True)


@torch.no_grad()
def evaluate(model, seqs, length, whitener, device, with_q=False):  # noqa: ANN001
    """Retrieval eval, three pools: recall@5 over ALL eval targets (hardest,
    context), recall@5_128 over true+127 seeded decoys (THE gate-comparable
    number), recall@5_doc over same-doc targets only (the topic-shortcut
    control — succession signal, not document identification). tgt_sim = mean
    pairwise target similarity, the platitude-gate reference for diversity.
    Runs in eval mode (no_grad does NOT disable dropout), restores the caller's
    mode; duplicate targets (cross-doc boilerplate) deduped before recall.
    with_q (v2): windows carry the question as row 0; the ill-posed q->step
    output is dropped (qwin_slices), so pairs-per-window stays length-1."""
    winf = _windows_q if with_q else _windows
    wins, win_doc = [], []
    for di, s in enumerate(seqs):
        for w in winf(s, length):
            wins.append(w)
            win_doc.append(di)
    if not wins:
        return {}
    was_training = model.training
    model.eval()
    preds, tgts = [], []
    for i in range(0, len(wins), 64):
        stack = torch.stack(wins[i : i + 64])
        b, ln, k, d = stack.shape
        wz = whitener.transform(stack.reshape(b * ln, k, d)).reshape(b, ln, k, d).to(device)
        pred = model(wz)
        pu, tu = qwin_slices(pred, wz) if with_q else (pred, wz[:, 1:])
        preds.append(model.pool(pu).reshape(-1, model.pool_proj.out_features))
        tgts.append(model.pool(tu).reshape(-1, model.pool_proj.out_features))
    model.train(was_training)
    p, t = torch.cat(preds), torch.cat(tgts)
    # each window contributes length-1 (pred, target) pairs, all from its doc
    # (with_q: length+1 rows -> length outputs -> length-1 kept after the drop)
    doc = torch.tensor(win_doc, device=t.device).repeat_interleave(length - 1)
    keep = _dedup_keep_idx(t)
    dropped = t.shape[0] - keep.shape[0]
    p, t, doc = p[keep], t[keep], doc[keep]
    within1 = _recall_within_doc(p, t, doc, topk=1)
    within5 = _recall_within_doc(p, t, doc, topk=5)
    return {
        "recall@1": round(recall_at_k(p, t, 1), 3),
        "recall@5": round(recall_at_k(p, t, 5), 3),
        "recall@5_128": round(_recall_subsampled(p, t, topk=5, pool=128), 3),
        # recall@1_doc is the length-robust topic-shortcut control (chance =
        # 1/doc_pool); recall@5_doc saturates to 1.0 when doc_pool <= 5 (short
        # reasoning traces), so it only discriminates for long docs.
        "recall@1_doc": round(within1["recall"], 3),
        "doc_chance": round(within1["chance"], 3),
        "recall@5_doc": round(within5["recall"], 3),
        "doc_pool": round(within5["pool"], 1),
        "diversity": round(prediction_diversity(p), 3),
        "tgt_sim": round(prediction_diversity(t), 3),
        "pool": p.shape[0],
        "dup_dropped": dropped,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--repo", default=None)
    ap.add_argument("--out-repo", default=None, help="HF repo to push predictor artifacts")
    ap.add_argument(
        "--out-subdir",
        default="stage2_predictor",
        help="path_in_repo for artifacts — distinct per parallel run so they don't clobber",
    )
    ap.add_argument("--n-docs", type=int, default=4000)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--corpus", choices=["web", "cot"], default="web")
    ap.add_argument("--dataset", default=None, help="cot: HF dataset (default openai/gsm8k)")
    ap.add_argument("--dataset-config", default=None, help="cot: HF config (default main)")
    ap.add_argument(
        "--text-field", default=None, help="cot: row field with the trace (default answer)"
    )
    ap.add_argument(
        "--unit",
        choices=["sentence", "line"],
        default=None,
        help="thought unit: sentence (web; long math solutions) or line (GSM8K "
        "step-per-line). Default: web->sentence, cot->line.",
    )
    ap.add_argument(
        "--eval-every",
        type=int,
        default=250,
        help="finer than 500 (validation overfit peaked at 500)",
    )
    ap.add_argument("--d-model", type=int, default=640)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument(
        "--with-question",
        action="store_true",
        help="v2: prepend the problem's own thought as row 0 of every window — "
        "v1 predicted steps without ever seeing the question",
    )
    ap.add_argument(
        "--hard-neg",
        type=float,
        default=0.0,
        help="v2: weight of the within-window hard-negative InfoNCE (same-doc "
        "sibling steps — the candidates recall@1_doc actually asks it to beat)",
    )
    ap.add_argument(
        "--whiten",
        choices=["off", "shrunk", "zca"],
        default="off",
        help="gist-space whitening: measured on smoke, raw recall@5=1.0 vs "
        "zca=0.3 (ZCA equalizes ~800 near-noise dims with the signal dims and "
        "amplifies the worst-estimated directions) — OFF until the real corpus "
        "shows anisotropy actually hurts; shrunk (0.1 toward spherical) is the "
        "bounded middle ground",
    )
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    unit = args.unit or ("line" if args.corpus == "cot" else "sentence")
    if args.corpus == "cot":
        # reasoning traces: shorter units-per-doc, steps can run longer than a
        # sentence. min_sents = window+1 so every kept trace yields >=1 window.
        max_span, max_sents, min_sents = 96, 16, args.window + 1
    else:
        max_span, max_sents, min_sents = (
            64,
            24,
            args.window + 1,
        )  # 64 = stage-1 training cap; 48 tripled truncation (9.1% vs 3.4%)
    if args.smoke:
        args.model_name, args.repo, args.n_docs, args.steps = "Qwen/Qwen2.5-0.5B", None, 40, 300
        args.window = 4 if args.corpus == "web" else 3
        max_span, max_sents, min_sents = 24, 12, args.window + 1
        print(f"=== SMOKE (tiny model, corpus={args.corpus} unit={unit}) ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _eval_smoke(device)  # crash device bugs in the eval path NOW, not post-encode
    quantize = device == "cuda" and not args.smoke
    pm, gist, tok = _load_stage1(args.model_name, args.repo, device, quantize)

    # ── encode corpus into gist sequences ────────────────────────────────────
    questions = None
    if args.with_question:
        if args.corpus != "cot":
            raise SystemExit("--with-question needs a cot dataset (question/solution pairs)")
        if args.smoke:
            sols = _smoke_cot_texts(args.n_docs)
            qa = [
                (f"Apply the operations to the starting items ({i}).", s)
                for i, s in enumerate(sols)
            ]
        else:
            qa = list(_doc_texts_qa(args.n_docs, args.dataset, args.dataset_config))
        questions = [q for q, _ in qa]
        docs_text = [s for _, s in qa]
    elif args.smoke:
        docs_text = (
            _smoke_cot_texts(args.n_docs) if args.corpus == "cot" else _smoke_texts(args.n_docs)
        )
    else:
        docs_text = list(
            _doc_texts(args.n_docs, args.corpus, args.dataset, args.dataset_config, args.text_field)
        )
    seqs = encode_corpus(
        pm,
        gist,
        tok,
        docs_text[: args.n_docs],
        max_span,
        max_sents,
        min_sents,
        device,
        unit,
        questions=questions,
    )
    print(f"encoded {len(seqs)} gist sequences", flush=True)
    # document-disjoint split
    n_eval = max(1, len(seqs) // 10)
    eval_seqs, train_seqs = seqs[:n_eval], seqs[n_eval:]

    # ── whitening (opt-in; fit per-slot on TRAIN gists only) ─────────────────
    k, hidden = train_seqs[0].shape[1], train_seqs[0].shape[2]
    if args.whiten == "off":
        whitener = IdentityWhitener()
        print("whitening OFF (raw gist space)", flush=True)
    else:
        shrink = 0.1 if args.whiten == "shrunk" else 0.0
        flat = torch.cat([s.reshape(-1, k, hidden) for s in train_seqs])  # [N, k, hidden]
        whitener = PerSlotWhitener.fit_streaming(iter(flat.split(4096)), k=k, shrink=shrink)
        print(
            f"fit {k} per-slot whiteners ({args.whiten}) on {flat.shape[0]} train gists",
            flush=True,
        )

    # ── train the predictor ──────────────────────────────────────────────────
    model = NextThoughtPredictor(
        d=hidden,
        k=k,
        d_model=256 if args.smoke else args.d_model,
        layers=2 if args.smoke else args.layers,
        heads=4 if args.smoke else 8,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    step, epoch = 0, 0
    best, best_state = {}, None
    while step < args.steps:
        for wz in _batches(
            train_seqs,
            args.window,
            8 if args.smoke else 64,
            whitener,
            seed=epoch,
            with_q=args.with_question,
        ):
            wz = wz.to(device)
            out = model(wz)
            pred, tgt = qwin_slices(out, wz) if args.with_question else (out, wz[:, 1:])
            pp = model.pool(pred).reshape(-1, model.pool_proj.out_features)
            tp = model.pool(tgt).reshape(-1, model.pool_proj.out_features)
            loss = 0.1 * regression_loss(pred, tgt) + info_nce_loss(pp, tp)
            if args.hard_neg > 0:
                # same-window sibling steps as isolated hard negatives
                ppw = model.pool(pred)  # [B, L-1, d_model]
                tpw = model.pool(tgt)
                loss = loss + args.hard_neg * info_nce_within(ppw, tpw)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % (20 if args.smoke else args.eval_every) == 0:
                ev = evaluate(
                    model, eval_seqs, args.window, whitener, device, with_q=args.with_question
                )
                print(f"[step {step}] loss {loss.item():.4f}  eval {ev}", flush=True)
                # gate reads the BEST eval checkpoint (pre-registered before
                # launch). Keyed on recall@1_doc — the within-doc succession
                # metric. recall@5_128 anti-correlates with it across training
                # (topic-matching saturates it early: A's 0.901 was at the
                # checkpoint WORST at succession), so selecting on it ships a
                # topic-matcher. recall@5_128 breaks ties.
                key = (ev.get("recall@1_doc", -1.0), ev.get("recall@5_128", -1.0))
                if key > (best.get("recall@1_doc", -1.0), best.get("recall@5_128", -1.0)):
                    best = {**ev, "step": step}
                    best_state = {
                        n: v.detach().cpu().clone() for n, v in model.state_dict().items()
                    }
            if step >= args.steps:
                break
        epoch += 1

    ev = evaluate(model, eval_seqs, args.window, whitener, device, with_q=args.with_question)
    fkey = (ev.get("recall@1_doc", -1.0), ev.get("recall@5_128", -1.0))
    if fkey > (best.get("recall@1_doc", -1.0), best.get("recall@5_128", -1.0)):
        best = {**ev, "step": step}
        best_state = None  # final model IS the best
    print(f"[FINAL] {ev}", flush=True)
    print(f"[BEST]  {best}", flush=True)
    print(
        f"GATE (on BEST): within-doc succession recall@1_doc ({best.get('recall@1_doc')}) "
        f"must beat its empirical blind-guess baseline doc_chance ({best.get('doc_chance')}) "
        f"— THE load-bearing test. recall@5_128 ({best.get('recall@5_128')}) > 0.40 is "
        "necessary but topic-matching alone saturates it at small doc_pool. "
        f"Platitude: diversity < tgt_sim ({best.get('tgt_sim')})."
    )

    if args.out_repo:
        if best_state is not None:
            model.load_state_dict(best_state)  # push the gated checkpoint
        _push_artifacts(
            model,
            whitener,
            args.out_repo,
            {
                "best": best,
                "final": ev,
                "whiten": args.whiten,
                "corpus": args.corpus,
                "unit": unit,
                # provenance (Fable held-out review: overlap must be checkable)
                "dataset": args.dataset,
                "n_docs": args.n_docs,
                "window": args.window,
                "with_question": args.with_question,
                "hard_neg": args.hard_neg,
                "d_model": args.d_model,
                "layers": args.layers,
            },
            args.out_subdir,
        )


def _doc_texts_qa(n, dataset=None, config=None, qfield=None, afield=None):  # noqa: ANN001
    """Stream n (question, solution) PAIRS from a cot dataset — the v2
    question-conditioning path. Defaults: gsm8k question/answer; other datasets
    problem/solution (OpenR1 layout)."""
    from datasets import load_dataset  # noqa: PLC0415

    name = dataset or "openai/gsm8k"
    cfg = config if config is not None else ("main" if name == "openai/gsm8k" else None)
    ds = load_dataset(name, cfg, split="train", streaming=True)
    qf = qfield or ("question" if name == "openai/gsm8k" else "problem")
    af = afield or ("answer" if name == "openai/gsm8k" else "solution")
    for seen, row in enumerate(ds):
        yield (row.get(qf) or "", row.get(af) or "")
        if seen + 1 >= n:
            break


def _doc_texts(n, corpus="web", dataset=None, config=None, field=None):  # noqa: ANN001
    """Stream n documents. web = FineWeb-Edu prose (field 'text'); cot =
    reasoning traces (default openai/gsm8k 'main' split 'train', field
    'answer' — the step-delimited solution)."""
    from datasets import load_dataset  # noqa: PLC0415

    if corpus == "cot":
        name = dataset or "openai/gsm8k"
        # config default 'main' is GSM8K-specific; other datasets take None
        cfg = config if config is not None else ("main" if name == "openai/gsm8k" else None)
        ds = load_dataset(name, cfg, split="train", streaming=True)
        field = field or ("answer" if name == "openai/gsm8k" else "solution")
    else:
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True
        )
        field = field or "text"
    for seen, row in enumerate(ds):
        yield row.get(field) or ""
        if seen + 1 >= n:
            break


_COT_TEMPLATES = [
    "Start with {a} items.\nAdd {b} more to get {a}+{b} = <<{a}+{b}={c}>>{c}.\n"
    "Double it: {c}*2 = <<{c}*2={d}>>{d}.\nRemove {b}: {d}-{b} = <<{d}-{b}={e}>>{e}.\n"
    "Split in half: {e}/2 = <<{e}/2={f}>>{f}.\n#### {f}",
]


def _smoke_cot_texts(n):  # noqa: ANN001
    """n DISTINCT synthetic GSM8K-style traces (multi-step, deterministic
    arithmetic succession, calc annotations + #### answer). Proves the cot
    split/encode/window path without a network dataset."""
    out = []
    for i in range(n):
        a, b = 3 + i, 2 + (i % 9)  # a strictly increasing -> every trace distinct
        c = a + b
        d = c * 2
        e = d - b
        f = e // 2
        out.append(_COT_TEMPLATES[0].format(a=a, b=b, c=c, d=d, e=e, f=f))
    return out


def _push_artifacts(model, whitener, repo, manifest, subdir="stage2_predictor"):  # noqa: ANN001
    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    from huggingface_hub import upload_folder  # noqa: PLC0415

    d = Path("/tmp/stage2_out")  # noqa: S108
    d.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), d / "predictor.pt")
    whitener.save(d / "whiteners.pt")
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
    upload_folder(repo_id=repo, folder_path=str(d), path_in_repo=subdir)
    print(f"pushed predictor artifacts to {repo}/{subdir}", flush=True)


if __name__ == "__main__":
    import os
    import sys

    main()
    sys.stdout.flush()
    os._exit(0)
