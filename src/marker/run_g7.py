"""G7 training run -- Stage 3 (see GIST_LM_PLAN.md "STAGES 2+3 DESIGN v3" and
"STOP THE LINE" gates). Trains the `g7` LoRA adapter + GistVocab on top of a
CLEAN frozen 4-bit base (no Stage-1 gist/render adapters -- G7 reads the
question as plain text and THINKS by emitting gist tokens, see gist_vocab.py)
to predict, autoregressively, each step's 8-per-slot dictionary ids, then
render the committed step's text.

Pipeline: load corpus (JSONL shards, schema owned by run_tokenize_corpus.py)
-> per epoch, two training sequences per solution (g7.sample_m) -> pack by
length, batch, two-gather loss (g7.two_gather_loss, never a full
base_vocab+8K+2-wide softmax) -> log running gist_ce/text_ce every 50 steps
-> checkpoint + push every N steps and at the end -> eval block 1
(teacher-forced next-id accuracy/exact-8 + majority/bigram/copy baselines)
-> eval block 2 (op-from-predicted-IDs via greedy history decode + the
stage-1 categorical NB, k_sizes=[K]*8 EXPLICIT -- never derived from the max
id seen in a possibly-small fit set) -> manifest ([G7 MANIFEST] {json}) ->
(non-smoke) push.

Smoke (offline, tiny UNTIED model, synthetic corpus, no network):
    PYTHONPATH=src python -m marker.run_g7 --smoke
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from marker.g7 import build_sequence, sample_m, two_gather_loss
from marker.gist_dict import fit_categorical_nb, predict_categorical_nb
from marker.gist_vocab import N_SLOTS, SLOT_ORDER, attach_g7
from marker.summaryprobe import OP_CLASSES, encode_labels, majority_rate, op_label

BASE_VOCAB = 152064  # Qwen2.5-7B config.vocab_size -- a DEFAULT/doc value only;
K = 4096  # every actual use reads base_vocab/K from the loaded model/dict.


# ── smoke: offline tokenizer + synthetic corpus (no network, no real weights) ──


class _SmokeTok:
    """A tiny deterministic whitespace tokenizer for the --smoke path: real
    Qwen tokenizers need a network fetch, and the smoke model's vocabulary is
    random anyway (tie_word_embeddings=False, untied random Qwen2 config) --
    only a stable word->id map is needed to exercise build_sequence."""

    def __init__(self, base_vocab: int) -> None:
        self.base_vocab = base_vocab
        self.eos_token_id = base_vocab - 1

    def __call__(self, text: str, add_special_tokens: bool = False):  # noqa: ARG002
        ids = [1 + (hash(w) % (self.base_vocab - 2)) for w in text.split()]
        return type("Enc", (), {"input_ids": ids})()


def smoke_records(n: int = 30, K: int = 6, seed: int = 0) -> list[dict]:
    """30 synthetic solutions matching the stage-2 JSONL schema: random ids
    (K small -- ALWAYS 8 slots, 2 slots is NOT a supported config), each
    step's text a real, parseable arithmetic relation (`a OP b = c`) so
    summaryprobe.op_label can score it, exactly as it would a real GSM8K
    step."""
    rng = random.Random(seed)
    ops = list(OP_CLASSES)
    records = []
    for i in range(n):
        n_steps = rng.randint(2, 4)
        steps, ids = [], []
        for s in range(n_steps):
            a, b = rng.randint(1, 20), rng.randint(1, 20)
            op = ops[rng.randrange(len(ops))]
            steps.append(f"step {s}: {a} {op} {b} = {a}")
            ids.append([[rng.randrange(K) for _ in range(N_SLOTS)]])
        records.append(
            {
                "src": "smoke",
                "doc_id": f"smoke_{i}",
                "question": f"question number {i} value {rng.randint(1, 99)}",
                "steps": steps,
                "ids": ids,
                "answer": "0",
                "n_groups": n_steps,
            }
        )
    return records


def _load_smoke_model(base_vocab: int, K: int):
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(0)
    cfg = AutoConfig.for_model(
        "qwen2",
        vocab_size=base_vocab,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        tie_word_embeddings=False,
    )
    if cfg.tie_word_embeddings:
        raise ValueError("smoke config unexpectedly tied embeddings")
    base = AutoModelForCausalLM.from_config(cfg, attn_implementation="sdpa").eval()
    mu = torch.randn(N_SLOTS, K, 32)
    pm, vocab = attach_g7(base, mu, r_lora=4, r_slot=4, r_id=3)
    return pm, vocab, _SmokeTok(base_vocab)


# ── real path: clean 4-bit base + dictionary mu (never touches network in tests) ──


def _load_base_clean(model_name: str, device: str, quantize: bool):
    """The frozen G7 base: NO Stage-1 gist/render adapters (run_stage2's
    `_load_stage1` attaches those; G7 never does -- see module docstring).
    Same 4-bit nf4 + bf16-compute pattern as run_stage2._load_stage1."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
    return base, tok


def load_mu_from_dict(path: str, K: int) -> torch.Tensor:
    """dict_kv_K4096.pt is a list of N_SLOTS per-slot dictionary entries
    (gist_dict.build_dict_kv's output), each carrying `mu_readout` fp16
    [K, d]. Stacks them into [8, K, d] fp32, NATURAL slot order (list index
    == slot index, matching how the dictionary was built one slot at a
    time)."""
    entries = torch.load(path, map_location="cpu")
    if len(entries) != N_SLOTS:
        raise ValueError(f"dictionary must have {N_SLOTS} per-slot entries, got {len(entries)}")
    mus = []
    for s, e in enumerate(entries):
        mu_s = e["mu_readout"].float()
        if mu_s.shape[0] != K:
            raise ValueError(f"slot {s}: dictionary K={mu_s.shape[0]} != requested K={K}")
        mus.append(mu_s)
    return torch.stack(mus, dim=0)


# ── batching ─────────────────────────────────────────────────────────────


def build_epoch_sequences(records: list[dict], tok, base_vocab: int, K: int, rng: random.Random):
    """Two build_sequence calls per solution (g7.sample_m)."""
    out = []
    for rec in records:
        n = len(rec["steps"])
        for m in sample_m(n, rng):
            out.append(build_sequence(rec, m, tok, base_vocab, K))
    return out


def pack_batches(seqs: list[dict], batch_size: int, seq_cap: int, pad_id: int = 0):
    """Sort-by-length then chunk (a simple pack-by-length, not a bin-packer):
    minimizes wasted padding within a batch while keeping the code small.
    Sequences longer than seq_cap are DROPPED (never silently truncated --
    truncating mid-sequence could cut a <think>/<commit> layout in half),
    counted in the returned `n_dropped`. Returns (batches, n_dropped);
    each batch is dict(input_ids [B,T] long, gist_mask/text_mask [B,T] bool,
    slot_of_pos [B,T] long, attention_mask [B,T] long)."""
    kept = [s for s in seqs if len(s["input_ids"]) <= seq_cap]
    n_dropped = len(seqs) - len(kept)
    kept.sort(key=lambda s: len(s["input_ids"]))
    batches = []
    for lo in range(0, len(kept), batch_size):
        chunk = kept[lo : lo + batch_size]
        max_t = max(len(s["input_ids"]) for s in chunk)
        input_ids, gist_mask, text_mask, slot_of_pos, attn = [], [], [], [], []
        for s in chunk:
            t = len(s["input_ids"])
            pad = max_t - t
            input_ids.append(s["input_ids"] + [pad_id] * pad)
            gist_mask.append(s["gist_mask"] + [False] * pad)
            text_mask.append(s["text_mask"] + [False] * pad)
            slot_of_pos.append(s["slot_of_pos"] + [-1] * pad)
            attn.append([1] * t + [0] * pad)
        batches.append(
            {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "gist_mask": torch.tensor(gist_mask, dtype=torch.bool),
                "text_mask": torch.tensor(text_mask, dtype=torch.bool),
                "slot_of_pos": torch.tensor(slot_of_pos, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
            }
        )
    return batches, n_dropped


def train_step(pm, head, batch, opt) -> tuple[float, float, float]:  # noqa: ANN001
    """One forward/backward/step. Shifts input_ids by one for next-token
    scoring: hidden[i] (from token i) predicts targets[i] = input_ids[i+1].
    Reads hidden states straight off the DECODER (pm.get_decoder()), never
    through the head wrapper -- computing full (base_vocab+8K+2)-wide logits
    for every position, most of which two_gather_loss would discard, is
    exactly the "never a full-width CE" cost the design forbids."""
    device = next(pm.parameters()).device
    ids = batch["input_ids"].to(device)
    attn = batch["attention_mask"].to(device)
    out = pm.get_decoder()(input_ids=ids, attention_mask=attn, use_cache=False)
    hidden = out.last_hidden_state[:, :-1].reshape(-1, out.last_hidden_state.shape[-1])
    targets = ids[:, 1:].reshape(-1)
    gist_mask = batch["gist_mask"][:, :-1].reshape(-1)
    text_mask = batch["text_mask"][:, :-1].reshape(-1)
    slot_of_pos = batch["slot_of_pos"][:, 1:].reshape(-1)  # slot of the TARGET, not the source

    loss, gist_ce, text_ce = two_gather_loss(
        head, hidden, targets, gist_mask, text_mask, slot_of_pos
    )
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
    opt.step()
    return float(loss.detach()), float(gist_ce.detach()), float(text_ce.detach())


# ── eval block 1: teacher-forced next-id accuracy/exact-8 + baselines ──────


def _doc_groups(records: list[dict]) -> list[list[list[int]]]:
    """Per-doc sequence of NATURAL-order 8-id groups, flattened across
    steps -- the unit both the bigram baseline and majority baseline fit
    on/predict over."""
    return [[g for step_groups in r["ids"] for g in step_groups] for r in records]


def majority_id_per_slot(groups: list[list[list[int]]], K: int) -> list[int]:
    counts = torch.zeros(N_SLOTS, K, dtype=torch.long)
    for doc in groups:
        for g in doc:
            for s in range(N_SLOTS):
                counts[s, g[s]] += 1
    return counts.argmax(dim=1).tolist()


def fit_bigram(groups: list[list[list[int]]], K: int) -> torch.Tensor:
    """Per-slot P(id_{n+1} | id_n) over consecutive groups WITHIN a doc
    (never across doc boundaries), Laplace alpha=1. [8, K, K]."""
    counts = torch.ones(N_SLOTS, K, K)
    for doc in groups:
        for a, b in zip(doc, doc[1:], strict=False):
            for s in range(N_SLOTS):
                counts[s, a[s], b[s]] += 1
    return counts / counts.sum(dim=2, keepdim=True)


def bigram_predict(probs: torch.Tensor, prev_group: list[int]) -> list[int]:
    return [int(probs[s, prev_group[s]].argmax()) for s in range(N_SLOTS)]


def bigram_ce(probs: torch.Tensor, groups: list[list[list[int]]]) -> float:
    losses = []
    for doc in groups:
        for a, b in zip(doc, doc[1:], strict=False):
            for s in range(N_SLOTS):
                losses.append(-torch.log(probs[s, a[s], b[s]].clamp_min(1e-12)))
    if not losses:
        raise ValueError("bigram_ce: no consecutive-group pairs to score")
    return float(torch.stack(losses).mean())


def next_id_accuracy(preds: list[list[int]], targets: list[list[int]]) -> float:
    if not preds:
        raise ValueError("next_id_accuracy: no (pred, target) pairs")
    hits = sum(
        p == t
        for pred, tgt in zip(preds, targets, strict=True)
        for p, t in zip(pred, tgt, strict=True)
    )
    return hits / (len(preds) * N_SLOTS)


def exact_group_rate(preds: list[list[int]], targets: list[list[int]]) -> float:
    if not preds:
        raise ValueError("exact_group_rate: no (pred, target) pairs")
    return sum(p == t for p, t in zip(preds, targets, strict=True)) / len(preds)


# ── eval block 2: op-from-predicted-IDs (stage-1 NB, k_sizes EXPLICIT) ─────


def op_labels_for_steps(steps: list[str]) -> list[str | None]:
    return [op_label(s) for s in steps]


def op_from_ids_nb(
    fit_ids: torch.Tensor, fit_ops: list[str], eval_ids: torch.Tensor, eval_ops: list[str], K: int
) -> dict:
    """op-from-predicted-IDs via the Stage-1 categorical NB
    (gist_dict.fit_categorical_nb/predict_categorical_nb), refit on THIS
    run's ids. k_sizes MUST be the dictionary's true K, [K]*8, built
    EXPLICITLY here -- NEVER derived from the max id seen in fit/eval
    (run_gist_dict.op_from_ids_nb's pattern, which is wrong for us: a small
    fit set can easily never draw id K-1 for some slot, and a derived
    k_sizes would then silently misclassify or index-error on it at eval
    time)."""
    k_sizes = [K] * N_SLOTS
    assert k_sizes == [K] * N_SLOTS, "k_sizes must be the dictionary's true K for every slot"
    y_fit = encode_labels(fit_ops)
    y_eval = encode_labels(eval_ops)
    model = fit_categorical_nb(fit_ids, y_fit, n_classes=len(OP_CLASSES), k_sizes=k_sizes)
    preds = predict_categorical_nb(model, eval_ids)
    return {
        "acc": round(float((preds == y_eval).float().mean()), 4),
        "majority": round(majority_rate(y_eval), 4),
    }


@torch.no_grad()
def teacher_forced_eval(pm, head, eval_seqs: list[dict], base_vocab: int, K: int):  # noqa: ANN001
    """Teacher-forced per-slot next-id accuracy + per-group exact-match over
    already-built sequences (g7.build_sequence output): argmax over the
    TARGET's own slot block at every gist position, regrouped into 8-wide
    NATURAL-order predictions (undoing SLOT_ORDER layout, like
    g7.parse_sequence) for the exact-group comparison. Returns
    (preds, targets), both list[list[int]] of NATURAL-order groups."""
    device = next(pm.parameters()).device
    rows, bias = head.vocab.output_rows_all()
    preds, targets = [], []
    for seq in eval_seqs:
        ids = torch.tensor([seq["input_ids"]], device=device)
        hidden = pm.get_decoder()(input_ids=ids, use_cache=False).last_hidden_state[0]
        gist_positions = [i for i, m in enumerate(seq["gist_mask"]) if m]
        for g0 in range(0, len(gist_positions) - N_SLOTS + 1, N_SLOTS):
            block_pos = gist_positions[g0 : g0 + N_SLOTS]
            pred_natural, tgt_natural = [0] * N_SLOTS, [0] * N_SLOTS
            for i in block_pos:
                s = seq["slot_of_pos"][i + 1]
                block_rows = rows[s * K : (s + 1) * K].to(hidden.dtype)
                block_bias = bias[s * K : (s + 1) * K].to(hidden.dtype)
                logit = hidden[i] @ block_rows.T + block_bias
                pred_natural[s] = int(logit.argmax())
                tgt_natural[s] = seq["input_ids"][i + 1] - base_vocab - s * K
            preds.append(pred_natural)
            targets.append(tgt_natural)
    return preds, targets


@torch.no_grad()
def greedy_predict_group(pm, head, prompt_ids: list[int], base_vocab: int, K: int) -> list[int]:  # noqa: ANN001
    """Greedy-predict one group's 8 ids (NATURAL order) given a prompt that
    already ends in the position right before the group (e.g. [...question,
    <think>, prior TRUE groups...] -- the "teacher-forced history" the build
    order specifies).

    Deliberately does NOT reuse GistLogitsProcessor/generate() here: that
    processor counts gist tokens *since <think>* in the WHOLE sequence, so a
    budget=1 group would immediately force <commit> once the prompt already
    carries prior groups (multi-step history). Simpler and correct: greedily
    decode one token at a time, masked to slot_for_position(t)'s own K-wide
    block via the same gather two_gather_loss uses -- no processor, no
    cumulative-history counting."""
    device = next(pm.parameters()).device
    vocab = head.vocab
    rows, bias = vocab.output_rows_all()
    ids = list(prompt_ids)
    natural = [0] * N_SLOTS
    for t in range(N_SLOTS):
        s = SLOT_ORDER[t]
        inp = torch.tensor([ids], device=device)
        hidden = pm.get_decoder()(input_ids=inp, use_cache=False).last_hidden_state[0, -1]
        block_rows = rows[s * K : (s + 1) * K].to(hidden.dtype)
        block_bias = bias[s * K : (s + 1) * K].to(hidden.dtype)
        j = int((hidden @ block_rows.T + block_bias).argmax())
        natural[s] = j
        ids.append(vocab.flat_id(s, j))
    return natural


def decode_text_step(logits_row: torch.Tensor, base_vocab: int) -> int:
    """Greedy-decode ONE text token from a full-width logits row, restricted
    to [:base_vocab] -- a reader/text decode must NEVER emit a
    gist/<think>/<commit> id even if the raw softmax puts more mass there
    (numerical noise, or an undertrained head at smoke scale)."""
    return int(logits_row[:base_vocab].argmax())


# ── smoke verdict ────────────────────────────────────────────────────────


def smoke_verdict(cells: dict) -> dict:
    """PASS iff train_gist_ce < 6.5 (ln(4096)=8.32; the bar is a real-run
    number ported into the smoke path per the build order -- loose by design
    at the smoke's own small K) AND train_gist_ce < bigram_ce AND
    next_id_acc > bigram_acc AND op_from_predicted > majority_op. Returns
    {"verdict": "PASS"|"FAIL", "reasons": [...]} -- reasons empty iff PASS."""
    reasons = []
    if not cells["train_gist_ce"] < 6.5:
        reasons.append(f"train_gist_ce {cells['train_gist_ce']:.3f} >= 6.5")
    if not cells["train_gist_ce"] < cells["bigram_ce"]:
        reasons.append(
            f"train_gist_ce {cells['train_gist_ce']:.3f} >= bigram_ce {cells['bigram_ce']:.3f}"
        )
    if not cells["next_id_acc"] > cells["bigram_acc"]:
        reasons.append(
            f"next_id_acc {cells['next_id_acc']:.3f} <= bigram_acc {cells['bigram_acc']:.3f}"
        )
    if not cells["op_from_predicted"] > cells["majority_op"]:
        reasons.append(
            f"op_from_predicted {cells['op_from_predicted']:.3f} <= majority_op {cells['majority_op']:.3f}"
        )
    return {"verdict": "FAIL" if reasons else "PASS", "reasons": reasons}


def _push_with_retry(repo_id, folder, path_in_repo):  # noqa: ANN001
    from marker.run_render import _push_with_retry as _impl  # noqa: PLC0415

    _impl(repo_id, folder, path_in_repo)


def _save_checkpoint(dir_path: Path, pm, vocab, meta: dict) -> None:  # noqa: ANN001
    from safetensors.torch import save_file  # noqa: PLC0415

    from marker.hf_push import write_manifest  # noqa: PLC0415

    d = Path(dir_path)
    d.mkdir(parents=True, exist_ok=True)
    pm.save_pretrained(str(d))
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in vocab.state_dict().items()},
        str(d / "gistvocab.safetensors"),
    )
    write_manifest(d, meta)


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--dict-path", default="dict_kv_K4096.pt")
    ap.add_argument("--corpus-dir", default=None)
    ap.add_argument("--eval-path", default="eval_gsm8k_test.jsonl")
    ap.add_argument(
        "--n-solutions",
        type=int,
        default=None,
        help="cap corpus records read (smoke-scale GPU runs)",
    )
    ap.add_argument("--out-repo", default=None)
    ap.add_argument("--K", type=int, default=K)
    ap.add_argument("--r-lora", type=int, default=16)
    ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--lr-vocab", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-cap", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--n-train-steps", type=int, default=None, help="cap total steps (smoke: 20)")
    ap.add_argument("--think-budget-steps", type=int, default=1)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--checkpoint-every", type=int, default=200)
    ap.add_argument("--eval-blocks", default="1,2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", default="/tmp/g7_cache")  # noqa: S108
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() and not args.smoke else "cpu"
    rng = random.Random(args.seed)
    eval_blocks = {int(x) for x in args.eval_blocks.split(",") if x}

    t0 = time.time()
    if args.smoke:
        K_run = 6
        n_train_steps = args.n_train_steps or 20
        smoke_base_vocab = 100  # small on purpose -- the smoke model's vocab is random anyway
        pm, vocab, tok = _load_smoke_model(smoke_base_vocab, K_run)
        base_vocab = vocab.base_vocab
        all_records = smoke_records(n=30, K=K_run, seed=args.seed)
        fit_records, eval_records = all_records[:20], all_records[20:]
    else:
        K_run = args.K
        n_train_steps = args.n_train_steps
        base, tok = _load_base_clean(args.model_name, device, quantize=(device == "cuda"))
        base_vocab = base.config.vocab_size
        mu = load_mu_from_dict(args.dict_path, K_run)
        pm, vocab = attach_g7(base, mu, r_lora=args.r_lora)
        if not args.corpus_dir:
            raise ValueError("--corpus-dir is required outside --smoke")
        all_records = [
            json.loads(line)
            for p in sorted(Path(args.corpus_dir).glob("*.jsonl"))
            for line in p.read_text().splitlines()
        ]
        if not all_records:
            raise ValueError(f"no records found under {args.corpus_dir}")
        if args.n_solutions is not None:
            all_records = all_records[: args.n_solutions]
        split = max(1, int(len(all_records) * 0.9))
        fit_records, eval_records = all_records[:split], all_records[split:]

    head = pm.get_output_embeddings()
    opt = torch.optim.AdamW(
        [
            {
                "params": [p for n, p in pm.named_parameters() if "lora_" in n and p.requires_grad],
                "lr": args.lr_lora,
            },
            {"params": list(vocab.parameters()), "lr": args.lr_vocab},
        ],
        weight_decay=0.01,
    )

    seqs = build_epoch_sequences(fit_records, tok, base_vocab, K_run, rng)
    for _ in range(1, args.epochs):
        seqs.extend(build_epoch_sequences(fit_records, tok, base_vocab, K_run, rng))
    batches, n_dropped = pack_batches(seqs, args.batch_size, args.seq_cap, pad_id=0)
    if not batches:
        raise ValueError(
            "no training batches after packing (corpus too small or seq_cap too tight)"
        )

    n_steps = n_train_steps or len(batches)
    running_gist, running_text, last_loss = [], [], (0.0, 0.0, 0.0)
    for step in range(n_steps):
        batch = batches[step % len(batches)]
        last_loss = train_step(pm, head, batch, opt)
        running_gist.append(last_loss[1])
        running_text.append(last_loss[2])
        if (step + 1) % args.log_every == 0 or step == n_steps - 1:
            print(
                f"[G7 STEP] step={step + 1} gist_ce={sum(running_gist) / len(running_gist):.4f} "
                f"text_ce={sum(running_text) / len(running_text):.4f}",
                flush=True,
            )
        if (step + 1) % args.checkpoint_every == 0 and not args.smoke and args.out_repo:
            ckpt_dir = Path(args.cache_dir) / f"step-{step + 1:07d}"
            _save_checkpoint(ckpt_dir, pm, vocab, {"step": step + 1})
            _push_with_retry(args.out_repo, str(ckpt_dir), f"g7_v0/step-{step + 1:07d}")

    train_gist_ce = sum(running_gist) / len(running_gist)
    train_text_ce = sum(running_text) / len(running_text)
    manifest: dict = {
        "smoke": args.smoke,
        "base_vocab": base_vocab,
        "K": K_run,
        "n_train_steps": n_steps,
        "n_dropped_sequences": n_dropped,
        "train_gist_ce": round(train_gist_ce, 4),
        "train_text_ce": round(train_text_ce, 4),
        "wall_s": round(time.time() - t0, 1),
    }

    doc_groups = _doc_groups(fit_records)
    eval_doc_groups = _doc_groups(eval_records) if eval_records else doc_groups
    eval_seqs = build_epoch_sequences(eval_records or fit_records, tok, base_vocab, K_run, rng)

    if 1 in eval_blocks:
        majority = majority_id_per_slot(doc_groups, K_run)
        bigram_probs = fit_bigram(doc_groups, K_run)
        preds_bigram, preds_copy, targets = [], [], []
        for doc in eval_doc_groups:
            for a, b in zip(doc, doc[1:], strict=False):
                preds_bigram.append(bigram_predict(bigram_probs, a))
                preds_copy.append(list(a))
                targets.append(b)
        model_preds, model_targets = teacher_forced_eval(pm, head, eval_seqs, base_vocab, K_run)
        cell1 = {
            "next_id_acc": next_id_accuracy(model_preds, model_targets) if model_preds else 0.0,
            "exact_group_rate": exact_group_rate(model_preds, model_targets)
            if model_preds
            else 0.0,
            "bigram_ce": bigram_ce(bigram_probs, doc_groups),
            "bigram_acc": next_id_accuracy(preds_bigram, targets) if targets else 0.0,
            "copy_acc": next_id_accuracy(preds_copy, targets) if targets else 0.0,
            "majority_id_per_slot": majority,
        }
        manifest["eval_block_1"] = cell1

    if 2 in eval_blocks:
        fit_ids_rows, fit_ops = [], []
        for r in fit_records:
            ops = op_labels_for_steps(r["steps"])
            for step_groups, op in zip(r["ids"], ops, strict=True):
                if op is not None:
                    fit_ids_rows.append(step_groups[0])
                    fit_ops.append(op)
        eval_ids_rows, eval_ops = [], []
        for r in eval_records or fit_records:
            ops = op_labels_for_steps(r["steps"])
            for step_groups, op in zip(r["ids"], ops, strict=True):
                if op is not None:
                    eval_ids_rows.append(step_groups[0])
                    eval_ops.append(op)
        cell2: dict = {}
        if fit_ids_rows and eval_ids_rows:
            fit_ids_t = torch.tensor(fit_ids_rows, dtype=torch.long)
            eval_ids_t = torch.tensor(eval_ids_rows, dtype=torch.long)
            nb_true = op_from_ids_nb(fit_ids_t, fit_ops, eval_ids_t, eval_ops, K_run)
            cell2["op_from_true_ids"] = nb_true

            predicted_rows = []
            for r in eval_records or fit_records:
                q_ids = list(tok(r["question"], add_special_tokens=False).input_ids)
                prompt = q_ids + [vocab.think_id]
                for step_groups in r["ids"]:
                    predicted_rows.append(greedy_predict_group(pm, head, prompt, base_vocab, K_run))
                    # teacher-forced history: extend with the TRUE ids, not the
                    # model's own prediction, per the build order's "greedy-predict
                    # each group of the true history".
                    prompt = prompt + [vocab.flat_id(s, step_groups[0][s]) for s in SLOT_ORDER]
            predicted_ids_t = torch.tensor(predicted_rows[: len(eval_ops)], dtype=torch.long)
            nb_pred = op_from_ids_nb(fit_ids_t, fit_ops, predicted_ids_t, eval_ops, K_run)
            cell2["op_from_predicted"] = nb_pred
        manifest["eval_block_2"] = cell2

    if args.smoke:
        cells = {
            "train_gist_ce": train_gist_ce,
            "bigram_ce": manifest.get("eval_block_1", {}).get("bigram_ce", float("inf")),
            "next_id_acc": manifest.get("eval_block_1", {}).get("next_id_acc", 0.0),
            "bigram_acc": manifest.get("eval_block_1", {}).get("bigram_acc", 0.0),
            "op_from_predicted": manifest.get("eval_block_2", {})
            .get("op_from_predicted", {})
            .get("acc", 0.0),
            "majority_op": manifest.get("eval_block_2", {})
            .get("op_from_predicted", {})
            .get("majority", 0.0),
        }
        manifest["smoke_verdict"] = smoke_verdict(cells)

    print(f"[G7 MANIFEST] {json.dumps(manifest, default=str)}", flush=True)

    if not args.smoke and args.out_repo:
        final_dir = Path(args.cache_dir) / "final"
        _save_checkpoint(final_dir, pm, vocab, manifest)
        _push_with_retry(args.out_repo, str(final_dir), "g7_v0")


if __name__ == "__main__":
    main()
