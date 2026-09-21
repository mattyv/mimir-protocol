"""Tests for run_g7.py: pure-logic baselines, packing, the NB k_sizes
invariant, and the --smoke end-to-end path (marked slow -- a real tiny
UNTIED model, no network)."""

from __future__ import annotations

import subprocess
import sys

import pytest
import torch

from marker.g7 import build_sequence
from marker.gist_vocab import N_SLOTS, SLOT_ORDER
from marker.run_g7 import (
    bigram_ce,
    bigram_predict,
    decode_text_step,
    exact_group_rate,
    fit_bigram,
    load_mu_from_dict,
    majority_id_per_slot,
    next_id_accuracy,
    op_from_ids_nb,
    pack_batches,
    predict_history_rows,
    slot7_confusion_split,
    smoke_records,
    smoke_verdict,
    verdict_cells,
)

K = 5


def _docs():
    # two docs, each a chain of groups over 8 slots, K=5
    return [
        [[0, 1, 2, 3, 4, 0, 1, 2], [1, 1, 2, 3, 4, 0, 1, 2], [1, 2, 2, 3, 4, 0, 1, 2]],
        [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]],
    ]


def test_majority_id_per_slot():
    m = majority_id_per_slot(_docs(), K)
    assert len(m) == N_SLOTS
    # slot 0: ids seen are 0,1,1,0,0 -> majority 0
    assert m[0] == 0
    # slot 1: ids seen are 1,1,2,0,0 -> majority 1 or 0 (tie broken by argmax, deterministic)
    assert m[1] in (0, 1)


def test_fit_bigram_and_predict_is_deterministic_and_normalized():
    probs = fit_bigram(_docs(), K)
    assert probs.shape == (N_SLOTS, K, K)
    assert torch.allclose(probs.sum(dim=2), torch.ones(N_SLOTS, K), atol=1e-5)
    pred = bigram_predict(probs, [0, 1, 2, 3, 4, 0, 1, 2])
    assert len(pred) == N_SLOTS
    assert all(0 <= p < K for p in pred)


def test_bigram_ce_positive_finite():
    probs = fit_bigram(_docs(), K)
    ce = bigram_ce(probs, _docs())
    assert ce > 0
    assert ce == ce  # not nan


def test_bigram_ce_rejects_no_pairs():
    with pytest.raises(ValueError, match="no consecutive"):
        bigram_ce(fit_bigram([[[0] * N_SLOTS]], K), [[[0] * N_SLOTS]])  # single-group doc


def test_next_id_accuracy_and_exact_group_rate():
    preds = [[0, 1, 2, 3, 4, 0, 1, 2], [9, 9, 9, 9, 9, 9, 9, 9]]
    targets = [[0, 1, 2, 3, 4, 0, 1, 2], [0, 0, 0, 0, 0, 0, 0, 0]]
    assert next_id_accuracy(preds, targets) == pytest.approx(0.5)
    assert exact_group_rate(preds, targets) == pytest.approx(0.5)


def test_next_id_accuracy_rejects_empty():
    with pytest.raises(ValueError, match="no"):
        next_id_accuracy([], [])


def test_pack_batches_drops_over_cap_and_pads_within_batch():
    tok = _FakeTok()
    rec = _rec()
    long_seq = build_sequence(rec, m=1, tok=tok, base_vocab=100, K=K)
    seqs = [long_seq, dict(long_seq)]
    seqs[1] = {**long_seq, "input_ids": long_seq["input_ids"] + [1, 2, 3]}
    batches, n_dropped = pack_batches(seqs, batch_size=2, seq_cap=len(long_seq["input_ids"]))
    assert n_dropped == 1  # the longer one exceeds the cap
    assert len(batches) == 1
    assert batches[0]["input_ids"].shape[0] == 1


def test_pack_batches_pads_to_batch_max_length():
    tok = _FakeTok()
    rec = _rec()
    short = build_sequence(rec, m=1, tok=tok, base_vocab=100, K=K)
    long = {
        **short,
        "input_ids": short["input_ids"] + [7],
        "gist_mask": short["gist_mask"] + [False],
        "text_mask": short["text_mask"] + [False],
        "slot_of_pos": short["slot_of_pos"] + [-1],
    }
    batches, n_dropped = pack_batches([short, long], batch_size=2, seq_cap=1000)
    assert n_dropped == 0
    b = batches[0]
    assert b["input_ids"].shape == (2, len(long["input_ids"]))
    assert b["attention_mask"][0].tolist() == [1] * len(short["input_ids"]) + [0]


class _FakeTok:
    eos_token_id = 999

    def __call__(self, text, add_special_tokens=False):  # noqa: ARG002
        ids = [1 + (hash(w) % 90) for w in text.split()]
        return type("Enc", (), {"input_ids": ids})()


def _rec():
    return {
        "src": "t",
        "doc_id": "d",
        "question": "a question",
        "steps": ["a step here"],
        "ids": [[[0, 1, 2, 3, 4, 0, 1, 2]]],
        "answer": "1",
        "n_groups": 1,
    }


# ── op_from_ids_nb: k_sizes assert, unseen-id smoke ────────────────────────


def test_op_from_ids_nb_scores_an_id_never_seen_in_fit():
    """The exact mechanical invariant the build order calls out: k_sizes is
    the dictionary's TRUE K ([K]*8), passed explicitly -- never derived from
    the max id seen -- so an id that never appears in the (tiny) fit set
    still gets a valid smoothed probability instead of an index error."""
    K_big = 4096
    torch.manual_seed(0)
    fit_ids = torch.randint(0, 100, (20, N_SLOTS))  # fit set never draws id 4095
    fit_ops = ["+"] * 10 + ["-"] * 10
    eval_ids = torch.full((2, N_SLOTS), K_big - 1, dtype=torch.long)  # id 4095, unseen in fit
    eval_ops = ["+", "-"]
    out = op_from_ids_nb(fit_ids, fit_ops, eval_ids, eval_ops, K_big)
    assert "acc" in out and 0.0 <= out["acc"] <= 1.0
    assert out["majority"] == pytest.approx(0.5)


# ── smoke_verdict ────────────────────────────────────────────────────────


def test_smoke_verdict_pass():
    cells = {
        "train_gist_ce": 1.0,
        "bigram_ce": 2.0,
        "next_id_acc": 0.5,
        "bigram_acc": 0.1,
        "op_from_predicted": 0.6,
        "majority_op": 0.25,
    }
    v = smoke_verdict(cells)
    assert v == {"verdict": "PASS", "reasons": []}


def test_smoke_verdict_fail_lists_every_failing_reason():
    cells = {
        "train_gist_ce": 7.0,  # >= 6.5
        "bigram_ce": 2.0,  # train_gist_ce also >= bigram_ce
        "next_id_acc": 0.1,
        "bigram_acc": 0.5,  # next_id_acc <= bigram_acc
        "op_from_predicted": 0.2,
        "majority_op": 0.25,  # op_from_predicted <= majority_op
    }
    v = smoke_verdict(cells)
    assert v["verdict"] == "FAIL"
    assert len(v["reasons"]) == 4


# ── decode_text_step ─────────────────────────────────────────────────────


def test_decode_text_step_never_emits_gist_or_control_id():
    base_vocab = 10
    row = torch.zeros(base_vocab + 40)
    row[base_vocab + 5] = 100.0  # a gist id has the highest raw logit
    row[3] = 1.0  # the best TEXT id
    assert decode_text_step(row, base_vocab) == 3


def test_smoke_records_shape_and_op_labels_parseable():
    from marker.summaryprobe import op_label

    recs = smoke_records(n=5, K=6, seed=1)
    assert len(recs) == 5
    for r in recs:
        assert len(r["ids"]) == len(r["steps"])
        for step_groups in r["ids"]:
            assert len(step_groups) == 1
            assert len(step_groups[0]) == N_SLOTS
            assert all(0 <= j < 6 for j in step_groups[0])
        # every synthetic step carries a real, parseable arithmetic relation
        assert all(op_label(s) is not None for s in r["steps"])


# ── load_mu_from_dict: the REAL artifact structure ────────────────────────


def test_load_mu_from_dict_accepts_the_saved_dict_structure(tmp_path):
    """dict_kv_K4096.pt on the hub is run_gist_dict.save_dict's output:
    {"cfg", "kind", "geometry", "slots": [entry per slot]} -- NOT a bare
    list. Loading must unwrap "slots" (a bare list is also accepted)."""
    K_d, d = 4, 6
    entries = [{"mu_readout": torch.randn(K_d, d).half(), "K": K_d} for _ in range(N_SLOTS)]
    saved = {"cfg": f"kv_K{K_d}", "kind": "kv", "geometry": {}, "slots": entries}
    p = tmp_path / "dict_kv.pt"
    torch.save(saved, p)
    mu = load_mu_from_dict(str(p), K_d)
    assert mu.shape == (N_SLOTS, K_d, d)
    assert mu.dtype == torch.float32

    torch.save(entries, p)  # bare list still works (tests/synthetic fixtures)
    assert load_mu_from_dict(str(p), K_d).shape == (N_SLOTS, K_d, d)


def test_load_mu_from_dict_rejects_wrong_kind(tmp_path):
    p = tmp_path / "dict_ro.pt"
    torch.save({"cfg": "ro_K4", "kind": "ro", "slots": []}, p)
    with pytest.raises(ValueError, match="kind"):
        load_mu_from_dict(str(p), 4)


# ── verdict_cells: the GPU rehearsal is SMOKE_RUN, not --smoke ────────────


def test_verdict_cells_built_from_any_manifest_with_both_blocks():
    manifest = {
        "eval_block_1": {"bigram_ce": 2.0, "next_id_acc": 0.5, "bigram_acc": 0.1},
        "eval_block_2": {"op_from_predicted": {"acc": 0.6, "majority": 0.25}},
    }
    cells = verdict_cells(1.0, manifest)
    assert cells is not None
    assert smoke_verdict(cells)["verdict"] == "PASS"


def test_verdict_cells_none_when_a_block_is_missing():
    assert verdict_cells(1.0, {"eval_block_1": {"bigram_ce": 2.0}}) is None
    assert verdict_cells(1.0, {}) is None


# ── slot-7 confusion split ────────────────────────────────────────────────


def test_slot7_confusion_split_buckets_by_mu_neighbourhood():
    # slot 7: ids 0 and 1 nearly identical (within 0.98 cosine), id 2 orthogonal
    d = 8
    mu = torch.zeros(N_SLOTS, 3, d)
    mu[7, 0, 0] = 1.0
    mu[7, 1, 0] = 1.0
    mu[7, 1, 1] = 0.01  # cos(mu[7,0], mu[7,1]) ~ 0.99995
    mu[7, 2, 2] = 1.0  # orthogonal to both
    # target id 0 (crowded, predicted wrong), target id 2 (isolated, right)
    preds = [[0] * 7 + [1], [0] * 7 + [2]]
    targets = [[0] * 8, [0] * 7 + [2]]
    out = slot7_confusion_split(preds, targets, mu, thresh=0.98)
    assert out["within_098"] == {"acc": 0.0, "n": 1}
    assert out["outside_098"] == {"acc": 1.0, "n": 1}


# ── predict_history_rows: alignment with op labels ────────────────────────


def test_predict_history_rows_aligns_with_op_labels(monkeypatch):
    """A record whose middle step has NO parseable op and whose steps carry
    MULTIPLE groups: exactly one prediction per op-labelled step, prompts
    grow by 8 x n_groups per step (true ids, whether or not the step was
    scored) -- the truncation shortcut this replaces mispaired every row
    after the first unlabelled or multi-group step."""
    import marker.run_g7 as rg

    prompts_seen = []

    def fake_greedy(pm, head, prompt_ids, base_vocab, K):  # noqa: ANN001, ARG001
        prompts_seen.append(list(prompt_ids))
        return [len(prompt_ids)] * N_SLOTS  # deterministic marker row

    monkeypatch.setattr(rg, "greedy_predict_group", fake_greedy)

    class _Vocab:
        think_id = 100 + N_SLOTS * 5

        def flat_id(self, s, j):
            return 100 + s * 5 + j

    class _Head:
        vocab = _Vocab()

    rec = {
        "question": "q one",
        "steps": ["1 + 2 = 3", "no operation here", "4 * 5 = 20"],
        "ids": [
            [[0] * N_SLOTS, [1] * N_SLOTS],  # step 1: TWO groups, op "+"
            [[2] * N_SLOTS],  # step 2: one group, NO op
            [[3] * N_SLOTS],  # step 3: one group, op "*"
        ],
    }
    tok = _FakeTok()
    rows, ops = predict_history_rows(None, _Head(), [rec], tok, 100, 5)
    assert ops == ["+", "*"]
    assert len(rows) == 2
    q_len = len(tok(rec["question"]).input_ids)
    # first prediction: question + <think> only
    assert len(prompts_seen[0]) == q_len + 1
    # second prediction: history holds ALL 3 prior groups (2 + 1), true ids
    assert len(prompts_seen[1]) == q_len + 1 + 3 * N_SLOTS
    # history groups are laid out in SLOT_ORDER with natural-slot flat ids
    first_group = prompts_seen[1][q_len + 1 : q_len + 1 + N_SLOTS]
    assert first_group == [100 + s * 5 + 0 for s in SLOT_ORDER]


# ── end-to-end smoke (real subprocess, no network, no push) ───────────────


@pytest.mark.slow
def test_smoke_end_to_end_prints_manifest_with_verdict():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "marker.run_g7",
            "--smoke",
            "--n-train-steps",
            "6",
            "--log-every",
            "3",
        ],
        cwd="/home/user/mimir-protocol",
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + "\n---STDERR---\n" + proc.stderr[-4000:]
    lines = [line for line in proc.stdout.splitlines() if line.startswith("[G7 MANIFEST]")]
    assert len(lines) == 1
    import json

    manifest = json.loads(lines[0][len("[G7 MANIFEST] ") :])
    assert manifest["smoke"] is True
    assert "smoke_verdict" in manifest
    assert manifest["smoke_verdict"]["verdict"] in ("PASS", "FAIL")


# ── slow: real attach path (tiny UNTIED model, no network) ────────────────


def _tiny_attach(base_vocab=100, K_run=6):
    from marker.run_g7 import _load_smoke_model

    return _load_smoke_model(base_vocab, K_run)


@pytest.mark.slow
def test_train_step_through_get_decoder_reaches_lora_and_vocab():
    """train_step reads hidden states off pm.get_decoder() -- this pins that
    the decoder it returns IS the LoRA-injected one (grads reach lora_*) and
    that the GistVocab masters actually move under the optimizer. If
    get_decoder ever bypassed PEFT, the LoRA would silently never train."""
    import torch as t

    from marker.g7 import build_sequence, sample_m
    from marker.run_g7 import pack_batches, smoke_records, train_step

    pm, vocab, tok = _tiny_attach()
    import random as _random

    rng = _random.Random(0)
    recs = smoke_records(n=4, K=6, seed=0)
    seqs = [
        build_sequence(r, m, tok, vocab.base_vocab, vocab.K)
        for r in recs
        for m in sample_m(len(r["steps"]), rng)
    ]
    batches, _ = pack_batches(seqs, batch_size=4, seq_cap=512)
    lora_named = [(n, p) for n, p in pm.named_parameters() if "lora_" in n and p.requires_grad]
    assert lora_named, "no trainable lora parameters found"
    head = pm.get_output_embeddings()
    opt = t.optim.AdamW(
        [
            {"params": [p for _, p in lora_named], "lr": 1e-2},
            {"params": list(vocab.parameters()), "lr": 1e-2},
        ]
    )
    a_before = vocab.A.detach().clone()
    lora_before = {n: p.detach().clone() for n, p in lora_named}
    loss, gist_ce, text_ce = train_step(pm, head, batches[0], opt)
    assert loss == loss and gist_ce == gist_ce and text_ce == text_ce  # finite
    assert not t.allclose(vocab.A, a_before), "GistVocab.A did not move"
    assert any(not t.allclose(p, lora_before[n]) for n, p in lora_named), (
        "no LoRA parameter moved -- get_decoder() bypassed the adapter"
    )


@pytest.mark.slow
def test_greedy_predict_group_cached_matches_uncached():
    """KV-cached greedy decode must produce the exact ids the uncached
    (full-prefix-per-token) reference does on the tiny fp32 model."""
    import torch as t

    from marker.gist_vocab import SLOT_ORDER as ORDER
    from marker.run_g7 import greedy_predict_group

    pm, vocab, tok = _tiny_attach()
    head = pm.get_output_embeddings()
    prompt = [5, 6, 7, vocab.think_id] + [vocab.flat_id(s, 1) for s in ORDER]

    with t.no_grad():
        cached = greedy_predict_group(pm, head, prompt, vocab.base_vocab, vocab.K)

        # uncached reference: recompute the whole prefix for every token
        rows, bias = vocab.output_rows_all()
        ids = list(prompt)
        ref = [0] * N_SLOTS
        for ti in range(N_SLOTS):
            s = ORDER[ti]
            hidden = pm.get_decoder()(input_ids=t.tensor([ids]), use_cache=False).last_hidden_state[
                0, -1
            ]
            block_rows = rows[s * vocab.K : (s + 1) * vocab.K].to(hidden.dtype)
            block_bias = bias[s * vocab.K : (s + 1) * vocab.K].to(hidden.dtype)
            j = int((hidden @ block_rows.T + block_bias).argmax())
            ref[s] = j
            ids.append(vocab.flat_id(s, j))
    assert cached == ref


@pytest.mark.slow
def test_checkpoint_roundtrip_and_resume(tmp_path):
    """_save_checkpoint -> load_g7_checkpoint restores the LoRA adapter and
    every GistVocab parameter into a FRESH attach (mu itself is never saved
    -- it is rebuilt from the dictionary); resume_from_repo picks the
    highest step via injected lister/fetcher."""
    import torch as t

    from marker.run_g7 import _save_checkpoint, resume_from_repo

    pm1, vocab1, _ = _tiny_attach()
    with t.no_grad():  # make the state distinguishable from a fresh init
        vocab1.A.add_(0.123)
        vocab1.c_in.add_(0.05)
        for n, p in pm1.named_parameters():
            if "lora_A" in n:
                p.add_(0.07)
    ck = tmp_path / "step-0000002"
    _save_checkpoint(ck, pm1, vocab1, {"step": 2})
    assert not (ck / "gistvocab.safetensors").stat().st_size > 10_000_000  # mu excluded

    pm2, vocab2, _ = _tiny_attach()
    assert not t.allclose(vocab2.A, vocab1.A)
    step = resume_from_repo(
        pm2,
        vocab2,
        "fake/repo",
        str(tmp_path),
        lister=lambda repo: [
            "g7_v0/step-0000001/gistvocab.safetensors",
            "g7_v0/step-0000002/gistvocab.safetensors",
            "unrelated.txt",
        ],
        fetcher=lambda repo, subdir, dest: ck,
    )
    assert step == 2
    assert t.allclose(vocab2.A, vocab1.A)
    assert t.allclose(vocab2.c_in, vocab1.c_in)
    l1 = {n: p for n, p in pm1.named_parameters() if "lora_A" in n}
    l2 = {n: p for n, p in pm2.named_parameters() if "lora_A" in n}
    assert l1.keys() == l2.keys() and all(t.allclose(l2[n], l1[n]) for n in l1)


@pytest.mark.slow
def test_resume_from_repo_fresh_when_listing_fails_or_empty():
    from marker.run_g7 import resume_from_repo

    pm, vocab, _ = _tiny_attach()

    def boom(repo):  # noqa: ANN001, ARG001
        raise OSError("no network")

    assert resume_from_repo(pm, vocab, "r", "/tmp/x", lister=boom) == 0
    assert resume_from_repo(pm, vocab, "r", "/tmp/x", lister=lambda r: ["a.txt"]) == 0
