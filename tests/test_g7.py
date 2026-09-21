"""Tests for g7.py: sequence layout, two-gather loss, generation-time slot
masking. Pure-logic tests run always; model-touching ones are marked slow (a
real tiny UNTIED Qwen2, no network)."""

from __future__ import annotations

import random

import pytest
import torch

from marker.g7 import (
    GistLogitsProcessor,
    build_sequence,
    parse_sequence,
    sample_m,
    two_gather_loss,
)
from marker.gist_vocab import N_SLOTS, SLOT_ORDER, attach_g7

BASE_VOCAB = 100
K = 5


class _FakeTok:
    """A minimal stand-in for an HF tokenizer: whitespace-split, one
    "token id" per word (deterministic, offline -- no network)."""

    eos_token_id = 999

    def __call__(self, text, add_special_tokens=False):  # noqa: ANN001, ARG002
        ids = [10 + (hash(w) % 50) for w in text.split()]
        return type("Enc", (), {"input_ids": ids})()


def _record(n_steps=3, groups_per_step=1, seed=0):
    rng = random.Random(seed)
    steps = [f"step {i} text words" for i in range(n_steps)]
    ids = [
        [[rng.randrange(K) for _ in range(N_SLOTS)] for _ in range(groups_per_step)]
        for _ in range(n_steps)
    ]
    return {
        "src": "test",
        "doc_id": "d0",
        "question": "what is the question",
        "steps": steps,
        "ids": ids,
        "answer": "42",
        "n_groups": n_steps * groups_per_step,
    }


# ── sample_m ─────────────────────────────────────────────────────────────


def test_sample_m_n1_returns_only_final():
    assert sample_m(1, random.Random(0)) == [1]


def test_sample_m_returns_n_and_earlier_step():
    rng = random.Random(0)
    ms = sample_m(5, rng)
    assert ms[0] == 5
    assert 1 <= ms[1] <= 4


def test_sample_m_rejects_n_lt_1():
    with pytest.raises(ValueError, match="n must be"):
        sample_m(0, random.Random(0))


# ── build_sequence / parse_sequence round trip ──────────────────────────


def test_build_sequence_layout_and_round_trip():
    rec = _record(n_steps=3, groups_per_step=1)
    tok = _FakeTok()
    seq = build_sequence(rec, m=2, tok=tok, base_vocab=BASE_VOCAB, K=K)
    ids = seq["input_ids"]

    think_id = BASE_VOCAB + N_SLOTS * K
    commit_id = think_id + 1
    assert ids.count(think_id) == 1
    assert ids.count(commit_id) == 1
    # groups of a step follow each other -- exactly 2 groups (m=2) * 8 ids
    # between <think> and <commit>.
    t_pos, c_pos = ids.index(think_id), ids.index(commit_id)
    assert c_pos - t_pos - 1 == 2 * N_SLOTS

    groups, text_ids = parse_sequence(ids, BASE_VOCAB, K)
    assert groups == [rec["ids"][0][0], rec["ids"][1][0]]
    # step m=2's text (0-indexed steps[1]) + eos
    expected_text = tok(rec["steps"][1]).input_ids + [tok.eos_token_id]
    assert text_ids == expected_text


def test_build_sequence_multi_group_step():
    rec = _record(n_steps=2, groups_per_step=3)
    tok = _FakeTok()
    seq = build_sequence(rec, m=1, tok=tok, base_vocab=BASE_VOCAB, K=K)
    groups, _ = parse_sequence(seq["input_ids"], BASE_VOCAB, K)
    assert groups == rec["ids"][0]  # all 3 groups of step 1


def test_build_sequence_rejects_m_out_of_range():
    rec = _record(n_steps=2)
    with pytest.raises(ValueError, match="out of range"):
        build_sequence(rec, m=3, tok=_FakeTok(), base_vocab=BASE_VOCAB, K=K)
    with pytest.raises(ValueError, match="out of range"):
        build_sequence(rec, m=0, tok=_FakeTok(), base_vocab=BASE_VOCAB, K=K)


def test_build_sequence_masks_exclude_question_think_commit():
    rec = _record(n_steps=2, groups_per_step=1)
    tok = _FakeTok()
    seq = build_sequence(rec, m=1, tok=tok, base_vocab=BASE_VOCAB, K=K)
    ids, gist_mask, text_mask = seq["input_ids"], seq["gist_mask"], seq["text_mask"]
    think_id = BASE_VOCAB + N_SLOTS * K
    commit_id = think_id + 1

    for i in range(len(ids) - 1):
        tgt = ids[i + 1]
        if tgt in (think_id, commit_id):
            assert not gist_mask[i] and not text_mask[i], f"loss scored a control target at {i}"
    # question tokens themselves are never a scored target either
    q_len = len(tok(rec["question"]).input_ids)
    for i in range(q_len - 1):
        assert not gist_mask[i] and not text_mask[i]
    # exactly one position scores the transition into text (<commit>'s
    # position, predicting the first text token)
    assert sum(text_mask) == len(tok(rec["steps"][0]).input_ids) + 1  # + eos
    assert sum(gist_mask) == N_SLOTS  # one group


def test_build_sequence_slot_of_pos_matches_slot_order():
    rec = _record(n_steps=1, groups_per_step=1)
    tok = _FakeTok()
    seq = build_sequence(rec, m=1, tok=tok, base_vocab=BASE_VOCAB, K=K)
    think_id = BASE_VOCAB + N_SLOTS * K
    t_pos = seq["input_ids"].index(think_id)
    slots_in_group = seq["slot_of_pos"][t_pos + 1 : t_pos + 1 + N_SLOTS]
    assert slots_in_group == SLOT_ORDER


def test_parse_sequence_rejects_missing_think():
    with pytest.raises(ValueError, match="<think>"):
        parse_sequence([1, 2, 3], BASE_VOCAB, K)


def test_parse_sequence_rejects_bad_group_length():
    think_id = BASE_VOCAB + N_SLOTS * K
    commit_id = think_id + 1
    with pytest.raises(ValueError, match="multiple of"):
        parse_sequence([think_id, BASE_VOCAB, commit_id], BASE_VOCAB, K)


# ── two_gather_loss ───────────────────────────────────────────────────────


class _FakeVocab:
    def __init__(self, n_slots, K, d_out):
        self.n_slots, self.K, self.d_out = n_slots, K, d_out
        self.base_vocab = BASE_VOCAB
        torch.manual_seed(0)
        self._rows = torch.randn(n_slots * K + 2, d_out)
        self._bias = torch.randn(n_slots * K + 2)

    def output_rows_all(self):
        return self._rows, self._bias


class _FakeHead:
    def __init__(self, d_out, text_vocab):
        self.vocab = _FakeVocab(N_SLOTS, K, d_out)
        torch.manual_seed(1)
        self._w = torch.randn(text_vocab, d_out)

    def base_head(self, h):
        return h @ self._w.T


def test_two_gather_loss_equals_full_width_masked_ce():
    d_out, text_vocab = 6, 12
    head = _FakeHead(d_out, text_vocab)
    torch.manual_seed(2)
    n = 5
    hidden = torch.randn(n, d_out)
    slot = 3
    local_id = 2
    target_flat = BASE_VOCAB + slot * K + local_id
    targets = torch.tensor([target_flat] * n)
    gist_mask = torch.tensor([True, True, False, False, False])
    text_mask = torch.tensor([False, False, True, True, False])
    # text targets must be valid base-vocab ids for the fake text head
    targets = targets.clone()
    targets[text_mask] = torch.tensor([1, 2])
    slot_of_pos = torch.full((n,), slot, dtype=torch.long)

    loss, gist_ce, text_ce = two_gather_loss(
        head, hidden, targets, gist_mask, text_mask, slot_of_pos
    )

    # full-width reference: base logits concat gist logits, -inf outside the
    # target's slot block, softmax CE against the GLOBAL (base_vocab-offset)
    # target id.
    rows, bias = head.vocab.output_rows_all()
    gist_logits_full = hidden @ rows.T + bias  # [n, 8K+2]
    full = torch.cat([head.base_head(hidden), gist_logits_full], dim=-1)
    neg_inf = torch.finfo(full.dtype).min
    block_lo = text_vocab + slot * K
    block_hi = block_lo + K
    masked = torch.full_like(full, neg_inf)
    masked[:, block_lo:block_hi] = full[:, block_lo:block_hi]
    import torch.nn.functional as F  # noqa: N812

    ref_gist_ce = F.cross_entropy(masked[gist_mask], targets[gist_mask] - BASE_VOCAB + text_vocab)

    assert torch.allclose(gist_ce, ref_gist_ce, atol=1e-5)
    assert n_total_consistent(loss, gist_ce, text_ce, int(gist_mask.sum()), int(text_mask.sum()))


def n_total_consistent(loss, gist_ce, text_ce, n_gist, n_text):
    n = n_gist + n_text
    expected = (text_ce * n_text + gist_ce * n_gist) / n
    return torch.allclose(loss, expected)


def test_two_gather_loss_rejects_no_scored_positions():
    d_out, text_vocab = 4, 8
    head = _FakeHead(d_out, text_vocab)
    hidden = torch.randn(3, d_out)
    targets = torch.zeros(3, dtype=torch.long)
    zeros = torch.zeros(3, dtype=torch.bool)
    with pytest.raises(ValueError, match="no scored positions"):
        two_gather_loss(head, hidden, targets, zeros, zeros, torch.zeros(3, dtype=torch.long))


def test_two_gather_loss_text_only():
    d_out, text_vocab = 4, 8
    head = _FakeHead(d_out, text_vocab)
    hidden = torch.randn(3, d_out)
    targets = torch.tensor([1, 2, 3])
    text_mask = torch.tensor([True, True, True])
    gist_mask = torch.zeros(3, dtype=torch.bool)
    loss, gist_ce, text_ce = two_gather_loss(
        head, hidden, targets, gist_mask, text_mask, torch.zeros(3, dtype=torch.long)
    )
    assert gist_ce.item() == 0.0
    assert torch.allclose(loss, text_ce)


# ── GistLogitsProcessor ──────────────────────────────────────────────────


def test_processor_masks_to_slot_block_then_forces_commit_then_text():
    budget = 2
    proc = GistLogitsProcessor(budget, BASE_VOCAB, K)
    think_id = proc.think_id
    commit_id = proc.commit_id
    vocab_width = BASE_VOCAB + N_SLOTS * K + 2

    seq = [think_id]
    for t in range(budget * N_SLOTS):
        row = torch.zeros(vocab_width)
        masked = proc._mask_row(seq, row)
        allowed = (masked > torch.finfo(row.dtype).min).nonzero(as_tuple=True)[0].tolist()
        slot = SLOT_ORDER[t % N_SLOTS]
        lo, hi = BASE_VOCAB + slot * K, BASE_VOCAB + (slot + 1) * K
        assert allowed == list(range(lo, hi))
        seq.append(lo)  # greedily emit the first allowed id of that slot

    # budget exhausted: only <commit> allowed
    row = torch.zeros(vocab_width)
    masked = proc._mask_row(seq, row)
    allowed = (masked > torch.finfo(row.dtype).min).nonzero(as_tuple=True)[0].tolist()
    assert allowed == [commit_id]
    seq.append(commit_id)

    # after <commit>: only text ids (< base_vocab)
    row = torch.zeros(vocab_width)
    masked = proc._mask_row(seq, row)
    allowed = (masked > torch.finfo(row.dtype).min).nonzero(as_tuple=True)[0].tolist()
    assert allowed == list(range(BASE_VOCAB))


def test_processor_requires_think_in_prompt():
    proc = GistLogitsProcessor(1, BASE_VOCAB, K)
    with pytest.raises(ValueError, match="<think>"):
        proc._mask_row([1, 2, 3], torch.zeros(BASE_VOCAB + N_SLOTS * K + 2))


def test_processor_rejects_bad_budget():
    with pytest.raises(ValueError, match="think_budget_steps"):
        GistLogitsProcessor(0, BASE_VOCAB, K)


@pytest.mark.slow
def test_processor_conformant_generation_on_tiny_wrapped_model():
    """End-to-end: real tiny UNTIED Qwen2 + attach_g7 + GistLogitsProcessor
    via HF generate(). Verifies the SCHEDULE (every gist id in its slot's
    block, exact group count, then <commit>, then text-only) -- never the
    specific ids chosen (the tiny model's logits are arbitrary noise)."""
    from transformers import AutoConfig, AutoModelForCausalLM, LogitsProcessorList

    torch.manual_seed(0)
    cfg = AutoConfig.for_model(
        "qwen2",
        vocab_size=BASE_VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        tie_word_embeddings=False,
    )
    assert cfg.tie_word_embeddings is False
    base = AutoModelForCausalLM.from_config(cfg, attn_implementation="sdpa").eval()
    mu = torch.randn(N_SLOTS, K, 32)
    pm, vocab = attach_g7(base, mu, r_lora=4, r_slot=4, r_id=3)

    budget = 2
    proc = GistLogitsProcessor(budget, vocab.base_vocab, vocab.K)
    prompt = torch.tensor([[5, 6, vocab.think_id]])
    new_tokens = budget * N_SLOTS + 1 + 3  # groups + <commit> + a few text tokens
    with torch.no_grad():
        out = pm.generate(
            input_ids=prompt,
            max_new_tokens=new_tokens,
            do_sample=False,
            logits_processor=LogitsProcessorList([proc]),
        )
    gen = out[0, prompt.shape[1] :].tolist()
    gist_part = gen[: budget * N_SLOTS]
    for t, flat in enumerate(gist_part):
        slot = SLOT_ORDER[t % N_SLOTS]
        lo, hi = vocab.base_vocab + slot * vocab.K, vocab.base_vocab + (slot + 1) * vocab.K
        assert lo <= flat < hi, f"position {t} landed outside slot {slot}'s block"
    assert gen[budget * N_SLOTS] == vocab.commit_id
    for flat in gen[budget * N_SLOTS + 1 :]:
        assert flat < vocab.base_vocab
