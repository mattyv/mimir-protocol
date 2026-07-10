"""Correctness tests for the gist bottleneck (see gist.py, GIST_PILOT_PLAN.md).

The 4D mask is the fiddly, failure-prone piece; a silent leak there fakes a
"gist works" result. The primary defense is asserting the mask VALUES directly
(model-free, unambiguous). The model-level leak/connect tests (slow, tiny
model) confirm behavior matches the mask.
"""

from __future__ import annotations

import torch

from marker.gist import (
    build_attention_mask,
    build_batch_labels,
    build_batch_mask,
    build_labels,
    build_leak_diagnostic_mask,
    gist_position_ids,
    make_pair,
    split_sentences,
)

NEG = float("-inf")


# ── Mask values: the direct, unambiguous leak test ──────────────────────────────


def test_mask_shape_and_diagonal_open():
    m = build_attention_mask(s=3, k=2, c=4)
    assert m.shape == (1, 1, 9, 9)
    for i in range(9):
        assert m[0, 0, i, i] == 0.0  # every token attends to itself


def test_continuation_is_blocked_from_span():
    s, k, c = 3, 2, 4
    m = build_attention_mask(s, k, c)[0, 0]
    for q in range(s + k, s + k + c):  # C queries
        for key in range(0, s):  # S keys
            assert m[q, key] == NEG, f"C {q} must not see S {key}"


def test_continuation_attends_to_gist_and_causal_self():
    s, k, c = 3, 2, 4
    m = build_attention_mask(s, k, c)[0, 0]
    for q in range(s + k, s + k + c):
        for g in range(s, s + k):  # gist keys open
            assert m[q, g] == 0.0
        for j in range(s + k, q + 1):  # causal within C open
            assert m[q, j] == 0.0
        for j in range(q + 1, s + k + c):  # future C blocked
            assert m[q, j] == NEG


def test_gist_attends_to_all_span_and_causal_gist():
    s, k, c = 3, 2, 4
    m = build_attention_mask(s, k, c)[0, 0]
    for g in range(s, s + k):
        for key in range(0, s):  # all of S open to gist
            assert m[g, key] == 0.0
        for j in range(s, g + 1):  # causal among gist
            assert m[g, j] == 0.0


def test_span_is_causal_within_itself():
    s, k, c = 3, 2, 4
    m = build_attention_mask(s, k, c)[0, 0]
    for q in range(s):
        for key in range(s):
            assert m[q, key] == (0.0 if key <= q else NEG)


def test_diagnostic_mask_additionally_blocks_gist_to_span():
    s, k, c = 3, 2, 4
    train = build_attention_mask(s, k, c)[0, 0]
    diag = build_leak_diagnostic_mask(s, k, c)[0, 0]
    # gist->S open in training, blocked in diagnostic
    assert train[s, 0] == 0.0
    assert diag[s, 0] == NEG
    # C->S blocked in both; C->gist open in both
    assert train[s + k, 0] == NEG and diag[s + k, 0] == NEG
    assert train[s + k, s] == 0.0 and diag[s + k, s] == 0.0


# ── Batched mask (Fable build-note #2): 2 samples, different s and c ─────────────


def test_batch_mask_shape_and_padding_keys_blocked():
    # sample 0: s=3,c=4 (full); sample 1: s=2,c=2 (span pad at 2, cont pad at 7,8)
    m = build_batch_mask([3, 2], [4, 2], k=2, max_s=3, max_c=4)
    assert m.shape == (2, 1, 9, 9)  # T = 3+2+4
    m1 = m[1, 0]
    # sample 1 padded keys: span-pad col 2, cont-pad cols 7,8 blocked for every
    # real query (rows 0,1 span; 3,4 gist; 5,6 cont)
    for q in (0, 1, 3, 4, 5, 6):
        for pad_key in (2, 7, 8):
            assert m1[q, pad_key] == NEG, f"row {q} must not see pad key {pad_key}"


def test_batch_mask_continuation_blocked_from_span_per_sample():
    m = build_batch_mask([3, 2], [4, 2], k=2, max_s=3, max_c=4)
    c0 = 5
    # sample 0: C rows 5..8 blocked from span 0..2
    for q in range(c0, c0 + 4):
        for key in range(0, 3):
            assert m[0, 0, q, key] == NEG
    # sample 1: C rows 5..6 (real) blocked from span real 0..1 AND span pad 2
    for q in range(c0, c0 + 2):
        for key in range(0, 3):
            assert m[1, 0, q, key] == NEG


def test_batch_mask_gist_sees_only_real_span():
    m = build_batch_mask([3, 2], [4, 2], k=2, max_s=3, max_c=4)
    g0 = 3
    # sample 1 gist (rows 3,4) sees real span cols 0,1 but NOT span-pad col 2
    for q in (g0, g0 + 1):
        assert m[1, 0, q, 0] == 0.0 and m[1, 0, q, 1] == 0.0
        assert m[1, 0, q, 2] == NEG


def test_batch_mask_no_all_blocked_rows():
    # every query row must have at least one open key (else softmax -> NaN)
    m = build_batch_mask([3, 2], [4, 2], k=2, max_s=3, max_c=4)
    for b in range(2):
        for q in range(9):
            assert (m[b, 0, q] == 0.0).any(), f"sample {b} row {q} fully blocked"


def test_batch_labels_only_real_continuation():
    labels = build_batch_labels([[10, 11, 12, 13], [20, 21]], max_s=3, k=2, max_c=4)
    assert labels.shape == (2, 9)
    assert labels[0, 5:].tolist() == [10, 11, 12, 13]  # full cont
    assert labels[1, 5:].tolist() == [20, 21, -100, -100]  # padded tail masked


def test_eval_masks_differ_only_in_continuation_access():
    # Fable #3: gist / full / none conditions keep C at the SAME positions and
    # differ ONLY in what C attends to among {span, gist}.
    args = ([3], [4], 2, 3, 4)  # s=3,c=4,k=2,max_s=3,max_c=4 -> C rows 5..8
    g = build_batch_mask(*args, cont_sees=frozenset({"gist"}))[0, 0]
    full = build_batch_mask(*args, cont_sees=frozenset({"gist", "span"}))[0, 0]
    none = build_batch_mask(*args, cont_sees=frozenset())[0, 0]
    for q in range(5, 9):  # continuation rows
        # gist: sees gist (3,4), not span (0..2)
        assert g[q, 3] == 0.0 and g[q, 0] == NEG
        # full: sees span AND gist
        assert full[q, 0] == 0.0 and full[q, 3] == 0.0
        # none: sees neither span nor gist (only causal C)
        assert none[q, 0] == NEG and none[q, 3] == NEG
        assert none[q, 5] == 0.0  # but still causal within C
    # span and gist rows identical across conditions (only C access changes)
    for q in range(0, 5):
        assert torch.equal(g[q], full[q]) and torch.equal(g[q], none[q])


# ── Labels & positions ──────────────────────────────────────────────────────────


def test_labels_only_on_continuation():
    labels = build_labels(s=3, k=2, cont_ids=[10, 11, 12])
    assert labels.shape == (1, 8)
    assert (labels[0, :5] == -100).all()  # S + gist masked
    assert labels[0, 5:].tolist() == [10, 11, 12]


def test_position_ids_contiguous():
    pos = gist_position_ids(3, 2, 4)
    assert pos.tolist() == [list(range(9))]


# ── Sentence pairing ────────────────────────────────────────────────────────────


def test_split_sentences_basic():
    out = split_sentences("Hello world. This is two! And three? ok")
    assert out == ["Hello world.", "This is two!", "And three?", "ok"]


class _FakeTok:
    def __call__(self, text, add_special_tokens=False):  # noqa: ANN001
        ids = [ord(ch) for ch in text if not ch.isspace()]
        return type("Enc", (), {"input_ids": ids})()


def test_make_pair_caps_and_min_length():
    tok = _FakeTok()
    span, cont = make_pair(tok, "abcdefgh", "xyzwv", max_span=4, max_cont=10, min_cont=3)
    assert span == [ord(c) for c in "abcd"]  # capped at 4
    assert cont == [ord(c) for c in "xyzwv"]


def test_make_pair_drops_short_continuation():
    tok = _FakeTok()
    assert make_pair(tok, "abc", "xy", max_span=4, max_cont=10, min_cont=3) is None


# ── Model-level behavior (slow, tiny model) — matches the mask ──────────────────


def _tiny_lm():
    from transformers import AutoModelForCausalLM  # noqa: PLC0415

    torch.manual_seed(0)
    from transformers import AutoConfig  # noqa: PLC0415

    cfg = AutoConfig.for_model(
        "qwen2",
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return AutoModelForCausalLM.from_config(cfg).eval()


def _cont_logits(model, span_ids, gist_emb, cont_ids, mask):  # noqa: ANN001
    s, k, c = len(span_ids), gist_emb.shape[0], len(cont_ids)
    embed = model.get_input_embeddings()
    span_e = embed(torch.tensor([span_ids]))
    cont_e = embed(torch.tensor([cont_ids]))
    inp = torch.cat([span_e, gist_emb.unsqueeze(0), cont_e], dim=1)
    pos = gist_position_ids(s, k, c)
    out = model(inputs_embeds=inp, attention_mask=mask.to(inp.dtype), position_ids=pos)
    return out.logits[0, s + k :]


import pytest  # noqa: E402


@pytest.mark.slow
def test_gist_connects_span_to_continuation():
    # With the TRAINING mask, changing S must move C's logits (S reaches C
    # through the gist — the bottleneck is live, not dead).
    model = _tiny_lm()
    torch.manual_seed(1)
    gist_emb = torch.randn(2, 32)
    cont = [5, 6, 7, 8]
    m = build_attention_mask(3, 2, 4, dtype=torch.float32)
    a = _cont_logits(model, [10, 11, 12], gist_emb, cont, m)
    b = _cont_logits(model, [40, 41, 42], gist_emb, cont, m)
    assert not torch.allclose(a, b, atol=1e-4), "gist carries nothing (dead bottleneck)"


@pytest.mark.slow
def test_no_direct_leak_under_diagnostic_mask():
    # Under the diagnostic mask (gist->S also blocked), the ONLY possible S->C
    # path is a direct C->S leak. So C logits MUST be invariant to S.
    model = _tiny_lm()
    torch.manual_seed(1)
    gist_emb = torch.randn(2, 32)
    cont = [5, 6, 7, 8]
    m = build_leak_diagnostic_mask(3, 2, 4, dtype=torch.float32)
    a = _cont_logits(model, [10, 11, 12], gist_emb, cont, m)
    b = _cont_logits(model, [40, 41, 42], gist_emb, cont, m)
    assert torch.allclose(a, b, atol=1e-4), "C leaks to S directly — mask bug"
