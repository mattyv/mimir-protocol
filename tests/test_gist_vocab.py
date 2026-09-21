"""Tests for gist_vocab.py (G7 vocabulary wiring, see GIST_LM_PLAN.md "STAGES
2+3 DESIGN v3"). Mechanical invariants only -- gradients flow, frozen rows
stay frozen, shapes/dtypes are right, generate() runs -- never the
experiment's actual numbers.

Model-touching tests are marked slow (a real (tiny) Qwen2 model, no network).
"""

from __future__ import annotations

import hashlib

import pytest
import torch

from marker.gist_vocab import (
    N_SLOTS,
    SLOT_ORDER,
    GistEmbedWrapper,
    GistHeadWrapper,
    GistVocab,
    attach_g7,
    gammas,
    slot_for_position,
)

# ── pure-logic tests (no model) ─────────────────────────────────────────────


def test_slot_for_position_covers_all_8_slots_in_one_cycle():
    seen = {slot_for_position(t) for t in range(N_SLOTS)}
    assert seen == set(range(N_SLOTS))


def test_slot_for_position_matches_slot_order_and_wraps():
    for t in range(24):
        assert slot_for_position(t) == SLOT_ORDER[t % N_SLOTS]


def test_slot_for_position_rejects_negative():
    with pytest.raises(ValueError, match="t_since_think"):
        slot_for_position(-1)


def test_gammas_scale_matched():
    torch.manual_seed(0)
    mu = torch.randn(N_SLOTS, 4, 6) * 3.0
    embed_w = torch.randn(50, 6) * 0.5
    head_w = torch.randn(50, 6) * 2.0
    gin, gout = gammas(embed_w, head_w, mu)
    assert gin == pytest.approx(embed_w.std().item() / mu.std().item())
    assert gout == pytest.approx(head_w.std().item() / mu.std().item())


def test_gist_vocab_rejects_wrong_slot_count():
    with pytest.raises(ValueError, match="8"):
        GistVocab(torch.zeros(4, 3, 5), d_model=5, d_out=5, base_vocab=10)


def _small_vocab(K=5, d_mu=6, base_vocab=100, seed=0):
    torch.manual_seed(seed)
    mu = torch.randn(N_SLOTS, K, d_mu)
    return GistVocab(mu, d_model=d_mu, d_out=d_mu, base_vocab=base_vocab, r_slot=4, r_id=3)


def test_flat_id_layout():
    v = _small_vocab()
    assert v.flat_id(0, 0) == v.base_vocab
    assert v.flat_id(1, 0) == v.base_vocab + v.K
    assert v.flat_id(7, v.K - 1) == v.base_vocab + 7 * v.K + v.K - 1
    assert v.think_id == v.base_vocab + N_SLOTS * v.K
    assert v.commit_id == v.think_id + 1


def test_input_rows_think_commit_use_their_own_rows():
    v = _small_vocab()
    rows = v.input_rows(torch.tensor([v.think_id, v.commit_id]))
    assert torch.allclose(rows[0], v.think_in)
    assert torch.allclose(rows[1], v.commit_in)


def test_input_rows_rejects_out_of_range_id():
    v = _small_vocab()
    with pytest.raises(ValueError, match="range"):
        v.input_rows(torch.tensor([v.commit_id + 1]))


def test_input_rows_empty_is_empty():
    v = _small_vocab()
    rows = v.input_rows(torch.zeros(0, dtype=torch.long))
    assert rows.shape == (0, v.d_model)


def test_output_rows_all_shape_and_order():
    v = _small_vocab()
    rows, bias = v.output_rows_all()
    assert rows.shape == (N_SLOTS * v.K + 2, v.d_out)
    assert bias.shape == (N_SLOTS * v.K + 2,)
    # flat_id(s, j) indexes rows/bias directly: s*K + j, then think, commit.
    assert torch.allclose(rows[-2], v.think_out)
    assert torch.allclose(rows[-1], v.commit_out)


def test_output_rows_zero_delta_at_init_leaves_pure_shared_projection():
    # c_out is zero-initialized, so at init output rows are exactly B . mu
    # (no per-id delta yet) -- pins the init story, not a trained value.
    v = _small_vocab()
    rows, bias = v.output_rows_all()
    mu_flat = v.mu.reshape(N_SLOTS * v.K, v.d_mu)
    expected = mu_flat @ v.B.T
    assert torch.allclose(rows[: N_SLOTS * v.K], expected, atol=1e-6)
    assert torch.allclose(bias[: N_SLOTS * v.K], torch.zeros(N_SLOTS * v.K))


def test_grad_flows_to_every_trainable_block():
    v = _small_vocab()
    flat_ids = torch.tensor([v.flat_id(0, 0), v.flat_id(3, 2), v.think_id, v.commit_id])
    in_rows = v.input_rows(flat_ids)
    out_rows, out_bias = v.output_rows_all()
    loss = in_rows.sum() + out_rows.sum() + out_bias.sum()
    loss.backward()
    for name in ["A", "B", "c_in", "c_out", "bias"]:
        g = getattr(v, name).grad
        assert g is not None, f"{name} got no grad"
        assert g.abs().sum() > 0, f"{name} grad is all-zero"
    # P/Q and U/V are both LoRA-style zero-init pairs (correction = basis @
    # (zero_code @ mu-or-nothing); e.g. corr = P @ (Q @ mu) with Q zero-init,
    # delta_in = U @ c_in with c_in zero-init). At THIS very first backward,
    # the RANDOM half of each pair (P, U, V) gets a genuinely zero gradient
    # -- dL/dP = grad_corr @ (Q@mu)^T = 0 when Q=0, same reasoning for U/V
    # against zero c_in/c_out (identical to why ordinary LoRA's A-side sees
    # no gradient at init when B is zero-init). Only the zero-init half (Q)
    # sees signal here; P/U/V are merely confirmed REACHABLE (grad exists,
    # value may be 0) -- test_grad_flows_to_P_and_U_V_once_codes_are_nonzero
    # below pins that they train once the codes move off zero.
    assert v.Q.grad is not None and v.Q.grad.abs().sum() > 0
    for name in ["P", "U", "V"]:
        assert getattr(v, name).grad is not None, f"{name} unreachable by backward"


def test_grad_flows_to_p_and_u_v_once_codes_are_nonzero():
    # Once Q/c_in/c_out move off zero (e.g. after one optimizer step), P/U/V
    # must also see gradient -- otherwise those bases never train.
    v = _small_vocab()
    with torch.no_grad():
        v.Q.add_(0.05)
        v.c_in.add_(0.05)
        v.c_out.add_(0.05)
    flat_ids = torch.tensor([v.flat_id(0, 0), v.flat_id(3, 2)])
    out_rows, _ = v.output_rows_all()
    (v.input_rows(flat_ids).sum() + out_rows.sum()).backward()
    for name in ["P", "U", "V"]:
        g = getattr(v, name).grad
        assert g is not None and g.abs().sum() > 0, f"{name} grad still zero"


# ── model-touching tests (tiny UNTIED Qwen2, no network) ────────────────────


def _tiny_qwen(vocab_size=200, hidden=64):
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(0)
    cfg = AutoConfig.for_model(
        "qwen2",
        vocab_size=vocab_size,
        hidden_size=hidden,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )
    assert cfg.tie_word_embeddings is False  # the smoke model MUST be untied
    model = AutoModelForCausalLM.from_config(cfg, attn_implementation="sdpa")
    return model.eval()


def _attach(base_vocab=200, hidden=64, K=5):
    base = _tiny_qwen(vocab_size=base_vocab, hidden=hidden)
    torch.manual_seed(1)
    mu = torch.randn(N_SLOTS, K, hidden)
    pm, vocab = attach_g7(base, mu, r_lora=4, r_slot=4, r_id=3)
    return pm, vocab, base_vocab, K


@pytest.mark.slow
def test_attach_g7_extends_vocab_size():
    pm, vocab, base_vocab, K = _attach()
    assert pm.config.vocab_size == base_vocab + N_SLOTS * K + 2


@pytest.mark.slow
def test_attach_g7_get_input_embeddings_is_the_wrapper():
    pm, vocab, *_ = _attach()
    assert isinstance(pm.get_input_embeddings(), GistEmbedWrapper)
    assert isinstance(pm.get_output_embeddings(), GistHeadWrapper)


@pytest.mark.slow
def test_attach_g7_backward_reaches_gistvocab_and_lora():
    pm, vocab, base_vocab, K = _attach()
    think = vocab.think_id
    gist0 = vocab.flat_id(0, 1)
    # gist0 must NOT be the last token: causal attention means a token past
    # the last-scored position (labels drop the final position) never
    # reaches the loss, which would make this a test of nothing.
    ids = torch.tensor([[5, 6, think, gist0, 7]])
    out = pm(input_ids=ids, labels=ids)
    out.loss.backward()
    assert vocab.A.grad is not None and vocab.A.grad.abs().sum() > 0
    assert vocab.c_in.grad is not None  # zero-init but must still be reachable
    lora_grads = [p.grad for n, p in pm.named_parameters() if "lora_" in n and p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in lora_grads)


@pytest.mark.slow
def test_attach_g7_frozen_text_rows_untouched_by_an_optimizer_step():
    pm, vocab, base_vocab, K = _attach()
    base_embed = pm.get_input_embeddings().base_embed
    before = hashlib.sha256(base_embed.weight.detach().numpy().tobytes()).hexdigest()

    think = vocab.think_id
    gist0 = vocab.flat_id(0, 1)
    ids = torch.tensor([[5, 6, think, gist0]])
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=1e-2)
    opt.zero_grad()
    out = pm(input_ids=ids, labels=ids)
    out.loss.backward()
    opt.step()

    after = hashlib.sha256(base_embed.weight.detach().numpy().tobytes()).hexdigest()
    assert before == after


@pytest.mark.slow
def test_attach_g7_logits_finite_single_dtype_across_concat():
    pm, vocab, base_vocab, K = _attach()
    think = vocab.think_id
    ids = torch.tensor([[5, 6, think]])
    with torch.no_grad():
        out = pm(input_ids=ids)
    assert out.logits.dtype == torch.float32
    assert torch.isfinite(out.logits).all()
    assert out.logits.shape[-1] == pm.config.vocab_size


@pytest.mark.slow
def test_attach_g7_generate_runs_end_to_end_with_extended_ids():
    pm, vocab, base_vocab, K = _attach()
    think = vocab.think_id
    ids = torch.tensor([[5, 6, think]])
    with torch.no_grad():
        out = pm.generate(input_ids=ids, max_new_tokens=4, do_sample=False)
    assert out.shape[1] == ids.shape[1] + 4
    assert (out < pm.config.vocab_size).all()
