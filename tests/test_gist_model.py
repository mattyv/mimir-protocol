"""Model-level tests for the gist LoRA + training forward (gist_model.py).

Slow (tiny Qwen2 on CPU): trainable-set is LoRA+gist only, the sdpa assertion,
a finite loss, loss decreasing over a few steps on a fixed batch, and the
3-PPL direction (full <= none). gap_closed math is unit-tested model-free.
"""

from __future__ import annotations

import pytest
import torch

from marker.gist_model import (
    assert_attn_impl,
    attach_gist,
    gap_closed,
    generate_from_gist,
    gist_forward,
    roll_spans,
    three_ppls,
    to_leaf_param,
    trainable_param_names,
)


def test_cross_doc_spans_never_donates_from_own_document():
    from marker.gist_model import cross_doc_spans

    docs = [
        [([1], [10]), ([2], [20])],  # doc 0 spans: [1],[2]
        [([3], [30])],  # doc 1 spans: [3]
        [([4], [40]), ([5], [50]), ([6], [60])],  # doc 2
    ]
    donors = cross_doc_spans(docs)
    assert len(donors) == 6  # aligned with the flattened pair list
    # doc 0's pairs borrow from doc 1; doc 1 from doc 2; doc 2 from doc 0
    assert donors[0] == [3] and donors[1] == [3]
    assert donors[2] == [4]
    assert donors[3] == [1] and donors[4] == [2] and donors[5] == [1]
    # none of the donors comes from the pair's own document
    flat_own_docs = [0, 0, 1, 2, 2, 2]
    for idx, d in enumerate(donors):
        own_spans = [p[0] for p in docs[flat_own_docs[idx]]]
        assert d not in own_spans, f"pair {idx} borrowed from its own doc"


def test_roll_spans_permutes_all_positions():
    # every continuation gets a DIFFERENT span (no fixed point) for n>=2
    spans = [[1], [2], [3]]
    rolled = roll_spans(spans)
    assert rolled == [[3], [1], [2]]
    assert all(r != o for r, o in zip(rolled, spans, strict=True))


# ── to_leaf_param: the GPU-only "non-leaf Tensor" optimizer crash ───────────────


def test_moved_param_is_non_leaf_but_to_leaf_param_fixes_it():
    p = torch.nn.Parameter(torch.randn(4, 8))
    # a device/dtype move returns a NON-leaf tensor that AdamW rejects (CPU
    # .to(cpu) is a no-op, so force it with a dtype move to reproduce on CPU)
    moved = p.to(torch.float64)
    assert not moved.is_leaf
    with pytest.raises(ValueError, match="non-leaf"):
        torch.optim.AdamW([moved])
    # to_leaf_param re-wraps as a leaf -> optimizer accepts it
    fixed = to_leaf_param(p, torch.device("cpu"))
    assert fixed.is_leaf and fixed.requires_grad
    torch.optim.AdamW([fixed])  # no raise


# ── model-free: gap_closed arithmetic ───────────────────────────────────────────


def test_gap_closed_full_match():
    assert gap_closed({"none": 100.0, "full": 10.0, "gist": 10.0}) == pytest.approx(1.0)


def test_gap_closed_no_help():
    assert gap_closed({"none": 100.0, "full": 10.0, "gist": 100.0}) == pytest.approx(0.0)


def test_gap_closed_half():
    assert gap_closed({"none": 100.0, "full": 20.0, "gist": 60.0}) == pytest.approx(0.5)


def test_gap_closed_degenerate_gap():
    # full not better than none -> gap undefined -> 0.0, no divide-by-zero
    assert gap_closed({"none": 10.0, "full": 10.0, "gist": 5.0}) == 0.0


# ── slow: tiny real model ───────────────────────────────────────────────────────


def _tiny_base():
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(0)
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
    model = AutoModelForCausalLM.from_config(cfg, attn_implementation="sdpa")
    return model.eval()


@pytest.mark.slow
def test_assert_attn_impl_rejects_flash():
    m = _tiny_base()
    assert_attn_impl(m)  # sdpa ok
    m.config._attn_implementation = "flash_attention_2"
    with pytest.raises(ValueError, match="4D masks"):
        assert_attn_impl(m)


@pytest.mark.slow
def test_only_lora_and_gist_are_trainable():
    base = _tiny_base()
    peft_model, gist = attach_gist(base, gist_k=4, r=4)
    trainable = trainable_param_names(peft_model)
    assert trainable, "no trainable params"
    assert all("lora" in n.lower() for n in trainable), (
        f"non-LoRA base param is trainable: {[n for n in trainable if 'lora' not in n.lower()]}"
    )
    assert gist.requires_grad


@pytest.mark.slow
def test_forward_finite_loss():
    base = _tiny_base()
    peft_model, gist = attach_gist(base, gist_k=4, r=4)
    loss = gist_forward(peft_model, gist, [[1, 2, 3], [4, 5]], [[6, 7, 8], [9, 10]])
    assert torch.isfinite(loss)


@pytest.mark.slow
def test_loss_decreases_over_steps():
    base = _tiny_base()
    peft_model, gist = attach_gist(base, gist_k=4, r=4)
    params = [p for p in peft_model.parameters() if p.requires_grad] + [gist]
    opt = torch.optim.AdamW(params, lr=1e-2)
    spans, conts = [[1, 2, 3, 4]], [[5, 6, 7, 8]]
    first = last = None
    for i in range(25):
        opt.zero_grad()
        loss = gist_forward(peft_model, gist, spans, conts)
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.item()
        last = loss.item()
    assert last < first, f"loss did not decrease: {first:.3f} -> {last:.3f}"


@pytest.mark.slow
def test_adapter_save_load_round_trip(tmp_path):
    # The resume path: save_bundle -> fresh model -> set_peft_model_state_dict
    # (NOT load_adapter, which raises on the existing 'default' name — Fable
    # pre-launch finding #1). Loss on the same batch must match exactly.
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    from marker.hf_push import save_bundle

    spans, conts = [[1, 2, 3, 4]], [[5, 6, 7, 8]]

    base = _tiny_base()
    pm, gist = attach_gist(base, gist_k=4, r=4)
    params = [p for p in pm.parameters() if p.requires_grad] + [gist]
    opt = torch.optim.AdamW(params, lr=1e-2)
    for _ in range(5):  # train so LoRA weights are nonzero (init B=0 is trivial)
        opt.zero_grad()
        gist_forward(pm, gist, spans, conts).backward()
        opt.step()
    save_bundle(tmp_path, pm, gist, {"step": 5})

    base2 = _tiny_base()  # same seed -> identical base
    pm2, gist2 = attach_gist(base2, gist_k=4, r=4)
    adapter_state = load_file(str(tmp_path / "adapter_model.safetensors"))
    set_peft_model_state_dict(pm2, adapter_state)
    gist2.data = load_file(str(tmp_path / "gist.safetensors"))["gist"]

    with torch.no_grad():
        l1 = gist_forward(pm, gist, spans, conts)
        l2 = gist_forward(pm2, gist2, spans, conts)
    assert torch.isclose(l1, l2, atol=1e-5), f"resume mismatch: {l1} vs {l2}"


@pytest.mark.slow
def test_encode_gist_shape_and_span_dependence():
    from marker.gist_model import encode_gist

    base = _tiny_base()
    pm, gist = attach_gist(base, gist_k=4, r=4)
    g = encode_gist(pm, gist, [[1, 2, 3], [4, 5]])
    assert g.shape == (2, 4, base.config.hidden_size)  # [B, k, hidden]
    # the gist encodes the span: a different span gives different gist vectors
    g2 = encode_gist(pm, gist, [[9, 8, 7], [4, 5]])
    assert not torch.allclose(g[0], g2[0], atol=1e-4)  # span 0 changed
    assert torch.allclose(g[1], g2[1], atol=1e-4)  # span 1 unchanged


@pytest.mark.slow
def test_gist_kv_extracts_per_layer_kv_at_gist_positions():
    # Stage-3 decode path: the continuation attends to the gist positions'
    # per-layer K/V (encode_gist returns only the top-layer readout). gist_kv
    # slices the k gist positions from every layer's cache -> an injectable
    # AxiomKV + the training-geometry continuation start + the last-gist-
    # position logits (the train-faithful first-token predictor).
    from marker.gist_model import gist_kv
    from marker.run_axiom_mlp_demo import _build_dynamic_cache

    base = _tiny_base()
    pm, gist = attach_gist(base, gist_k=4, r=4)
    kv, cont_start, first_logits = gist_kv(pm, gist, [1, 2, 3])
    assert cont_start == 3 + 4  # continuation resumes at span_len + k (training layout)
    assert first_logits.shape == (base.config.vocab_size,)
    assert kv.n_layers == base.config.num_hidden_layers
    for kmat, vmat in zip(kv.keys, kv.values, strict=True):
        # [B=1, n_kv_heads, k=4 gist positions, head_dim]
        assert kmat.shape[0] == 1 and kmat.shape[2] == 4
        assert vmat.shape[2] == 4
        assert kmat.shape[1] == base.config.num_key_value_heads
    # span-specificity enters at DEPTH: layer-0 gist K/V is just the projected
    # (constant) gist embeddings — span-independent — and only becomes
    # span-specific at deeper layers once the gist positions have attended to
    # the span. So the last layer must differ across spans; layer 0 must not.
    kv2, _, _ = gist_kv(pm, gist, [9, 8, 7])
    assert torch.allclose(kv.keys[0], kv2.keys[0], atol=1e-4)  # layer 0: span-independent
    assert not torch.allclose(kv.keys[-1], kv2.keys[-1], atol=1e-4)  # last layer: span-specific
    # injectable through the existing runtime (builds a DynamicCache, no raise)
    cache = _build_dynamic_cache(kv, torch.device("cpu"))
    assert cache is not None


@pytest.mark.slow
def test_cache_decode_is_logit_parity_with_training_forward():
    # THE Stage-3 injection gold test (Fable review): decoding from the
    # injected gist KV must reproduce — to float tolerance — the logits of a
    # full training-style [span|gist|cont] forward with the training mask
    # (cont_sees={'gist'}). Parity proves the cache path is faithful to
    # training; it catches positional-geometry bugs (gist keys are RoPE'd at
    # [span_len, span_len+k), so continuation MUST decode from span_len+k,
    # not from the cache length) that a run-and-no-crash test cannot.
    from marker.gist import build_batch_mask
    from marker.gist_model import gist_kv
    from marker.run_axiom_mlp_demo import _build_dynamic_cache

    base = _tiny_base()
    pm, gist = attach_gist(base, gist_k=4, r=4)
    span, cont = [1, 2, 3], [5, 6, 7]
    k, max_s, max_c = 4, len(span), len(cont)

    # full training-style forward: [span | gist | cont], cont sees gist only
    embed = pm.get_input_embeddings()
    span_e = embed(torch.tensor([span]))
    cont_e = embed(torch.tensor([cont]))
    gist_e = gist.to(span_e.dtype).unsqueeze(0)
    inp = torch.cat([span_e, gist_e, cont_e], dim=1)
    mask = build_batch_mask(
        [max_s], [max_c], k, max_s, max_c, cont_sees=frozenset({"gist"}), dtype=inp.dtype
    )
    pos = torch.arange(inp.shape[1]).unsqueeze(0)
    with torch.no_grad():
        full = pm(inputs_embeds=inp, attention_mask=mask, position_ids=pos).logits[0]

    # cache path: inject gist KV, decode cont with explicit training positions
    kv, cont_start, first_logits = gist_kv(pm, gist, span)
    assert torch.allclose(first_logits, full[max_s + k - 1], atol=1e-4), (
        "first-token logits (last gist position) diverge from the training forward"
    )
    cache = _build_dynamic_cache(kv, torch.device("cpu"))
    with torch.no_grad():
        out = pm(
            torch.tensor([cont]),
            past_key_values=cache,
            position_ids=torch.arange(cont_start, cont_start + max_c).unsqueeze(0),
            use_cache=True,
        )
    # cache logits at cont position j must match the full forward at [max_s+k+j]
    assert torch.allclose(out.logits[0], full[max_s + k :], atol=1e-4), (
        f"cache-decode diverges from training forward: "
        f"max|Δ|={float((out.logits[0] - full[max_s + k :]).abs().max()):.5f}"
    )


@pytest.mark.slow
def test_decode_from_gist_kv_greedy_matches_manual_rollout():
    # decode_from_gist_kv's greedy loop must equal a manual training-geometry
    # rollout: first token = argmax(first_logits), then feed each token at
    # explicit positions cont_start, cont_start+1, ... over the injected cache.
    from marker.gist_model import decode_from_gist_kv, gist_kv
    from marker.run_axiom_mlp_demo import _build_dynamic_cache

    base = _tiny_base()
    pm, gist = attach_gist(base, gist_k=4, r=4)
    kv, cont_start, first_logits = gist_kv(pm, gist, [1, 2, 3])
    gen = decode_from_gist_kv(pm, kv, cont_start, first_logits, max_new=4)
    assert 1 <= len(gen) <= 4 and all(isinstance(t, int) for t in gen)

    # manual rollout
    want = [int(first_logits.argmax())]
    cache = _build_dynamic_cache(kv, torch.device("cpu"))
    past = cache
    with torch.no_grad():
        for j in range(3):
            out = pm(
                torch.tensor([[want[-1]]]),
                past_key_values=past,
                position_ids=torch.tensor([[cont_start + j]]),
                use_cache=True,
            )
            past = out.past_key_values
            want.append(int(out.logits[0, -1].argmax()))
    assert gen == want[: len(gen)]


@pytest.mark.slow
def test_nll_under_gist_kv_matches_training_forward():
    # the PPL-based ceiling metric: teacher-forced NLL of the continuation
    # under the injected gist KV must equal the gist-only NLL from a full
    # [span|gist|cont] training forward (same parity guarantee as the logit
    # test, now on the loss the gist was trained with).
    import torch.nn.functional as F  # noqa: N812

    from marker.gist import build_batch_mask
    from marker.gist_model import gist_kv, nll_under_gist_kv

    base = _tiny_base()
    pm, gist = attach_gist(base, gist_k=4, r=4)
    span, cont = [1, 2, 3], [5, 6, 7]
    k, max_s, max_c = 4, len(span), len(cont)

    embed = pm.get_input_embeddings()
    inp = torch.cat(
        [
            embed(torch.tensor([span])),
            gist.to(embed.weight.dtype).unsqueeze(0),
            embed(torch.tensor([cont])),
        ],
        dim=1,
    )
    mask = build_batch_mask(
        [max_s], [max_c], k, max_s, max_c, cont_sees=frozenset({"gist"}), dtype=inp.dtype
    )
    pos = torch.arange(inp.shape[1]).unsqueeze(0)
    with torch.no_grad():
        logits = pm(inputs_embeds=inp, attention_mask=mask, position_ids=pos).logits[0]
    # positions predicting cont: last gist position + each cont position but last
    pred = logits[max_s + k - 1 : max_s + k - 1 + max_c]
    want = F.cross_entropy(pred, torch.tensor(cont)).item()

    kv, cont_start, first_logits = gist_kv(pm, gist, span)
    got = nll_under_gist_kv(pm, kv, cont_start, first_logits, cont)
    assert abs(got - want) < 1e-4, f"NLL parity broken: {got} vs {want}"


@pytest.mark.slow
def test_decode_from_gist_kv_respects_stop_ids():
    from marker.gist_model import decode_from_gist_kv, gist_kv

    base = _tiny_base()
    pm, gist = attach_gist(base, gist_k=4, r=4)
    kv, cont_start, first_logits = gist_kv(pm, gist, [1, 2, 3])
    full = decode_from_gist_kv(pm, kv, cont_start, first_logits, max_new=6)
    stopped = decode_from_gist_kv(
        pm, kv, cont_start, first_logits, max_new=6, stop_ids={full[0]}
    )
    assert stopped == [full[0]]  # halts on the stop token (inclusive)


@pytest.mark.slow
def test_generate_from_gist_runs_and_respects_max_new():
    base = _tiny_base()
    pm, gist = attach_gist(base, gist_k=4, r=4)
    gen = generate_from_gist(pm, gist, [1, 2, 3], max_new=5)
    assert 1 <= len(gen) <= 5
    assert all(isinstance(t, int) for t in gen)


@pytest.mark.slow
def test_three_ppls_direction_full_le_none():
    base = _tiny_base()
    peft_model, gist = attach_gist(base, gist_k=4, r=4)
    # train a little so 'full' (raw span visible) genuinely beats 'none'
    params = [p for p in peft_model.parameters() if p.requires_grad] + [gist]
    opt = torch.optim.AdamW(params, lr=1e-2)
    spans, conts = [[1, 2, 3, 4]], [[5, 6, 7, 8]]
    for _ in range(30):
        opt.zero_grad()
        gist_forward(
            peft_model, gist, spans, conts, cont_sees=frozenset({"gist", "span"})
        ).backward()
        opt.step()
    ppls = three_ppls(peft_model, gist, spans, conts)
    assert all(torch.isfinite(torch.tensor(v)) for v in ppls.values())
    assert ppls["full"] <= ppls["none"] + 1e-3, ppls
