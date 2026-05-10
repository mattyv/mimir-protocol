"""Mechanical invariants for APE (Adaptive Parallel Encoding).

APE adds three dials to fix attention-entropy collapse at 3+ stacked
prefixes (Yang et al ICLR 2025, arxiv 2502.05431):

  1. Shared prefix prepended before all axioms (one consistent
     attention-sink).
  2. Q-scale: multiply Q by a scalar to sharpen / quiet attention. In
     this v1 we collapse APE's separate temperature T and scale S into
     one knob `q_scale`. q_scale > 1 → sharper, q_scale < 1 → softer,
     q_scale == 1 → no-op (must equal vanilla forward).

These tests assert mechanical invariants. Whether APE actually fixes
the 3-prefix loop is the demo's job, not these tests.
"""

from __future__ import annotations

import hashlib

import pytest
import torch


@pytest.fixture(scope="module")
def tiny_model():
    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "Qwen/Qwen2.5-0.5B"
    try:
        tok = AutoTokenizer.from_pretrained(name)
        mdl = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).eval()
    except Exception as e:
        pytest.skip(f"could not load {name}: {e}")
    return mdl, tok


def _state_checksum(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


# ---------- q_scale hook ----------


def test_q_scale_hook_no_op_at_one(tiny_model):
    """q_scale=1.0 must produce byte-identical logits to vanilla forward."""
    from transformers.cache_utils import DynamicCache

    from marker.ape import install_q_scale_hook

    model, tok = tiny_model
    ids = tok("Hello world test sentence here.", return_tensors="pt", add_special_tokens=False)
    ids = ids.input_ids
    with torch.no_grad():
        ref = model(ids, past_key_values=DynamicCache(), use_cache=True)

    handles = install_q_scale_hook(model, q_scale=1.0)
    try:
        with torch.no_grad():
            out = model(ids, past_key_values=DynamicCache(), use_cache=True)
    finally:
        for h in handles:
            h.remove()
    assert torch.allclose(out.logits, ref.logits, atol=1e-5), (
        "q_scale=1.0 should be a no-op vs vanilla forward"
    )


def test_q_scale_hook_changes_logits(tiny_model):
    """q_scale != 1.0 must produce different logits than vanilla. Proves
    the hook is actually doing something. (Whether scaling Q sharpens the
    softmax is a math identity: softmax(c*x) is sharper than softmax(x)
    for c>1 — no need to assert it via captured attentions, which sdpa
    doesn't expose.)
    """
    from transformers.cache_utils import DynamicCache

    from marker.ape import install_q_scale_hook

    model, tok = tiny_model
    ids = tok("Hello world test sentence here.", return_tensors="pt", add_special_tokens=False)
    ids = ids.input_ids
    with torch.no_grad():
        ref = model(ids, past_key_values=DynamicCache(), use_cache=True)

    handles = install_q_scale_hook(model, q_scale=3.0)
    try:
        with torch.no_grad():
            out = model(ids, past_key_values=DynamicCache(), use_cache=True)
    finally:
        for h in handles:
            h.remove()
    diff = (out.logits - ref.logits).abs().max().item()
    assert diff > 1e-2, f"q_scale=3.0 should change logits noticeably; max diff = {diff:.6f}"


def test_q_scale_hook_does_not_change_weights(tiny_model):
    """Hook installation/removal must leave model weights byte-identical."""
    from marker.ape import install_q_scale_hook

    model, _ = tiny_model
    before = _state_checksum(model)
    handles = install_q_scale_hook(model, q_scale=2.0)
    for h in handles:
        h.remove()
    after = _state_checksum(model)
    assert before == after, "q_scale hook changed model weights"


# ---------- generate_with_ape integration ----------


def test_generate_with_ape_runs_three_prefix(tiny_model):
    """Sanity rail: tiny model, 3 axioms, APE active → produces output
    without crashing. Numerical correctness is the demo's job.
    """
    from marker.ape import generate_with_ape
    from marker.prefix_tuning import Prefix

    model, tok = tiny_model
    layers = list(range(model.config.num_hidden_layers))
    prefixes = [
        Prefix.from_description(model, tok, txt, target_layers=layers)
        for txt in [
            "Alpha publishes balances to Kafka every 250ms.",
            "Beta consumes balances and computes margin.",
            "Gamma signs orders if margin is sufficient.",
        ]
    ]

    out = generate_with_ape(
        model=model,
        tokenizer=tok,
        prompt="What does Alpha do?",
        prefixes=prefixes,
        shared_prefix_text="System: answer concisely.",
        q_scale=1.5,
        max_new=10,
    )
    assert isinstance(out, str)


def test_generate_with_ape_q_scale_one_matches_no_shared_prefix(tiny_model):
    """When q_scale=1.0 AND shared_prefix_text="" (empty), APE must equal
    the existing generate_with_prefixes output exactly.
    """
    from marker.ape import generate_with_ape
    from marker.prefix_tuning import Prefix, generate_with_prefixes

    model, tok = tiny_model
    layers = list(range(model.config.num_hidden_layers))
    prefixes = [
        Prefix.from_description(model, tok, "Alpha publishes balances.", target_layers=layers),
        Prefix.from_description(model, tok, "Beta consumes them.", target_layers=layers),
    ]
    prompt = "What does Alpha do?"

    ref = generate_with_prefixes(model, tok, prompt, prefixes, max_new=8, rope_correct=True)
    out = generate_with_ape(
        model=model,
        tokenizer=tok,
        prompt=prompt,
        prefixes=prefixes,
        shared_prefix_text="",
        q_scale=1.0,
        max_new=8,
    )
    assert out == ref, f"APE no-op mode should equal vanilla: {out!r} vs {ref!r}"
