"""Round-trip test for single-vector axiom save/load.

A single-vector axiom is (seed embedding row + per-layer bolt-on adapters).
Saving and reloading onto a fresh model must reproduce the exact same forward
behavior — the trained seed vector and adapter weights survive the trip, and
the seed token is re-registered so its id resolves correctly.
"""

from __future__ import annotations

import pytest
import torch


@pytest.fixture
def fresh_model():
    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "Qwen/Qwen2.5-0.5B"
    try:
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).eval()
    except Exception as e:
        pytest.skip(f"could not load {name}: {e}")
    return model, tokenizer


def _logits(model, tokenizer, text):
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    with torch.no_grad():
        return model(ids).logits.clone()


def test_save_load_round_trip(fresh_model, tmp_path):
    """Save a trained-ish single-vector axiom, reload into a clean model, and
    confirm logits on a seeded prompt match within tight tolerance."""
    from marker.bolt_selector import install_bolt_hooks, make_bolt_selector, remove_bolt_hooks
    from marker.seed_token import register_seed_token, seed_embedding
    from marker.single_vector_store import (
        load_single_vector_axiom,
        save_single_vector_axiom,
    )

    model, tokenizer = fresh_model
    seed = register_seed_token(model, tokenizer, "BalancePublisher")
    bolt = make_bolt_selector(model, seed, r=8, skill_mode=False)

    # Simulate "trained" state: nonzero seed vector + nonzero adapter weights.
    with torch.no_grad():
        seed_embedding(model, seed).copy_(torch.full_like(seed_embedding(model, seed), 0.05))
        for ad in bolt.adapters:
            ad.up.weight.fill_(0.001)
            ad.down.weight.fill_(0.002)

    prompt = "Q: Tell me about <BalancePublisher>.\nA:"
    handles = install_bolt_hooks(model, bolt)
    try:
        ref_logits = _logits(model, tokenizer, prompt)
    finally:
        remove_bolt_hooks(handles)

    path = tmp_path / "BalancePublisher.pt"
    save_single_vector_axiom(model, seed, bolt, path)
    assert path.exists()

    # Fresh model + tokenizer — nothing carried over in memory.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "Qwen/Qwen2.5-0.5B"
    tok2 = AutoTokenizer.from_pretrained(name)
    mdl2 = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).eval()

    seed2, bolt2 = load_single_vector_axiom(path, mdl2, tok2)
    assert seed2.name == "BalancePublisher"
    assert bolt2.skill_mode is False
    assert len(bolt2.adapters) == mdl2.config.num_hidden_layers

    handles2 = install_bolt_hooks(mdl2, bolt2)
    try:
        loaded_logits = _logits(mdl2, tok2, prompt)
    finally:
        remove_bolt_hooks(handles2)

    assert torch.allclose(ref_logits, loaded_logits, atol=1e-5), (
        f"round-trip logits diverged: max diff {(ref_logits - loaded_logits).abs().max().item()}"
    )


def test_skill_mode_flag_persists(fresh_model, tmp_path):
    """skill_mode must survive the save/load round trip."""
    from marker.bolt_selector import make_bolt_selector
    from marker.seed_token import register_seed_token
    from marker.single_vector_store import (
        load_single_vector_axiom,
        save_single_vector_axiom,
    )

    model, tokenizer = fresh_model
    seed = register_seed_token(model, tokenizer, "ilp_for")
    bolt = make_bolt_selector(model, seed, r=8, skill_mode=True)

    path = tmp_path / "ilp_for.pt"
    save_single_vector_axiom(model, seed, bolt, path)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "Qwen/Qwen2.5-0.5B"
    tok2 = AutoTokenizer.from_pretrained(name)
    mdl2 = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).eval()

    _seed2, bolt2 = load_single_vector_axiom(path, mdl2, tok2)
    assert bolt2.skill_mode is True
