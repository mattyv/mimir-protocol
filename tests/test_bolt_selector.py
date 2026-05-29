"""Mechanical invariants and CPU smoke-training test for BoltSelector.

The bolt-on selector is a per-layer low-rank adapter added in parallel to each
transformer block. It is gated on the seed token id being present in the
input — for any forward pass without the seed token, the bolt-on is invisible.

Six tests assert the wiring (zero-init no-op, gating, gradient flow, restore)
and one slow CPU smoke test verifies that loss decreases when training
seed + bolt-on jointly on real BalancePublisher Q+A pairs. Quality of the
trained model is NOT asserted here — that requires a 7B run.
"""

from __future__ import annotations

import pytest
import torch


@pytest.fixture
def tiny_model():
    """Fresh Qwen 2.5-0.5B + tokenizer per test. Both tests in this file
    mutate the tokenizer (add_tokens) and the model (resize embeddings,
    install hooks), so isolation is required."""
    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "Qwen/Qwen2.5-0.5B"
    try:
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).eval()
    except Exception as e:
        pytest.skip(f"could not load {name}: {e}")
    return model, tokenizer


def _logits_for(model, tokenizer, text):
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    with torch.no_grad():
        return model(ids).logits.clone()


def _with_seed(model, tokenizer, name="BalancePublisher"):
    from marker.seed_token import register_seed_token

    seed = register_seed_token(model, tokenizer, name)
    return seed


# ───────────────────────────────────────────────────────────── unit invariants


def test_pass_count_matches_layer_count(tiny_model):
    """One adapter per transformer block."""
    from marker.bolt_selector import make_bolt_selector

    model, tokenizer = tiny_model
    seed = _with_seed(model, tokenizer)
    bolt = make_bolt_selector(model, seed, r=4)
    assert len(bolt.adapters) == model.config.num_hidden_layers


def test_zero_init_is_no_op(tiny_model):
    """With bolt-on installed but zero-init (up.weight all zero), the residual
    addition is exactly zero. Logits must match the hookless forward."""
    from marker.bolt_selector import install_bolt_hooks, make_bolt_selector

    model, tokenizer = tiny_model
    seed = _with_seed(model, tokenizer)

    # Baseline: no hooks
    prompt = f"Tell me about <{seed.name}>."
    baseline = _logits_for(model, tokenizer, prompt)

    bolt = make_bolt_selector(model, seed, r=4)
    handles = install_bolt_hooks(model, bolt)
    try:
        with_hooks = _logits_for(model, tokenizer, prompt)
    finally:
        for h in handles:
            h.remove()

    assert torch.equal(baseline, with_hooks), (
        "zero-init bolt-on changed logits — up.weight not zero-init?"
    )


def test_bolt_does_not_fire_without_seed_in_input(tiny_model):
    """A prompt that does not contain the seed token must produce baseline
    logits even with bolt hooks installed AND adapter weights perturbed."""
    from marker.bolt_selector import install_bolt_hooks, make_bolt_selector

    model, tokenizer = tiny_model
    seed = _with_seed(model, tokenizer)

    # Sanity: this prompt contains nothing about the seed
    prompt = "The cat sat on the mat."
    seed_ids = tokenizer.encode(f"<{seed.name}>", add_special_tokens=False)
    assert seed_ids[0] not in tokenizer.encode(prompt, add_special_tokens=False)

    baseline = _logits_for(model, tokenizer, prompt)

    bolt = make_bolt_selector(model, seed, r=4)
    # Perturb so a non-gating implementation would visibly diverge
    with torch.no_grad():
        for ad in bolt.adapters:
            ad.up.weight.fill_(0.01)
    handles = install_bolt_hooks(model, bolt)
    try:
        with_hooks = _logits_for(model, tokenizer, prompt)
    finally:
        for h in handles:
            h.remove()

    assert torch.equal(baseline, with_hooks), (
        "bolt-on fired on a prompt without the seed token — gating broken"
    )


def test_bolt_fires_when_seed_present_after_perturbation(tiny_model):
    """After perturbing the up projection, a prompt containing the seed token
    must produce logits that differ from the hookless baseline."""
    from marker.bolt_selector import install_bolt_hooks, make_bolt_selector

    model, tokenizer = tiny_model
    seed = _with_seed(model, tokenizer)

    prompt = f"Tell me about <{seed.name}>."
    baseline = _logits_for(model, tokenizer, prompt)

    bolt = make_bolt_selector(model, seed, r=4)
    with torch.no_grad():
        for ad in bolt.adapters:
            ad.up.weight.fill_(0.01)
    handles = install_bolt_hooks(model, bolt)
    try:
        with_hooks = _logits_for(model, tokenizer, prompt)
    finally:
        for h in handles:
            h.remove()

    assert not torch.equal(baseline, with_hooks), (
        "perturbed bolt-on did not change logits on a seeded prompt"
    )


def test_gradient_flows_only_to_bolt_and_seed(tiny_model):
    """Backward on a seeded forward pass must put gradient on (a) at least
    one bolt adapter parameter per layer and (b) the seed row of the
    embedding. No other model parameter may end up with nonzero grad.

    Note: up.weight is zero-init so down.weight gets ZERO grad on the very
    first backward (chain back to down passes through up·0 = 0). We run two
    forward+backward passes with one optimizer step between, so by the
    second pass up is nonzero and both projections receive grad.
    """
    from marker.bolt_selector import bolt_parameters, install_bolt_hooks, make_bolt_selector

    model, tokenizer = tiny_model
    seed = _with_seed(model, tokenizer)

    bolt = make_bolt_selector(model, seed, r=4)
    assert bolt.adapters[0].down.weight.abs().sum() > 0  # sanity: random init

    handles = install_bolt_hooks(model, bolt)
    try:
        prompt = f"Tell me about <{seed.name}>."
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids

        # Pass 1: zero-init up means down.weight will have zero grad. We
        # still want to confirm up.weight gets a real gradient, so do the
        # first backward and check.
        model(ids).logits.sum().backward()
        assert bolt.adapters[0].up.weight.grad.abs().sum().item() > 0

        # Manually set up.weight to a small nonzero value so the chain back
        # to down.weight is now alive. (We avoid using an optimizer step
        # here because the raw logits.sum() gradient is ~1e7 and any real
        # step would explode the activations on the next forward.) Clear
        # grads after manual perturbation; do a fresh backward.
        with torch.no_grad():
            for ad in bolt.adapters:
                ad.up.weight.fill_(1e-4)
        for p in [
            *bolt_parameters(bolt),
            model.get_input_embeddings().weight,
        ]:
            if p.grad is not None:
                p.grad = None

        # Pass 2: both up and down should now receive grad
        model(ids).logits.sum().backward()
    finally:
        for h in handles:
            h.remove()

    # Both projections in every adapter receive grad on the second pass
    for i, ad in enumerate(bolt.adapters):
        for pname, p in ad.named_parameters():
            assert p.grad is not None, f"adapter {i}.{pname}: grad is None"
            assert p.grad.abs().sum().item() > 0, (
                f"adapter {i}.{pname}: grad is zero (expected nonzero)"
            )

    # Seed row of embedding has gradient (Phase 1 invariant)
    embed = model.get_input_embeddings()
    assert embed.weight.grad is not None
    assert embed.weight.grad[seed.token_id].abs().sum().item() > 0, (
        "seed row grad is zero — seed embedding not learning"
    )

    # Other rows of embedding still zero (Phase 1 grad mask)
    for other_id in (0, 1, 100, 1000):
        if other_id != seed.token_id:
            assert embed.weight.grad[other_id].abs().sum().item() == 0, (
                f"embed row {other_id} got grad; Phase 1 mask broken"
            )

    # No other model param has nonzero grad. Walk all parameters NOT in the
    # bolt and NOT the (tied) embedding.
    bolt_param_ids = {id(p) for ad in bolt.adapters for p in ad.parameters()}
    embed_ptr = embed.weight.data_ptr()
    for name, p in model.named_parameters():
        if id(p) in bolt_param_ids:
            continue
        if p.data_ptr() == embed_ptr:
            continue
        if p.grad is None:
            continue
        assert p.grad.abs().sum().item() == 0, (
            f"model parameter {name} got nonzero grad (bolt or seed should be the only learners)"
        )


def test_hook_remove_is_full_restore(tiny_model):
    """After remove_bolt_hooks, forward output must be byte-equal to the
    pre-install baseline — proving the hooks are cleanly detached."""
    from marker.bolt_selector import install_bolt_hooks, make_bolt_selector

    model, tokenizer = tiny_model
    seed = _with_seed(model, tokenizer)

    prompt = f"Tell me about <{seed.name}>."
    baseline = _logits_for(model, tokenizer, prompt)

    bolt = make_bolt_selector(model, seed, r=4)
    # Perturb so any leftover hook would produce visibly different output
    with torch.no_grad():
        for ad in bolt.adapters:
            ad.up.weight.fill_(0.01)

    handles = install_bolt_hooks(model, bolt)
    for h in handles:
        h.remove()

    after_remove = _logits_for(model, tokenizer, prompt)
    assert torch.equal(baseline, after_remove), (
        "logits differ after hooks removed — hook lifecycle leaked state"
    )


def test_skill_mode_keeps_firing_on_decode_steps(tiny_model):
    """Skill mode: once the seed token has been seen in a prefill pass, the
    bolt-on must keep firing on subsequent single-token decode passes (which
    do NOT contain the seed token). Fact mode must fall silent on those same
    decode passes.

    We simulate the prefill→decode sequence by calling the embedding pre-hook
    directly with prefill-shaped then decode-shaped input ids, and inspecting
    the resulting `_fire_this_pass` flag."""
    from marker.bolt_selector import _make_embedding_pre_hook, make_bolt_selector

    model, tokenizer = tiny_model
    seed = _with_seed(model, tokenizer)

    prefill_ids = torch.tensor([[1, 2, seed.token_id, 3]])  # contains seed
    decode_ids = torch.tensor([[42]])  # a normal token, no seed

    # Fact mode: fires on prefill (seed present), silent on decode.
    fact_bolt = make_bolt_selector(model, seed, r=4)
    assert fact_bolt.skill_mode is False
    fact_hook = _make_embedding_pre_hook(fact_bolt)
    fact_hook(None, (prefill_ids,))
    assert fact_bolt._fire_this_pass is True, "fact mode should fire on seeded prefill"
    fact_hook(None, (decode_ids,))
    assert fact_bolt._fire_this_pass is False, "fact mode should be silent on decode step"

    # Skill mode: fires on prefill, STAYS firing through decode steps.
    skill_bolt = make_bolt_selector(model, seed, r=4, skill_mode=True)
    assert skill_bolt.skill_mode is True
    skill_hook = _make_embedding_pre_hook(skill_bolt)
    skill_hook(None, (prefill_ids,))
    assert skill_bolt._fire_this_pass is True, "skill mode should fire on seeded prefill"
    skill_hook(None, (decode_ids,))
    assert skill_bolt._fire_this_pass is True, "skill mode should keep firing on decode step"
    # A second decode step still fires.
    skill_hook(None, (decode_ids,))
    assert skill_bolt._fire_this_pass is True, "skill mode should keep firing across decode steps"


def test_skill_mode_does_not_fire_if_seed_never_seen(tiny_model):
    """Skill mode must not fire on a decode step if the seed token was never
    in the prefill — a generation about something else should stay clean."""
    from marker.bolt_selector import _make_embedding_pre_hook, make_bolt_selector

    model, tokenizer = tiny_model
    seed = _with_seed(model, tokenizer)

    prefill_no_seed = torch.tensor([[1, 2, 3, 4]])  # no seed
    decode_ids = torch.tensor([[42]])

    skill_bolt = make_bolt_selector(model, seed, r=4, skill_mode=True)
    skill_hook = _make_embedding_pre_hook(skill_bolt)
    skill_hook(None, (prefill_no_seed,))
    assert skill_bolt._fire_this_pass is False, "skill mode fired with no seed in prefill"
    skill_hook(None, (decode_ids,))
    assert skill_bolt._fire_this_pass is False, "skill mode latched on without ever seeing seed"


# ─────────────────────────────────────────────────────── CPU smoke training


@pytest.mark.slow
def test_loss_decreases_on_balance_publisher(tiny_model):
    """End-to-end learnability: training seed + bolt jointly on real
    BalancePublisher Q+A pairs must drive the cross-entropy loss meaningfully
    down over ~300 CPU steps. We assert mean(last 10) < 0.5 * mean(first 10).
    This is not a quality claim — only that the architecture is learnable."""
    from marker.bolt_selector import make_bolt_selector, train_seed_and_bolt
    from marker.run_soft_prompt_plus_v4_demo import TEST_AXIOMS

    model, tokenizer = tiny_model
    seed = _with_seed(model, tokenizer)
    bolt = make_bolt_selector(model, seed, r=16)

    # Build Q+A pairs from BalancePublisher facts. Replace the literal term
    # in each question with the seed token so the gate fires every step.
    axiom = TEST_AXIOMS[0]
    assert axiom["name"] == "BalancePublisher"
    seed_token_str = f"<{seed.name}>"
    qa_pairs: list[tuple[str, str]] = []
    for f in axiom["facts"]:
        for q in f["questions_train"]:
            q_with_seed = q.replace("BalancePublisher", seed_token_str)
            qa_pairs.append((q_with_seed, f["answer"]))

    losses = train_seed_and_bolt(
        model,
        tokenizer,
        seed,
        bolt,
        qa_pairs,
        n_steps=300,
        lr=1e-3,
    )

    assert len(losses) >= 20
    first_window = sum(losses[:10]) / 10
    last_window = sum(losses[-10:]) / 10
    assert last_window < 0.5 * first_window, (
        f"loss did not decrease meaningfully: first10={first_window:.3f} last10={last_window:.3f}"
    )
