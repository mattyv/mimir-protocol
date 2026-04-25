"""Mechanical invariants for the capture/inject hook.

These tests assert the *plumbing* — not the experiment outcome. They guard
against regressions that would silently invalidate every downstream result:

- A zero-vector injection must be a no-op (otherwise the hook's add path is
  broken or has a side effect).
- A non-zero injection must change the logits (otherwise the hook isn't
  actually wired to the residual stream).
- Capture must return a tensor of the right shape at the configured layer
  (otherwise we're capturing the wrong thing — wrong layer, wrong position,
  wrong dtype).
"""

import numpy as np
import pytest
import torch

from poc.hooks import HookedModel

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
LAYER = 8
D_MODEL = 768  # GPT-2 small


@pytest.fixture(scope="module")
def hooked() -> HookedModel:
    torch.manual_seed(0)
    np.random.seed(0)
    return HookedModel(model_name="gpt2", layer=LAYER, device=DEVICE)


def test_capture_shape(hooked: HookedModel) -> None:
    h = hooked.capture("Hello world")
    assert h.shape == (D_MODEL,), f"expected ({D_MODEL},), got {h.shape}"
    assert h.dtype == np.float32


def test_zero_inject_is_noop(hooked: HookedModel) -> None:
    prompt = "The quick brown fox"
    base = hooked.logits_at(prompt, vec=None, alpha=0.0)
    zero = np.zeros(D_MODEL, dtype=np.float32)
    shifted = hooked.logits_at(prompt, vec=zero, alpha=1.0)
    assert np.allclose(base, shifted, atol=1e-5), (
        f"zero-vec injection changed logits by max {np.abs(base - shifted).max():.2e}"
    )


def test_random_inject_changes_logits(hooked: HookedModel) -> None:
    prompt = "The quick brown fox"
    base = hooked.logits_at(prompt, vec=None, alpha=0.0)
    rng = np.random.default_rng(42)
    rand = rng.standard_normal(D_MODEL).astype(np.float32)
    rand /= np.linalg.norm(rand)
    shifted = hooked.logits_at(prompt, vec=rand, alpha=5.0)
    assert not np.allclose(base, shifted, atol=1e-3), (
        "random injection at α=5 produced no change — hook is not wired"
    )


def test_inject_state_does_not_leak_between_calls(hooked: HookedModel) -> None:
    """After an injecting call, the next call with vec=None must be clean."""
    prompt = "The quick brown fox"
    rng = np.random.default_rng(1)
    rand = rng.standard_normal(D_MODEL).astype(np.float32)

    base_before = hooked.logits_at(prompt, vec=None, alpha=0.0)
    _ = hooked.logits_at(prompt, vec=rand, alpha=5.0)
    base_after = hooked.logits_at(prompt, vec=None, alpha=0.0)

    assert np.allclose(base_before, base_after, atol=1e-5), (
        "injection state leaked into a subsequent vec=None call"
    )


def test_capture_is_deterministic(hooked: HookedModel) -> None:
    h1 = hooked.capture("Determinism check")
    h2 = hooked.capture("Determinism check")
    assert np.allclose(h1, h2, atol=1e-6)


def test_capture_layers_returns_requested_layers(hooked: HookedModel) -> None:
    out = hooked.capture_layers("Hello world", layers=[4, 6, 8, 10])
    assert set(out.keys()) == {4, 6, 8, 10}
    for layer, h in out.items():
        assert h.shape == (D_MODEL,), f"layer {layer}: shape {h.shape}"
        assert h.dtype == np.float32


def test_capture_layers_matches_hook_capture(hooked: HookedModel) -> None:
    """capture() (hook-based, layer 8) should match capture_layers()[8]
    (hidden_states-based) — both read the residual stream after block 8."""
    prompt = "Consistency between capture paths"
    h_hook = hooked.capture(prompt)
    h_hs = hooked.capture_layers(prompt, layers=[LAYER])[LAYER]
    assert np.allclose(h_hook, h_hs, atol=1e-4), (
        f"hook vs hidden_states diverge by max {np.abs(h_hook - h_hs).max():.2e}"
    )


def test_generate_extends_prompt_by_n_tokens(hooked: HookedModel) -> None:
    prompt = "Once upon a time"
    n = 5
    out = hooked.generate(prompt, vec=None, alpha=0.0, n=n)
    base_ids = hooked.tok(prompt, add_special_tokens=False).input_ids
    full_ids = hooked.tok(out, add_special_tokens=False).input_ids
    assert len(full_ids) == len(base_ids) + n, (
        f"expected {len(base_ids) + n} tokens, got {len(full_ids)}: {out!r}"
    )


def test_generate_is_deterministic(hooked: HookedModel) -> None:
    prompt = "The capital of France is"
    a = hooked.generate(prompt, vec=None, alpha=0.0, n=5)
    b = hooked.generate(prompt, vec=None, alpha=0.0, n=5)
    assert a == b


def test_inject_at_position_default_matches_last(hooked: HookedModel) -> None:
    """logits_at without an inject_position argument must match the previous
    behaviour (inject at last token)."""
    prompt = "The quick brown fox jumps over"
    rng = np.random.default_rng(0)
    vec = rng.standard_normal(D_MODEL).astype(np.float32)
    vec /= np.linalg.norm(vec)
    a = hooked.logits_at(prompt, vec=vec, alpha=2.0)
    b = hooked.logits_at(prompt, vec=vec, alpha=2.0, inject_position=-1)
    assert np.allclose(a, b, atol=1e-6)


def test_inject_at_different_positions_diverges(hooked: HookedModel) -> None:
    """Injecting at position 1 and at the last position must produce different
    logits — otherwise position is not actually being honoured."""
    prompt = "The quick brown fox jumps over"
    rng = np.random.default_rng(0)
    vec = rng.standard_normal(D_MODEL).astype(np.float32)
    vec /= np.linalg.norm(vec)
    a = hooked.logits_at(prompt, vec=vec, alpha=5.0, inject_position=1)
    b = hooked.logits_at(prompt, vec=vec, alpha=5.0, inject_position=-1)
    assert not np.allclose(a, b, atol=1e-3)


def test_inject_at_explicit_position_zero_vec_is_noop(hooked: HookedModel) -> None:
    prompt = "The quick brown fox"
    base = hooked.logits_at(prompt, vec=None, alpha=0.0)
    zero = np.zeros(D_MODEL, dtype=np.float32)
    shifted = hooked.logits_at(prompt, vec=zero, alpha=1.0, inject_position=2)
    assert np.allclose(base, shifted, atol=1e-5)


def test_log_probs_at_returns_normalised_distribution(hooked: HookedModel) -> None:
    prompt = "The quick brown fox"
    log_probs = hooked.log_probs_at(prompt, vec=None, alpha=0.0)
    # exp of log-probs should sum to ~1.
    total = float(np.exp(log_probs).sum())
    assert abs(total - 1.0) < 1e-4, f"log-probs do not normalise: sum={total}"


def test_inject_at_position_list_matches_single(hooked: HookedModel) -> None:
    """Passing [pos] should match passing pos directly."""
    prompt = "The quick brown fox jumps over"
    rng = np.random.default_rng(0)
    vec = rng.standard_normal(D_MODEL).astype(np.float32)
    vec /= np.linalg.norm(vec)
    a = hooked.logits_at(prompt, vec=vec, alpha=2.0, inject_position=2)
    b = hooked.logits_at(prompt, vec=vec, alpha=2.0, inject_position=[2])
    assert np.allclose(a, b, atol=1e-6)


def test_inject_at_multiple_positions_diverges_from_single(hooked: HookedModel) -> None:
    """Injecting at [2, -1] should differ from injecting only at 2."""
    prompt = "The quick brown fox jumps over"
    rng = np.random.default_rng(0)
    vec = rng.standard_normal(D_MODEL).astype(np.float32)
    vec /= np.linalg.norm(vec)
    single = hooked.logits_at(prompt, vec=vec, alpha=2.0, inject_position=2)
    multi = hooked.logits_at(prompt, vec=vec, alpha=2.0, inject_position=[2, -1])
    assert not np.allclose(single, multi, atol=1e-3)


def test_log_prob_shift_zero_for_uniform_logit_tilt(hooked: HookedModel) -> None:
    """If injection only adds a constant to all logits, log-probs are
    invariant. We test the inverse: a real injection that produces logit
    shifts should produce *some* nonzero log-prob shift somewhere in vocab.
    This guards against accidentally implementing a metric that washes out
    real signal."""
    prompt = "The capital of France is"
    rng = np.random.default_rng(0)
    vec = rng.standard_normal(D_MODEL).astype(np.float32)
    vec /= np.linalg.norm(vec)
    base = hooked.log_probs_at(prompt, vec=None, alpha=0.0)
    shifted = hooked.log_probs_at(prompt, vec=vec, alpha=10.0)
    max_abs_shift = float(np.abs(shifted - base).max())
    assert max_abs_shift > 0.01, f"injection produced no log-prob shift; max={max_abs_shift}"


def test_capture_at_position_returns_correct_shape(hooked: HookedModel) -> None:
    h = hooked.capture_at_position("Hello world from a python test", layer=LAYER, position=0)
    assert h.shape == (D_MODEL,)
    assert h.dtype == np.float32


def test_capture_at_last_position_matches_capture_layers(hooked: HookedModel) -> None:
    prompt = "consistency between position-aware and last-token capture"
    h_last = hooked.capture_layers(prompt, layers=[LAYER])[LAYER]
    h_pos = hooked.capture_at_position(prompt, layer=LAYER, position=-1)
    assert np.allclose(h_last, h_pos, atol=1e-5)


def test_find_token_positions_locates_acronym(hooked: HookedModel) -> None:
    """JOTP encodes to 3 BPE tokens; find_token_positions returns the last."""
    prompt = "The team relied on JOTP daily and JOTP weekly."
    positions = hooked.find_token_positions(prompt, "JOTP")
    assert len(positions) == 2, f"expected 2 occurrences, got {positions}"
    decoded_at = [
        hooked.tok.decode(hooked.tok(prompt, add_special_tokens=False).input_ids[: p + 1])
        for p in positions
    ]
    assert all("JOTP" in d for d in decoded_at), decoded_at


def test_generate_with_large_alpha_eventually_perturbs(hooked: HookedModel) -> None:
    """Under a large-enough α, greedy output must differ from baseline.
    This is the only way to confirm injection is actually active during
    generation (and not just on the first step)."""
    prompt = "The capital of France is"
    rng = np.random.default_rng(7)
    vec = rng.standard_normal(D_MODEL).astype(np.float32)
    vec /= np.linalg.norm(vec)
    base = hooked.generate(prompt, vec=None, alpha=0.0, n=5)
    shifted = hooked.generate(prompt, vec=vec, alpha=200.0, n=5)
    assert base != shifted
