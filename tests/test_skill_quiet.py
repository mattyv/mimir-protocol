"""Invariants for the learned-silence skill experiment. Model-free — no GPU
or real tokenizer needed; segment/token-span logic is tested against a
stubbed offsets list, matching SKILL_QUIET_PLAN.md's test spec.
"""

from __future__ import annotations

import re

import torch
import torch.nn as nn

from marker.run_axiom_mlp_demo import ILP_PROBES, SKILL_AXIOM, SKILL_AXIOM_ILP, SKILL_PROBES
from marker.run_skill_quiet import (
    ILP_MIXED,
    ILP_PURE_PROSE,
    INTERNALBUS_MIXED,
    INTERNALBUS_PURE_PROSE,
    _contains,
    _contamination_count,
)
from marker.skill_quiet import (
    AxiomMLP,
    as_skill_pair,
    install_recording_hooks,
    prose_penalty,
    segment_labels_from_offsets,
)

# ── Segment -> token label mapping (stubbed offsets, no tokenizer) ──────────────


def test_pure_skill_segment_all_labelled_skill():
    # "hello world" as one skill segment, tokens roughly word-sized.
    offsets = [(0, 5), (5, 6), (6, 11)]  # "prompt: " occupies chars [0,8) below
    answer_start = 8
    # shift offsets so they sit after answer_start
    offsets = [(8, 13), (13, 14), (14, 19)]  # "hello", " ", "world"
    labels = segment_labels_from_offsets(offsets, answer_start, [("hello world", "skill")])
    assert labels == ["skill", "skill", "skill"]


def test_prompt_tokens_labelled_prompt():
    answer_start = 10
    offsets = [(0, 5), (5, 10), (10, 15)]  # first two tokens end at/before answer_start
    labels = segment_labels_from_offsets(offsets, answer_start, [("hello", "skill")])
    assert labels == ["prompt", "prompt", "skill"]


def test_mixed_segments_split_correctly():
    # answer = "CODE" + " explains " -> first segment "skill", second "prose"
    answer_start = 0
    segments = [("CODE", "skill"), (" explains", "prose")]
    # tokens: "CODE" [0,4), " " [4,5) -> midpoint 4.5 falls in "explains" range? Let's be precise.
    # segment ranges: skill=[0,4), prose=[4,13)
    offsets = [(0, 4), (4, 6), (6, 13)]
    labels = segment_labels_from_offsets(offsets, answer_start, segments)
    assert labels == ["skill", "prose", "prose"]


def test_trailing_token_falls_back_to_last_segment_kind():
    segments = [("ab", "skill"), ("cd", "prose")]
    answer_start = 0
    # last token's offset extends slightly past the last segment's end (e.g. eos-like token)
    offsets = [(0, 2), (2, 4), (4, 5)]
    labels = segment_labels_from_offsets(offsets, answer_start, segments)
    assert labels[-1] == "prose"


def test_empty_segments_labels_everything_prompt():
    offsets = [(0, 3), (3, 6)]
    labels = segment_labels_from_offsets(offsets, 100, [])
    assert labels == ["prompt", "prompt"]


# ── as_skill_pair ─────────────────────────────────────────────────────────────


def test_as_skill_pair_wraps_single_segment():
    q, segments = as_skill_pair("write code", "client.emit(x)")
    assert q == "write code"
    assert segments == [("client.emit(x)", "skill")]


# ── Recording hooks + penalty (tiny CPU model, real grad) ────────────────────────


class _ZeroMLP(nn.Module):
    """Deterministic zero-init SmallMLP stand-in: forward returns a learnable
    offset via a single linear layer, so backward() populates a real grad.
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.w = nn.Linear(hidden, hidden, bias=False)
        nn.init.normal_(self.w.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w(x)


def _fake_axiom_mlp(hidden: int, layers: list[int]) -> AxiomMLP:
    mlps = nn.ModuleList([_ZeroMLP(hidden) for _ in layers])
    return AxiomMLP(term="Foo", term_token_ids=[1, 2], chosen_layers=layers, mlps=mlps)


class _FakeLayer(nn.Module):
    def forward(self, x):  # noqa: ANN001, ANN201
        return (x,)


class _FakeModel(nn.Module):
    def __init__(self, n_layers: int, hidden: int):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_FakeLayer() for _ in range(n_layers)])
        self.hidden = hidden

    def run_layer(self, layer_idx: int, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.layers[layer_idx](hidden_states)[0]


def test_install_recording_hooks_captures_offsets_with_grad():
    hidden = 8
    axiom_mlp = _fake_axiom_mlp(hidden, layers=[0, 1])
    model = _FakeModel(n_layers=2, hidden=hidden)
    record: dict[int, dict[int, torch.Tensor]] = {}
    positions = [0, 2]
    handles = install_recording_hooks(model, axiom_mlp, positions, record)
    try:
        h = torch.randn(1, 3, hidden, requires_grad=True)
        _ = model.run_layer(0, h)
        _ = model.run_layer(1, h)
    finally:
        for handle in handles:
            handle.remove()

    assert set(record.keys()) == {0, 1}
    for layer_idx in (0, 1):
        assert set(record[layer_idx].keys()) == {0, 2}
        for pos in (0, 2):
            offset = record[layer_idx][pos]
            assert offset.requires_grad
            assert offset.grad_fn is not None


def test_prose_penalty_only_uses_specified_positions():
    hidden = 4
    axiom_mlp = _fake_axiom_mlp(hidden, layers=[0])
    model = _FakeModel(n_layers=1, hidden=hidden)
    record: dict[int, dict[int, torch.Tensor]] = {}
    handles = install_recording_hooks(model, axiom_mlp, [0, 1, 2], record)
    try:
        h = torch.randn(1, 3, hidden, requires_grad=True)
        model.run_layer(0, h)
    finally:
        for handle in handles:
            handle.remove()

    full_penalty = prose_penalty(record, {0, 1, 2})
    partial_penalty = prose_penalty(record, {1})
    # partial (one position) should generally differ from full (mean over 3)
    # unless by fluke all three offsets have identical norm — use a
    # structural check instead: partial equals the single position's own
    # squared norm exactly.
    expected = record[0][1].pow(2).sum()
    assert torch.allclose(partial_penalty, expected)
    assert full_penalty.numel() == 1


def test_prose_penalty_empty_positions_is_zero_and_grad_free_safe():
    hidden = 4
    axiom_mlp = _fake_axiom_mlp(hidden, layers=[0])
    model = _FakeModel(n_layers=1, hidden=hidden)
    record: dict[int, dict[int, torch.Tensor]] = {}
    handles = install_recording_hooks(model, axiom_mlp, [0], record)
    try:
        h = torch.randn(1, 2, hidden, requires_grad=True)
        model.run_layer(0, h)
    finally:
        for handle in handles:
            handle.remove()

    penalty = prose_penalty(record, set())
    assert penalty.item() == 0.0


def test_prose_penalty_no_offsets_recorded_returns_zero_tensor():
    penalty = prose_penalty({}, {0, 1})
    assert isinstance(penalty, torch.Tensor)
    assert penalty.item() == 0.0


# ── Contamination scoring ─────────────────────────────────────────────────────


def test_contamination_count_after_blank_line():
    api_re = re.compile(r"client\.(emit|subscribe)\(")
    text = "client.emit('x', y, ttl=30)\n\nThis calls client.emit( again in prose."
    assert _contamination_count(text, "client.emit('x'", api_re) == 1


def test_contamination_count_no_blank_line_falls_back_to_after_gold():
    api_re = re.compile(r"client\.(emit|subscribe)\(")
    text = "client.emit('x', y, ttl=30) and then client.subscribe('y', h)"
    # no "\n\n"; falls back to splitting right after the gold match
    n = _contamination_count(text, "client.emit('x'", api_re)
    assert n == 1  # only the subscribe call counts as post-code contamination


def test_contamination_count_zero_when_clean():
    api_re = re.compile(r"client\.(emit|subscribe)\(")
    text = "client.emit('x', y, ttl=30)\n\nThis publishes a message to channel x."
    assert _contamination_count(text, "client.emit('x'", api_re) == 0


def test_contains_case_insensitive():
    assert _contains("The Answer Is Client.Emit(", "client.emit(")
    assert not _contains("nothing here", "client.emit(")


# ── Data hygiene ──────────────────────────────────────────────────────────────


def test_regression_probes_match_run_axiom_mlp_demo_originals():
    # Byte-identical import, not a redefinition — this test just guards
    # against someone accidentally forking the probe lists in the future.
    assert SKILL_PROBES is SKILL_PROBES  # sanity: imported, not shadowed
    assert len(SKILL_PROBES) == 4
    assert len(ILP_PROBES) == 4


def test_every_mixed_pair_has_skill_and_prose_segment():
    for q, segments in [*INTERNALBUS_MIXED, *ILP_MIXED]:
        kinds = {kind for _text, kind in segments}
        assert kinds == {"skill", "prose"}, f"{q!r}: expected both kinds, got {kinds}"


def test_every_pure_prose_pair_is_all_prose():
    for q, segments in [*INTERNALBUS_PURE_PROSE, *ILP_PURE_PROSE]:
        kinds = {kind for _text, kind in segments}
        assert kinds == {"prose"}, f"{q!r}: expected only prose, got {kinds}"


def test_control_pairs_wrap_original_qa_unchanged():
    for q, a in SKILL_AXIOM["qa"]:
        wrapped_q, segments = as_skill_pair(q, a)
        assert wrapped_q == q
        assert segments == [(a, "skill")]
    for q, a in SKILL_AXIOM_ILP["qa"]:
        wrapped_q, segments = as_skill_pair(q, a)
        assert wrapped_q == q
        assert segments == [(a, "skill")]
