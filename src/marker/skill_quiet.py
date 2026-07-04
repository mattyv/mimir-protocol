"""Teach skill axioms to disengage on prose ("learned silence").

See SKILL_QUIET_PLAN.md for the full design and pre-registered pass/kill
criteria.

Problem: skill_mode fires the skill MLP at every decode step from trigger
until EOS (run_axiom_mlp_demo.generate_with_mlp / .train). Works for
single-artifact answers (CE runs through EOS, so "finish the pattern, then
stop" is learned). Breaks on mixed answers ("write the call, then explain
in plain English") — the MLP keeps steering through the whole explanation.

Fix: the MLP is already input-conditional (reads the residual, emits an
offset). It bleeds because training never contained a moment where the
correct output was near-zero offset. This module extends training data with
segment-labelled answers (kind = "skill" | "prose") and an explicit penalty
on offset magnitude at "prose" positions, so quietness becomes a stated
training objective rather than an emergent hope.

Does not modify run_axiom_mlp_demo.py; reuses its SmallMLP, AxiomMLP,
compute_axiom_kv, _build_dynamic_cache, _find_term_positions, TEMPLATE.
"""

from __future__ import annotations

import random

import torch

from marker.run_axiom_mlp_demo import (
    TEMPLATE,
    AxiomMLP,
    _build_dynamic_cache,
    _find_term_positions,
)

# A training pair for skill_quiet: (question, segments). segments is a list
# of (text, kind) tuples, kind in {"skill", "prose"}, whose text concatenates
# to the full answer. A pure-code pair is [(answer, "skill")]; a pure-prose
# pair is [(answer, "prose")]; a mixed pair has both.
SegmentPair = tuple[str, list[tuple[str, str]]]


def as_skill_pair(q: str, a: str) -> SegmentPair:
    """Wrap an existing (question, answer) pair as a single "skill" segment —
    how arm C (control) reuses the current pure-code training recipe unchanged.
    """
    return (q, [(a, "skill")])


# ── Segment -> token-span mapping ───────────────────────────────────────────────


def segment_labels_from_offsets(
    offsets: list[tuple[int, int]],
    answer_start_char: int,
    segments: list[tuple[str, str]],
) -> list[str]:
    """Pure logic (no tokenizer needed): given each token's (start_char,
    end_char) in the full text and the char offset where the answer begins,
    label each token "prompt" or the kind of the segment its midpoint falls
    in. Segments are assumed contiguous starting at answer_start_char.

    Testable with a stubbed offsets list — this is what tests/test_skill_quiet.py
    exercises directly, independent of tokenize_segments' real-tokenizer path.
    """
    seg_ranges: list[tuple[int, int, str]] = []
    cur = answer_start_char
    for text, kind in segments:
        seg_ranges.append((cur, cur + len(text), kind))
        cur += len(text)

    labels: list[str] = []
    for start_c, end_c in offsets:
        if end_c <= answer_start_char:
            labels.append("prompt")
            continue
        mid = (start_c + end_c) / 2
        kind = None
        for s, e, k in seg_ranges:
            if s <= mid < e:
                kind = k
                break
        if kind is None:
            # Trailing/edge token (e.g. right at the final boundary) — fall
            # back to the last segment's kind rather than mislabel as prompt.
            kind = seg_ranges[-1][2] if seg_ranges else "prompt"
        labels.append(kind)
    return labels


def tokenize_segments(
    tokenizer,  # noqa: ANN001
    q_text: str,
    segments: list[tuple[str, str]],
    device: torch.device,
) -> tuple[torch.Tensor, int, list[str]]:
    """Tokenize q_text + " " + answer_text as ONE string (matching the
    existing convention in run_axiom_mlp_demo.train — q_ids and the shared
    prefix of full_ids are assumed to tokenize identically) and label every
    token "prompt" / "skill" / "prose" via segment_labels_from_offsets.

    Returns (full_ids (1, T), q_len, token_labels (length T)).
    """
    q_ids = tokenizer(q_text, add_special_tokens=False).input_ids
    q_len = len(q_ids)

    answer_text = "".join(text for text, _ in segments)
    full_text = q_text + " " + answer_text
    enc = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    answer_start_char = len(q_text) + 1  # +1 for the separating space

    labels = segment_labels_from_offsets(enc["offset_mapping"], answer_start_char, segments)
    full_ids = torch.tensor([enc["input_ids"]], device=device)
    return full_ids, q_len, labels


# ── Recording hooks + penalty ────────────────────────────────────────────────────


def _make_recording_hook(mlp, positions: list[int], layer_record: dict[int, torch.Tensor]):  # noqa: ANN001
    """Same offset computation as run_axiom_mlp_demo._make_layer_hook, but
    also stashes each position's offset tensor (kept on-graph) into
    layer_record so the caller can compute a penalty over a subset of
    positions after the forward pass.
    """

    def hook(module, input, output):  # noqa: ANN001
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        new_hidden = hidden
        for pos in positions:
            if pos >= new_hidden.shape[1]:
                continue
            offset = mlp(new_hidden[:, pos, :].float()).to(new_hidden.dtype)
            layer_record[pos] = offset
            new_hidden = torch.cat(
                [
                    new_hidden[:, :pos, :],
                    new_hidden[:, pos : pos + 1, :] + offset.unsqueeze(1),
                    new_hidden[:, pos + 1 :, :],
                ],
                dim=1,
            )
        return (new_hidden,) + output[1:] if is_tuple else new_hidden

    return hook


def install_recording_hooks(
    model,  # noqa: ANN001
    axiom_mlp: AxiomMLP,
    positions: list[int],
    record: dict[int, dict[int, torch.Tensor]],
):
    """record[layer_idx][pos] = offset tensor, populated as hooks fire."""
    handles = []
    for layer_idx, mlp in zip(axiom_mlp.chosen_layers, axiom_mlp.mlps, strict=True):
        layer_record: dict[int, torch.Tensor] = {}
        record[layer_idx] = layer_record
        h = model.model.layers[layer_idx].register_forward_hook(
            _make_recording_hook(mlp, positions, layer_record)
        )
        handles.append(h)
    return handles


def prose_penalty(
    record: dict[int, dict[int, torch.Tensor]], prose_positions: set[int]
) -> torch.Tensor:
    """mean over prose_positions of (sum over hooked layers of ||offset||^2).

    Returns a zero tensor (not a Python 0.0) if prose_positions is empty, so
    it composes into `loss = ce + lam * penalty` without a type break — and
    stays on the right device/dtype by inheriting from any recorded offset.
    """
    if not prose_positions:
        for layer_record in record.values():
            for offset in layer_record.values():
                return offset.new_zeros(())
        return torch.zeros(())

    per_position_sums = []
    for pos in prose_positions:
        layer_sq_norms = [
            layer_record[pos].pow(2).sum()
            for layer_record in record.values()
            if pos in layer_record
        ]
        if layer_sq_norms:
            per_position_sums.append(torch.stack(layer_sq_norms).sum())
    if not per_position_sums:
        for layer_record in record.values():
            for offset in layer_record.values():
                return offset.new_zeros(())
        return torch.zeros(())
    return torch.stack(per_position_sums).mean()


# ── Training ──────────────────────────────────────────────────────────────────


def train_skill_quiet(
    model,  # noqa: ANN001
    tokenizer,  # noqa: ANN001
    axiom_mlp: AxiomMLP,
    segment_pairs: list[SegmentPair],
    n_steps: int = 3000,
    lr_start: float = 3e-5,
    lr_end: float = 3e-6,
    weight_decay: float = 0.05,
    lam: float = 0.0,
    seed: int = 42,
) -> list[float]:
    """Like run_axiom_mlp_demo.train, but answers are segment-labelled and a
    penalty on offset magnitude at "prose"-labelled answer positions is added
    to the loss (weight lam). lam=0 reduces to CE-only training (arm A);
    arm C additionally uses only as_skill_pair-wrapped pairs (no prose data
    at all) to reproduce the current recipe unchanged.
    """
    for p in model.parameters():
        p.requires_grad_(False)

    device = next(model.parameters()).device
    optim = torch.optim.AdamW(axiom_mlp.mlps.parameters(), lr=lr_start, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=n_steps, eta_min=lr_end)
    eos_id = tokenizer.eos_token_id
    rng = random.Random(seed)
    losses: list[float] = []
    skipped = 0

    for _ in range(n_steps):
        q, segments = rng.choice(segment_pairs)
        q_text = TEMPLATE.format(q=q)
        full_ids, q_len, token_labels = tokenize_segments(tokenizer, q_text, segments, device)
        if eos_id is not None:
            full_ids = torch.cat([full_ids, torch.tensor([[eos_id]], device=device)], dim=1)
            token_labels = [*token_labels, token_labels[-1] if token_labels else "skill"]

        positions = _find_term_positions(full_ids[:, :q_len], axiom_mlp.term_token_ids)
        # skill_mode always true for this module's use case, but stay defensive.
        if axiom_mlp.skill_mode:
            positions += list(range(q_len, full_ids.shape[1]))
        if not positions:
            skipped += 1
            continue

        labels = torch.full_like(full_ids, -100)
        labels[0, q_len:] = full_ids[0, q_len:]
        prose_positions = {i for i in range(q_len, full_ids.shape[1]) if token_labels[i] == "prose"}

        record: dict[int, dict[int, torch.Tensor]] = {}
        handles = install_recording_hooks(model, axiom_mlp, positions, record)
        try:
            optim.zero_grad()
            kv_cache = (
                _build_dynamic_cache(axiom_mlp.kv, device) if axiom_mlp.kv is not None else None
            )
            ce_loss = model(full_ids, past_key_values=kv_cache, labels=labels).loss
            penalty = prose_penalty(record, prose_positions)
            loss = ce_loss + lam * penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(axiom_mlp.mlps.parameters(), max_norm=1.0)
            optim.step()
            scheduler.step()
            losses.append(float(loss.item()))
        finally:
            for h in handles:
                h.remove()

    if skipped:
        print(f"  WARNING: skipped {skipped} steps (term not found in prompt)")
    if not losses:
        raise RuntimeError(
            f"all {n_steps} training steps skipped — term {axiom_mlp.term!r} "
            "never found in any training prompt"
        )
    return losses


# ── Eval: decode with per-token offset-norm trace ────────────────────────────────


@torch.no_grad()
def decode_with_norm_trace(
    model,  # noqa: ANN001
    tokenizer,  # noqa: ANN001
    prompt: str,
    axiom_mlp: AxiomMLP,
    max_new: int = 120,
) -> tuple[str, list[float]]:
    """Greedy decode with the skill's hooks active throughout (mirrors
    run_axiom_mlp_demo.generate_with_mlp's skill_mode decode loop), recording
    the mean L2 offset norm (averaged across the axiom's hooked layers) at
    every generated token position. This is the plan's star diagnostic: a
    quiet skill's trace should collapse at the code/prose boundary.

    Returns (generated_text, norm_trace) where norm_trace[i] is the mean
    offset norm at generated position i (position 0 in the single-token
    decode step, since skill hooks fire at [0] each step).
    """
    device = next(model.parameters()).device
    ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    positions = _find_term_positions(ids, axiom_mlp.term_token_ids)
    term_found = bool(positions)

    kv_cache = _build_dynamic_cache(axiom_mlp.kv, device) if axiom_mlp.kv is not None else None

    prefill_record: dict[int, dict[int, torch.Tensor]] = {}
    handles = (
        install_recording_hooks(model, axiom_mlp, positions, prefill_record) if positions else []
    )
    try:
        out = model(ids, past_key_values=kv_cache, use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
    finally:
        for h in handles:
            h.remove()

    norm_trace: list[float] = []
    out_ids = [next_tok]
    for _ in range(max_new - 1):
        step_record: dict[int, dict[int, torch.Tensor]] = {}
        decode_handles = (
            install_recording_hooks(model, axiom_mlp, [0], step_record) if term_found else []
        )
        out = model(next_tok, past_key_values=past, use_cache=True)
        for h in decode_handles:
            h.remove()

        if step_record:
            offsets_at_0 = [
                layer_record[0] for layer_record in step_record.values() if 0 in layer_record
            ]
            mean_norm = (
                torch.stack([o.norm() for o in offsets_at_0]).mean().item() if offsets_at_0 else 0.0
            )
        else:
            mean_norm = 0.0
        norm_trace.append(mean_norm)

        past = out.past_key_values
        next_tok = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        out_ids.append(next_tok)
        if int(next_tok.item()) == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(torch.cat(out_ids, dim=1)[0], skip_special_tokens=True).strip()
    return text, norm_trace
