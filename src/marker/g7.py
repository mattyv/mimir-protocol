"""G7 sequence layout, two-gather loss, and generation-time slot masking (see
GIST_LM_PLAN.md "STAGES 2+3 DESIGN v3").

Consumes the stage-2 corpus schema (owned by gist_tokenizer.py /
run_tokenize_corpus.py, NOT built here): one JSON record per solution --
{"src", "doc_id", "question", "steps": [str], "ids": [[[8 ints] per group]
per step], "answer", "n_groups"} -- `ids` in NATURAL slot order (index s is
dictionary slot s's id; the SLOT_ORDER permutation is a LAYOUT/emission
detail applied here, never baked into the stored corpus).

Layout: `[question] <think> g^(1)..g^(m) <commit> [step m text] <eos>` -- ONE
<think>, ONE <commit>, gist groups for steps 1..m run back to back (never
interleave text between them), then the text of step m (the step being
committed) is scored as ordinary next-token text, including <eos>. No loss on
the question, <think> or <commit> as PREDICTION TARGETS.
"""

from __future__ import annotations

import random

import torch

from marker.gist_vocab import N_SLOTS, SLOT_ORDER, slot_for_position


def sample_m(n: int, rng: random.Random) -> list[int]:
    """The two-per-epoch rule: m=n (commit the final step) plus one more m
    drawn uniformly from {1..n-1} (an earlier commit point, so the model
    sees mid-chain commits, not only whole-solution ones). n==1 has no
    earlier step, so only [1] is returned."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if n == 1:
        return [1]
    return [n, rng.randint(1, n - 1)]


def _flat(base_vocab: int, K: int, s: int, j: int) -> int:
    return base_vocab + s * K + j


def build_sequence(record: dict, m: int, tok, base_vocab: int, K: int) -> dict:  # noqa: ANN001
    """One training sequence committing step `m` of `record`. Returns
    dict(input_ids, gist_mask, text_mask, slot_of_pos) -- four equal-length
    lists. `gist_mask[i]`/`text_mask[i]` is True iff input_ids[i+1] (the
    token position i's logit is scored against) is a gist id / a post-commit
    text id; both are False at the question, <think> and <commit> targets.
    `slot_of_pos[i]` is the NATURAL slot of the gist token AT position i (-1
    elsewhere) -- callers needing the target's slot for a gist_mask position
    i read `slot_of_pos[i + 1]`.

    Deviation from the build order's literal 3-arg signature
    `build_sequence(record, m, tok)`: base_vocab/K are threaded in explicitly
    (never hardcoded -- matches the build order's own "BASE_VOCAB ... read
    from the tokenizer/model config at runtime" instruction) since a plain
    HF tokenizer carries neither."""
    steps = record["steps"]
    if not (1 <= m <= len(steps)):
        raise ValueError(f"m={m} out of range for {len(steps)} steps")
    think_id = base_vocab + N_SLOTS * K
    commit_id = think_id + 1

    q_ids = list(tok(record["question"], add_special_tokens=False).input_ids)
    input_ids = q_ids + [think_id]
    slot_of_pos = [-1] * len(input_ids)

    for step_groups in record["ids"][:m]:
        for group in step_groups:
            if len(group) != N_SLOTS:
                raise ValueError(f"group must carry {N_SLOTS} ids, got {len(group)}")
            for t in range(N_SLOTS):
                s = SLOT_ORDER[t]
                input_ids.append(_flat(base_vocab, K, s, group[s]))
                slot_of_pos.append(s)

    input_ids.append(commit_id)
    slot_of_pos.append(-1)

    step_text_ids = list(tok(steps[m - 1], add_special_tokens=False).input_ids)
    eos_id = tok.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer has no eos_token_id")
    input_ids.extend(step_text_ids)
    input_ids.append(eos_id)
    slot_of_pos.extend([-1] * (len(step_text_ids) + 1))

    n = len(input_ids)
    text_start = n - (len(step_text_ids) + 1)  # index of the first text/eos token
    gist_mask = [False] * n
    text_mask = [False] * n
    for i in range(n - 1):
        tgt_pos = i + 1
        if tgt_pos >= text_start:
            text_mask[i] = True
        elif input_ids[tgt_pos] not in (think_id, commit_id) and input_ids[tgt_pos] >= base_vocab:
            gist_mask[i] = True

    return {
        "input_ids": input_ids,
        "gist_mask": gist_mask,
        "text_mask": text_mask,
        "slot_of_pos": slot_of_pos,
    }


def parse_sequence(ids: list[int], base_vocab: int, K: int) -> tuple[list[list[int]], list[int]]:
    """Inverse of the [<think> gist-groups <commit> text] portion of
    build_sequence: ids (the FULL sequence, or just the post-question tail --
    only the first <think>/<commit> pair found is used) -> (groups, text_ids)
    where groups[i] is step (i+1)'s NATURAL-order 8-id group and text_ids is
    everything after <commit> (including <eos> if present)."""
    think_id = base_vocab + N_SLOTS * K
    commit_id = think_id + 1
    if think_id not in ids:
        raise ValueError("sequence has no <think>")
    t_pos = ids.index(think_id)
    if commit_id not in ids[t_pos:]:
        raise ValueError("sequence has no <commit> after <think>")
    c_pos = ids.index(commit_id, t_pos)
    gist_span = ids[t_pos + 1 : c_pos]
    if len(gist_span) % N_SLOTS != 0:
        raise ValueError(f"gist span length {len(gist_span)} is not a multiple of {N_SLOTS}")

    groups = []
    for g0 in range(0, len(gist_span), N_SLOTS):
        block = gist_span[g0 : g0 + N_SLOTS]
        natural = [0] * N_SLOTS
        for t, flat in enumerate(block):
            s = SLOT_ORDER[t]
            j = flat - base_vocab - s * K
            if not (0 <= j < K):
                raise ValueError(f"id {flat} at layout position {t} is not slot {s}'s block")
            natural[s] = j
        groups.append(natural)
    return groups, ids[c_pos + 1 :]


def two_gather_loss(
    head,  # noqa: ANN001 -- a GistHeadWrapper (base_head + vocab)
    hidden: torch.Tensor,
    targets: torch.Tensor,
    gist_mask: torch.Tensor,
    text_mask: torch.Tensor,
    slot_of_pos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Next-token CE over positions flattened across batch/time (caller's
    job to flatten). Two gathers, never one full (base_vocab + 8K + 2)-wide
    softmax:
    - text: base_head(hidden) CE at text_mask positions.
    - gist: CE from ONLY the target's own K-wide slot block (h @
      rows[slot].T + bias[slot], rows/bias sliced from
      head.vocab.output_rows_all()) at gist_mask positions.
    slot_of_pos[i] must be the NATURAL slot of targets[i] wherever
    gist_mask[i] is True (build_sequence's convention: slot of the token AT
    position i+1, read at index i+1 -- pass `slot_of_pos[1:]` alongside
    `targets[1:]`/logits `hidden[:-1]` the way a caller normally would for
    next-token scoring). Returns (loss, gist_ce, text_ce); loss is the mean
    over both position sets combined."""
    import torch.nn.functional as func

    device = hidden.device
    targets = targets.to(device)
    gist_mask = gist_mask.to(device=device, dtype=torch.bool)
    text_mask = text_mask.to(device=device, dtype=torch.bool)
    slot_of_pos = slot_of_pos.to(device)

    zero = hidden.new_tensor(0.0)
    n_text = int(text_mask.sum())
    text_ce = zero
    if n_text:
        # .float(): CE in bf16 over a 150k-wide (text) or 4096-wide (gist)
        # softmax adds ~1e-2 noise -- cheap to do the log-softmax in fp32.
        text_logits = head.base_head(hidden[text_mask]).float()
        text_ce = func.cross_entropy(text_logits, targets[text_mask])

    n_gist = int(gist_mask.sum())
    gist_ce = zero
    if n_gist:
        vocab = head.vocab
        rows, bias = vocab.output_rows_all()
        gist_hidden = hidden[gist_mask]
        gist_targets = targets[gist_mask]
        slots = slot_of_pos[gist_mask]
        losses = []
        for s in torch.unique(slots).tolist():
            sel = slots == s
            block_rows = rows[s * vocab.K : (s + 1) * vocab.K].to(gist_hidden.dtype)
            block_bias = bias[s * vocab.K : (s + 1) * vocab.K].to(gist_hidden.dtype)
            block_logits = (gist_hidden[sel] @ block_rows.T + block_bias).float()
            local_target = gist_targets[sel] - (vocab.base_vocab + s * vocab.K)
            losses.append(func.cross_entropy(block_logits, local_target, reduction="sum"))
        gist_ce = torch.stack(losses).sum() / n_gist

    n_total = n_text + n_gist
    if n_total == 0:
        raise ValueError(
            "two_gather_loss: no scored positions (gist_mask and text_mask both all-False)"
        )
    loss = (text_ce * n_text + gist_ce * n_gist) / n_total
    return loss, gist_ce, text_ce


class GistLogitsProcessor:
    """HF LogitsProcessor: masks generation to the schedule
    `<think> [8 * think_budget_steps gist ids, slot-masked] <commit> [text]`.
    After <think>: for the current gist position `t` (0-indexed since
    <think>), allow only ids in slot_for_position(t)'s K-wide block; once
    `8 * think_budget_steps` gist ids have been emitted, allow only
    <commit> (forced commit at budget). After <commit>: allow only text ids
    (< base_vocab) -- the caller's stopping criteria (max_new_tokens /
    eos_token_id) end the text phase; this processor only masks logits, it
    never emits a token itself."""

    def __init__(self, think_budget_steps: int, base_vocab: int, K: int) -> None:
        if think_budget_steps < 1:
            raise ValueError("think_budget_steps must be >= 1")
        self.think_budget_steps = think_budget_steps
        self.base_vocab = base_vocab
        self.K = K
        self.think_id = base_vocab + N_SLOTS * K
        self.commit_id = self.think_id + 1

    def _mask_row(self, seq: list[int], row: torch.Tensor) -> torch.Tensor:
        neg_inf = torch.finfo(row.dtype).min
        masked = torch.full_like(row, neg_inf)
        if self.commit_id in seq:
            masked[: self.base_vocab] = row[: self.base_vocab]
            return masked
        if self.think_id not in seq:
            raise ValueError("GistLogitsProcessor requires <think> already in the prompt")
        think_pos = len(seq) - 1 - seq[::-1].index(self.think_id)
        n_emitted = len(seq) - 1 - think_pos
        total_budget = self.think_budget_steps * N_SLOTS
        if n_emitted >= total_budget:
            masked[self.commit_id] = row[self.commit_id]
            return masked
        slot = slot_for_position(n_emitted)
        lo, hi = self.base_vocab + slot * self.K, self.base_vocab + (slot + 1) * self.K
        masked[lo:hi] = row[lo:hi]
        return masked

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        out = scores.clone()
        for b in range(input_ids.shape[0]):
            out[b] = self._mask_row(input_ids[b].tolist(), out[b])
        return out
