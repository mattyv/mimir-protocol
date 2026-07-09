"""Stage-1 gist compression: attention mask, labels, batch construction.

See GIST_PILOT_PLAN.md. Layout of one training example:

    [ span S (<=64 subwords) ] [ g1 .. gk gist slots ] [ continuation C ]

The gist slots are a BOTTLENECK: they attend to S, and C attends only to the
gist (never to S directly). So S's information reaches C exclusively through
the k gist KVs — that is what makes the gist a compressed stand-in for S.

Correctness note (correcting GIST_PILOT_PLAN's stated invariant): the plan
said "randomize S ⇒ C logits unchanged, else the mask leaks." That is
BACKWARDS. Because the gist attends to S and C attends to the gist, S reaches
C *through the gist by design* — randomizing S SHOULD move C's logits. C
being invariant to S would mean the gist carries nothing (a dead bottleneck).
The real tests are:
  (1) C never attends to S DIRECTLY — assert the mask values (model-free).
  (2) the gist actually connects S→C — C logits DO move when S changes.
  (3) no DIRECT leak — under a diagnostic mask that ALSO blocks gist→S, the
      only remaining S→C path is the (bug) direct one, so randomizing S must
      then leave C unchanged.
See tests/test_gist.py.
"""

from __future__ import annotations

import re

import torch

_NEG = float("-inf")


# ── Attention mask ───────────────────────────────────────────────────────────────


def build_attention_mask(
    s: int, k: int, c: int, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Additive [1, 1, T, T] mask (0 = attend, -inf = block) for the
    span/gist/continuation layout, T = s + k + c. Positions:
        S   = [0, s)          causal within S
        gist= [s, s+k)        attends to all of S + causal among gist
        C   = [s+k, T)        attends to gist + causal within C; S is BLOCKED

    The mask is causal EVERYWHERE except one extra rule: C→S is blocked (pure
    causality would allow it — that block is the whole bottleneck)."""
    t = s + k + c
    q = torch.arange(t).unsqueeze(1)
    key = torch.arange(t).unsqueeze(0)
    allowed = key <= q  # causal base

    # Block C (query >= s+k) from attending to S (key < s).
    c_rows = q >= (s + k)
    s_cols = key < s
    allowed = allowed & ~(c_rows & s_cols)

    mask = torch.zeros(t, t, dtype=dtype)
    mask.masked_fill_(~allowed, _NEG)
    return mask.unsqueeze(0).unsqueeze(0)


def build_leak_diagnostic_mask(
    s: int, k: int, c: int, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """The training mask, ADDITIONALLY blocking gist→S. This severs the
    legitimate S→gist→C route, leaving only the (bug) direct C→S path. Under
    this mask, C logits MUST be invariant to S; if they move, C is leaking to
    S directly. Diagnostic only — never used for training."""
    mask = build_attention_mask(s, k, c, dtype).clone()
    # gist queries [s, s+k) must not attend to S keys [0, s)
    mask[0, 0, s : s + k, 0:s] = _NEG
    return mask


def build_labels(s: int, k: int, cont_ids: list[int]) -> torch.Tensor:
    """[1, T] label tensor: -100 everywhere except the continuation positions,
    which carry cont_ids. CE is computed on C only."""
    c = len(cont_ids)
    labels = torch.full((1, s + k + c), -100, dtype=torch.long)
    labels[0, s + k :] = torch.tensor(cont_ids, dtype=torch.long)
    return labels


def gist_position_ids(s: int, k: int, c: int) -> torch.Tensor:
    """Contiguous [1, T] positions — the gist slots occupy real sequence
    positions between S and C (RoPE sees a normal monotone sequence)."""
    return torch.arange(s + k + c).unsqueeze(0)


# ── Sentence pairing (data prep, model-free) ─────────────────────────────────────

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Cheap sentence split (fallback when blingfire is absent). Splits on
    .!? followed by whitespace; drops empties."""
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]


def make_pair(
    tokenizer,  # noqa: ANN001
    sentence: str,
    continuation_text: str,
    max_span: int,
    max_cont: int,
    min_cont: int,
) -> tuple[list[int], list[int]] | None:
    """Tokenize (span, continuation). Span capped at max_span subwords;
    continuation capped at max_cont and required to be at least min_cont
    tokens (short tails are dropped -> returns None)."""
    span = tokenizer(sentence, add_special_tokens=False).input_ids[:max_span]
    cont = tokenizer(continuation_text, add_special_tokens=False).input_ids[:max_cont]
    if len(span) == 0 or len(cont) < min_cont:
        return None
    return span, cont
