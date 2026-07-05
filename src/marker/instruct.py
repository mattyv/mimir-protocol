"""Instruct-model portability: Phase 0 + Phase 1 machinery.

See INSTRUCT_PLAN.md. Phase 1 tests whether an anti-refusal preamble (plain
tokens) lets a chat/instruct model answer confidently from an injected axiom
KV while keeping BOUNDARY discipline — the refusal-prior hypothesis.

Layout (single axiom, Phase 1):
    [ <|im_start|>system\nAbout {term}:\n{description}\n ]   <- injected KV
                                                                (sink at pos 0)
    [ {preamble}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>    <- live tokens
      \n<|im_start|>assistant\n ]

The description rides in the KV (never in the visible prompt). The preamble is
plain live tokens, so it necessarily follows the description in Phase 1 — a
real limitation the plan notes (preamble-first needs the Phase-2 meta-KV).

Reuses run_axiom_mlp_demo.compute_axiom_kv / _build_dynamic_cache. No new
training. Everything here is inference-only.
"""

from __future__ import annotations

import re

import torch

from marker.run_axiom_mlp_demo import AxiomKV, _build_dynamic_cache, compute_axiom_kv

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

# ── Decline detection for BOUNDARY scoring ──────────────────────────────────────
_DECLINE_MARKERS = [
    "doesn't specify",
    "does not specify",
    "not specified",
    "isn't specified",
    "is not specified",
    "no information",
    "doesn't mention",
    "does not mention",
    "not mentioned",
    "does not provide",
    "doesn't provide",
    "cannot determine",
    "can't determine",
    "unable to",
    "don't have",
    "do not have",
    "no details",
    "not covered",
    "does not say",
    "doesn't say",
    "isn't covered",
]


def matches(answer: str, gold: str) -> bool:
    """Digit-boundary substring match (the run-3 scoring fix): gold '10' must
    not match inside '100'."""
    return re.search(rf"(?<!\d){re.escape(gold.lower())}(?!\d)", answer.lower()) is not None


def declined(answer: str) -> bool:
    """True if the answer declines / says the material doesn't cover it — the
    correct BOUNDARY behavior."""
    low = answer.lower()
    return any(m in low for m in _DECLINE_MARKERS)


# ── KV construction ─────────────────────────────────────────────────────────────


def chat_system_prefix(term: str, description: str) -> str:
    """The open system block that goes into the injected KV: system-start
    (attention sink) + the axiom description, left unclosed so the live
    tokens continue the same block."""
    return f"{IM_START}system\nAbout {term}:\n{description}\n"


def chat_live_suffix(question: str, preamble: str | None) -> str:
    """Live tokens after the KV: optional plain-token preamble (closing the
    system block), then the user turn and the assistant generation prompt."""
    pre = f"\n{preamble}" if preamble else ""
    return f"{pre}{IM_END}\n{IM_START}user\n{question}{IM_END}\n{IM_START}assistant\n"


def encode_chat_axiom_kv(model, tokenizer, term: str, description: str) -> AxiomKV:  # noqa: ANN001
    """Compute the KV for the open chat system block (Phase 1 axiom carrier).
    term='' path of compute_axiom_kv encodes the text verbatim (no 'About X'
    auto-prefix), which is what we want since we build the prefix ourselves."""
    return compute_axiom_kv(model, tokenizer, chat_system_prefix(term, description), term="")


def base_axiom_kv(model, tokenizer, term: str, description: str) -> AxiomKV:  # noqa: ANN001
    """Reference (base-model) carrier: the existing 'About {term}:\n{desc}'
    KV — exactly the validated base path, for apples-to-apples parity."""
    return compute_axiom_kv(model, tokenizer, description, term=term)


# ── Decode (fresh cache per call — run-3 discipline) ─────────────────────────────


@torch.no_grad()
def decode_with_kv(
    model,  # noqa: ANN001
    tokenizer,  # noqa: ANN001
    kv: AxiomKV,
    live_text: str,
    max_new: int,
    stop_id: int | None,
) -> str:
    """Greedy decode of live_text with a FRESH DynamicCache built from kv.
    stop_id lets chat runs halt on <|im_end|> rather than the base eos."""
    device = next(model.parameters()).device
    ids = tokenizer(live_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    cache = _build_dynamic_cache(kv, device)

    out = model(ids, past_key_values=cache, use_cache=True)
    past = out.past_key_values
    next_tok = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
    out_ids = [next_tok]
    eos_id = tokenizer.eos_token_id
    for _ in range(max_new - 1):
        out = model(next_tok, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        tok_id = int(next_tok.item())
        out_ids.append(next_tok)
        if tok_id == eos_id or (stop_id is not None and tok_id == stop_id):
            break
    return tokenizer.decode(torch.cat(out_ids, dim=1)[0], skip_special_tokens=True).strip()


def im_end_id(tokenizer) -> int | None:  # noqa: ANN001
    """Token id for <|im_end|> if present (chat models), else None (base)."""
    tid = tokenizer.convert_tokens_to_ids(IM_END)
    unk = tokenizer.unk_token_id
    if tid is None or (unk is not None and tid == unk):
        return None
    return tid


# ── Positional invariant (Phase 0 check, model-free) ─────────────────────────────


def injected_position_ranges(kv_len: int, live_len: int) -> tuple[range, range]:
    """The KV occupies positions [0, kv_len); live tokens occupy
    [kv_len, kv_len+live_len). Returned for the positional-invariant test:
    ranges must be non-overlapping and monotone (no key ever shares a slot)."""
    return range(0, kv_len), range(kv_len, kv_len + live_len)
