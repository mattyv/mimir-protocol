"""Gist fidelity probe helpers (the Fable-specced ~$1 discriminator).

The reconstitute result (gist_render == none) has three live explanations —
(a) the 8-slot gist lacks relational structure, (b) the render reader never
learned structure (token-F1 + ledger + first-token priming certify LITERALS,
not relations), (c) plain per-step error compounding. These helpers score the
thing no metric ever measured: RELATIONS (a op b = c), plus the token masking
for a structure-only NLL contrast. Model-free, CPU-tested.
"""

from __future__ import annotations

import re

import torch

_REL = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s*([+\-*/x×÷])\s*(\d[\d,]*(?:\.\d+)?)\s*=\s*(\d[\d,]*(?:\.\d+)?)"
)
_OPS = {"x": "*", "×": "*", "÷": "/"}


def extract_relations(text: str) -> list[str]:
    """The step's arithmetic relations as normalized 'a|op|b|c' strings, in
    order. Commas stripped, x/×->*, ÷->/. This is the structure token-F1 is
    blind to: 'right numbers, wrong operator' changes the relation string."""
    out = []
    for a, op, b, c in _REL.findall(text):
        norm = lambda s: s.replace(",", "")  # noqa: E731
        out.append(f"{norm(a)}|{_OPS.get(op, op)}|{norm(b)}|{norm(c)}")
    return out


def relation_score(pred_text: str, gold_text: str) -> dict:
    """How much of the gold step's arithmetic STRUCTURE survives in a
    reconstruction. exact = fraction of gold relations present verbatim in the
    prediction; op_seq = does the operator sequence match. None-safe: a gold
    step with no relations contributes nothing (n_gold 0)."""
    gold = extract_relations(gold_text)
    pred = extract_relations(pred_text)
    if not gold:
        return {"n_gold": 0, "exact": None, "op_seq": None}
    pred_set = set(pred)
    exact = sum(1 for g in gold if g in pred_set) / len(gold)
    gold_ops = [g.split("|")[1] for g in gold]
    pred_ops = [p.split("|")[1] for p in pred]
    return {"n_gold": len(gold), "exact": round(exact, 3), "op_seq": gold_ops == pred_ops}


def digit_token_mask(token_strs: list[str]) -> torch.Tensor:
    """True where a token contains a digit — the LITERAL tokens the ledger
    hands the reader for free. The structure-NLL contrast averages CE over the
    complement (~the relation/operator/word tokens the gist must supply)."""
    return torch.tensor([any(ch.isdigit() for ch in t) for t in token_strs], dtype=torch.bool)


def per_token_ce(pm, thought_kv, cont_start: int, ledger_ids: list[int], span_ids: list[int]):  # noqa: ANN001
    """ledger_render_nll's layout (visible ledger prefix, then the span), but
    returning the PER-TOKEN CE over the span targets instead of the mean —
    so callers can mask digit tokens and read structure-only NLL. Teacher-
    forced; render adapter must be active."""
    import torch.nn.functional as F  # noqa: N812, PLC0415
    from transformers import DynamicCache  # noqa: PLC0415

    if len(span_ids) < 2:
        raise ValueError("need >= 2 span tokens")
    device = next(pm.parameters()).device
    cache = DynamicCache()
    for i in range(thought_kv.n_layers):
        cache.update(thought_kv.keys[i].to(device), thought_kv.values[i].to(device), i)
    seq = list(ledger_ids) + list(span_ids)
    inp = seq[:-1]
    pos = torch.arange(cont_start, cont_start + len(inp), device=device).unsqueeze(0)
    out = pm(
        torch.tensor([inp], device=device),
        past_key_values=cache,
        position_ids=pos,
        use_cache=True,
    )
    start = len(ledger_ids) - 1 if ledger_ids else 0
    logits = out.logits[0][start : start + (len(span_ids) - (0 if ledger_ids else 1))]
    tgt = span_ids if ledger_ids else span_ids[1:]
    ce = F.cross_entropy(logits, torch.tensor(tgt, device=device), reduction="none")
    return ce, tgt  # per-target-token CE, aligned with tgt ids
