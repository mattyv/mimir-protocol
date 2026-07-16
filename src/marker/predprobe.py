"""PREDPROBE helpers (Fable-specced): does a PREDICTED gist, rendered to words,
keep its arithmetic structure? Model-free logic only -- the parts of
run_predprobe.py that are pure enough to CPU-test without loading a model.

Render reads per-layer K/V (gist_kv); the predictor emits a final-layer
SUMMARY. A predicted thought only reaches render via
predict_step -> bridge(summary) -> KV ("Path B"), which is NOT the
gist_kv(true text) distribution render was trained on ("Path A"). The five
conditions below (see run_predprobe.py) separate predictor error from the
bridge hop:

    true_gistkv     gist_kv(true step n)                    -- Path A, native ceiling
    true_bridged    encode_gist(true n) -> bridge            -- Path B on truth
    wrong_bridged   encode_gist(step from a DIFFERENT doc)   -- Path B floor
    pred_bridged    predict_step(true ctx) -> bridge          -- Path B on prediction
    noised_bridged  bridge(noised(true summary, ratio=1.0))  -- magnitude-matched noise

bridged_condition is the ONE conversion path shared by the last four -- so a
cont_start or dtype bug can't silently diverge between them (the two named
plumbing traps: bridged KV must decode from cont_start=k, the bridge's OWN
canonical position frame, never a span-dependent value; and must be cast to
the model's attention kv_dtype before it reaches attention).
"""

from __future__ import annotations

import torch

# published render-of-true-gist relations-exact ceiling (the reconstitute
# experiment, validated). REFERENCE ONLY -- reported in the manifest for
# context, never gated on: the recomputed n>=1 ceiling (this run's true_gistkv
# rel) legitimately drifts from it (different subset; and the easy set's
# ceiling was never 0.92). Gate 0 instead uses an absolute floor (C_FLOOR): a
# real plumbing bug (wrong cont_start / dtype) craters Path A to near-floor,
# not to 0.85.
CEILING = 0.92
C_FLOOR = 0.8


def scorable_ns(m: int) -> list[int]:
    """Step indices n a (capped) doc of m steps can be SCORED at. predict_step
    needs >=1 prior step, so n=0 is unscorable for EVERY condition -- including
    true_gistkv (Fable spec: restrict all five conditions to n>=1, and
    recompute the ceiling on this same n>=1 subset, since the published 0.92
    included n=0 steps)."""
    return list(range(1, m))


def bridged_condition(bridge, vec: torch.Tensor, kv_dtype: torch.dtype):  # noqa: ANN001
    """The ONE Path-B conversion: vec [k, d] -> bridge -> KV cast to the
    model's attention dtype, decoded from the bridge's CANONICAL position
    frame cont_start = bridge.k (never span-dependent -- see
    bridge_injection_nll's position-frame note in bridge.py). Shared by all
    four bridged conditions (true/wrong/pred/noised) so cont_start and dtype
    can't drift between them; true_bridged doubles as the runtime canary."""
    kv = bridge(vec)
    kv = type(kv)(
        kv.n_layers,
        [k.to(kv_dtype) for k in kv.keys],
        [v.to(kv_dtype) for v in kv.values],
    )
    return kv, bridge.k


def pick_cross_doc_step(
    doc_idx: int, doc_lengths: list[int], gen: torch.Generator, step_idx: int | None = None
) -> tuple[int, int]:
    """The wrong_bridged pairing: a (doc, step) index from a DIFFERENT document
    than doc_idx -- never a shift-by-1 within the SAME doc (Fable: adjacent
    GSM8K steps share numbers and inflate the floor). Falls back to doc_idx
    itself only when it's the sole document available (mirrors run_bridge's
    "shuffled" control's own `while dj == di and len(docs) > 1` guard) -- and
    in that fallback never returns `step_idx` itself (the current step's own
    summary would make W silently equal true_bridged and erase H)."""
    n_docs = len(doc_lengths)
    dj = doc_idx
    while dj == doc_idx and n_docs > 1:
        dj = int(torch.randint(0, n_docs, (1,), generator=gen))
    sj = int(torch.randint(0, doc_lengths[dj], (1,), generator=gen))
    while dj == doc_idx and step_idx is not None and sj == step_idx and doc_lengths[dj] > 1:
        sj = int(torch.randint(0, doc_lengths[dj], (1,), generator=gen))
    return dj, sj


def relation_gate(w: float, h: float, p: float) -> str:
    """Gate 2/3/4 on relations-exact margins over the wrong-gist floor W, with
    headroom H = true_bridged - wrong_bridged. GREEN: the predicted-thought
    condition keeps >=70% of H. RED: keeps <=30%. YELLOW: between."""
    if p >= w + 0.7 * h:
        return "GREEN"
    if p <= w + 0.3 * h:
        return "RED"
    return "YELLOW"


def struct_nll_gate(delta_b: float, delta_p: float) -> str:
    """Struct-NLL mirror of relation_gate. delta_X = NLL(wrong) - NLL(X) (bigger
    = more improvement over the floor). GREEN if pred keeps >=50% of the
    true_bridged NLL improvement, RED if <=20%. delta_b<=0 means true_bridged
    itself has NO NLL headroom over wrong -- there's no fraction of nothing to
    judge, so this reads YELLOW rather than raising or dividing by <=0
    (mirrors ladder_gap_closed's non-positive-denominator guard)."""
    if delta_b <= 0:
        return "YELLOW"
    if delta_p >= 0.5 * delta_b:
        return "GREEN"
    if delta_p <= 0.2 * delta_b:
        return "RED"
    return "YELLOW"


def gated_verdict(  # noqa: PLR0913
    c: float | None,
    b: float | None,
    w: float | None,
    p: float | None,
    nll_w: float | None,
    nll_b: float | None,
    nll_p: float | None,
    c_floor: float = C_FLOOR,
) -> dict:
    """The PREDPROBE gated read (Fable spec), in strict gate order. This is a
    CONVENIENCE field only -- it must never gate the run's exit code; the
    human + Fable read the actual C/B/W/P/N numbers in the manifest.

    Any None input (a set where no step had an extractable relation, or no
    struct tokens) -> INSUFFICIENT_DATA, never coerced to 0.0 (which would
    fake a gate-0 INVALID, or for W=None a fake headroom H=B).

    Gate 0: C below `c_floor` absolute -- the harness itself is suspect
        (cont_start/dtype), stop before any science. An absolute floor, NOT
        "near the published 0.92": C here IS the recomputed n>=1 ceiling, it
        legitimately drifts from the published number (different subset, and
        the easy set's ceiling was never 0.92), while a real plumbing bug
        craters C to near-floor -- which the absolute check catches.
    Gate 1: H = B-W < 0.15, or B < 0.5 absolute -- the bridge is the wall;
        pred_bridged is unreadable through it, don't read P.
    Gates 2-4: relations-exact AND struct-NLL must BOTH land GREEN (or both
        RED) for a decisive verdict -- any disagreement is YELLOW ("inspect
        samples"), never a green flattered by only one metric."""
    if any(x is None for x in (c, b, w, p, nll_w, nll_b, nll_p)):
        return {"gate": None, "verdict": "INSUFFICIENT_DATA"}
    h = b - w
    if c < c_floor:
        return {"h": round(h, 4), "gate": 0, "verdict": "INVALID_HARNESS_CHECK_PLUMBING"}
    if h < 0.15 or b < 0.5:
        return {"h": round(h, 4), "gate": 1, "verdict": "BRIDGE_IS_WALL"}
    delta_b, delta_p = nll_w - nll_b, nll_w - nll_p
    rel_band = relation_gate(w, h, p)
    nll_band = struct_nll_gate(delta_b, delta_p)
    if rel_band == nll_band == "GREEN":
        gate, verdict = 2, "GREEN"
    elif rel_band == nll_band == "RED":
        gate, verdict = 3, "RED"
    else:
        gate, verdict = 4, "YELLOW"
    return {
        "h": round(h, 4),
        "gate": gate,
        "verdict": verdict,
        "rel_band": rel_band,
        "nll_band": nll_band,
    }
