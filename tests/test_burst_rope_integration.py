"""Slow integration check: rope_shift_keys must match the MODEL's actual RoPE.

burst_true's validity hinges on this — it injects a gist thought rotated to a
placement position, and if our rotation convention (theta, rotate-half split)
disagrees with the model's, the injected keys land at the wrong angle and the
oracle arm is meaningless. Oracle: rotating gist_kv(span)'s keys by Δ must equal
encoding the same span with gist_start shifted by Δ (which re-rotates via the
model's own position_ids). Tiny CPU model.
"""

from __future__ import annotations

import pytest
import torch

from marker.burst import rope_shift_keys, rope_theta


@pytest.mark.slow
def test_rope_shift_matches_model_gist_positions():
    from marker.gist_model import attach_gist, gist_kv, to_leaf_param
    from tests.test_gist_model import _tiny_base

    base = _tiny_base()
    pm, gist = attach_gist(base, gist_k=4, r=4)
    gist = to_leaf_param(gist, "cpu")
    theta = rope_theta(base.config)

    span = [5, 6, 7, 8, 9]  # span_len = 5
    kv0, _, _ = gist_kv(pm, gist, span)  # keys RoPE'd at [5, 5+4)
    delta = 11
    kv_shifted, _, _ = gist_kv(pm, gist, span, gist_start=len(span) + delta)  # at [5+11, ...)

    for layer in range(kv0.n_layers):
        ours = rope_shift_keys(kv0.keys[layer], delta, theta)
        assert torch.allclose(ours, kv_shifted.keys[layer], atol=1e-4), (
            f"layer {layer}: rope_shift_keys disagrees with the model's RoPE — "
            "burst_true would inject at the wrong angle"
        )
        # values are untouched by RoPE
        assert torch.allclose(kv0.values[layer], kv_shifted.values[layer], atol=1e-4)
