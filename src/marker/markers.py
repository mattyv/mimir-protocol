"""Marker-wrap utilities for axiom-name extraction.

The hypothesis: capturing residuals at a designated *closing marker*
position gives a cleaner readout of the term-in-axiom-context than
capturing at the natural subword position (which we already tested and
which produced more uniform tilt, not less).

We use simple text markers that tokenise to existing-vocab tokens —
no new tokens to install. The closing marker becomes a structural
aggregation point similar in spirit to BERT's [CLS].
"""

from __future__ import annotations

import re

OPEN_MARKER = "[["
CLOSE_MARKER = "]]"


def wrap_term_in_paraphrase(paraphrase: str, term_variants: list[str]) -> str:
    """Wrap every occurrence of any term variant with markers.

    Variants are tried longest-first so "Just Out of Time Processing"
    matches before "JOTP" within the same string. Already-wrapped
    occurrences are skipped (idempotent).
    """
    sorted_variants = sorted(term_variants, key=len, reverse=True)
    out = paraphrase
    for variant in sorted_variants:
        # Already-wrapped instances (preceded by the open marker) are skipped.
        # Build a regex with negative-lookbehind for the open marker.
        pattern = re.compile(
            r"(?<!"
            + re.escape(OPEN_MARKER)
            + r")"
            + re.escape(variant)
            + r"(?!"
            + re.escape(CLOSE_MARKER)
            + r")"
        )
        out = pattern.sub(f"{OPEN_MARKER}{variant}{CLOSE_MARKER}", out)
    return out


def find_close_marker_positions(token_ids: list[int], close_token_ids: list[int]) -> list[int]:
    """Find every position where the close-marker token sequence ends.

    Returns the index of the *last* token of each close-marker occurrence
    in the BPE-encoded prompt. Used as the capture site.
    """
    if not close_token_ids:
        return []
    out: list[int] = []
    n = len(close_token_ids)
    for i in range(len(token_ids) - n + 1):
        if token_ids[i : i + n] == close_token_ids:
            out.append(i + n - 1)
    return out
