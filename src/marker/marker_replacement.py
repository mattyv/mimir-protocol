"""Marker replacement: substitute a registered axiom term with an opaque
placeholder string before sending the prompt to the model.

Why:
  - The model's lexical priors on multi-word terms (e.g., 'Balance
    Publisher' → 'balance sheet') dominate the residual stream at the
    term position. Our small contrastive vector v fights against that
    prior.
  - If we replace the term with an opaque marker the model has never
    seen ('AXIOMTAG0001'), the residual at that position has zero lexical
    priors. Vector injection lands on a clean canvas.
  - The marker carries no semantic content — that's the point. The
    vector v has to do all the work, which is the project's invariant.

This is *not* prompt injection (no semantic content added to the prompt).
It's prompt SUBSTITUTION to remove competing lexical priors.

User-facing UX:
  - User types 'Balance Publisher'.
  - System swaps to marker before model inference.
  - Model output containing the marker is restored to 'Balance Publisher'
    before being shown to the user.
"""

from __future__ import annotations


def _make_marker_string(idx: int) -> str:
    """Generate an opaque marker string. Uses uppercase consonant blocks
    + digits — looks like a serial number / SKU rather than a meaningful
    word. Avoids common dictionary words and avoids prefixes like 'AXIOM'
    or 'TAG' that would suggest a fabricated system name to the model.

    Empirically tested: 'AXIOMTAG' caused the model to invent an
    'AXIOM system' prior and reference it in outputs. 'ZQVXM####'-style
    serial codes don't trigger that fabrication.
    """
    # Cycle through opaque consonant prefixes
    prefixes = ["ZQVXM", "JKWPF", "QXBHN", "VWGTR", "MFZPK"]
    return f"{prefixes[idx % len(prefixes)]}{idx:04d}"


class MarkerRegistry:
    """Manages bidirectional mapping between axiom terms and marker strings.

    Markers are assigned monotonically. Idempotent: assigning the same
    term twice returns the same marker.
    """

    def __init__(self) -> None:
        self._term_to_marker: dict[str, str] = {}
        self._marker_to_term: dict[str, str] = {}

    def assign(self, term: str) -> str:
        """Reserve a marker string for `term`. Returns the marker. If
        `term` was already assigned, returns the existing marker."""
        if term in self._term_to_marker:
            return self._term_to_marker[term]
        marker = _make_marker_string(len(self._term_to_marker))
        self._term_to_marker[term] = marker
        self._marker_to_term[marker] = term
        return marker

    def marker_for(self, term: str) -> str:
        """Look up the marker for an already-assigned term."""
        return self._term_to_marker[term]

    def term_for(self, marker: str) -> str:
        """Look up the original term for a marker."""
        return self._marker_to_term[marker]

    def rewrite_prompt(self, prompt: str, term: str) -> str:
        """Replace the FIRST occurrence of `term` in `prompt` with its
        registered marker. No-op if `term` not in `prompt`."""
        if term not in prompt:
            return prompt
        marker = self._term_to_marker.get(term)
        if marker is None:
            marker = self.assign(term)
        return prompt.replace(term, marker, 1)

    def restore_output(self, text: str) -> str:
        """Replace ALL occurrences of any registered marker in `text`
        with their original terms. Used on model output before display."""
        for marker, term in self._marker_to_term.items():
            text = text.replace(marker, term)
        return text


def find_marker_position(tokenizer, rewritten_prompt: str, marker: str) -> int:  # noqa: ANN001
    """Find the LAST token-position of the marker's tokenization in the
    rewritten prompt. The injection point sits on the marker's final
    sub-token so attention from later positions has the most-resolved
    semantic representation to read.

    Returns -1 if the marker can't be found in the tokenized sequence.
    """
    ids = tokenizer(rewritten_prompt, add_special_tokens=False).input_ids
    marker_ids = tokenizer(marker, add_special_tokens=False).input_ids
    if not marker_ids:
        return -1
    # Search from the end for the marker token sequence
    n, m = len(ids), len(marker_ids)
    for i in range(n - m, -1, -1):
        if ids[i : i + m] == marker_ids:
            return i + m - 1  # last sub-token of the marker
    # Fallback: search for any single marker sub-token
    for tid in marker_ids:
        if tid in ids:
            # Return last occurrence
            for i in range(len(ids) - 1, -1, -1):
                if ids[i] == tid:
                    return i
    return -1
