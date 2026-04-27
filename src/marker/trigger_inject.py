"""Token-trigger-based injection — the runtime path.

The user types free text. We tokenize, scan for any registered axiom
term as a token-id subsequence, and inject the term's meaning vector at
exactly those positions during the forward pass.

No user-facing markers. The build pipeline captures meaning vectors at
the end of each paraphrase (the model's integrated reading of the
description). Vectors live in a side memory keyed by term name.

Default behaviour: simple additive injection at one chosen layer for
every term match. The elaborate variants (DAG / decoupled-layer /
prior-subtraction / multi-position) were tested and rejected — see
FAILED_IDEAS.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class Entry:
    name: str
    token_id_variants: list[list[int]]
    vector: np.ndarray | None


@dataclass
class Registry:
    entries: list[Entry] = field(default_factory=list)

    def _add_term(
        self,
        name: str,
        token_id_variants: list[list[int]],
        vector: np.ndarray | None,
    ) -> None:
        self.entries.append(Entry(name, token_id_variants, vector))

    def get(self, name: str) -> Entry | None:
        for e in self.entries:
            if e.name == name:
                return e
        return None

    def register(
        self,
        name: str,
        term_variants: list[str],
        vector: np.ndarray,
        tokenizer,  # noqa: ANN001
    ) -> None:
        """Tokenise each surface variant with and without a leading space and
        store the unique token-id sequences. Both forms are needed because BPE
        produces different ids depending on whether the term is preceded by
        whitespace."""
        all_variants: list[list[int]] = []
        for v in term_variants:
            for prefix in ("", " "):
                ids = tokenizer(prefix + v, add_special_tokens=False).input_ids
                if ids and ids not in all_variants:
                    all_variants.append(ids)
        self.entries.append(Entry(name, all_variants, vector))


def find_matches(ids: list[int], registry: Registry) -> list[tuple[int, int, str]]:
    """Greedy longest-non-overlapping match across all registered terms.

    Returns list of (start, end, name) where the match covers ids[start:end].
    """
    candidates: list[tuple[int, int, str]] = []
    for entry in registry.entries:
        for variant in entry.token_id_variants:
            n = len(variant)
            if n == 0:
                continue
            for i in range(len(ids) - n + 1):
                if ids[i : i + n] == variant:
                    candidates.append((i, i + n, entry.name))
    candidates.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    chosen: list[tuple[int, int, str]] = []
    cursor = 0
    for start, end, name in candidates:
        if start < cursor:
            continue
        chosen.append((start, end, name))
        cursor = end
    return chosen


class TriggerInjector:
    """Wraps a model + tokenizer + registry. Scans for term matches in the
    current input_ids during the forward pass and injects at those positions."""

    def __init__(
        self,
        model,  # noqa: ANN001
        tokenizer,  # noqa: ANN001
        layer: int,
        registry: Registry,
        alpha: float = 30.0,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        self.registry = registry
        self.alpha = alpha
        self._handles: list = []
        self._current_ids: list[int] | None = None
        self._vectors: dict[str, torch.Tensor] = {
            e.name: torch.tensor(e.vector, dtype=torch.float32)
            for e in registry.entries
            if e.vector is not None
        }

    def _hook(self, module, inputs, output):  # noqa: ARG002, ANN001
        if self._current_ids is None or self.alpha == 0.0:
            return output
        h = output[0] if isinstance(output, tuple) else output
        seq_len = h.shape[1]
        ids = self._current_ids
        # If KV cache shrinks h to length-1 (incremental decode), align to the
        # tail of the known sequence. Multi-token term matches won't be found
        # in a length-1 window, so injection naturally short-circuits during
        # decode — the prefill modification carries forward via KV cache.
        if seq_len < len(ids):
            ids_window = ids[-seq_len:]
        else:
            ids_window = ids
        matches = find_matches(ids_window, self.registry)
        if not matches:
            return output
        h = h.clone()
        for start, end, name in matches:
            v = self._vectors.get(name)
            if v is None:
                continue
            v_dev = v.to(dtype=h.dtype, device=h.device)
            for p in range(start, end):
                if 0 <= p < seq_len:
                    h[:, p, :] = h[:, p, :] + self.alpha * v_dev
        if isinstance(output, tuple):
            return (h, *output[1:])
        return h

    def attach(self) -> None:
        m = self.model
        if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
            m = m.base_model.model
        target = m.model.layers[self.layer]
        self._handles.append(target.register_forward_hook(self._hook))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 60) -> str:
        """KV-cache-aware greedy decoding. Hook fires during prefill at term
        token positions; cached KV in upper layers carries the modification
        forward through generation. O(N) instead of O(N²)."""
        device = next(self.model.parameters()).device
        ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(
            device
        )
        self._current_ids = ids[0].tolist()
        self.attach()
        try:
            out = self.model(ids, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            ids = torch.cat([ids, nxt], dim=1)
            self._current_ids = ids[0].tolist()
            if int(nxt.item()) == self.tokenizer.eos_token_id:
                return ""
            for _ in range(max_new_tokens - 1):
                out = self.model(nxt, past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
                ids = torch.cat([ids, nxt], dim=1)
                self._current_ids = ids[0].tolist()
                if int(nxt.item()) == self.tokenizer.eos_token_id:
                    break
            full = self.tokenizer.decode(ids[0], skip_special_tokens=True)
            return full[len(prompt) :]
        finally:
            self.detach()
            self._current_ids = None
