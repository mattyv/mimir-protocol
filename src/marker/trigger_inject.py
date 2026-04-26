"""Token-trigger-based injection.

The runtime path: the user types free text. We tokenize, scan for any
registered axiom term as a token-id subsequence, and inject the term's
concept vector at exactly those positions during the forward pass.

No user-facing markers. Markers (`[[...]]`) are scaffolding for *building*
the keys via marker-anchored extraction; they do not appear at inference.

Side memory: {term_name -> (token_id_variants, vector)}.
Routing:     scan input_ids and generated_ids for matches.
Injection:   forward hook at chosen layer adds alpha*vector at each matched
             position.
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
    prior: np.ndarray | None = None


@dataclass
class Registry:
    entries: list[Entry] = field(default_factory=list)

    def _add_term(
        self,
        name: str,
        token_id_variants: list[list[int]],
        vector: np.ndarray | None,
        prior: np.ndarray | None = None,
    ) -> None:
        self.entries.append(Entry(name, token_id_variants, vector, prior))

    def register(
        self,
        name: str,
        term_variants: list[str],
        vector: np.ndarray,
        tokenizer,  # noqa: ANN001
        prior: np.ndarray | None = None,
    ) -> None:
        """Tokenise each surface variant with and without a leading space and
        store the unique token-id sequences. Both forms are needed because BPE
        produces different ids depending on whether the term is preceded by
        whitespace.

        `prior`, if provided, is a unit-norm direction representing the model's
        baseline reading of the term in unmarked text. The injector subtracts a
        scaled projection onto this direction before adding the concept vector,
        which removes the surface-form prior fight."""
        all_variants: list[list[int]] = []
        for v in term_variants:
            for prefix in ("", " "):
                ids = tokenizer(prefix + v, add_special_tokens=False).input_ids
                if ids and ids not in all_variants:
                    all_variants.append(ids)
        self.entries.append(Entry(name, all_variants, vector, prior))


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
    # Sort: by start asc, then by length desc so longest wins at each start.
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
    current input_ids on every forward call and injects at those positions."""

    def __init__(
        self,
        model,  # noqa: ANN001
        tokenizer,  # noqa: ANN001
        layer: int,
        registry: Registry,
        alpha: float = 30.0,
        beta: float = 0.0,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        self.registry = registry
        self.alpha = alpha
        self.beta = beta
        self._handle = None
        self._current_ids: list[int] | None = None
        self._vectors: dict[str, torch.Tensor] = {
            e.name: torch.tensor(e.vector, dtype=torch.float32)
            for e in registry.entries
            if e.vector is not None
        }
        self._priors: dict[str, torch.Tensor] = {
            e.name: torch.tensor(e.prior, dtype=torch.float32)
            for e in registry.entries
            if e.prior is not None
        }

    def _hook(self, module, inputs, output):  # noqa: ARG002, ANN001
        if self._current_ids is None:
            return output
        h = output[0] if isinstance(output, tuple) else output
        seq_len = h.shape[1]
        ids = self._current_ids
        # If KV cache shrinks h to length-1 (incremental decode), align to the
        # tail of the known sequence.
        if seq_len < len(ids):
            ids_window = ids[-seq_len:]
            offset = len(ids) - seq_len
        else:
            ids_window = ids
            offset = 0
        matches = find_matches(ids_window, self.registry)
        if not matches:
            return output
        h = h.clone()
        for start, end, name in matches:
            v = self._vectors[name].to(dtype=h.dtype, device=h.device)
            u = self._priors.get(name)
            if u is not None:
                u = u.to(dtype=h.dtype, device=h.device)
            for p in range(start, end):
                pos = p
                if not (0 <= pos < seq_len):
                    continue
                row = h[:, pos, :]
                if u is not None and self.beta != 0.0:
                    coef = (row * u).sum(dim=-1, keepdim=True)
                    row = row - self.beta * coef * u
                h[:, pos, :] = row + self.alpha * v
        _ = offset  # kept for symmetry; window-local indexing is correct
        if isinstance(output, tuple):
            return (h, *output[1:])
        return h

    def attach(self) -> None:
        # Resolve through PEFT wrapping if present.
        m = self.model
        if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
            m = m.base_model.model
        target = m.model.layers[self.layer]
        self._handle = target.register_forward_hook(self._hook)

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 60) -> str:
        device = next(self.model.parameters()).device
        ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(
            device
        )
        self._current_ids = ids[0].tolist()
        self.attach()
        try:
            for _ in range(max_new_tokens):
                logits = self.model(ids).logits[0, -1]
                nxt = logits.argmax().unsqueeze(0).unsqueeze(0)
                ids = torch.cat([ids, nxt], dim=1)
                self._current_ids = ids[0].tolist()
                if int(nxt.item()) == self.tokenizer.eos_token_id:
                    break
            full = self.tokenizer.decode(ids[0], skip_special_tokens=True)
            return full[len(prompt) :]
        finally:
            self.detach()
            self._current_ids = None
