"""Token-trigger-based injection.

Runtime path: the user types free text. We tokenize, scan for any
registered axiom term as a token-id subsequence, and inject the term's
meaning vector at exactly those positions during the forward pass.

No user-facing markers. The build pipeline captures meaning vectors at
the end of each paraphrase (the model's integrated reading of the
description). Vectors live in a side memory keyed by term name.

Optional capabilities (all off by default — used to experiment with
performance improvements without affecting the simple path):

- Per-entry components: a list of other registered names. With dag=True
  the injector can fire the component vectors alongside the root.
- Per-entry prior: a unit-norm direction representing the term's prior
  reading; subtracted with weight beta before the additive injection.
- Decoupled layer for components: components fire at inner_layer instead
  of layer, so the two contributions don't sum-and-dilute at the same
  residual position.
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
    components: tuple[str, ...] = ()


@dataclass
class Registry:
    entries: list[Entry] = field(default_factory=list)

    def _add_term(
        self,
        name: str,
        token_id_variants: list[list[int]],
        vector: np.ndarray | None,
        prior: np.ndarray | None = None,
        components: tuple[str, ...] = (),
    ) -> None:
        self.entries.append(Entry(name, token_id_variants, vector, prior, components))

    def get(self, name: str) -> Entry | None:
        for e in self.entries:
            if e.name == name:
                return e
        return None

    def expand_with_components(self, name: str, _seen: set[str] | None = None) -> list[str]:
        """Return [name, *recursive components] in injection order. Cycle-safe."""
        if _seen is None:
            _seen = set()
        if name in _seen:
            return []
        _seen.add(name)
        out = [name]
        e = self.get(name)
        if e is not None:
            for comp in e.components:
                out.extend(self.expand_with_components(comp, _seen))
        return out

    def register(
        self,
        name: str,
        term_variants: list[str],
        vector: np.ndarray,
        tokenizer,  # noqa: ANN001
        prior: np.ndarray | None = None,
        components: tuple[str, ...] = (),
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
        self.entries.append(Entry(name, all_variants, vector, prior, tuple(components)))


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
    current input_ids during the forward pass and injects at those positions.

    Default behaviour: simple additive injection at `layer` for matched terms.
    Optional flags (dag, inner_alpha, inner_layer, beta) enable the more
    elaborate variants explored during the architecture experiments."""

    def __init__(
        self,
        model,  # noqa: ANN001
        tokenizer,  # noqa: ANN001
        layer: int,
        registry: Registry,
        alpha: float = 30.0,
        beta: float = 0.0,
        dag: bool = False,
        inner_alpha: float | None = None,
        inner_layer: int | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        self.registry = registry
        self.alpha = alpha
        self.beta = beta
        self.dag = dag
        self.inner_alpha = inner_alpha
        self.inner_layer = inner_layer
        self._handles: list = []
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

    def _make_hook(self, mode: str):  # noqa: ANN202
        """Build a forward hook for the given mode:
        - "all":        inject root + components together (single-layer DAG)
        - "root":       inject only the root term
        - "components": inject only the components (skip root)
        """

        def _hook(module, inputs, output):  # noqa: ARG001, ANN001
            if self._current_ids is None or self.alpha == 0.0:
                return output
            h = output[0] if isinstance(output, tuple) else output
            seq_len = h.shape[1]
            ids = self._current_ids
            if seq_len < len(ids):
                ids_window = ids[-seq_len:]
            else:
                ids_window = ids
            matches = find_matches(ids_window, self.registry)
            if not matches:
                return output
            h = h.clone()
            for start, end, name in matches:
                if self.dag:
                    expanded = self.registry.expand_with_components(name)
                    if mode == "root":
                        names_to_inject = expanded[:1]
                    elif mode == "components":
                        names_to_inject = expanded[1:]
                    else:
                        names_to_inject = expanded
                else:
                    names_to_inject = [] if mode == "components" else [name]
                if not names_to_inject:
                    continue
                if self.inner_alpha is None:
                    per_alpha = self.alpha / len(names_to_inject)
                    alphas = [per_alpha] * len(names_to_inject)
                else:
                    alphas = []
                    for i, _ in enumerate(names_to_inject):
                        if mode == "components":
                            alphas.append(self.inner_alpha)
                        else:
                            alphas.append(self.alpha if i == 0 else self.inner_alpha)
                for p in range(start, end):
                    if not (0 <= p < seq_len):
                        continue
                    row = h[:, p, :]
                    u = self._priors.get(name)
                    if u is not None and self.beta != 0.0:
                        u_dev = u.to(dtype=h.dtype, device=h.device)
                        coef = (row * u_dev).sum(dim=-1, keepdim=True)
                        row = row - self.beta * coef * u_dev
                    for inj_name, a in zip(names_to_inject, alphas):
                        v_inj = self._vectors.get(inj_name)
                        if v_inj is None:
                            continue
                        v_dev = v_inj.to(dtype=h.dtype, device=h.device)
                        row = row + a * v_dev
                    h[:, p, :] = row
            if isinstance(output, tuple):
                return (h, *output[1:])
            return h

        return _hook

    def attach(self) -> None:
        m = self.model
        if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
            m = m.base_model.model
        layers = m.model.layers
        decoupled = self.dag and self.inner_layer is not None and self.inner_layer != self.layer
        if decoupled:
            self._handles.append(layers[self.layer].register_forward_hook(self._make_hook("root")))
            self._handles.append(
                layers[self.inner_layer].register_forward_hook(self._make_hook("components"))
            )
        else:
            self._handles.append(layers[self.layer].register_forward_hook(self._make_hook("all")))

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
