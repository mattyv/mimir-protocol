"""Multi-hook runtime injector that consumes a list of AxiomPlan.

Each plan can carry one or more mechanisms (eop / steer / disambig),
each at its own layer with its own alpha. AutoInjector groups them by
layer and registers one forward hook per layer. The hook scans for any
plan's term tokens in the input and applies that plan's vector(s) at
the matched positions, with the per-mechanism alpha.

All this on top of the existing find_matches subsequence scanner. The
runtime is otherwise the same KV-cache-aware greedy decode as
TriggerInjector — hooks fire during prefill, KV cache carries the
modification through generation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch

from marker.axiom_plan import AxiomPlan
from marker.trigger_inject import Registry, find_matches


@dataclass
class _LayerSpec:
    """One injection point at a single layer: which term ids to match,
    which vector to add, with what alpha."""

    term_id_variants: list[list[int]]
    vector: torch.Tensor
    alpha: float


class AutoInjector:
    """Runtime injector that applies multiple plans at once.

    For each plan, for each mechanism in the plan's stack, AutoInjector
    will inject that mechanism's vector at the term's tokens at the
    mechanism's layer with the mechanism's alpha. Plans are independent:
    different terms get matched independently, and one prompt can fire
    several plans if multiple registered terms appear in it.
    """

    def __init__(
        self,
        model,  # noqa: ANN001
        tokenizer,  # noqa: ANN001
        plans: list[AxiomPlan],
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.plans = plans
        self._handles: list = []
        self._current_ids: list[int] | None = None
        # We reuse find_matches() which scans against a Registry. Build
        # one Registry per layer so each layer's hook only matches the
        # relevant axioms.
        self._layer_specs: dict[int, list[_LayerSpec]] = defaultdict(list)
        for plan in plans:
            term_id_variants = self._tokenize_term_variants(plan.term_variants)
            for _kind, mech in plan.mechanisms.items():
                self._layer_specs[mech["layer"]].append(
                    _LayerSpec(
                        term_id_variants=term_id_variants,
                        vector=torch.tensor(mech["vector"], dtype=torch.float32),
                        alpha=float(mech["alpha"]),
                    )
                )

    def _tokenize_term_variants(self, variants: list[str]) -> list[list[int]]:
        """Tokenize each term variant with and without a leading space, dedupe."""
        out: list[list[int]] = []
        for v in variants:
            for prefix in ("", " "):
                ids = self.tokenizer(prefix + v, add_special_tokens=False).input_ids
                if ids and ids not in out:
                    out.append(ids)
        return out

    def _make_hook(self, specs: list[_LayerSpec]):  # noqa: ANN202
        def hook(module, inputs, output):  # noqa: ANN001, ARG001
            if self._current_ids is None:
                return output
            # Skip cheaply if every spec at this layer has alpha=0.
            if all(s.alpha == 0.0 for s in specs):
                return output
            h = output[0] if isinstance(output, tuple) else output
            seq_len = h.shape[1]
            ids = self._current_ids
            ids_window = ids[-seq_len:] if seq_len < len(ids) else ids
            modified = False
            h_new = h
            for spec in specs:
                if spec.alpha == 0.0:
                    continue
                # Build a one-axiom registry for find_matches.
                tmp = Registry()
                tmp._add_term("_t", spec.term_id_variants, vector=None)
                matches = find_matches(ids_window, tmp)
                if not matches:
                    continue
                if not modified:
                    h_new = h.clone()
                    modified = True
                v_dev = spec.vector.to(dtype=h.dtype, device=h.device)
                for start, end, _ in matches:
                    for p in range(start, end):
                        if 0 <= p < seq_len:
                            h_new[:, p, :] = h_new[:, p, :] + spec.alpha * v_dev
            if not modified:
                return output
            if isinstance(output, tuple):
                return (h_new, *output[1:])
            return h_new

        return hook

    def attach(self) -> None:
        m = self.model
        if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
            m = m.base_model.model
        layers = m.model.layers
        for layer_idx, specs in self._layer_specs.items():
            self._handles.append(layers[layer_idx].register_forward_hook(self._make_hook(specs)))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 60) -> str:
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
