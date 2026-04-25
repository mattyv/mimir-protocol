"""GPT-2 small wrapped with a forward hook on a single layer's residual output.

Two operations:
  - capture(prompt) -> last-token residual at the configured layer
  - logits_at(prompt, vec, alpha) -> next-token logits, optionally with
    `alpha * vec` added to the last-token residual at the configured layer

Hook position rationale lives in docs/mimir-axiom-design-rationale.md §6 and §7.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class HookedModel:
    def __init__(self, model_name: str = "gpt2", layer: int = 8, device: str = "cpu") -> None:
        self.device = device
        self.layer = layer
        self.tok = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self._captured: torch.Tensor | None = None
        self._inject_vec: torch.Tensor | None = None
        self._inject_alpha: float = 0.0
        self._inject_positions: list[int] = [-1]

        self._handle = self.model.transformer.h[layer].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):  # noqa: ARG002
        h = output[0] if isinstance(output, tuple) else output
        self._captured = h.detach().clone()
        if self._inject_vec is not None:
            h = h.clone()
            for pos in self._inject_positions:
                h[:, pos, :] = h[:, pos, :] + self._inject_alpha * self._inject_vec
            if isinstance(output, tuple):
                return (h, *output[1:])
            return h
        return output

    def _encode(self, prompt: str) -> torch.Tensor:
        return self.tok(prompt, return_tensors="pt").input_ids.to(self.device)

    @torch.no_grad()
    def capture(self, prompt: str) -> np.ndarray:
        """Last-token residual at the configured layer, as float32 numpy."""
        self._inject_vec = None
        ids = self._encode(prompt)
        self.model(ids)
        assert self._captured is not None
        return self._captured[0, -1].cpu().float().numpy()

    @torch.no_grad()
    def logits_at(
        self,
        prompt: str,
        vec: np.ndarray | None,
        alpha: float,
        inject_position: int | list[int] = -1,
    ) -> np.ndarray:
        """Next-token logits at the final position, with optional injection.

        `inject_position` controls *where* in the sequence the injection lands
        (default -1 = last token, matching the v2 spec). Pass a list to inject
        at multiple positions in the same forward pass — useful for asserting
        a concept signal across an entire span (e.g. concept-onward). The
        captured logits are always read from the final position.
        """
        if vec is None:
            self._inject_vec = None
            self._inject_alpha = 0.0
        else:
            self._inject_vec = torch.tensor(vec, device=self.device, dtype=torch.float32)
            self._inject_alpha = float(alpha)
        self._inject_positions = (
            [inject_position] if isinstance(inject_position, int) else list(inject_position)
        )
        ids = self._encode(prompt)
        out = self.model(ids).logits[0, -1].cpu().float().numpy()
        # Reset so the next capture() / logits_at(None) is clean.
        self._inject_vec = None
        self._inject_alpha = 0.0
        self._inject_positions = [-1]
        return out

    @torch.no_grad()
    def log_probs_at(
        self,
        prompt: str,
        vec: np.ndarray | None,
        alpha: float,
        inject_position: int | list[int] = -1,
    ) -> np.ndarray:
        """Same as logits_at but returns log-softmax over vocabulary.

        Log-prob shifts cancel any uniform additive logit tilt: if injection
        adds a constant c to all logits, logsumexp also shifts by ~c, so
        log p = logit - logsumexp is invariant. This is the right metric for
        distinguishing "boosts vocabulary uniformly" from "selectively shifts
        probability mass."
        """
        logits = self.logits_at(prompt, vec=vec, alpha=alpha, inject_position=inject_position)
        # log_softmax in numpy: x - logsumexp(x)
        m = logits.max()
        return (logits - (m + np.log(np.exp(logits - m).sum()))).astype(np.float32)

    @torch.no_grad()
    def generate(self, prompt: str, vec: np.ndarray | None, alpha: float, n: int) -> str:
        """Greedy decode `n` tokens with optional injection held active across
        all steps. Injection fires at the last position of each forward pass —
        which advances by one token per step — so the axiom signal is asserted
        throughout generation, per v2 spec §5.
        """
        if vec is None:
            self._inject_vec = None
            self._inject_alpha = 0.0
        else:
            self._inject_vec = torch.tensor(vec, device=self.device, dtype=torch.float32)
            self._inject_alpha = float(alpha)
        ids = self._encode(prompt)
        for _ in range(n):
            out = self.model(ids).logits[0, -1]
            nxt = out.argmax().unsqueeze(0).unsqueeze(0)
            ids = torch.cat([ids, nxt], dim=1)
        self._inject_vec = None
        self._inject_alpha = 0.0
        return self.tok.decode(ids[0])

    @torch.no_grad()
    def capture_layers(self, prompt: str, layers: list[int]) -> dict[int, np.ndarray]:
        """Last-token residuals at multiple layers in a single forward pass.

        Uses HF's output_hidden_states rather than the injection hook. In
        GPT-2, hidden_states[i+1] is the output of block i (the same signal
        the hook captures), so layer L corresponds to hidden_states[L+1].
        """
        self._inject_vec = None
        ids = self._encode(prompt)
        out = self.model(ids, output_hidden_states=True)
        result: dict[int, np.ndarray] = {}
        for layer in layers:
            h = out.hidden_states[layer + 1]
            result[layer] = h[0, -1].cpu().float().numpy()
        return result

    @torch.no_grad()
    def capture_at_position(self, prompt: str, layer: int, position: int) -> np.ndarray:
        """Residual at an arbitrary token position. Negative positions count
        from the end (-1 == last). Used for variant experiments where we
        capture at the position of a specific term rather than at end-of-prompt.
        """
        self._inject_vec = None
        ids = self._encode(prompt)
        out = self.model(ids, output_hidden_states=True)
        h = out.hidden_states[layer + 1]
        return h[0, position].cpu().float().numpy()

    def find_token_positions(self, prompt: str, target: str) -> list[int]:
        """Return positions in the BPE-encoded prompt where `target` appears.

        Tries both the sentence-start tokenization and the with-leading-space
        tokenization, since GPT-2 BPE encodes them differently. Returns the
        position of the *last* token of each match (deduplicated).
        """
        prompt_ids = self.tok(prompt, add_special_tokens=False).input_ids
        candidates = [
            self.tok(target, add_special_tokens=False).input_ids,
            self.tok(" " + target, add_special_tokens=False).input_ids,
        ]
        seen: set[int] = set()
        for target_ids in candidates:
            if not target_ids:
                continue
            for i in range(len(prompt_ids) - len(target_ids) + 1):
                if prompt_ids[i : i + len(target_ids)] == target_ids:
                    seen.add(i + len(target_ids) - 1)
        return sorted(seen)

    def close(self) -> None:
        self._handle.remove()
