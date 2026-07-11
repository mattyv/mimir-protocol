"""Gist-space whitening (Stage-2 predictor, spec killer #2: anisotropy).

Gist / sentence-embedding space is cone-shaped — a large shared mean and
variance concentrated in a few directions, so raw regression wastes capacity
fighting the distortion. Whitening maps gists to zero-mean, identity-covariance
via g' = W (g - mu), W = Sigma^{-1/2} (symmetric ZCA whitening — keeps the
transform as close to identity as possible, unlike PCA whitening which
rotates). Fit once over ~1M training gists offline; store (mu, W, W_inv) as
part of the model contract. All prediction and losses run in whitened space;
invert before Stage-3 injection.
"""

from __future__ import annotations

from pathlib import Path

import torch


class Whitener:
    """ZCA whitener: transform() -> zero-mean identity-cov; inverse() undoes it."""

    def __init__(self, mean: torch.Tensor, w: torch.Tensor, w_inv: torch.Tensor):
        self.mean = mean
        self.w = w  # Sigma^{-1/2}
        self.w_inv = w_inv  # Sigma^{1/2}

    @staticmethod
    def _from_moments(n: int, s1: torch.Tensor, s2: torch.Tensor, eps: float) -> Whitener:
        """Build a Whitener from accumulated moments: n samples, s1 = sum(x),
        s2 = sum(x x^T). cov = (s2 - n mu mu^T)/(n-1)."""
        mean = s1 / n
        cov = (s2 - n * torch.outer(mean, mean)) / (n - 1)
        evals, evecs = torch.linalg.eigh(cov)
        evals = evals.clamp_min(eps)
        inv_sqrt = evecs @ torch.diag(evals.rsqrt()) @ evecs.T
        sqrt = evecs @ torch.diag(evals.sqrt()) @ evecs.T
        return Whitener(mean.float(), inv_sqrt.float(), sqrt.float())

    @classmethod
    def fit(cls, gists: torch.Tensor, eps: float = 1e-5) -> Whitener:
        """Fit from [N, d] gists in memory. eps floors eigenvalues for numerical
        safety (degenerate/near-zero-variance directions)."""
        g = gists.double()
        return cls._from_moments(g.shape[0], g.sum(0), g.T @ g, eps)

    @classmethod
    def fit_streaming(cls, chunks, eps: float = 1e-5) -> Whitener:  # noqa: ANN001
        """Fit from an iterable of [chunk_n, d] gist chunks, accumulating
        running moments (mean + outer-product sums) so the full [N, d] matrix
        is never materialized — required at corpus scale (Fable to-do)."""
        n = 0
        s1 = s2 = None
        for chunk in chunks:
            c = chunk.double()
            if s1 is None:
                s1 = torch.zeros(c.shape[1], dtype=torch.double)
                s2 = torch.zeros(c.shape[1], c.shape[1], dtype=torch.double)
            n += c.shape[0]
            s1 += c.sum(0)
            s2 += c.T @ c
        if n < 2:
            raise ValueError("need >= 2 samples to fit a whitener")
        return cls._from_moments(n, s1, s2, eps)

    def transform(self, g: torch.Tensor) -> torch.Tensor:
        return (g - self.mean) @ self.w.T

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.w_inv.T + self.mean

    def save(self, path: str | Path) -> None:
        torch.save({"mean": self.mean, "w": self.w, "w_inv": self.w_inv}, str(path))

    @classmethod
    def load(cls, path: str | Path) -> Whitener:
        d = torch.load(str(path), weights_only=True)
        return cls(d["mean"], d["w"], d["w_inv"])


class PerSlotWhitener:
    """One ZCA whitener per gist slot index (Fable steer #3) — slot 1 and slot
    8 are distributionally different, so each is whitened against its own
    statistics. Operates on [N, k, d] gist tensors."""

    def __init__(self, whiteners: list[Whitener]):
        self.whiteners = whiteners

    @classmethod
    def fit(cls, gists: torch.Tensor, eps: float = 1e-5) -> PerSlotWhitener:
        """gists [N, k, d] -> k whiteners (one per slot index)."""
        k = gists.shape[1]
        return cls([Whitener.fit(gists[:, s, :], eps) for s in range(k)])

    @classmethod
    def fit_streaming(cls, chunks, k: int, eps: float = 1e-5) -> PerSlotWhitener:  # noqa: ANN001
        """chunks yield [chunk_n, k, d]; accumulate per-slot moments in one
        pass over the stream (each chunk consumed once)."""
        n = 0
        s1 = s2 = None
        for chunk in chunks:
            c = chunk.double()
            if s1 is None:
                d = c.shape[2]
                s1 = torch.zeros(k, d, dtype=torch.double)
                s2 = torch.zeros(k, d, d, dtype=torch.double)
            n += c.shape[0]
            for s in range(k):
                cs = c[:, s, :]
                s1[s] += cs.sum(0)
                s2[s] += cs.T @ cs
        if n < 2:
            raise ValueError("need >= 2 samples to fit a whitener")
        return cls([Whitener._from_moments(n, s1[s], s2[s], eps) for s in range(k)])

    def transform(self, g: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.whiteners[s].transform(g[:, s, :]) for s in range(g.shape[1])], 1)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.whiteners[s].inverse(z[:, s, :]) for s in range(z.shape[1])], 1)

    def save(self, path: str | Path) -> None:
        torch.save(
            [{"mean": w.mean, "w": w.w, "w_inv": w.w_inv} for w in self.whiteners], str(path)
        )

    @classmethod
    def load(cls, path: str | Path) -> PerSlotWhitener:
        ws = torch.load(str(path), weights_only=True)
        return cls([Whitener(d["mean"], d["w"], d["w_inv"]) for d in ws])
