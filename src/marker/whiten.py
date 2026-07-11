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

    @classmethod
    def fit(cls, gists: torch.Tensor, eps: float = 1e-5) -> Whitener:
        """Fit from [N, d] gists. eps floors eigenvalues for numerical safety
        (degenerate/near-zero-variance directions)."""
        g = gists.double()
        mean = g.mean(0)
        centered = g - mean
        cov = (centered.T @ centered) / (g.shape[0] - 1)
        # symmetric eigendecomposition; cov is PSD
        evals, evecs = torch.linalg.eigh(cov)
        evals = evals.clamp_min(eps)
        inv_sqrt = evecs @ torch.diag(evals.rsqrt()) @ evecs.T
        sqrt = evecs @ torch.diag(evals.sqrt()) @ evecs.T
        return cls(mean.float(), inv_sqrt.float(), sqrt.float())

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
