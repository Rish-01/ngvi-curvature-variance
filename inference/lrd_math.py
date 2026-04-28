from __future__ import annotations

import math
import torch


def lrd_reparameterize(
    mu: torch.Tensor,
    log_s: torch.Tensor,
    V: torch.Tensor,
    n_samples: int,
) -> torch.Tensor:
    """z ~ N(mu, D + V V^T) with the standard reparam above."""
    Ddim, r = V.shape
    s = log_s.exp()
    eps1 = torch.randn(n_samples, Ddim, device=mu.device, dtype=mu.dtype)
    eps2 = torch.randn(n_samples, r, device=mu.device, dtype=mu.dtype)
    return mu + s * eps1 + eps2 @ V.T


def lrd_log_prob(
    z: torch.Tensor,
    mu: torch.Tensor,
    log_s: torch.Tensor,
    V: torch.Tensor,
) -> torch.Tensor:
    """
    Batched log N(z; mu, D + V V^T) for z of shape (S, D).
    """
    if z.dim() == 1:
        z = z.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    s = log_s.exp()
    d = (s * s).clamp(min=1e-30)
    d_inv = 1.0 / d
    Dlat = z.shape[1]
    r = V.shape[1]
    I = torch.eye(r, device=V.device, dtype=V.dtype)
    W = V * d.sqrt().reciprocal().unsqueeze(1)
    K = I + W.T @ W
    logdet_d = d.log().sum()
    logdet_k = torch.linalg.slogdet(K)[1]
    logdet_sigma = logdet_d + logdet_k

    delta = z - mu
    alpha = delta * d_inv
    quad1 = (delta * alpha).sum(-1)
    b = alpha @ V
    w = torch.linalg.solve(K, b.T)
    quad2 = (b * w.T).sum(-1)
    quad = quad1 - quad2

    two_pi = torch.as_tensor(2.0 * math.pi, device=z.device, dtype=z.dtype)
    out = -0.5 * (Dlat * two_pi.log() + logdet_sigma) - 0.5 * quad
    if squeeze:
        return out.squeeze(0)
    return out
