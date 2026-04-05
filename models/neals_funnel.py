"""
Neal's Funnel
-------------
A synthetic hierarchical model with extreme curvature anisotropy.

    v   ~ Normal(0, 3)
    x_i ~ Normal(0, exp(v/2))   for i = 1, ..., D-1

The marginal geometry is highly non-Gaussian: the variance of each x_i
is controlled by v, creating a funnel-shaped posterior that is pathological
for inference algorithms relying on Euclidean geometry.

Reference: Neal (2003), "Slice Sampling", Annals of Statistics.
"""

import numpy as np
import torch
import torch.nn as nn


class NealsFunnel:
    """
    Neal's Funnel log-joint and geometry utilities.

    Parameters
    ----------
    D : int
        Total dimension (1 funnel variable v + D-1 normal variables x).
    v_scale : float
        Prior std of v. Default is 3 (Neal's original setting).
    """

    def __init__(self, D: int = 10, v_scale: float = 3.0):
        assert D >= 2, "D must be at least 2 (1 funnel var + 1 data var)."
        self.D = D
        self.v_scale = v_scale

    # ------------------------------------------------------------------
    # Log-joint
    # ------------------------------------------------------------------

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(v, x) for a batch of samples.

        Parameters
        ----------
        z : Tensor of shape (..., D)
            z[..., 0] is v; z[..., 1:] are the x_i's.

        Returns
        -------
        Tensor of shape (...)
            Log joint probability for each sample.
        """
        v = z[..., 0]           # (...,)
        x = z[..., 1:]          # (..., D-1)

        # log p(v)
        log_pv = -0.5 * (v / self.v_scale) ** 2 - torch.log(
            torch.tensor(self.v_scale * (2 * torch.pi) ** 0.5)
        )

        # log p(x | v): each x_i ~ Normal(0, exp(v/2)); Var(x_i) = exp(v)
        log_var = v
        var_x = torch.exp(log_var).unsqueeze(-1)  # (..., 1), broadcast over x dims
        log_px = -0.5 * (self.D - 1) * log_var - 0.5 * (
            (x ** 2 / var_x).sum(-1)
        )
        log_px -= 0.5 * (self.D - 1) * torch.log(torch.tensor(2 * torch.pi))

        return log_pv + log_px

    # ------------------------------------------------------------------
    # Geometry utilities
    # ------------------------------------------------------------------

    def true_variance(self, v_samples: torch.Tensor) -> torch.Tensor:
        """
        Analytic conditional variance of x given v: Var(x_i | v) = exp(v).

        Useful for sanity-checking posterior approximations.
        """
        return torch.exp(v_samples)

    def condition_number_analytic(self, v: float) -> float:
        """
        Rough analytic condition number of the Hessian at a point.

        The Hessian eigenvalues scale as 1/v_scale^2 (for v) and
        exp(-v) (for x). Their ratio gives a measure of ill-conditioning.

        Parameters
        ----------
        v : float
            Value of the funnel variable.

        Returns
        -------
        float
            Ratio of largest to smallest eigenvalue magnitude.
        """
        lambda_v = 1.0 / self.v_scale ** 2
        lambda_x = np.exp(-v)       # 1 / Var(x|v)
        return max(lambda_v, lambda_x) / min(lambda_v, lambda_x)

    def hessian(self, z: torch.Tensor) -> torch.Tensor:
        """
        Exact Hessian of -log_prob at z via autograd.

        Parameters
        ----------
        z : Tensor of shape (D,)  [single point, requires_grad]

        Returns
        -------
        Tensor of shape (D, D)
        """
        z = z.detach().requires_grad_(True)
        lp = self.log_prob(z)
        grad = torch.autograd.grad(lp, z, create_graph=True)[0]
        H = torch.stack([
            torch.autograd.grad(grad[i], z, retain_graph=True)[0]
            for i in range(self.D)
        ])
        return -H  # Hessian of negative log prob

    def fisher_spectrum(self, z: torch.Tensor) -> torch.Tensor:
        """
        Eigenvalues of the empirical Fisher ≈ Hessian at z.

        Returns eigenvalues in descending order.
        """
        H = self.hessian(z)
        eigvals = torch.linalg.eigvalsh(H)
        return eigvals.flip(0)

    # ------------------------------------------------------------------
    # Sampling (ground truth via reparameterization)
    # ------------------------------------------------------------------

    def sample(self, n: int, rng: torch.Generator = None) -> torch.Tensor:
        """
        Draw exact samples from the prior p(v, x).

        Parameters
        ----------
        n : int
        rng : optional torch.Generator

        Returns
        -------
        Tensor of shape (n, D)
        """
        kwargs = {"generator": rng} if rng is not None else {}
        v = torch.randn(n, **kwargs) * self.v_scale          # (n,)
        x = torch.randn(n, self.D - 1, **kwargs) * torch.exp(v / 2).unsqueeze(1)
        return torch.cat([v.unsqueeze(1), x], dim=1)