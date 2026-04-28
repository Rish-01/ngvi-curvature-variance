"""
Variance-reduced BBVI (antithetic MC + sticking-the-landing)
------------------------------------------------------------
Extends BBVI + Adam with optional antithetic sampling for the reparameterized
noise and the STL gradient estimator (Roeder et al., 2017).
"""

import torch

from .bbvi_adam import BBVIAdam
from .lrd_math import lrd_log_prob, lrd_reparameterize


class VarianceReducedBBVI(BBVIAdam):
    """
    BBVIAdam with antithetic pairs and/or sticking-the-landing for log q(z|phi).
    """

    def __init__(
        self,
        model,
        D: int,
        lr: float = 1e-3,
        n_samples: int = 10,
        betas: tuple = (0.9, 0.999),
        eps_adam: float = 1e-8,
        use_antithetic: bool = True,
        use_stl: bool = True,
        variational_family: str = "mean_field",
        low_rank: int = 3,
        v_init_scale: float = 0.01,
    ):
        super().__init__(
            model, D, lr=lr, n_samples=n_samples, betas=betas, eps_adam=eps_adam,
            variational_family=variational_family, low_rank=low_rank, v_init_scale=v_init_scale,
        )
        self.use_antithetic = use_antithetic
        self.use_stl = use_stl

    def _log_q(self, z: torch.Tensor, sigma: torch.Tensor | None) -> torch.Tensor:
        """Mean-field Gaussian log density; z may be detached for STL."""
        pi = torch.tensor(2 * torch.pi, device=z.device, dtype=z.dtype)
        if self.variational_family == "mean_field":
            return (
                -0.5 * ((z - self.mu) ** 2 / (sigma ** 2)).sum(-1)
                - self.log_sigma.sum()
                - 0.5 * self.D * torch.log(pi)
            )
        return lrd_log_prob(z, self.mu, self.log_s, self.V)

    def elbo(self) -> torch.Tensor:
        if self.variational_family == "mean_field":
            sigma = torch.exp(self.log_sigma)

        if self.use_antithetic:
            n_pair = self.n_samples // 2
            eps_half = torch.randn(max(1, n_pair), self.D)
            eps = torch.cat([eps_half, -eps_half], dim=0)
            if self.n_samples % 2 == 1:
                eps = torch.cat([eps, torch.randn(1, self.D)], dim=0)
            if eps.shape[0] > self.n_samples:
                eps = eps[: self.n_samples]
        else:
            eps = torch.randn(self.n_samples, self.D)

        if self.variational_family == "mean_field":
            z = self.mu + sigma * eps
        else:
            # Antithetic in latent space for low-rank factor noise.
            if self.use_antithetic:
                n_pair = self.n_samples // 2
                r = self.low_rank
                eps1_h = torch.randn(max(1, n_pair), self.D)
                eps2_h = torch.randn(max(1, n_pair), r)
                eps1 = torch.cat([eps1_h, -eps1_h], dim=0)
                eps2 = torch.cat([eps2_h, -eps2_h], dim=0)
                if self.n_samples % 2 == 1:
                    eps1 = torch.cat([eps1, torch.randn(1, self.D)], dim=0)
                    eps2 = torch.cat([eps2, torch.randn(1, r)], dim=0)
                eps1 = eps1[: self.n_samples]
                eps2 = eps2[: self.n_samples]
                s = torch.exp(self.log_s)
                z = self.mu + s * eps1 + eps2 @ self.V.T
            else:
                z = lrd_reparameterize(self.mu, self.log_s, self.V, self.n_samples)
        log_joint = self.model.log_prob(z)
        z_q = z.detach() if self.use_stl else z
        log_q = self._log_q(z_q, sigma if self.variational_family == "mean_field" else None)
        return (log_joint - log_q).mean()

    def estimate_grad_var(self) -> float:
        per_sample_grads = []

        for _ in range(self.n_samples):
            self.optimizer.zero_grad()
            if self.variational_family == "mean_field":
                sigma = torch.exp(self.log_sigma)
                if self.use_antithetic:
                    eps_half = torch.randn(self.n_samples // 2, self.D)
                    eps = torch.cat([eps_half, -eps_half], dim=0)
                else:
                    eps = torch.randn(self.n_samples, self.D)
                z = self.mu + sigma * eps
            else:
                z = lrd_reparameterize(self.mu, self.log_s, self.V, self.n_samples)
            log_joint = self.model.log_prob(z)
            z_q = z.detach() if self.use_stl else z
            log_q = self._log_q(z_q, sigma if self.variational_family == "mean_field" else None)
            loss = -(log_joint - log_q).mean()
            loss.backward()
            if self.variational_family == "mean_field":
                g = torch.cat(
                    [self.mu.grad.detach().clone(), self.log_sigma.grad.detach().clone()]
                )
            else:
                g = torch.cat([
                    self.mu.grad.detach().flatten().clone(),
                    self.log_s.grad.detach().flatten().clone(),
                    self.V.grad.detach().flatten().clone(),
                ])
            per_sample_grads.append(g)

        grads = torch.stack(per_sample_grads)
        return grads.var(dim=0).norm().item()
