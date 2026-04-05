"""
Eight Schools
-------------
A classical hierarchical Gaussian model (Rubin, 1981) that exhibits
funnel-like geometry in the centered parameterization.

Centered parameterization (CP):
    mu      ~ Normal(0, 5)
    log_tau ~ Normal(0, 1)          [tau = exp(log_tau)]
    theta_j ~ Normal(mu, tau)       for j = 1, ..., J
    y_j     ~ Normal(theta_j, sigma_j)   [sigma_j observed]

Non-centered parameterization (NCP):
    mu      ~ Normal(0, 5)
    log_tau ~ Normal(0, 1)
    eta_j   ~ Normal(0, 1)          [theta_j = mu + tau * eta_j]
    y_j     ~ Normal(mu + tau * eta_j, sigma_j)

The NCP breaks the funnel correlation between tau and theta, making
it much better conditioned and suitable for BBVI / HMC.

Reference: Rubin (1981); Betancourt & Girolami (2015).
"""

import torch


# Observed data (Rubin, 1981)
_Y = torch.tensor([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
_SIGMA = torch.tensor([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])


class EightSchools:
    """
    Eight Schools hierarchical model in centered or non-centered form.

    Parameters
    ----------
    parameterization : str
        'centered' or 'noncentered'.
    y : Tensor of shape (J,), optional
        Observed treatment effects. Defaults to Rubin's original data.
    sigma : Tensor of shape (J,), optional
        Observed standard errors. Defaults to Rubin's original data.

    Latent variable layout
    ----------------------
    Centered (D = J + 2):
        z[0]   = mu
        z[1]   = log_tau
        z[2:]  = theta_1, ..., theta_J

    Non-centered (D = J + 2):
        z[0]   = mu
        z[1]   = log_tau
        z[2:]  = eta_1, ..., eta_J
    """

    def __init__(
        self,
        parameterization: str = "noncentered",
        y: torch.Tensor = None,
        sigma: torch.Tensor = None,
    ):
        assert parameterization in (
            "centered",
            "noncentered",
        ), "parameterization must be 'centered' or 'noncentered'."
        self.parameterization = parameterization
        self.y = y if y is not None else _Y.clone()
        self.sigma = sigma if sigma is not None else _SIGMA.clone()
        self.J = len(self.y)
        self.D = self.J + 2  # mu, log_tau, theta/eta x J

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _unpack(self, z: torch.Tensor):
        """Split z into (mu, log_tau, theta_or_eta)."""
        mu = z[..., 0]
        log_tau = z[..., 1]
        phi = z[..., 2:]        # theta (CP) or eta (NCP)
        return mu, log_tau, phi

    def _to_theta(self, mu, log_tau, phi) -> torch.Tensor:
        """Convert phi to theta (no-op for CP; reparameterize for NCP)."""
        tau = torch.exp(log_tau)
        if self.parameterization == "centered":
            return phi
        else:
            return mu.unsqueeze(-1) + tau.unsqueeze(-1) * phi

    # ------------------------------------------------------------------
    # Log-joint
    # ------------------------------------------------------------------

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(params, y) for a batch of samples.

        Parameters
        ----------
        z : Tensor of shape (..., D)

        Returns
        -------
        Tensor of shape (...)
        """
        mu, log_tau, phi = self._unpack(z)
        tau = torch.exp(log_tau)
        theta = self._to_theta(mu, log_tau, phi)

        # Priors
        log_p_mu = _normal_log_prob(mu, 0.0, 5.0)
        log_p_log_tau = _normal_log_prob(log_tau, 0.0, 1.0)

        if self.parameterization == "centered":
            # theta_j ~ Normal(mu, tau)
            log_p_theta = _normal_log_prob(
                phi, mu.unsqueeze(-1), tau.unsqueeze(-1)
            ).sum(-1)
        else:
            # eta_j ~ Normal(0, 1)
            log_p_theta = _normal_log_prob(phi, 0.0, 1.0).sum(-1)
            # Jacobian: d(theta)/d(eta) = tau for each j → log|J| = J * log_tau
            log_p_theta += self.J * log_tau

        # Likelihood: y_j ~ Normal(theta_j, sigma_j)
        log_lik = _normal_log_prob(
            self.y, theta, self.sigma
        ).sum(-1)

        return log_p_mu + log_p_log_tau + log_p_theta + log_lik

    # ------------------------------------------------------------------
    # Geometry utilities
    # ------------------------------------------------------------------

    def hessian(self, z: torch.Tensor) -> torch.Tensor:
        """
        Exact Hessian of -log_prob at z via autograd.

        Parameters
        ----------
        z : Tensor of shape (D,)

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
        return -H

    def fisher_spectrum(self, z: torch.Tensor) -> torch.Tensor:
        """Eigenvalues of the empirical Fisher ≈ Hessian, descending."""
        H = self.hessian(z)
        eigvals = torch.linalg.eigvalsh(H)
        return eigvals.flip(0)

    def condition_number(self, z: torch.Tensor) -> float:
        """Condition number of the Hessian at z."""
        eigvals = self.fisher_spectrum(z).abs()
        return (eigvals.max() / eigvals.min()).item()

    # ------------------------------------------------------------------
    # Reparameterization helpers
    # ------------------------------------------------------------------

    def to_noncentered(self, z_cp: torch.Tensor) -> torch.Tensor:
        """
        Convert centered samples to non-centered coordinates.

        Parameters
        ----------
        z_cp : Tensor of shape (..., D) in centered parameterization.

        Returns
        -------
        Tensor of shape (..., D) in non-centered parameterization.
        """
        mu, log_tau, theta = self._unpack(z_cp)
        tau = torch.exp(log_tau)
        eta = (theta - mu.unsqueeze(-1)) / tau.unsqueeze(-1)
        return torch.cat([mu.unsqueeze(-1), log_tau.unsqueeze(-1), eta], dim=-1)

    def to_centered(self, z_ncp: torch.Tensor) -> torch.Tensor:
        """
        Convert non-centered samples to centered coordinates.

        Parameters
        ----------
        z_ncp : Tensor of shape (..., D) in non-centered parameterization.

        Returns
        -------
        Tensor of shape (..., D) in centered parameterization.
        """
        mu, log_tau, eta = self._unpack(z_ncp)
        tau = torch.exp(log_tau)
        theta = mu.unsqueeze(-1) + tau.unsqueeze(-1) * eta
        return torch.cat([mu.unsqueeze(-1), log_tau.unsqueeze(-1), theta], dim=-1)


# ------------------------------------------------------------------
# Module-level utility
# ------------------------------------------------------------------

def _normal_log_prob(
    x: torch.Tensor, mu: float | torch.Tensor, sigma: float | torch.Tensor
) -> torch.Tensor:
    """Elementwise log Normal(x; mu, sigma)."""
    return (
        -0.5 * ((x - mu) / sigma) ** 2
        - torch.log(torch.as_tensor(sigma, dtype=x.dtype))
        - 0.5 * torch.log(torch.tensor(2 * torch.pi))
    )