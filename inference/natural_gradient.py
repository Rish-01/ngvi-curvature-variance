"""
Natural Gradient Variational Inference
----------------------------------------
Preconditions ELBO gradients with the inverse Fisher Information Matrix
of the variational family, yielding the steepest ascent direction in the
space of probability distributions (Amari, 1998).

Theory
------
For a mean-field Gaussian q(z; phi) = prod_i N(z_i; mu_i, sigma_i^2),
the variational parameters are phi = (mu, sigma).  The natural gradient
is:

    phi_{t+1} = phi_t + eta * F(phi_t)^{-1} * grad_phi L(phi_t)

where F(phi) is the Fisher Information Matrix:

    F(phi) = E_{q_phi}[ (d/dphi log q_phi(z)) (d/dphi log q_phi(z))^T ]

For mean-field Gaussian with params (mu, log_sigma), F has a closed form:

    F_mu       = diag(1 / sigma^2)
    F_log_sigma = diag(2)            [constant wrt sigma for log parameterization]

In the (mu, sigma) parameterization the block-diagonal Fisher is:

    F = block_diag( diag(sigma^{-2}),  diag(2 * sigma^{-2}) )

Inverting is O(D) -- no matrix factorization needed.

This gives the exact natural gradient for the mean-field Gaussian family.

Reference: Amari, "Natural Gradient Works Efficiently in Learning",
           Neural Computation, 1998.
"""

import time
import torch


class NaturalGradientVI:
    """
    Natural Gradient VI for a mean-field Gaussian variational family.

    Parameters
    ----------
    model : object
        Must implement .log_prob(z) -> Tensor of shape (n_samples,).
    D : int
        Latent dimension.
    lr : float
        Step size (learning rate) for natural gradient updates.
    n_samples : int
        Monte Carlo samples per gradient estimate.
    damping : float
        Added to Fisher diagonal before inversion for numerical stability.
        Acts as a trust-region: larger values -> more conservative steps.
    """

    def __init__(
        self,
        model,
        D: int,
        lr: float = 1e-2,
        n_samples: int = 10,
        damping: float = 1e-4,
    ):
        self.model = model
        self.D = D
        self.lr = lr
        self.n_samples = n_samples
        self.damping = damping

        # Variational parameters (no optimizer -- we update manually)
        self.mu = torch.zeros(D)
        self.log_sigma = torch.zeros(D)

        # Diagnostics
        self.elbo_history = []
        self.grad_var_history = []
        self.fisher_cond_history = []
        self.wall_times = []

    # ------------------------------------------------------------------
    # Fisher Information (closed form for mean-field Gaussian)
    # ------------------------------------------------------------------

    def fisher_diagonal(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the closed-form Fisher diagonal for (mu, log_sigma).

        For q = N(mu, diag(sigma^2)):
            F_mu_ii        = 1 / sigma_i^2
            F_logsigma_ii  = 2.0   (constant)

        Returns
        -------
        f_mu : Tensor (D,)
        f_log_sigma : Tensor (D,)
        """
        sigma_sq = torch.exp(2 * self.log_sigma)               # (D,)
        f_mu = 1.0 / sigma_sq                                  # (D,)
        f_log_sigma = torch.full((self.D,), 2.0)               # (D,)
        return f_mu, f_log_sigma

    def fisher_condition_number(self) -> float:
        """
        Condition number of the Fisher diagonal (max / min eigenvalue).
        Useful as a curvature diagnostic.
        """
        f_mu, f_ls = self.fisher_diagonal()
        all_diag = torch.cat([f_mu, f_ls]) + self.damping
        return (all_diag.max() / all_diag.min()).item()

    # ------------------------------------------------------------------
    # ELBO gradient (Euclidean)
    # ------------------------------------------------------------------

    def elbo_and_grad(self) -> tuple[float, torch.Tensor, torch.Tensor]:
        """
        Compute ELBO and Euclidean gradients wrt (mu, log_sigma).

        Returns
        -------
        elbo_val : float
        grad_mu : Tensor (D,)
        grad_log_sigma : Tensor (D,)
        """
        mu = self.mu.clone().requires_grad_(True)
        log_sigma = self.log_sigma.clone().requires_grad_(True)
        sigma = torch.exp(log_sigma)

        eps = torch.randn(self.n_samples, self.D)
        z = mu + sigma * eps

        log_joint = self.model.log_prob(z)                     # (S,)
        log_q = (
            -0.5 * (eps ** 2).sum(-1)
            - log_sigma.sum()
            - 0.5 * self.D * torch.log(torch.tensor(2 * torch.pi))
        )                                                      # (S,)

        elbo = (log_joint - log_q).mean()
        elbo.backward()

        return (
            elbo.item(),
            mu.grad.detach().clone(),
            log_sigma.grad.detach().clone(),
        )

    # ------------------------------------------------------------------
    # Natural gradient step
    # ------------------------------------------------------------------

    def step(self) -> float:
        """
        Perform one natural gradient update.

        Returns the current ELBO value.
        """
        elbo_val, grad_mu, grad_ls = self.elbo_and_grad()

        # Fisher diagonal + damping
        f_mu, f_ls = self.fisher_diagonal()
        f_mu_damp = f_mu + self.damping
        f_ls_damp = f_ls + self.damping

        # Natural gradient = F^{-1} * grad
        nat_grad_mu = grad_mu / f_mu_damp
        nat_grad_ls = grad_ls / f_ls_damp

        # Gradient ascent
        self.mu = self.mu + self.lr * nat_grad_mu
        self.log_sigma = self.log_sigma + self.lr * nat_grad_ls

        return elbo_val

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        n_iters: int = 2000,
        log_every: int = 100,
        track_grad_var: bool = False,
        track_fisher_cond: bool = False,
    ) -> dict:
        """
        Run the full optimization loop.

        Parameters
        ----------
        n_iters : int
        log_every : int
        track_grad_var : bool
            Estimate gradient variance (extra forward passes).
        track_fisher_cond : bool
            Record condition number of the Fisher at each log step.

        Returns
        -------
        dict with keys: mu, sigma, elbo, grad_var, fisher_cond, wall_time
        """
        t0 = time.time()

        for i in range(1, n_iters + 1):
            elbo_val = self.step()

            if i % log_every == 0:
                self.elbo_history.append(elbo_val)
                self.wall_times.append(time.time() - t0)

                if track_fisher_cond:
                    self.fisher_cond_history.append(
                        self.fisher_condition_number()
                    )

                if track_grad_var:
                    self.grad_var_history.append(
                        self.estimate_grad_var()
                    )

                print(
                    f"[NaturalGradVI] iter {i:5d} | ELBO {elbo_val:+.4f}"
                    + (f" | cond(F) {self.fisher_cond_history[-1]:.2e}"
                       if track_fisher_cond else "")
                    + (f" | grad_var {self.grad_var_history[-1]:.4e}"
                       if track_grad_var else "")
                )

        return self.results()

    # ------------------------------------------------------------------
    # Results and diagnostics
    # ------------------------------------------------------------------

    def results(self) -> dict:
        return {
            "mu": self.mu.detach().clone(),
            "sigma": torch.exp(self.log_sigma).detach().clone(),
            "elbo": self.elbo_history,
            "grad_var": self.grad_var_history,
            "fisher_cond": self.fisher_cond_history,
            "wall_time": self.wall_times,
        }

    def estimate_grad_var(self) -> float:
        """
        Estimate ||Var(natural_grad)||_2 across single-sample estimates.
        """
        per_sample_nat_grads = []
        f_mu, f_ls = self.fisher_diagonal()
        f_mu_damp = f_mu + self.damping
        f_ls_damp = f_ls + self.damping

        for _ in range(self.n_samples):
            mu = self.mu.clone().requires_grad_(True)
            log_sigma = self.log_sigma.clone().requires_grad_(True)
            sigma = torch.exp(log_sigma)

            eps = torch.randn(1, self.D)
            z = mu + sigma * eps
            log_joint = self.model.log_prob(z)
            log_q = (
                -0.5 * (eps ** 2).sum(-1)
                - log_sigma.sum()
                - 0.5 * self.D * torch.log(torch.tensor(2 * torch.pi))
            )
            elbo = (log_joint - log_q).mean()
            elbo.backward()

            ng = torch.cat([
                mu.grad.detach() / f_mu_damp,
                log_sigma.grad.detach() / f_ls_damp,
            ])
            per_sample_nat_grads.append(ng)

        grads = torch.stack(per_sample_nat_grads)
        return grads.var(dim=0).norm().item()

    def max_stable_lr(
        self,
        lr_grid: list = None,
        n_probe_iters: int = 200,
    ) -> float:
        """Find the largest stable learning rate via grid search."""
        if lr_grid is None:
            lr_grid = [10 ** (-k / 2) for k in range(0, 8)]  # 1.0 → ~3e-4

        for lr in sorted(lr_grid, reverse=True):
            probe = NaturalGradientVI(
                self.model, self.D, lr=lr,
                n_samples=self.n_samples, damping=self.damping
            )
            probe.mu.data = self.mu.clone()
            probe.log_sigma.data = self.log_sigma.clone()

            try:
                for _ in range(n_probe_iters):
                    val = probe.step()
                    if not torch.isfinite(torch.tensor(val)):
                        raise ValueError("diverged")
                return lr
            except (ValueError, RuntimeError):
                continue

        return lr_grid[-1]