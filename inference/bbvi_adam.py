"""
BBVI + Adam (Euclidean baseline)
---------------------------------
Maximizes the ELBO using reparameterized Monte Carlo gradients,
optimized with Adam.

ELBO:
    L(phi) = E_{q_phi}[log p(z, x) - log q_phi(z)]
           = E_{eps~N(0,I)}[log p(mu + sigma * eps, x)
                            - log q_phi(mu + sigma * eps)]

Reparameterization trick:
    z = mu + sigma * eps,  eps ~ N(0, I)
    => unbiased, low-variance gradient wrt (mu, sigma)

The variational family is a mean-field Gaussian:
    q(z) = prod_i N(z_i; mu_i, sigma_i^2)

Parameters stored as (mu, log_sigma) for unconstrained optimization.
sigma = exp(log_sigma) enforces positivity.

Reference: Kingma & Welling, "Auto-Encoding Variational Bayes", ICLR 2014.
"""

import time
import torch


class BBVIAdam:
    """
    Black-Box Variational Inference with Adam optimizer.

    Parameters
    ----------
    model : object
        Must implement .log_prob(z) -> Tensor of shape (n_samples,)
        where z has shape (n_samples, D).
    D : int
        Latent dimension.
    lr : float
        Adam learning rate.
    n_samples : int
        Number of Monte Carlo samples per gradient estimate.
    betas : tuple
        Adam (beta1, beta2).
    eps_adam : float
        Adam numerical stability constant.
    """

    def __init__(
        self,
        model,
        D: int,
        lr: float = 1e-3,
        n_samples: int = 10,
        betas: tuple = (0.9, 0.999),
        eps_adam: float = 1e-8,
    ):
        self.model = model
        self.D = D
        self.lr = lr
        self.n_samples = n_samples
        self.betas = betas
        self.eps_adam = eps_adam

        # Variational parameters: (mu, log_sigma)
        self.mu = torch.zeros(D, requires_grad=True)
        self.log_sigma = torch.zeros(D, requires_grad=True)

        self.optimizer = torch.optim.Adam(
            [self.mu, self.log_sigma],
            lr=lr,
            betas=betas,
            eps=eps_adam,
        )

        # Diagnostics
        self.elbo_history = []
        self.grad_var_history = []
        self.wall_times = []

    # ------------------------------------------------------------------
    # Core ELBO and gradient
    # ------------------------------------------------------------------

    def elbo(self) -> torch.Tensor:
        """
        Estimate the ELBO via reparameterized Monte Carlo.

        Returns
        -------
        Scalar tensor (negative ELBO for minimization).
        """
        sigma = torch.exp(self.log_sigma)                      # (D,)

        # Reparameterize: z = mu + sigma * eps
        eps = torch.randn(self.n_samples, self.D)              # (S, D)
        z = self.mu + sigma * eps                              # (S, D)

        # log p(z, x)
        log_joint = self.model.log_prob(z)                     # (S,)

        # log q(z) = sum_i N(z_i; mu_i, sigma_i)
        log_q = -0.5 * (eps ** 2).sum(-1) \
                - self.log_sigma.sum() \
                - 0.5 * self.D * torch.log(torch.tensor(2 * torch.pi))   # (S,)

        elbo_samples = log_joint - log_q                       # (S,)
        return elbo_samples.mean()

    def estimate_grad_var(self) -> float:
        """
        Estimate ||Var(grad)||_2 across n_samples independent gradient estimates.

        Returns the Frobenius norm of the empirical per-coordinate variance.
        """
        per_sample_grads = []

        for _ in range(self.n_samples):
            self.optimizer.zero_grad()
            sigma = torch.exp(self.log_sigma)
            eps = torch.randn(1, self.D)
            z = self.mu + sigma * eps
            log_joint = self.model.log_prob(z)
            log_q = -0.5 * (eps ** 2).sum(-1) \
                    - self.log_sigma.sum() \
                    - 0.5 * self.D * torch.log(torch.tensor(2 * torch.pi))
            loss = -(log_joint - log_q).mean()
            loss.backward()
            g = torch.cat([self.mu.grad.detach().clone(),
                           self.log_sigma.grad.detach().clone()])
            per_sample_grads.append(g)

        grads = torch.stack(per_sample_grads)                  # (S, 2D)
        var = grads.var(dim=0)                                 # (2D,)
        return var.norm().item()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def step(self) -> float:
        """Perform a single Adam update. Returns current ELBO value."""
        self.optimizer.zero_grad()
        loss = -self.elbo()
        loss.backward()
        self.optimizer.step()
        return -loss.item()

    def fit(
        self,
        n_iters: int = 2000,
        log_every: int = 100,
        track_grad_var: bool = False,
    ) -> dict:
        """
        Run the full optimization loop.

        Parameters
        ----------
        n_iters : int
            Number of gradient steps.
        log_every : int
            Record diagnostics every this many iterations.
        track_grad_var : bool
            If True, estimate gradient variance at each log step
            (adds extra forward passes).

        Returns
        -------
        dict with keys: elbo, grad_var, wall_time, mu, sigma
        """
        t0 = time.time()

        for i in range(1, n_iters + 1):
            elbo_val = self.step()

            if i % log_every == 0:
                self.elbo_history.append(elbo_val)
                self.wall_times.append(time.time() - t0)

                if track_grad_var:
                    self.grad_var_history.append(self.estimate_grad_var())

                print(
                    f"[BBVIAdam] iter {i:5d} | ELBO {elbo_val:+.4f}"
                    + (f" | grad_var {self.grad_var_history[-1]:.4e}"
                       if track_grad_var else "")
                )

        return self.results()

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def results(self) -> dict:
        """Return current variational parameters and diagnostic history."""
        return {
            "mu": self.mu.detach().clone(),
            "sigma": torch.exp(self.log_sigma).detach().clone(),
            "elbo": self.elbo_history,
            "grad_var": self.grad_var_history,
            "wall_time": self.wall_times,
        }

    def max_stable_lr(
        self,
        lr_grid: list = None,
        n_probe_iters: int = 200,
    ) -> float:
        """
        Grid-search the largest learning rate that doesn't diverge.

        Parameters
        ----------
        lr_grid : list of floats
            Learning rates to probe (descending). Defaults to log-spaced
            grid from 1e-1 to 1e-4.
        n_probe_iters : int
            Iterations per probe.

        Returns
        -------
        float : Largest stable learning rate found.
        """
        if lr_grid is None:
            lr_grid = [10 ** (-k / 2) for k in range(2, 9)]  # 0.1 → ~3e-4

        for lr in sorted(lr_grid, reverse=True):
            mu0 = self.mu.detach().clone()
            ls0 = self.log_sigma.detach().clone()

            probe = BBVIAdam(self.model, self.D, lr=lr,
                             n_samples=self.n_samples)
            probe.mu.data.copy_(mu0)
            probe.log_sigma.data.copy_(ls0)

            try:
                for _ in range(n_probe_iters):
                    val = probe.step()
                    if not torch.isfinite(torch.tensor(val)):
                        raise ValueError("diverged")
                return lr
            except (ValueError, RuntimeError):
                continue

        return lr_grid[-1]