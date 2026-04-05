"""
Diagonal Fisher Quasi-Natural Gradient VI with Damping
--------------------------------------------------------
A computationally efficient middle ground between Adam (Euclidean) and
full Natural Gradient VI.

Theory
------
Full NGVI requires inverting the D×D Fisher matrix at O(D^3) cost.
Here we approximate the Fisher with its diagonal, estimated empirically
from MC gradient samples, and add a damping term lambda*I for stability:

    F_approx(phi) = diag( E[ (d/dphi log q_phi)^2 ] ) + lambda * I

The quasi-natural gradient update is then:

    phi_{t+1} = phi_t + eta * F_approx(phi_t)^{-1} * grad_phi L(phi_t)

This is O(D) per step -- identical to Adam -- but uses actual gradient
second-moment information from the current variational distribution
rather than a running exponential average (as Adam does).

Two Fisher estimation modes are supported:

  'analytic':  Closed-form diagonal for mean-field Gaussian (exact).
               F_mu_ii = 1/sigma_i^2,  F_logsigma_ii = 2.

  'empirical': Estimated from the MC gradient samples already drawn for
               the ELBO. No extra model evaluations needed. Useful when
               the variational family is extended beyond mean-field.

Damping lambda plays a dual role:
  - Prevents division by near-zero Fisher entries (numerical safety).
  - Interpolates between natural (lambda->0) and Euclidean (lambda->inf)
    updates. Ablation over lambda directly measures the preconditioning
    benefit vs variance tradeoff.

References
----------
- Amari (1998): Natural Gradient Works Efficiently in Learning.
- Martens (2014): New Insights and Perspectives on the Natural Gradient.
- Khan & Lin (2017): Conjugate-Computation Variational Inference.
"""

import time
import torch


class DiagonalFisherVI:
    """
    Diagonal-Fisher quasi-natural gradient VI.

    Parameters
    ----------
    model : object
        Must implement .log_prob(z) -> Tensor of shape (n_samples,).
    D : int
        Latent dimension.
    lr : float
        Step size for quasi-natural gradient updates.
    n_samples : int
        MC samples per ELBO gradient estimate.
    damping : float
        Regularization added to Fisher diagonal (lambda in the theory).
        Key ablation parameter: vary to isolate preconditioning benefit.
    fisher_mode : str
        'analytic' (closed-form, exact for mean-field Gaussian) or
        'empirical' (estimated from MC samples, general purpose).
    fisher_ema : float
        Exponential moving average coefficient for the empirical Fisher
        diagonal. 0 = no averaging (fresh each step), 0.9 = slow update.
        Ignored when fisher_mode='analytic'.
    """

    def __init__(
        self,
        model,
        D: int,
        lr: float = 1e-2,
        n_samples: int = 10,
        damping: float = 1e-3,
        fisher_mode: str = "analytic",
        fisher_ema: float = 0.9,
    ):
        assert fisher_mode in ("analytic", "empirical"), \
            "fisher_mode must be 'analytic' or 'empirical'."

        self.model = model
        self.D = D
        self.lr = lr
        self.n_samples = n_samples
        self.damping = damping
        self.fisher_mode = fisher_mode
        self.fisher_ema = fisher_ema

        # Variational parameters
        self.mu = torch.zeros(D)
        self.log_sigma = torch.zeros(D)

        # EMA state for empirical Fisher
        self._fish_mu_ema = torch.ones(D)
        self._fish_ls_ema = torch.ones(D)
        self._ema_initialized = False

        # Diagnostics
        self.elbo_history = []
        self.grad_var_history = []
        self.fisher_diag_history = []   # (iter, 2D) -- tracks Fisher evolution
        self.condition_history = []
        self.wall_times = []

    # ------------------------------------------------------------------
    # Fisher diagonal
    # ------------------------------------------------------------------

    def _analytic_fisher(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Closed-form Fisher diagonal for mean-field Gaussian.

        F_mu_ii       = 1 / sigma_i^2
        F_logsigma_ii = 2.0
        """
        sigma_sq = torch.exp(2 * self.log_sigma)
        return 1.0 / sigma_sq, torch.full((self.D,), 2.0)

    def _empirical_fisher(
        self,
        grad_mu_samples: torch.Tensor,
        grad_ls_samples: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate Fisher diagonal from per-sample score gradients.

        Parameters
        ----------
        grad_mu_samples : Tensor (S, D)  per-sample grad wrt mu
        grad_ls_samples : Tensor (S, D)  per-sample grad wrt log_sigma

        Returns
        -------
        f_mu : Tensor (D,)
        f_ls : Tensor (D,)
        """
        f_mu_raw = (grad_mu_samples ** 2).mean(0)              # (D,)
        f_ls_raw = (grad_ls_samples ** 2).mean(0)              # (D,)

        if not self._ema_initialized:
            self._fish_mu_ema = f_mu_raw.detach().clone()
            self._fish_ls_ema = f_ls_raw.detach().clone()
            self._ema_initialized = True
        else:
            a = self.fisher_ema
            self._fish_mu_ema = a * self._fish_mu_ema + (1 - a) * f_mu_raw.detach()
            self._fish_ls_ema = a * self._fish_ls_ema + (1 - a) * f_ls_raw.detach()

        return self._fish_mu_ema.clone(), self._fish_ls_ema.clone()

    def get_fisher_diagonal(
        self,
        grad_mu_samples: torch.Tensor = None,
        grad_ls_samples: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the (possibly damped) Fisher diagonal.

        Returns (f_mu, f_log_sigma), each of shape (D,), with damping applied.
        """
        if self.fisher_mode == "analytic":
            f_mu, f_ls = self._analytic_fisher()
        else:
            assert grad_mu_samples is not None, \
                "Empirical Fisher requires grad samples."
            f_mu, f_ls = self._empirical_fisher(grad_mu_samples, grad_ls_samples)

        return f_mu + self.damping, f_ls + self.damping

    def condition_number(self) -> float:
        """Condition number of the current (damped) Fisher diagonal."""
        f_mu, f_ls = self.get_fisher_diagonal()
        all_diag = torch.cat([f_mu, f_ls])
        return (all_diag.max() / all_diag.min()).item()

    # ------------------------------------------------------------------
    # ELBO and per-sample gradients
    # ------------------------------------------------------------------

    def elbo_and_grad(
        self,
    ) -> tuple[float, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor]:
        """
        Compute ELBO, mean gradient, and per-sample gradients.

        Returns
        -------
        elbo_val : float
        grad_mu : Tensor (D,)       mean gradient wrt mu
        grad_ls : Tensor (D,)       mean gradient wrt log_sigma
        grad_mu_samples : Tensor (S, D)
        grad_ls_samples : Tensor (S, D)
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

        # Per-sample ELBO for empirical Fisher
        elbo_samples = log_joint - log_q                       # (S,)

        # Per-sample gradients (needed for empirical Fisher)
        grad_mu_samples = []
        grad_ls_samples = []

        if self.fisher_mode == "empirical":
            for s in range(self.n_samples):
                mu2 = self.mu.clone().requires_grad_(True)
                ls2 = self.log_sigma.clone().requires_grad_(True)
                sg2 = torch.exp(ls2)
                z_s = mu2 + sg2 * eps[s]
                lj_s = self.model.log_prob(z_s.unsqueeze(0))
                lq_s = (-0.5 * (eps[s] ** 2).sum()
                        - ls2.sum()
                        - 0.5 * self.D * torch.log(torch.tensor(2 * torch.pi)))
                es = (lj_s.squeeze() - lq_s)
                es.backward()
                grad_mu_samples.append(mu2.grad.detach().clone())
                grad_ls_samples.append(ls2.grad.detach().clone())

            grad_mu_samples = torch.stack(grad_mu_samples)     # (S, D)
            grad_ls_samples = torch.stack(grad_ls_samples)     # (S, D)
        else:
            grad_mu_samples = None
            grad_ls_samples = None

        # Mean ELBO gradient
        elbo_mean = elbo_samples.mean()
        elbo_mean.backward()

        return (
            elbo_mean.item(),
            mu.grad.detach().clone(),
            log_sigma.grad.detach().clone(),
            grad_mu_samples,
            grad_ls_samples,
        )

    # ------------------------------------------------------------------
    # Update step
    # ------------------------------------------------------------------

    def step(self) -> float:
        """
        Perform one quasi-natural gradient update.

        Returns current ELBO value.
        """
        elbo_val, grad_mu, grad_ls, gmu_s, gls_s = self.elbo_and_grad()

        f_mu, f_ls = self.get_fisher_diagonal(gmu_s, gls_s)

        # Quasi-natural gradient = (diag(F) + lambda*I)^{-1} * grad
        qng_mu = grad_mu / f_mu
        qng_ls = grad_ls / f_ls

        self.mu = self.mu + self.lr * qng_mu
        self.log_sigma = self.log_sigma + self.lr * qng_ls

        return elbo_val

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        n_iters: int = 2000,
        log_every: int = 100,
        track_grad_var: bool = False,
        track_fisher: bool = False,
    ) -> dict:
        """
        Run optimization.

        Parameters
        ----------
        n_iters : int
        log_every : int
        track_grad_var : bool
        track_fisher : bool
            Record the Fisher diagonal at each log step. Useful for
            understanding how preconditioning evolves.
        """
        t0 = time.time()

        for i in range(1, n_iters + 1):
            elbo_val = self.step()

            if i % log_every == 0:
                self.elbo_history.append(elbo_val)
                self.wall_times.append(time.time() - t0)

                if track_fisher:
                    f_mu, f_ls = self.get_fisher_diagonal()
                    self.fisher_diag_history.append(
                        torch.cat([f_mu, f_ls]).detach().clone()
                    )
                    self.condition_history.append(self.condition_number())

                if track_grad_var:
                    self.grad_var_history.append(self.estimate_grad_var())

                print(
                    f"[DiagFisherVI] iter {i:5d} | ELBO {elbo_val:+.4f}"
                    + (f" | cond(F_diag) {self.condition_history[-1]:.2e}"
                       if track_fisher else "")
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
            "fisher_diag": self.fisher_diag_history,
            "fisher_cond": self.condition_history,
            "wall_time": self.wall_times,
        }

    def estimate_grad_var(self) -> float:
        """Estimate ||Var(quasi-natural grad)||_2 from single-sample estimates."""
        f_mu, f_ls = self.get_fisher_diagonal()
        per_sample = []

        for _ in range(self.n_samples):
            mu = self.mu.clone().requires_grad_(True)
            ls = self.log_sigma.clone().requires_grad_(True)
            sg = torch.exp(ls)
            eps = torch.randn(1, self.D)
            z = mu + sg * eps
            lj = self.model.log_prob(z)
            lq = (-0.5 * (eps ** 2).sum(-1)
                  - ls.sum()
                  - 0.5 * self.D * torch.log(torch.tensor(2 * torch.pi)))
            (lj - lq).mean().backward()
            ng = torch.cat([mu.grad.detach() / f_mu,
                            ls.grad.detach() / f_ls])
            per_sample.append(ng)

        grads = torch.stack(per_sample)
        return grads.var(dim=0).norm().item()

    def damping_sweep(
        self,
        damping_grid: list = None,
        n_iters: int = 500,
    ) -> dict:
        """
        Ablation: run short optimization for each damping value.

        Directly measures the preconditioning benefit vs Euclidean
        baseline by comparing final ELBO across damping levels.
        lambda -> 0 approaches true natural gradient;
        lambda -> inf approaches Euclidean (Adam-like) step.

        Returns
        -------
        dict mapping damping value -> final ELBO
        """
        if damping_grid is None:
            damping_grid = [1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0]

        results = {}
        for lam in damping_grid:
            probe = DiagonalFisherVI(
                self.model, self.D,
                lr=self.lr, n_samples=self.n_samples,
                damping=lam, fisher_mode=self.fisher_mode,
            )
            probe.mu = self.mu.clone()
            probe.log_sigma = self.log_sigma.clone()

            final_elbo = None
            for _ in range(n_iters):
                final_elbo = probe.step()
            results[lam] = final_elbo
            print(f"  damping={lam:.1e} -> ELBO {final_elbo:+.4f}")

        return results