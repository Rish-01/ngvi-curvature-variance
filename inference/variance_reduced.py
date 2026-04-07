"""
Variance-Reduced BBVI
---------------------
Combines two complementary variance-reduction techniques:

1. Antithetic sampling (Owen, 2013):
   Draw S//2 base noise samples eps ~ N(0, I), then use both eps and -eps.
   For symmetric integrands the mirrored samples cancel noise, roughly
   halving variance at no extra model evaluations.

2. "Sticking the Landing" / STL (Roeder et al., 2017):
   The standard ELBO gradient includes a score-function component from
   differentiating -log q_phi(z) through phi at fixed z. This term has
   zero expectation but inflates variance. STL removes it by substituting
   the analytic entropy H(q_phi) for the MC entropy estimate:

       ELBO = E_q[log p(z, x)] + H(q_phi)

   For mean-field Gaussian: H = sum_i log_sigma_i + D/2 * (1 + log 2pi).
   Gradient wrt log_sigma is then a constant 1 per dimension, eliminating
   the noisy (eps_i^2 - 1) score term.

Neither technique requires extra model evaluations.

References
----------
- Roeder, Wu, Duvenaud. "Sticking the landing: Simple, lower-variance
  gradient estimators for variational inference." NeurIPS 2017.
- Owen. "Monte Carlo Theory, Methods and Examples." Stanford, 2013.
"""

import time
import torch


class VarianceReducedBBVI:
    """
    Variance-reduced BBVI with antithetic sampling and STL estimator.

    Parameters
    ----------
    model : object
        Must implement .log_prob(z) -> Tensor of shape (n_samples,).
    D : int
        Latent dimension.
    lr : float
        Adam learning rate.
    n_samples : int
        Total MC samples per step. With antithetic=True, n_samples//2
        base samples are drawn and mirrored to give n_samples total.
    use_antithetic : bool
        Enable antithetic (mirrored) sampling.
    use_stl : bool
        Enable "sticking the landing" analytic entropy estimator.
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
        use_antithetic: bool = True,
        use_stl: bool = True,
        betas: tuple = (0.9, 0.999),
        eps_adam: float = 1e-8,
    ):
        self.model = model
        self.D = D
        self.lr = lr
        self.n_samples = n_samples
        self.use_antithetic = use_antithetic
        self.use_stl = use_stl

        self.mu = torch.zeros(D, requires_grad=True)
        self.log_sigma = torch.zeros(D, requires_grad=True)

        self.optimizer = torch.optim.Adam(
            [self.mu, self.log_sigma],
            lr=lr, betas=betas, eps=eps_adam,
        )

        self.elbo_history = []
        self.grad_var_history = []
        self.wall_times = []

    # ------------------------------------------------------------------
    # ELBO
    # ------------------------------------------------------------------

    def elbo(self) -> torch.Tensor:
        """
        Estimate the ELBO with optional antithetic sampling and STL.

        Returns a scalar tensor (differentiable wrt mu, log_sigma).
        """
        sigma = torch.exp(self.log_sigma)

        if self.use_antithetic:
            half = max(self.n_samples // 2, 1)
            eps_pos = torch.randn(half, self.D)
            eps = torch.cat([eps_pos, -eps_pos], dim=0)        # (2*half, D)
        else:
            eps = torch.randn(self.n_samples, self.D)

        z = self.mu + sigma * eps                              # (S, D)
        log_joint = self.model.log_prob(z)                     # (S,)

        if self.use_stl:
            # Analytic entropy: H(q) = sum(log_sigma) + D/2 * (1 + log 2pi)
            # Gradient wrt log_sigma = 1 per dim (no noisy score term)
            entropy = (
                self.log_sigma.sum()
                + 0.5 * self.D * (1.0 + torch.log(torch.tensor(2.0 * torch.pi)))
            )
            return log_joint.mean() + entropy
        else:
            log_q = (
                -0.5 * (eps ** 2).sum(-1)
                - self.log_sigma.sum()
                - 0.5 * self.D * torch.log(torch.tensor(2.0 * torch.pi))
            )
            return (log_joint - log_q).mean()

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def step(self) -> float:
        """One Adam step on the variance-reduced ELBO. Returns ELBO value."""
        self.optimizer.zero_grad()
        loss = -self.elbo()
        loss.backward()
        self.optimizer.step()
        return -loss.item()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

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
        log_every : int
        track_grad_var : bool
            Estimate gradient variance at each log step (extra passes).

        Returns
        -------
        dict with keys: mu, sigma, elbo, grad_var, wall_time
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
                    f"[VarReducedBBVI] iter {i:5d} | ELBO {elbo_val:+.4f}"
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
            "wall_time": self.wall_times,
        }

    def estimate_grad_var(self) -> float:
        """
        Estimate ||Var(grad)||_2 from n_samples single-sample gradient estimates.
        """
        per_sample_grads = []

        for _ in range(self.n_samples):
            self.optimizer.zero_grad()
            eps = torch.randn(1, self.D)
            sigma = torch.exp(self.log_sigma.detach())
            z = self.mu + sigma * eps
            log_joint = self.model.log_prob(z)

            if self.use_stl:
                entropy = (
                    self.log_sigma.sum()
                    + 0.5 * self.D * (1.0 + torch.log(torch.tensor(2.0 * torch.pi)))
                )
                loss = -(log_joint.mean() + entropy)
            else:
                log_q = (
                    -0.5 * (eps ** 2).sum(-1)
                    - self.log_sigma.sum()
                    - 0.5 * self.D * torch.log(torch.tensor(2.0 * torch.pi))
                )
                loss = -(log_joint - log_q).mean()

            loss.backward()
            g = torch.cat([
                self.mu.grad.detach().clone(),
                self.log_sigma.grad.detach().clone(),
            ])
            per_sample_grads.append(g)

        grads = torch.stack(per_sample_grads)
        return grads.var(dim=0).norm().item()
