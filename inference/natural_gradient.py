import math
import time
import torch

from .lrd_math import lrd_log_prob, lrd_reparameterize


class NaturalGradientVI:
    """
    Natural-gradient VI with selectable variational family:
      - mean_field: exact diagonal Fisher in (mu, log_sigma)
      - low_rank_diag: empirical full Fisher in flattened parameters
    """

    def __init__(
        self,
        model,
        D: int,
        lr: float = 1e-2,
        n_samples: int = 10,
        damping: float = 1e-4,
        variational_family: str = "mean_field",
        low_rank: int = 3,
        fisher_ema: float = 0.9,
        v_init_scale: float = 0.01,
        use_antithetic: bool = False,
        use_stl: bool = False,
    ):
        self.model = model
        self.D = D
        self.lr = lr
        self.n_samples = n_samples
        self.damping = damping
        self.variational_family = variational_family
        self.low_rank = low_rank
        self.fisher_ema = fisher_ema
        self.use_antithetic = use_antithetic
        self.use_stl = use_stl
        if self.variational_family not in ("mean_field", "low_rank_diag"):
            raise ValueError("variational_family must be 'mean_field' or 'low_rank_diag'.")

        self.mu = torch.zeros(D)
        if self.variational_family == "mean_field":
            self.log_sigma = torch.zeros(D)
            self.log_s = None
            self.V = None
        else:
            self.log_s = torch.zeros(D)
            self.V = v_init_scale * torch.randn(D, low_rank)
            self.log_sigma = None

        self._fish_ema = None
        self._ema_initialized = False

        self.elbo_history = []
        self.grad_var_history = []
        self.fisher_cond_history = []
        self.wall_times = []

    def _sample_and_logq(self, mu, scale_param, V=None, n_samples=None):
        n = self.n_samples if n_samples is None else n_samples
        use_pairs = self.use_antithetic and n >= 2
        n_pair = n // 2 if use_pairs else 0
        if self.variational_family == "mean_field":
            sigma = torch.exp(scale_param)
            if use_pairs:
                eps_half = torch.randn(n_pair, self.D)
                eps = torch.cat([eps_half, -eps_half], dim=0)
                if n % 2 == 1:
                    eps = torch.cat([eps, torch.randn(1, self.D)], dim=0)
                eps = eps[:n]
            else:
                eps = torch.randn(n, self.D)
            z = mu + sigma * eps
            if self.use_stl:
                z_q = z.detach()
                log_two_pi_half = -0.5 * self.D * math.log(2 * math.pi)
                log_q = (
                    -0.5 * (((z_q - mu) / sigma) ** 2).sum(-1)
                    - scale_param.sum()
                    + log_two_pi_half
                )
            else:
                log_q = (
                    -0.5 * (eps**2).sum(-1)
                    - scale_param.sum()
                    - 0.5 * self.D * torch.log(torch.tensor(2 * torch.pi))
                )
            return z, log_q
        if use_pairs:
            eps1_h = torch.randn(n_pair, self.D)
            eps2_h = torch.randn(n_pair, self.low_rank)
            eps1 = torch.cat([eps1_h, -eps1_h], dim=0)
            eps2 = torch.cat([eps2_h, -eps2_h], dim=0)
            if n % 2 == 1:
                eps1 = torch.cat([eps1, torch.randn(1, self.D)], dim=0)
                eps2 = torch.cat([eps2, torch.randn(1, self.low_rank)], dim=0)
            eps1 = eps1[:n]
            eps2 = eps2[:n]
            s = torch.exp(scale_param)
            z = mu + s * eps1 + eps2 @ V.T
        else:
            z = lrd_reparameterize(mu, scale_param, V, n)
        z_q = z.detach() if self.use_stl else z
        log_q = lrd_log_prob(z_q, mu, scale_param, V)
        return z, log_q

    def _param_size(self) -> int:
        if self.variational_family == "mean_field":
            return 2 * self.D
        return 2 * self.D + self.D * self.low_rank

    def fisher_diagonal(self) -> torch.Tensor:
        if self.variational_family == "mean_field":
            sigma_sq = torch.exp(2 * self.log_sigma)
            f_mu = 1.0 / sigma_sq
            f_ls = torch.full((self.D,), 2.0)
            return torch.cat([f_mu, f_ls]) + self.damping
        raise RuntimeError("fisher_diagonal() is only valid for variational_family='mean_field'.")

    def fisher_matrix(self) -> torch.Tensor:
        if self.variational_family == "mean_field":
            return torch.diag(self.fisher_diagonal())
        if not self._ema_initialized:
            p = self._param_size()
            self._fish_ema = torch.eye(p)
            self._ema_initialized = True
        p = self._fish_ema.shape[0]
        eye = torch.eye(p, device=self._fish_ema.device, dtype=self._fish_ema.dtype)
        return self._fish_ema + self.damping * eye

    def fisher_condition_number(self) -> float:
        if self.variational_family == "mean_field":
            f = self.fisher_diagonal()
            return (f.max() / f.min()).item()
        F = self.fisher_matrix()
        return torch.linalg.cond(F).item()

    def _flatten_params(self, mu, scale_param, V=None):
        if self.variational_family == "mean_field":
            return torch.cat([mu, scale_param])
        return torch.cat([mu, scale_param, V.reshape(-1)])

    def _unflatten_and_set(self, flat: torch.Tensor):
        D = self.D
        self.mu = self.mu + self.lr * flat[:D]
        if self.variational_family == "mean_field":
            self.log_sigma = self.log_sigma + self.lr * flat[D:2 * D]
            return
        self.log_s = self.log_s + self.lr * flat[D:2 * D]
        self.V = self.V + self.lr * flat[2 * D:].reshape(D, self.low_rank)

    def elbo_and_grad(self) -> tuple[float, torch.Tensor]:
        mu = self.mu.clone().requires_grad_(True)
        if self.variational_family == "mean_field":
            scale = self.log_sigma.clone().requires_grad_(True)
            V = None
        else:
            scale = self.log_s.clone().requires_grad_(True)
            V = self.V.clone().requires_grad_(True)
        z, log_q = self._sample_and_logq(mu, scale, V)
        log_joint = self.model.log_prob(z)
        elbo = (log_joint - log_q).mean()
        elbo.backward()
        if self.variational_family == "mean_field":
            grad = self._flatten_params(mu.grad.detach().clone(), scale.grad.detach().clone())
        else:
            grad = self._flatten_params(
                mu.grad.detach().clone(),
                scale.grad.detach().clone(),
                V.grad.detach().clone(),
            )
        return elbo.item(), grad

    def _update_empirical_fisher(self):
        per_sample = []
        for _ in range(self.n_samples):
            mu = self.mu.clone().requires_grad_(True)
            if self.variational_family == "mean_field":
                scale = self.log_sigma.clone().requires_grad_(True)
                V = None
            else:
                scale = self.log_s.clone().requires_grad_(True)
                V = self.V.clone().requires_grad_(True)
            z, log_q = self._sample_and_logq(mu, scale, V, n_samples=1)
            # Fisher metric uses score of q: E[grad log q grad log q^T].
            val = log_q.mean()
            val.backward()
            if self.variational_family == "mean_field":
                g = self._flatten_params(mu.grad.detach(), scale.grad.detach())
            else:
                g = self._flatten_params(mu.grad.detach(), scale.grad.detach(), V.grad.detach())
            per_sample.append(g)
        g = torch.stack(per_sample)
        raw = (g.T @ g) / g.shape[0]
        if not self._ema_initialized:
            self._fish_ema = raw.clone()
            self._ema_initialized = True
        else:
            a = self.fisher_ema
            self._fish_ema = a * self._fish_ema + (1 - a) * raw

    def step(self) -> float:
        if self.variational_family == "low_rank_diag":
            self._update_empirical_fisher()
        elbo_val, grad = self.elbo_and_grad()
        if self.variational_family == "mean_field":
            f = self.fisher_diagonal()
            nat_grad = grad / f
        else:
            F = self.fisher_matrix()
            nat_grad = torch.linalg.solve(F, grad)
        self._unflatten_and_set(nat_grad)
        return elbo_val

    def fit(
        self,
        n_iters: int = 2000,
        log_every: int = 100,
        track_grad_var: bool = False,
        track_fisher_cond: bool = False,
    ) -> dict:
        t0 = time.time()
        for i in range(1, n_iters + 1):
            elbo_val = self.step()
            if i % log_every == 0:
                self.elbo_history.append(elbo_val)
                self.wall_times.append(time.time() - t0)
                if track_fisher_cond:
                    self.fisher_cond_history.append(self.fisher_condition_number())
                if track_grad_var:
                    self.grad_var_history.append(self.estimate_grad_var())
                print(
                    f"[NaturalGradVI] iter {i:5d} | ELBO {elbo_val:+.4f}"
                    + (f" | cond(F) {self.fisher_cond_history[-1]:.2e}" if track_fisher_cond else "")
                    + (f" | grad_var {self.grad_var_history[-1]:.4e}" if track_grad_var else "")
                )
        return self.results()

    def results(self) -> dict:
        out = {
            "mu": self.mu.detach().clone(),
            "sigma": (
                torch.exp(self.log_sigma).detach().clone()
                if self.variational_family == "mean_field"
                else torch.exp(self.log_s).detach().clone()
            ),
            "elbo": self.elbo_history,
            "grad_var": self.grad_var_history,
            "fisher_cond": self.fisher_cond_history,
            "wall_time": self.wall_times,
        }
        if self.variational_family == "low_rank_diag":
            out["V"] = self.V.detach().clone()
        return out

    def estimate_grad_var(self) -> float:
        use_diag = self.variational_family == "mean_field"
        if use_diag:
            f = self.fisher_diagonal()
        else:
            F = self.fisher_matrix()
        per = []
        for _ in range(self.n_samples):
            mu = self.mu.clone().requires_grad_(True)
            if self.variational_family == "mean_field":
                scale = self.log_sigma.clone().requires_grad_(True)
                V = None
            else:
                scale = self.log_s.clone().requires_grad_(True)
                V = self.V.clone().requires_grad_(True)
            z, log_q = self._sample_and_logq(mu, scale, V, n_samples=self.n_samples)
            val = (self.model.log_prob(z) - log_q).mean()
            val.backward()
            if self.variational_family == "mean_field":
                g = self._flatten_params(mu.grad.detach(), scale.grad.detach())
            else:
                g = self._flatten_params(mu.grad.detach(), scale.grad.detach(), V.grad.detach())
            if use_diag:
                per.append(g / f)
            else:
                per.append(torch.linalg.solve(F, g))
        return torch.stack(per).var(dim=0, unbiased=False).norm().item()

    def max_stable_lr(self, lr_grid: list = None, n_probe_iters: int = 200) -> float:
        if lr_grid is None:
            lr_grid = [10 ** (-k / 2) for k in range(0, 8)]
        for lr in sorted(lr_grid, reverse=True):
            probe = NaturalGradientVI(
                self.model, self.D, lr=lr, n_samples=self.n_samples, damping=self.damping,
                variational_family=self.variational_family, low_rank=self.low_rank, fisher_ema=self.fisher_ema,
                use_antithetic=self.use_antithetic, use_stl=self.use_stl,
            )
            probe.mu = self.mu.clone()
            if self.variational_family == "mean_field":
                probe.log_sigma = self.log_sigma.clone()
            else:
                probe.log_s = self.log_s.clone()
                probe.V = self.V.clone()
            try:
                for _ in range(n_probe_iters):
                    val = probe.step()
                    if not torch.isfinite(torch.tensor(val)):
                        raise ValueError("diverged")
                return lr
            except (ValueError, RuntimeError):
                continue
        return lr_grid[-1]
