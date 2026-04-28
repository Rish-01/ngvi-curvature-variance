import time
import torch

from .lrd_math import lrd_log_prob, lrd_reparameterize


class DiagonalFisherVI:
    """
    Diagonal empirical-Fisher quasi-natural gradient VI.

    Unlike NaturalGradientVI's mean-field closed-form Fisher, this class always
    uses empirical diagonal Fisher with EMA smoothing for all supported families.
    """

    def __init__(
        self,
        model,
        D: int,
        lr: float = 1e-2,
        n_samples: int = 10,
        damping: float = 1e-3,
        fisher_ema: float = 0.9,
        variational_family: str = "mean_field",
        low_rank: int = 3,
        v_init_scale: float = 0.01,
    ):
        self.model = model
        self.D = D
        self.lr = lr
        self.n_samples = n_samples
        self.damping = damping
        self.fisher_ema = fisher_ema
        self.variational_family = variational_family
        self.low_rank = low_rank
        if self.variational_family not in ("mean_field", "low_rank_diag"):
            raise ValueError("variational_family must be 'mean_field' or 'low_rank_diag'.")

        self.mu = torch.zeros(D)
        if self.variational_family == "mean_field":
            self.log_sigma = torch.zeros(D)
            self.log_s = None
            self.V = None
            self._fisher_ema = torch.ones(2 * D)
        else:
            self.log_s = torch.zeros(D)
            self.V = v_init_scale * torch.randn(D, low_rank)
            self.log_sigma = None
            self._fisher_ema = torch.ones(2 * D + D * low_rank)
        self._ema_initialized = False

        self.elbo_history = []
        self.grad_var_history = []
        self.fisher_diag_history = []
        self.condition_history = []
        self.wall_times = []

    def _sample_and_logq(self, mu, scale_param, V=None, n_samples=None):
        n = self.n_samples if n_samples is None else n_samples
        if self.variational_family == "mean_field":
            sigma = torch.exp(scale_param)
            eps = torch.randn(n, self.D)
            z = mu + sigma * eps
            log_q = (
                -0.5 * (eps ** 2).sum(-1)
                - scale_param.sum()
                - 0.5 * self.D * torch.log(torch.tensor(2 * torch.pi))
            )
            return z, log_q
        z = lrd_reparameterize(mu, scale_param, V, n)
        log_q = lrd_log_prob(z, mu, scale_param, V)
        return z, log_q

    def _flatten(self, mu, scale, V=None):
        if self.variational_family == "mean_field":
            return torch.cat([mu, scale])
        return torch.cat([mu, scale, V.reshape(-1)])

    def _apply_step(self, qng: torch.Tensor):
        D = self.D
        self.mu = self.mu + self.lr * qng[:D]
        if self.variational_family == "mean_field":
            self.log_sigma = self.log_sigma + self.lr * qng[D:2 * D]
        else:
            self.log_s = self.log_s + self.lr * qng[D:2 * D]
            self.V = self.V + self.lr * qng[2 * D:].reshape(D, self.low_rank)

    def _empirical_fisher_step(self):
        per = []
        for _ in range(self.n_samples):
            mu = self.mu.clone().requires_grad_(True)
            if self.variational_family == "mean_field":
                scale = self.log_sigma.clone().requires_grad_(True)
                V = None
            else:
                scale = self.log_s.clone().requires_grad_(True)
                V = self.V.clone().requires_grad_(True)
            z, log_q = self._sample_and_logq(mu, scale, V, 1)
            elbo = (self.model.log_prob(z) - log_q).mean()
            elbo.backward()
            if self.variational_family == "mean_field":
                g = self._flatten(mu.grad.detach(), scale.grad.detach())
            else:
                g = self._flatten(mu.grad.detach(), scale.grad.detach(), V.grad.detach())
            per.append(g)
        g = torch.stack(per)
        raw = g.pow(2).mean(0)
        if not self._ema_initialized:
            self._fisher_ema = raw.clone()
            self._ema_initialized = True
        else:
            a = self.fisher_ema
            self._fisher_ema = a * self._fisher_ema + (1 - a) * raw

    def get_fisher_diagonal(self) -> torch.Tensor:
        return self._fisher_ema + self.damping

    def condition_number(self) -> float:
        f = self.get_fisher_diagonal()
        return (f.max() / f.min()).item()

    def _elbo_and_grad(self) -> tuple[float, torch.Tensor]:
        mu = self.mu.clone().requires_grad_(True)
        if self.variational_family == "mean_field":
            scale = self.log_sigma.clone().requires_grad_(True)
            V = None
        else:
            scale = self.log_s.clone().requires_grad_(True)
            V = self.V.clone().requires_grad_(True)
        z, log_q = self._sample_and_logq(mu, scale, V)
        elbo = (self.model.log_prob(z) - log_q).mean()
        elbo.backward()
        if self.variational_family == "mean_field":
            grad = self._flatten(mu.grad.detach(), scale.grad.detach())
        else:
            grad = self._flatten(mu.grad.detach(), scale.grad.detach(), V.grad.detach())
        return elbo.item(), grad

    def step(self) -> float:
        self._empirical_fisher_step()
        elbo_val, grad = self._elbo_and_grad()
        f = self.get_fisher_diagonal()
        self._apply_step(grad / f)
        return elbo_val

    def fit(
        self,
        n_iters: int = 2000,
        log_every: int = 100,
        track_grad_var: bool = False,
        track_fisher: bool = False,
    ) -> dict:
        t0 = time.time()
        for i in range(1, n_iters + 1):
            elbo_val = self.step()
            if i % log_every == 0:
                self.elbo_history.append(elbo_val)
                self.wall_times.append(time.time() - t0)
                if track_fisher:
                    f = self.get_fisher_diagonal()
                    self.fisher_diag_history.append(f.detach().clone())
                    self.condition_history.append(self.condition_number())
                if track_grad_var:
                    self.grad_var_history.append(self.estimate_grad_var())
                print(
                    f"[DiagFisherVI] iter {i:5d} | ELBO {elbo_val:+.4f}"
                    + (f" | cond(F_diag) {self.condition_history[-1]:.2e}" if track_fisher else "")
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
            "fisher_diag": self.fisher_diag_history,
            "fisher_cond": self.condition_history,
            "wall_time": self.wall_times,
        }
        if self.variational_family == "low_rank_diag":
            out["V"] = self.V.detach().clone()
        return out

    def estimate_grad_var(self) -> float:
        f = self.get_fisher_diagonal()
        per = []
        for _ in range(self.n_samples):
            mu = self.mu.clone().requires_grad_(True)
            if self.variational_family == "mean_field":
                scale = self.log_sigma.clone().requires_grad_(True)
                V = None
            else:
                scale = self.log_s.clone().requires_grad_(True)
                V = self.V.clone().requires_grad_(True)
            z, log_q = self._sample_and_logq(mu, scale, V, 1)
            elbo = (self.model.log_prob(z) - log_q).mean()
            elbo.backward()
            if self.variational_family == "mean_field":
                g = self._flatten(mu.grad.detach(), scale.grad.detach())
            else:
                g = self._flatten(mu.grad.detach(), scale.grad.detach(), V.grad.detach())
            per.append(g / f)
        return torch.stack(per).var(dim=0).norm().item()

