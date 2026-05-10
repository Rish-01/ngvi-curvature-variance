"""
Curvature-Variance study runner for BBVI vs natural gradients.

Key design:
  - Training/update time is measured from method.step() only.
  - Hessian curvature probes are computed out-of-band and logged separately.
  - Outputs are written under: experiments/results/curvature_variance_study/
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

# Ensure repo root is importable when script is run directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference import get_inference_method
from inference.lrd_math import lrd_log_prob
from models import EightSchools, NealsFunnel


RESULTS_DIR = REPO_ROOT / "experiments" / "results" / "curvature_variance_study"
CSV_DIR = RESULTS_DIR / "csv"
FIGS_DIR = RESULTS_DIR / "figs"
CSV_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# Legend: large type + spare bottom margin so labels never sit under data.
_LEGEND_AX_KW = {
    "frameon": False,
    "fontsize": 12,
    "handlelength": 2.4,
    "handletextpad": 0.75,
    "columnspacing": 1.6,
}
_LEGEND_FIG_KW = {**_LEGEND_AX_KW, "fontsize": 13}


@dataclass
class Scenario:
    key: str
    label: str
    config_path: str
    param: str | None = None


@dataclass
class MethodSpec:
    run_id: str
    method_name: str
    label: str
    overrides: dict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-iters", type=int, default=3000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--probe-every", type=int, default=200)
    p.add_argument("--eval-mc", type=int, default=256)
    p.add_argument("--hess-mc", type=int, default=64)
    p.add_argument("--ref-tol", type=float, default=0.5, help="Reference line: best observed ELBO minus this tolerance.")
    p.add_argument("--run-main", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--run-variance-axis", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--run-curvature-axis", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--main-families", type=str, default="mean_field,low_rank_diag")
    p.add_argument(
        "--variance-n-samples",
        type=int,
        default=None,
        help=(
            "MC samples per training step for variance-axis runs only. "
            "Default: bbvi_adam n_samples from the scenario YAML."
        ),
    )
    p.add_argument(
        "--variance-scenario",
        type=str,
        default="funnel",
        help="Scenario key for variance-axis runs (e.g., funnel, schools_cp, schools_ncp).",
    )
    p.add_argument(
        "--curvature-scenarios",
        type=str,
        default="funnel,schools_cp,schools_ncp",
        help="Comma-separated scenario keys for curvature-axis runs.",
    )
    return p.parse_args()


def load_cfg(path: str) -> dict:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, param: str | None = None):
    name = cfg["model"]["name"]
    if name == "neals_funnel":
        return NealsFunnel(D=cfg["model"]["D"], v_scale=cfg["model"]["v_scale"])
    if name == "eight_schools":
        p = param or "noncentered"
        return EightSchools(
            parameterization=p,
            y=torch.tensor(cfg["model"]["y"]),
            sigma=torch.tensor(cfg["model"]["sigma"]),
        )
    raise ValueError(f"Unknown model name: {name}")


def method_kwargs(
    cfg: dict,
    method_name: str,
    *,
    family: str | None = None,
    low_rank: int | None = None,
    overrides: dict | None = None,
) -> dict:
    raw = cfg["methods"][method_name].copy()
    raw["variational_family"] = family or cfg.get("variational", {}).get("family", "mean_field")
    raw["low_rank"] = int(low_rank if low_rank is not None else cfg.get("variational", {}).get("low_rank_dim", 3))
    if overrides:
        raw.update(overrides)
    return raw


def eval_elbo(method, model, n_mc: int) -> float:
    mu = method.mu.detach().reshape(-1)
    family = getattr(method, "variational_family", "mean_field")
    D = method.D
    device, dtype = mu.device, mu.dtype
    with torch.no_grad():
        if family == "mean_field":
            log_sigma = method.log_sigma.detach().reshape(-1)
            sigma = torch.exp(log_sigma)
            eps = torch.randn(n_mc, D, device=device, dtype=dtype)
            z = mu + sigma * eps
            log_joint = model.log_prob(z)
            log_q = (
                -0.5 * (eps ** 2).sum(-1)
                - log_sigma.sum()
                - 0.5 * D * torch.log(torch.tensor(2 * torch.pi, device=device, dtype=dtype))
            )
        else:
            log_s = method.log_s.detach().reshape(-1)
            V = method.V.detach()
            s = torch.exp(log_s)
            eps1 = torch.randn(n_mc, D, device=device, dtype=dtype)
            eps2 = torch.randn(n_mc, V.shape[1], device=device, dtype=dtype)
            z = mu + s * eps1 + eps2 @ V.T
            log_joint = model.log_prob(z)
            log_q = lrd_log_prob(z, mu, log_s, V)
        return (log_joint - log_q).mean().item()


def _flatten_method_params(method: object) -> torch.Tensor:
    if method.variational_family == "mean_field":
        return torch.cat([method.mu.detach().reshape(-1), method.log_sigma.detach().reshape(-1)])
    return torch.cat(
        [
            method.mu.detach().reshape(-1),
            method.log_s.detach().reshape(-1),
            method.V.detach().reshape(-1),
        ]
    )


def _unpack_flat(flat: torch.Tensor, method: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    d = method.D
    if method.variational_family == "mean_field":
        mu = flat[:d]
        log_sigma = flat[d : 2 * d]
        return mu, log_sigma, None
    r = method.low_rank
    mu = flat[:d]
    log_s = flat[d : 2 * d]
    v = flat[2 * d :].reshape(d, r)
    return mu, log_s, v


def _elbo_from_flat(flat: torch.Tensor, method: object, model: object, n_mc: int) -> torch.Tensor:
    mu, scale, V = _unpack_flat(flat, method)
    d = method.D
    if method.variational_family == "mean_field":
        sigma = torch.exp(scale)
        eps = torch.randn(n_mc, d, device=flat.device, dtype=flat.dtype)
        z = mu + sigma * eps
        log_joint = model.log_prob(z)
        log_q = (
            -0.5 * (eps ** 2).sum(-1)
            - scale.sum()
            - 0.5 * d * torch.log(torch.tensor(2 * torch.pi, device=flat.device, dtype=flat.dtype))
        )
        return (log_joint - log_q).mean()
    s = torch.exp(scale)
    eps1 = torch.randn(n_mc, d, device=flat.device, dtype=flat.dtype)
    eps2 = torch.randn(n_mc, method.low_rank, device=flat.device, dtype=flat.dtype)
    z = mu + s * eps1 + eps2 @ V.T
    log_joint = model.log_prob(z)
    log_q = lrd_log_prob(z, mu, scale, V)
    return (log_joint - log_q).mean()


def probe_hessian_curvature(method: object, model: object, hess_mc: int) -> dict:
    """
    Compute Hessian diagnostics of -ELBO at current variational parameters.
    Probe is intentionally out-of-band and should not affect optimizer state.
    """
    x0 = _flatten_method_params(method).clone().detach().requires_grad_(True)

    def neg_elbo_from_x(x: torch.Tensor) -> torch.Tensor:
        return -_elbo_from_flat(x, method, model, hess_mc)

    H = torch.autograd.functional.hessian(neg_elbo_from_x, x0)
    H = 0.5 * (H + H.T)
    # Eigh can fail on noisy/ill-conditioned Hessians; use robust fallbacks.
    H64 = H.to(dtype=torch.float64)
    H64 = torch.nan_to_num(H64, nan=0.0, posinf=1e12, neginf=-1e12)
    eye = torch.eye(H64.shape[0], device=H64.device, dtype=H64.dtype)
    eigvals = None
    for jitter in (0.0, 1e-10, 1e-8, 1e-6, 1e-4):
        try:
            eigvals = torch.linalg.eigvalsh(H64 + jitter * eye)
            break
        except torch._C._LinAlgError:
            continue

    if eigvals is None:
        # Last-resort stable proxy from singular values.
        svals = torch.linalg.svdvals(H64)
        if svals.numel() == 0:
            return {
                "lambda_max": float("nan"),
                "lambda_min": float("nan"),
                "cond_abs": float("nan"),
                "trace_h": float("nan"),
            }
        lmax = svals.max().item()
        lmin = svals.min().item()
        eps = 1e-12
        cond_abs = abs(lmax) / max(abs(lmin), eps)
        return {
            "lambda_max": lmax,
            "lambda_min": -lmin,
            "cond_abs": cond_abs,
            "trace_h": H64.diag().sum().item(),
        }

    lmax = eigvals[-1].item()
    lmin = eigvals[0].item()
    eps = 1e-12
    cond_abs = abs(lmax) / max(abs(lmin), eps)
    return {
        "lambda_max": lmax,
        "lambda_min": lmin,
        "cond_abs": cond_abs,
        "trace_h": eigvals.sum().item(),
    }


def save_table_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_curve_plot(
    path: Path,
    title: str,
    ylabel: str,
    series: list[tuple[str, list[int], list[float]]],
    logy: bool = False,
    ref_y: float | None = None,
    ref_label: str = "Reference",
):
    fig, ax = plt.subplots(figsize=(8.0, 5.35))
    for label, xs, ys in series:
        x_clean = []
        y_clean = []
        for x, y in zip(xs, ys):
            if np.isfinite(y):
                x_clean.append(x)
                y_clean.append(y)
        if not x_clean:
            continue
        if logy:
            ax.semilogy(x_clean, y_clean, label=label)
        else:
            ax.plot(x_clean, y_clean, label=label)
    if ref_y is not None and np.isfinite(ref_y):
        ax.axhline(ref_y, linestyle="--", linewidth=1.1, color="black", alpha=0.8, label=ref_label)
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    hz = ax.get_legend_handles_labels()
    if hz[0]:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=min(4, max(2, len(hz[0]))),
            borderpad=0.7,
            **_LEGEND_AX_KW,
        )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_tiled_main_plot(
    path: Path,
    scenario_label: str,
    family: str,
    elbo_series: list[tuple[str, list[int], list[float]]],
    curv_series: list[tuple[str, list[int], list[float]]],
    ref_elbo: float | None = None,
):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6))

    # Left panel: ELBO
    for label, xs, ys in elbo_series:
        x_clean = []
        y_clean = []
        for x, y in zip(xs, ys):
            if np.isfinite(y):
                x_clean.append(x)
                y_clean.append(y)
        if x_clean:
            axes[0].plot(x_clean, y_clean, label=label)
    if ref_elbo is not None and np.isfinite(ref_elbo):
        axes[0].axhline(ref_elbo, linestyle="--", linewidth=1.1, color="black", alpha=0.8, label="Reference")
    axes[0].set_title("ELBO")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("ELBO (eval)")
    axes[0].grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    axes[0].spines[["top", "right"]].set_visible(False)

    # Right panel: curvature proxy
    for label, xs, ys in curv_series:
        x_clean = []
        y_clean = []
        for x, y in zip(xs, ys):
            if np.isfinite(y) and y > 0:
                x_clean.append(x)
                y_clean.append(y)
        if x_clean:
            axes[1].semilogy(x_clean, y_clean, label=label)
    axes[1].set_title("Hessian conditioning proxy")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel(r"$|\lambda_{\max}| / \max(|\lambda_{\min}|,\epsilon)$")
    axes[1].grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    axes[1].spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=4,
            borderpad=0.7,
            **_LEGEND_FIG_KW,
        )
    fig.suptitle(f"{scenario_label} [{family}] — ELBO and Curvature", y=1.02)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.31)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_family_overview_tile(
    path: Path,
    family: str,
    scenario_specs: list[tuple[str, str]],
    ref_tol: float,
):
    """
    Save a 3x3 dashboard:
      row 1 = ELBO curves for (funnel, schools_cp, schools_ncp) + reference line
      row 2 = curvature proxy curves for same scenarios
      row 3 = gradient-variance curves for same scenarios
    """
    fig, axes = plt.subplots(3, 3, figsize=(18.5, 13.25))
    any_legend_handles = None
    any_legend_labels = None

    family_tag = "mf" if family == "mean_field" else "lrd"
    for col, (scenario_key, scenario_label) in enumerate(scenario_specs):
        perf_path = CSV_DIR / f"main_{scenario_key}_{family_tag}_perf.csv"
        curv_path = CSV_DIR / f"main_{scenario_key}_{family_tag}_curvature.csv"
        if not perf_path.exists() or not curv_path.exists():
            axes[0, col].set_visible(False)
            axes[1, col].set_visible(False)
            continue

        # Load from CSV to avoid relying on in-memory run order.
        import csv as _csv

        perf_rows = []
        with open(perf_path, newline="") as f:
            reader = _csv.DictReader(f)
            for r in reader:
                perf_rows.append(r)

        curv_rows = []
        with open(curv_path, newline="") as f:
            reader = _csv.DictReader(f)
            for r in reader:
                curv_rows.append(r)

        run_ids = sorted({r["run_id"] for r in perf_rows})
        finite_elbos = [float(r["eval_elbo"]) for r in perf_rows if np.isfinite(float(r["eval_elbo"]))]
        ref_elbo = (max(finite_elbos) - ref_tol) if finite_elbos else None
        for run_id in run_ids:
            label = next((r["label"] for r in perf_rows if r["run_id"] == run_id), run_id)
            p = [r for r in perf_rows if r["run_id"] == run_id]
            p.sort(key=lambda x: int(x["iter"]))
            xs = [int(r["iter"]) for r in p]
            ys = [float(r["eval_elbo"]) for r in p]
            x_clean = [x for x, y in zip(xs, ys) if np.isfinite(y)]
            y_clean = [y for y in ys if np.isfinite(y)]
            if x_clean:
                axes[0, col].plot(x_clean, y_clean, label=label)

            gys = [float(r["grad_var"]) for r in p]
            gx_clean = [x for x, y in zip(xs, gys) if np.isfinite(y) and y > 0]
            gy_clean = [y for y in gys if np.isfinite(y) and y > 0]
            if gx_clean:
                axes[2, col].semilogy(gx_clean, gy_clean, label=label)

            c = [r for r in curv_rows if r["run_id"] == run_id]
            c.sort(key=lambda x: int(x["iter"]))
            cxs = [int(r["iter"]) for r in c]
            cys = [float(r["cond_abs"]) for r in c]
            cx_clean = [x for x, y in zip(cxs, cys) if np.isfinite(y) and y > 0]
            cy_clean = [y for y in cys if np.isfinite(y) and y > 0]
            if cx_clean:
                axes[1, col].semilogy(cx_clean, cy_clean, label=label)

        if ref_elbo is not None and np.isfinite(ref_elbo):
            axes[0, col].axhline(
                ref_elbo,
                linestyle="--",
                linewidth=1.1,
                color="black",
                alpha=0.8,
                label=f"Reference (best - {ref_tol:g})",
            )

        axes[0, col].set_title(f"{scenario_label} — ELBO")
        axes[0, col].set_xlabel("Iteration")
        axes[0, col].set_ylabel("ELBO (eval)")
        axes[0, col].grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
        axes[0, col].spines[["top", "right"]].set_visible(False)

        axes[1, col].set_title(f"{scenario_label} — Curvature")
        axes[1, col].set_xlabel("Iteration")
        axes[1, col].set_ylabel(r"$|\lambda_{\max}| / \max(|\lambda_{\min}|,\epsilon)$")
        axes[1, col].grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
        axes[1, col].spines[["top", "right"]].set_visible(False)

        axes[2, col].set_title(f"{scenario_label} — Gradient variance")
        axes[2, col].set_xlabel("Iteration")
        axes[2, col].set_ylabel(r"$||\mathrm{Var}(g)||_2$")
        axes[2, col].grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
        axes[2, col].spines[["top", "right"]].set_visible(False)

        handles, labels = axes[0, col].get_legend_handles_labels()
        if handles and any_legend_handles is None:
            any_legend_handles = handles
            any_legend_labels = labels

    if any_legend_handles:
        fig.legend(
            any_legend_handles,
            any_legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.007),
            ncol=4,
            borderpad=0.75,
            **_LEGEND_FIG_KW,
        )
    fig.suptitle(f"Main Comparison Dashboard [{family}]", y=1.01)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.26)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_variance_optimizer_grid(
    path: Path,
    scenario_label: str,
    column_overlays: dict[str, dict[str, tuple[list[int], list[float], list[float]]]],
    n_samples: int,
):
    """
    Variance-axis: columns = VR mode (base / antithetic / STL / both).
    Rows = ELBO vs iteration (top), gradient variance vs iteration (bottom, log scale).
    Each panel overlays BBVI, NGVI, and Diagonal Fisher with the same VR flags for that column.
    """
    col_order = ["base", "antithetic", "stl", "antithetic_stl"]
    col_titles = {
        "base": "Base (no VR)",
        "antithetic": "Antithetic",
        "stl": "STL",
        "antithetic_stl": "Antithetic + STL",
    }
    legend_order = ["BBVI + Adam", "Natural Gradient", "Diagonal Fisher"]

    fig, axes = plt.subplots(2, 4, figsize=(20.0, 9.2))

    for j, key in enumerate(col_order):
        overlays = column_overlays.get(key, {})
        for label in legend_order:
            curve = overlays.get(label)
            if curve is None:
                continue
            xs, elbos, gvars = curve
            x_elbo = [x for x, y in zip(xs, elbos) if np.isfinite(y)]
            y_elbo = [y for y in elbos if np.isfinite(y)]
            leg = label if j == 0 else "_nolegend_"
            if x_elbo:
                axes[0, j].plot(x_elbo, y_elbo, label=leg)
            xv = [x for x, y in zip(xs, gvars) if np.isfinite(y) and y > 0]
            yv = [y for y in gvars if np.isfinite(y) and y > 0]
            if xv:
                axes[1, j].semilogy(xv, yv, label="_nolegend_")

        axes[0, j].set_title(f"{col_titles[key]} — ELBO")
        axes[0, j].set_xlabel("Iteration")
        axes[0, j].set_ylabel("ELBO (eval)")
        axes[0, j].grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
        axes[0, j].spines[["top", "right"]].set_visible(False)

        axes[1, j].set_title(f"{col_titles[key]} — Gradient variance")
        axes[1, j].set_xlabel("Iteration")
        axes[1, j].set_ylabel(r"$||\mathrm{Var}(g)||_2$")
        axes[1, j].grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
        axes[1, j].spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
            ncol=3,
            borderpad=0.72,
            **_LEGEND_FIG_KW,
        )

    fig.suptitle(f"Variance-axis (optimizers overlaid) [{scenario_label}] — n_samples={n_samples}", y=1.02)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _perf_curves_xy(out: dict) -> tuple[list[int], list[float], list[float]]:
    perf_by_iter = {r["iter"]: r for r in out["perf"]}
    xs = [r["iter"] for r in out["perf"]]
    elbos = [perf_by_iter[i]["eval_elbo"] for i in xs]
    gvars = [perf_by_iter[i]["grad_var"] for i in xs]
    return xs, elbos, gvars


def run_single(
    model,
    cfg: dict,
    spec: MethodSpec,
    *,
    family: str,
    low_rank: int,
    n_iters: int,
    log_every: int,
    probe_every: int,
    eval_mc: int,
    hess_mc: int,
    seed: int,
    probe_hessian: bool = True,
) -> dict:
    torch.manual_seed(seed)
    kwargs = method_kwargs(cfg, spec.method_name, family=family, low_rank=low_rank, overrides=spec.overrides)
    method = get_inference_method(spec.method_name, model=model, D=model.D, **kwargs)

    perf_rows = []
    curv_rows = []
    train_time_cum = 0.0
    probe_time_cum = 0.0
    # Initialization (before any training step): keeps plots/CSVs aligned with true start.
    perf_rows.append(
        {
            "iter": 0,
            "eval_elbo": eval_elbo(method, model, eval_mc),
            "grad_var": method.estimate_grad_var(),
            "train_time_s": train_time_cum,
        }
    )
    for i in tqdm(range(1, n_iters + 1), desc=spec.label, leave=False):
        t_train0 = time.perf_counter()
        method.step()
        train_time_cum += time.perf_counter() - t_train0

        if i % log_every == 0:
            ev = eval_elbo(method, model, eval_mc)
            gv = method.estimate_grad_var()
            perf_rows.append(
                {
                    "iter": i,
                    "eval_elbo": ev,
                    "grad_var": gv,
                    "train_time_s": train_time_cum,
                }
            )

        if probe_hessian and i % probe_every == 0:
            t_probe0 = time.perf_counter()
            cm = probe_hessian_curvature(method, model, hess_mc)
            probe_time = time.perf_counter() - t_probe0
            probe_time_cum += probe_time
            curv_rows.append(
                {
                    "iter": i,
                    "lambda_max": cm["lambda_max"],
                    "lambda_min": cm["lambda_min"],
                    "cond_abs": cm["cond_abs"],
                    "trace_h": cm["trace_h"],
                    "probe_time_s": probe_time,
                    "probe_time_cum_s": probe_time_cum,
                    "hess_mc": hess_mc,
                }
            )

    return {"perf": perf_rows, "curv": curv_rows}


def run_main_block(args, scenario: Scenario):
    cfg = load_cfg(scenario.config_path)
    model = build_model(cfg, scenario.param)
    low_rank = int(cfg.get("variational", {}).get("low_rank_dim", 3))
    families = [x.strip() for x in args.main_families.split(",") if x.strip()]
    if not families:
        families = [cfg.get("variational", {}).get("family", "mean_field")]

    shared_specs = [
        MethodSpec("bbvi", "bbvi_adam", "BBVI + Adam", {}),
        MethodSpec("bbvi_vr", "variance_reduced", "BBVI + Adam + VR", {"use_antithetic": True, "use_stl": False}),
        MethodSpec("ngvi", "natural_gradient", "Natural Gradient", {"use_antithetic": False, "use_stl": False}),
        MethodSpec("ngvi_vr", "natural_gradient", "Natural Gradient + VR", {"use_antithetic": True, "use_stl": False}),
    ]

    for family in families:
        base_specs = list(shared_specs)
        if family == "low_rank_diag":
            # Diagonal-Fisher is only distinct from NGVI in correlated families.
            base_specs.extend(
                [
                    MethodSpec("diag", "diagonal_fisher", "Diagonal Fisher", {"use_antithetic": False, "use_stl": False}),
                    MethodSpec("diag_vr", "diagonal_fisher", "Diagonal Fisher + VR", {"use_antithetic": True, "use_stl": False}),
                ]
            )
        rows_perf = []
        rows_curv = []
        perf_series = []
        curv_series = []
        for spec in base_specs:
            out = run_single(
                model,
                cfg,
                spec,
                family=family,
                low_rank=low_rank,
                n_iters=args.n_iters,
                log_every=args.log_every,
                probe_every=args.probe_every,
                eval_mc=args.eval_mc,
                hess_mc=args.hess_mc,
                seed=args.seed,
            )
            for row in out["perf"]:
                rows_perf.append(
                    {
                        "scenario": scenario.key,
                        "family": family,
                        "run_id": spec.run_id,
                        "label": spec.label,
                        **row,
                    }
                )
            for row in out["curv"]:
                rows_curv.append(
                    {
                        "scenario": scenario.key,
                        "family": family,
                        "run_id": spec.run_id,
                        "label": spec.label,
                        **row,
                    }
                )
            perf_series.append((spec.label, [r["iter"] for r in out["perf"]], [r["eval_elbo"] for r in out["perf"]]))
            curv_series.append((spec.label, [r["iter"] for r in out["curv"]], [r["cond_abs"] for r in out["curv"]]))

        finite_elbos = [r["eval_elbo"] for r in rows_perf if np.isfinite(r["eval_elbo"])]
        ref_elbo = (max(finite_elbos) - args.ref_tol) if finite_elbos else None

        family_tag = "mf" if family == "mean_field" else "lrd"
        save_table_csv(
            CSV_DIR / f"main_{scenario.key}_{family_tag}_perf.csv",
            rows_perf,
            ["scenario", "family", "run_id", "label", "iter", "eval_elbo", "grad_var", "train_time_s"],
        )
        save_table_csv(
            CSV_DIR / f"main_{scenario.key}_{family_tag}_curvature.csv",
            rows_curv,
            [
                "scenario",
                "family",
                "run_id",
                "label",
                "iter",
                "lambda_max",
                "lambda_min",
                "cond_abs",
                "trace_h",
                "probe_time_s",
                "probe_time_cum_s",
                "hess_mc",
            ],
        )
        save_curve_plot(
            FIGS_DIR / f"main_{scenario.key}_{family_tag}_elbo.png",
            f"ELBO vs iteration — {scenario.label} [{family}]",
            "ELBO (eval)",
            perf_series,
            logy=False,
            ref_y=ref_elbo,
            ref_label=f"Reference (best - {args.ref_tol:g})",
        )
        save_curve_plot(
            FIGS_DIR / f"main_{scenario.key}_{family_tag}_curvature_cond.png",
            f"Hessian condition proxy vs iteration — {scenario.label} [{family}]",
            r"$|\lambda_{\max}| / \max(|\lambda_{\min}|,\epsilon)$",
            curv_series,
            logy=True,
        )
        save_tiled_main_plot(
            FIGS_DIR / f"main_{scenario.key}_{family_tag}_tiled.png",
            scenario.label,
            family,
            perf_series,
            curv_series,
            ref_elbo=ref_elbo,
        )


def run_variance_axis_block(args, scenario: Scenario):
    cfg = load_cfg(scenario.config_path)
    model = build_model(cfg, scenario.param)
    low_rank = int(cfg.get("variational", {}).get("low_rank_dim", 3))
    n_samp = args.variance_n_samples
    if n_samp is None:
        n_samp = int(cfg["methods"]["bbvi_adam"].get("n_samples", 20))

    # Columns vary VR knobs; curves overlay optimizers (BBVI / NGVI / Diag-Fisher).
    vr_modes = [
        ("base", False, False),
        ("antithetic", True, False),
        ("stl", False, True),
        ("antithetic_stl", True, True),
    ]
    legend_keys = {
        "bbvi": "BBVI + Adam",
        "ngvi": "Natural Gradient",
        "diag": "Diagonal Fisher",
    }
    rows = []
    column_overlays: dict[str, dict[str, tuple[list[int], list[float], list[float]]]] = {}

    vr_overrides_core = {"n_samples": n_samp}

    for variance_method, use_anti, use_stl in vr_modes:
        column_overlays[variance_method] = {}
        desc_prefix = variance_method.replace("_", " ")

        # BBVI + Adam baseline uses bbvi_adam only when VR is off.
        if not use_anti and not use_stl:
            spec_bbvi = MethodSpec(
                f"va_{variance_method}_bbvi",
                "bbvi_adam",
                f"{desc_prefix}: BBVI + Adam ns={n_samp}",
                {**vr_overrides_core},
            )
        else:
            spec_bbvi = MethodSpec(
                f"va_{variance_method}_bbvi",
                "variance_reduced",
                f"{desc_prefix}: BBVI + Adam ns={n_samp}",
                {**vr_overrides_core, "use_antithetic": use_anti, "use_stl": use_stl},
            )

        spec_ngvi = MethodSpec(
            f"va_{variance_method}_ngvi",
            "natural_gradient",
            f"{desc_prefix}: Natural Gradient ns={n_samp}",
            {**vr_overrides_core, "use_antithetic": use_anti, "use_stl": use_stl},
        )
        spec_diag = MethodSpec(
            f"va_{variance_method}_diag",
            "diagonal_fisher",
            f"{desc_prefix}: Diagonal Fisher ns={n_samp}",
            {**vr_overrides_core, "use_antithetic": use_anti, "use_stl": use_stl},
        )

        for opt_key, spec in (
            ("bbvi", spec_bbvi),
            ("ngvi", spec_ngvi),
            ("diag", spec_diag),
        ):
            out = run_single(
                model,
                cfg,
                spec,
                family="mean_field",
                low_rank=low_rank,
                n_iters=args.n_iters,
                log_every=args.log_every,
                probe_every=args.probe_every,
                eval_mc=args.eval_mc,
                hess_mc=args.hess_mc,
                seed=args.seed,
                probe_hessian=False,
            )
            final_perf = out["perf"][-1]
            probe_cum = out["curv"][-1]["probe_time_cum_s"] if out["curv"] else 0.0
            xs, elbos, gvars = _perf_curves_xy(out)
            column_overlays[variance_method][legend_keys[opt_key]] = (xs, elbos, gvars)

            rows.append(
                {
                    "scenario": scenario.key,
                    "run_id": spec.run_id,
                    "label": spec.label,
                    "optimizer": opt_key,
                    "variance_method": variance_method,
                    "n_samples": n_samp,
                    "final_eval_elbo": final_perf["eval_elbo"],
                    "final_grad_var": final_perf["grad_var"],
                    "final_hess_cond_abs": float("nan"),
                    "final_lambda_max": float("nan"),
                    "final_lambda_min": float("nan"),
                    "train_time_s": final_perf["train_time_s"],
                    "probe_time_cum_s": probe_cum,
                }
            )

    save_table_csv(
        CSV_DIR / f"variance_axis_{scenario.key}.csv",
        rows,
        [
            "scenario",
            "run_id",
            "label",
            "optimizer",
            "variance_method",
            "n_samples",
            "final_eval_elbo",
            "final_grad_var",
            "final_hess_cond_abs",
            "final_lambda_max",
            "final_lambda_min",
            "train_time_s",
            "probe_time_cum_s",
        ],
    )
    save_variance_optimizer_grid(
        FIGS_DIR / f"variance_axis_{scenario.key}_optimizer_grid.png",
        scenario.label,
        column_overlays,
        n_samples=n_samp,
    )


def run_curvature_axis_block(args, scenarios: list[Scenario]):
    rows = []
    specs = [
        MethodSpec("bbvi", "bbvi_adam", "BBVI + Adam", {}),
        MethodSpec("ngvi", "natural_gradient", "Natural Gradient", {"use_antithetic": False, "use_stl": False}),
    ]
    for sc in scenarios:
        cfg = load_cfg(sc.config_path)
        model = build_model(cfg, sc.param)
        low_rank = int(cfg.get("variational", {}).get("low_rank_dim", 3))
        for spec in specs:
            out = run_single(
                model,
                cfg,
                spec,
                family="mean_field",
                low_rank=low_rank,
                n_iters=args.n_iters,
                log_every=args.log_every,
                probe_every=args.probe_every,
                eval_mc=args.eval_mc,
                hess_mc=args.hess_mc,
                seed=args.seed,
            )
            final_perf = out["perf"][-1]
            final_curv = out["curv"][-1] if out["curv"] else {
                "cond_abs": float("nan"),
                "lambda_max": float("nan"),
                "lambda_min": float("nan"),
                "probe_time_cum_s": 0.0,
            }
            rows.append(
                {
                    "scenario": sc.key,
                    "label": sc.label,
                    "run_id": spec.run_id,
                    "method": spec.label,
                    "final_eval_elbo": final_perf["eval_elbo"],
                    "final_grad_var": final_perf["grad_var"],
                    "final_hess_cond_abs": final_curv["cond_abs"],
                    "final_lambda_max": final_curv["lambda_max"],
                    "final_lambda_min": final_curv["lambda_min"],
                    "train_time_s": final_perf["train_time_s"],
                    "probe_time_cum_s": final_curv["probe_time_cum_s"],
                }
            )

    save_table_csv(
        CSV_DIR / "curvature_axis_summary.csv",
        rows,
        [
            "scenario",
            "label",
            "run_id",
            "method",
            "final_eval_elbo",
            "final_grad_var",
            "final_hess_cond_abs",
            "final_lambda_max",
            "final_lambda_min",
            "train_time_s",
            "probe_time_cum_s",
        ],
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    all_scenarios = [
        Scenario("funnel", "Neal's Funnel", "experiments/configs/neals_funnel.yaml", None),
        Scenario("schools_cp", "Eight Schools (CP)", "experiments/configs/eight_schools.yaml", "centered"),
        Scenario("schools_ncp", "Eight Schools (NCP)", "experiments/configs/eight_schools.yaml", "noncentered"),
    ]
    scenario_by_key = {s.key: s for s in all_scenarios}

    t0 = time.time()
    if args.run_main:
        for sc in all_scenarios:
            print(f"[main] {sc.label}")
            run_main_block(args, sc)

    if args.run_variance_axis:
        sc_key = args.variance_scenario.strip()
        if sc_key not in scenario_by_key:
            raise ValueError(f"Unknown variance scenario key: {sc_key}")
        sc = scenario_by_key[sc_key]
        print(f"[variance-axis] {sc.label}")
        run_variance_axis_block(args, sc)

    if args.run_curvature_axis:
        keys = [x.strip() for x in args.curvature_scenarios.split(",") if x.strip()]
        chosen = [scenario_by_key[k] for k in keys]
        print("[curvature-axis] " + ", ".join(s.label for s in chosen))
        run_curvature_axis_block(args, chosen)

    # Cross-scenario method comparison in one image (3 scenarios x 2 metrics).
    if args.run_main:
        scenario_specs = [
            ("funnel", "Funnel"),
            ("schools_cp", "Schools CP"),
            ("schools_ncp", "Schools NCP"),
        ]
        save_family_overview_tile(
            FIGS_DIR / "main_overview_mf.png",
            "mean_field",
            scenario_specs,
            args.ref_tol,
        )
        save_family_overview_tile(
            FIGS_DIR / "main_overview_lrd.png",
            "low_rank_diag",
            scenario_specs,
            args.ref_tol,
        )

    print(f"Done. Wrote outputs to {RESULTS_DIR} (csv/, figs/) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
