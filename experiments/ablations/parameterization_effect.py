"""
Section 5.4 – Parameterization Effect
--------------------------------------
Compares ELBO convergence of all four inference methods on the Eight Schools
model under both the centered (CP) and non-centered (NCP) parameterizations.

Expected finding: NGVI's advantage over Euclidean baselines (Adam, Variance-
Reduced) diminishes or vanishes in NCP, where the posterior is already
well-conditioned and Fisher preconditioning provides little extra benefit.

Usage:
    python experiments/ablations/parameterization_effect.py

Outputs:
    results/parameterization_effect.png
    results/parameterization_effect.pkl
    results/parameterization_effect.log
"""

import os
import sys
import pickle
import logging

import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project root on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from models.eight_schools import EightSchools
from inference.bbvi_adam import BBVIAdam
from inference.natural_gradient import NaturalGradientVI
from inference.diagonal_fisher import DiagonalFisherVI
from inference.variance_reduced import VarianceReducedBBVI


# ── Config ────────────────────────────────────────────────────────────────────

_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "configs", "eight_schools.yaml"
)
with open(_CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

exp_cfg    = cfg["parameterization_comparison"]
budget     = cfg["budget"]
N_ITERS    = exp_cfg.get("n_iters", budget["n_iters"])
LOG_EVERY  = budget["log_every"]
SEED       = cfg.get("seed", 42)
METHOD_CFGS = cfg["methods"]
METHOD_NAMES = exp_cfg["methods"]          # order from yaml
PARAMETERIZATIONS = ["centered", "noncentered"]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Logging: stdout + file ────────────────────────────────────────────────────

log_path = os.path.join(RESULTS_DIR, "parameterization_effect.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, mode="w"),
    ],
)
log = logging.getLogger()

# Redirect bare print() calls through the logger so everything goes to the file
def print(*args, **kwargs):  # noqa: A001
    log.info(" ".join(str(a) for a in args))


# ── Method factory ────────────────────────────────────────────────────────────

def make_method(name: str, model, D: int):
    c = METHOD_CFGS[name]
    if name == "bbvi_adam":
        return BBVIAdam(
            model, D,
            lr=float(c["lr"]), n_samples=int(c["n_samples"]),
            betas=tuple(float(b) for b in c["betas"]),
            eps_adam=float(c["eps_adam"]),
        )
    if name == "natural_gradient":
        return NaturalGradientVI(
            model, D,
            lr=float(c["lr"]), n_samples=int(c["n_samples"]),
            damping=float(c["damping"]),
        )
    if name == "diagonal_fisher":
        return DiagonalFisherVI(
            model, D,
            lr=float(c["lr"]), n_samples=int(c["n_samples"]),
            damping=float(c["damping"]),
            fisher_mode=c.get("fisher_mode", "analytic"),
            fisher_ema=float(c.get("fisher_ema", 0.9)),
        )
    if name == "variance_reduced":
        return VarianceReducedBBVI(
            model, D,
            lr=float(c["lr"]), n_samples=int(c["n_samples"]),
            use_antithetic=bool(c.get("use_antithetic", True)),
            use_stl=bool(c.get("use_stl", True)),
        )
    raise ValueError(f"Unknown method: {name}")


# ── Run ───────────────────────────────────────────────────────────────────────

results = {}   # {parameterization: {method_name: result_dict}}

for param in PARAMETERIZATIONS:
    results[param] = {}
    model = EightSchools(parameterization=param)
    D = model.D

    print(f"\n{'='*60}")
    print(f"  Parameterization: {param.upper()}  (D={D})")
    print(f"{'='*60}")

    for name in METHOD_NAMES:
        print(f"\n--- {name} ---")
        torch.manual_seed(SEED)
        method = make_method(name, model, D)
        res = method.fit(n_iters=N_ITERS, log_every=LOG_EVERY)
        results[param][name] = res
        print(f"  Final ELBO: {res['elbo'][-1]:+.4f}")


# ── Save ──────────────────────────────────────────────────────────────────────

pkl_path = os.path.join(RESULTS_DIR, "parameterization_effect.pkl")
with open(pkl_path, "wb") as f:
    pickle.dump(results, f)
print(f"\nResults saved to {pkl_path}")


# ── Plot ──────────────────────────────────────────────────────────────────────

METHOD_LABELS = {
    "bbvi_adam":        "BBVI + Adam",
    "natural_gradient": "Natural Gradient VI",
    "diagonal_fisher":  "Diagonal-Fisher VI",
    "variance_reduced": "Variance-Reduced BBVI",
}
PARAM_STYLES = {
    "centered":    {"color": "#d62728", "linestyle": "-",  "label": "CP (centered)"},
    "noncentered": {"color": "#1f77b4", "linestyle": "--", "label": "NCP (non-centered)"},
}

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
iters = np.arange(LOG_EVERY, N_ITERS + 1, LOG_EVERY)

for ax, name in zip(axes, METHOD_NAMES):
    for param in PARAMETERIZATIONS:
        elbo = results[param][name]["elbo"]
        style = PARAM_STYLES[param]
        ax.plot(
            iters[: len(elbo)], elbo,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            label=style["label"],
        )
    ax.set_title(METHOD_LABELS[name], fontsize=11)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("ELBO")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig.suptitle(
    "Eight Schools: CP vs. NCP — ELBO Convergence Across Methods",
    fontsize=13, fontweight="bold",
)
plt.tight_layout()

plot_path = os.path.join(RESULTS_DIR, "parameterization_effect.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Figure saved to {plot_path}")

# ── Summary table ─────────────────────────────────────────────────────────────

print("\n\n" + "=" * 57)
print("  SUMMARY")
print("=" * 57)
for param in PARAMETERIZATIONS:
    label = "CENTERED" if param == "centered" else "NON-CENTERED"
    print(f"\n  {label}")
    print(f"  {'Method':<25} {'Final ELBO':>12} {'Best ELBO':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")
    for name in METHOD_NAMES:
        elbo = results[param][name]["elbo"]
        print(f"  {METHOD_LABELS[name]:<25} {elbo[-1]:>12.4f} {max(elbo):>12.4f}")

print("\n  CP vs NCP gap (Final ELBO — higher gap = more NGVI benefit in CP)")
print(f"  {'Method':<25} {'CP':>8} {'NCP':>8} {'gap':>8}")
print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
for name in METHOD_NAMES:
    cp  = results["centered"][name]["elbo"][-1]
    ncp = results["noncentered"][name]["elbo"][-1]
    print(f"  {METHOD_LABELS[name]:<25} {cp:>8.2f} {ncp:>8.2f} {ncp-cp:>+8.2f}")

print(f"\nLog saved to {log_path}")
