"""
Postprocess curvature-variance study outputs (reads run_curvature_variance_study CSVs).

Writes summary CSVs under results_dir/csv/ and figures under results_dir/figs/.
ELBO threshold for iters/time-to-threshold: --threshold-mode weakest (default, shared
min over methods' best ELBO minus --tol) or global (shared max minus --tol).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN_ID_ORDER = ("bbvi", "bbvi_vr", "diag", "diag_vr", "ngvi", "ngvi_vr")


def parse_args():
    default_results_dir = Path(__file__).resolve().parent / "results" / "curvature_variance_study"
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=str, default=str(default_results_dir))
    p.add_argument(
        "--tol",
        type=float,
        default=1.0,
        help="Subtracted from the shared ELBO bar (weakest: min(best)−tol; global: max−tol).",
    )
    p.add_argument(
        "--threshold-mode",
        choices=("weakest", "global"),
        default="weakest",
        help="weakest: all methods share target = min(run best ELBO) − tol. global: max − tol.",
    )
    return p.parse_args()


def auc_trapz(xs: np.ndarray, ys: np.ndarray) -> float:
    if xs.size < 2 or ys.size < 2:
        return float("nan")
    return float(np.trapz(ys, xs))


def time_to_threshold(df: pd.DataFrame, threshold: float) -> float:
    hit = df[df["eval_elbo"] >= threshold]
    return float("nan") if hit.empty else float(hit["train_time_s"].iloc[0])


def iters_to_threshold(df: pd.DataFrame, threshold: float) -> float:
    hit = df[df["eval_elbo"] >= threshold]
    return float("nan") if hit.empty else float(hit["iter"].iloc[0])


def load_main_frames(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    csv_dir = results_dir / "csv"
    search_dir = csv_dir if csv_dir.exists() else results_dir
    perf_files = sorted(search_dir.glob("main_*_perf.csv"))
    curv_files = sorted(search_dir.glob("main_*_curvature.csv"))
    if not perf_files or not curv_files:
        raise FileNotFoundError("Missing main_*_perf.csv or main_*_curvature.csv outputs.")
    perf = pd.concat((pd.read_csv(pf) for pf in perf_files), ignore_index=True)
    curv = pd.concat((pd.read_csv(cf) for cf in curv_files), ignore_index=True)
    return perf, curv


def group_elbo_minmax_best(perf: pd.DataFrame) -> dict[tuple[str, str], tuple[float, float]]:
    """Per (scenario, family): (min over run_ids of best ELBO, max over all logged ELBO)."""
    out: dict[tuple[str, str], tuple[float, float]] = {}
    for (scenario, family), sf in perf.groupby(["scenario", "family"]):
        bests: list[float] = []
        for _, g in sf.groupby("run_id"):
            fe = g[np.isfinite(g["eval_elbo"])]
            if not fe.empty:
                bests.append(float(fe["eval_elbo"].max()))
        if not bests:
            out[(scenario, family)] = (float("nan"), float("nan"))
        else:
            fe_all = sf[np.isfinite(sf["eval_elbo"])]
            mx = float(fe_all["eval_elbo"].max()) if not fe_all.empty else float("nan")
            out[(scenario, family)] = (min(bests), mx)
    return out


def summarize_main(
    perf: pd.DataFrame,
    curv: pd.DataFrame,
    tol: float,
    *,
    threshold_mode: str = "weakest",
) -> pd.DataFrame:
    bounds = group_elbo_minmax_best(perf)
    rows: list[dict] = []
    for key, g in perf.groupby(["scenario", "family", "run_id", "label"]):
        g = g.sort_values("iter")
        finite = g[np.isfinite(g["eval_elbo"]) & np.isfinite(g["train_time_s"])]
        sk = (key[0], key[1])
        gmin, gmax = bounds.get(sk, (np.nan, np.nan))
        base = gmin if threshold_mode == "weakest" else gmax
        target = float(base - tol) if np.isfinite(base) else float("nan")

        if finite.empty:
            rows.append(
                {
                    "scenario": key[0],
                    "family": key[1],
                    "run_id": key[2],
                    "label": key[3],
                    "final_eval_elbo": np.nan,
                    "best_eval_elbo": np.nan,
                    "final_grad_var": np.nan,
                    "auc_elbo_time": np.nan,
                    "iters_to_threshold": np.nan,
                    "time_to_threshold_s": np.nan,
                    "target_elbo": target,
                    "group_min_best_eval_elbo": gmin,
                    "group_max_eval_elbo": gmax,
                    "elbo_threshold_tol": tol,
                    "elbo_threshold_mode": threshold_mode,
                    "final_train_time_s": np.nan,
                    "final_probe_time_cum_s": np.nan,
                    "final_hess_cond_abs": np.nan,
                }
            )
            continue

        final_eval = float(finite["eval_elbo"].iloc[-1])
        best_eval = float(finite["eval_elbo"].max())
        final_grad_var = float(finite["grad_var"].iloc[-1]) if np.isfinite(finite["grad_var"].iloc[-1]) else np.nan
        auc = auc_trapz(finite["train_time_s"].to_numpy(), finite["eval_elbo"].to_numpy())
        ttt = time_to_threshold(finite, target)
        itt = iters_to_threshold(finite, target)

        cg = curv[
            (curv["scenario"] == key[0])
            & (curv["family"] == key[1])
            & (curv["run_id"] == key[2])
            & (curv["label"] == key[3])
        ].sort_values("iter")
        final_probe_time = float(cg["probe_time_cum_s"].iloc[-1]) if not cg.empty else np.nan
        final_cond = float(cg["cond_abs"].iloc[-1]) if not cg.empty else np.nan

        rows.append(
            {
                "scenario": key[0],
                "family": key[1],
                "run_id": key[2],
                "label": key[3],
                "final_eval_elbo": final_eval,
                "best_eval_elbo": best_eval,
                "final_grad_var": final_grad_var,
                "auc_elbo_time": auc,
                "iters_to_threshold": itt,
                "time_to_threshold_s": ttt,
                "target_elbo": target,
                "group_min_best_eval_elbo": gmin,
                "group_max_eval_elbo": gmax,
                "elbo_threshold_tol": tol,
                "elbo_threshold_mode": threshold_mode,
                "final_train_time_s": float(finite["train_time_s"].iloc[-1]),
                "final_probe_time_cum_s": final_probe_time,
                "final_hess_cond_abs": final_cond,
            }
        )
    return pd.DataFrame(rows)


def summarize_break_even(main_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scenario, family), g in main_summary.groupby(["scenario", "family"]):
        ng = g[g["run_id"] == "ngvi"]
        dg = g[g["run_id"] == "diag"]
        if ng.empty or dg.empty:
            continue
        ng, dg = ng.iloc[0], dg.iloc[0]
        rows.append(
            {
                "scenario": scenario,
                "family": family,
                "delta_auc_elbo_time_ng_minus_diag": ng["auc_elbo_time"] - dg["auc_elbo_time"],
                "delta_best_elbo_ng_minus_diag": ng["best_eval_elbo"] - dg["best_eval_elbo"],
                "delta_time_to_threshold_ng_minus_diag_s": ng["time_to_threshold_s"] - dg["time_to_threshold_s"],
                "ngvi_justified_by_time": int(
                    np.isfinite(ng["time_to_threshold_s"])
                    and np.isfinite(dg["time_to_threshold_s"])
                    and ng["time_to_threshold_s"] <= dg["time_to_threshold_s"]
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_variance_axis(results_dir: Path) -> pd.DataFrame:
    csv_dir = results_dir / "csv"
    search_dir = csv_dir if csv_dir.exists() else results_dir
    files = sorted(search_dir.glob("variance_axis_*.csv"))
    if not files:
        return pd.DataFrame()
    rows = []
    for path in files:
        df = pd.read_csv(path)
        if df.empty:
            continue
        scenario_key = (
            str(df["scenario"].iloc[0]) if "scenario" in df.columns else path.stem.replace("variance_axis_", "")
        )
        n_samp = int(df["n_samples"].iloc[0]) if "n_samples" in df.columns else 0

        if "optimizer" in df.columns and set(df["optimizer"].astype(str)).issuperset({"bbvi", "ngvi", "diag"}):
            for vm in ["base", "antithetic", "stl", "antithetic_stl"]:
                sub = df[(df["variance_method"] == vm) & (df["optimizer"].isin(["bbvi", "ngvi", "diag"]))]
                if len(sub) < 3:
                    continue
                by_opt = sub.set_index("optimizer")
                bb, ng, dg = by_opt.loc["bbvi"], by_opt.loc["ngvi"], by_opt.loc["diag"]
                rows.append(
                    {
                        "scenario": scenario_key,
                        "n_samples": n_samp,
                        "variance_method": vm,
                        "bbvi_final_elbo": bb["final_eval_elbo"],
                        "ngvi_final_elbo": ng["final_eval_elbo"],
                        "diag_final_elbo": dg["final_eval_elbo"],
                        "bbvi_final_grad_var": bb["final_grad_var"],
                        "ngvi_final_grad_var": ng["final_grad_var"],
                        "diag_final_grad_var": dg["final_grad_var"],
                    }
                )
        else:
            g = df

            def row_for(run_prefix: str) -> pd.Series | None:
                m = g[g["run_id"] == run_prefix]
                if not m.empty:
                    return m.iloc[0]
                alt = g[g["run_id"].str.startswith(run_prefix)]
                return alt.iloc[0] if not alt.empty else None

            b, a, s, c = row_for("bbvi_base"), row_for("bbvi_anti"), row_for("bbvi_stl"), row_for("bbvi_anti_stl")
            if b is None or a is None or s is None or c is None:
                continue
            rows.append(
                {
                    "scenario": scenario_key,
                    "n_samples": n_samp,
                    "base_final_elbo": b["final_eval_elbo"],
                    "antithetic_final_elbo": a["final_eval_elbo"],
                    "stl_final_elbo": s["final_eval_elbo"],
                    "antithetic_stl_final_elbo": c["final_eval_elbo"],
                    "base_to_antithetic_variance_factor": b["final_grad_var"] / max(a["final_grad_var"], 1e-12),
                    "base_to_stl_variance_factor": b["final_grad_var"] / max(s["final_grad_var"], 1e-12),
                    "base_to_antithetic_stl_variance_factor": b["final_grad_var"] / max(c["final_grad_var"], 1e-12),
                    "antithetic_gain_over_base": a["final_eval_elbo"] - b["final_eval_elbo"],
                    "stl_gain_over_base": s["final_eval_elbo"] - b["final_eval_elbo"],
                    "antithetic_stl_gain_over_base": c["final_eval_elbo"] - b["final_eval_elbo"],
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    if "variance_method" in out.columns:
        return out.sort_values(["scenario", "n_samples", "variance_method"])
    return out.sort_values(["scenario", "n_samples"])


def summarize_curvature_gain(results_dir: Path) -> pd.DataFrame:
    csv_dir = results_dir / "csv"
    search_dir = csv_dir if csv_dir.exists() else results_dir
    path = search_dir / "curvature_axis_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    rows = []
    for scenario, g in df.groupby("scenario"):
        bb = g[g["run_id"] == "bbvi"]
        ng = g[g["run_id"] == "ngvi"]
        if bb.empty or ng.empty:
            continue
        b, n = bb.iloc[0], ng.iloc[0]
        rows.append(
            {
                "scenario": scenario,
                "bbvi_final_elbo": b["final_eval_elbo"],
                "ngvi_final_elbo": n["final_eval_elbo"],
                "ngvi_gain_over_bbvi": n["final_eval_elbo"] - b["final_eval_elbo"],
                "bbvi_grad_var": b["final_grad_var"],
                "ngvi_grad_var": n["final_grad_var"],
                "bbvi_hess_cond": b["final_hess_cond_abs"],
                "ngvi_hess_cond": n["final_hess_cond_abs"],
                "curvature_proxy_for_plot": b["final_hess_cond_abs"],
            }
        )
    return pd.DataFrame(rows)


def plot_gain_vs_curvature(curv_gain: pd.DataFrame, out_path: Path):
    if curv_gain.empty:
        return
    x = np.log(np.maximum(curv_gain["curvature_proxy_for_plot"].to_numpy(), 1e-12))
    y = curv_gain["ngvi_gain_over_bbvi"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.scatter(x, y, s=45)
    for _, row in curv_gain.iterrows():
        ax.annotate(
            str(row["scenario"]),
            (np.log(max(row["curvature_proxy_for_plot"], 1e-12)), row["ngvi_gain_over_bbvi"]),
            fontsize=8,
        )
    if len(x) >= 2 and np.isfinite(x).all() and np.isfinite(y).all():
        m, b = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 100)
        ax.plot(xx, m * xx + b, linestyle="--")
    ax.set_xlabel("log(curvature proxy)")
    ax.set_ylabel("NGVI gain over BBVI (final ELBO)")
    ax.set_title("NGVI Gain vs Curvature")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _pivot_run_bars(main_summary: pd.DataFrame, value_col: str, ylabel: str, title: str, out_path: Path):
    df = main_summary[np.isfinite(main_summary[value_col])].copy()
    if df.empty:
        return
    pivot = df.pivot_table(index=["scenario", "family"], columns="run_id", values=value_col, aggfunc="mean")
    methods = [c for c in RUN_ID_ORDER if c in pivot.columns]
    if not methods:
        return
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    x = np.arange(len(pivot.index))
    w = 0.12
    for i, m in enumerate(methods):
        ax.bar(x + (i - (len(methods) - 1) / 2) * w, pivot[m].to_numpy(), width=w, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}/{f}" for s, f in pivot.index], rotation=22, ha="right", fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.32),
        ncol=4,
        fontsize=11,
        frameon=False,
        handlelength=2.0,
        handletextpad=0.68,
        columnspacing=1.35,
        borderpad=0.55,
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.29)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    if not results_dir.exists() and not results_dir.is_absolute():
        alt = (Path(__file__).resolve().parent / results_dir).resolve()
        if alt.exists():
            results_dir = alt
    csv_dir = results_dir / "csv"
    figs_dir = results_dir / "figs"
    if not csv_dir.exists():
        csv_dir = results_dir
    figs_dir.mkdir(parents=True, exist_ok=True)

    perf, curv = load_main_frames(results_dir)
    main_summary = summarize_main(perf, curv, tol=args.tol, threshold_mode=args.threshold_mode)

    main_summary.to_csv(csv_dir / "summary_main_metrics.csv", index=False)
    summarize_break_even(main_summary).to_csv(csv_dir / "summary_ngvi_vs_diag_break_even.csv", index=False)
    summarize_variance_axis(results_dir).to_csv(csv_dir / "summary_variance_axis_effects.csv", index=False)
    curv_gain = summarize_curvature_gain(results_dir)
    curv_gain.to_csv(csv_dir / "summary_curvature_gain.csv", index=False)
    plot_gain_vs_curvature(curv_gain, figs_dir / "plot_ngvi_gain_vs_curvature.png")
    _pivot_run_bars(
        main_summary,
        "auc_elbo_time",
        "AUC(ELBO vs train time)",
        "Cost-normalized ELBO performance",
        figs_dir / "plot_auc_time_by_method.png",
    )
    suffix = " (min best ELBO − tol)" if args.threshold_mode == "weakest" else " (max ELBO − tol)"
    _pivot_run_bars(
        main_summary,
        "iters_to_threshold",
        "Iterations to threshold",
        f"Convergence Iterations by Method{suffix}",
        figs_dir / "plot_iters_to_convergence.png",
    )
    _pivot_run_bars(
        main_summary,
        "time_to_threshold_s",
        "Train time to threshold (s)",
        f"Convergence Time by Method{suffix}",
        figs_dir / "plot_time_to_convergence.png",
    )

    print(f"Wrote summary outputs to {results_dir} (csv/, figs/)")


if __name__ == "__main__":
    main()
