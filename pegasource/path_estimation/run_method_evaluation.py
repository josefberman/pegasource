#!/usr/bin/env python3
"""Generate synthetic datasets (run_1 … run_N) and path estimation; write per-method CSVs.

Each ``method_eval/run_<i>/`` holds ``run_<i>_observations.csv``, ``run_<i>_true_path.csv``,
plots from generation, ``metrics.json``, and ``figures/`` from evaluation.

Writes ``method_eval/<method>.csv`` with metric names as rows and ``run_1`` … ``run_N`` columns,
and ``method_eval/metric_bars/<metric>.png`` — one bar chart per metric (x = methods, mean height, std error bars).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluate import run_evaluation
from .generate_synthetic_datasets import generate_dataset

DEFAULT_METHODS = (
    "dijkstra,astar,hmm,kf,ekf,ukf,particle,lstm,transformer,gnn"
)

# Scalar metrics from ``path_estimation.metrics.compute_all_metrics`` (exclude nested ``meta``).
METRIC_KEYS: List[str] = [
    "rmse_m",
    "mae_m",
    "mae_east_m",
    "mae_north_m",
    "hausdorff_m",
    "path_length_true_m",
    "path_length_est_m",
    "length_ratio",
    "endpoint_error_m",
    "discrete_frechet_m",
    "dtw_m",
]


def _parse_methods(s: str) -> List[str]:
    return [m.strip().lower() for m in s.split(",") if m.strip()]


def _write_method_csvs(
    eval_dir: Path,
    summaries: List[Dict[str, Any]],
    methods: List[str],
    run_cols: List[str],
) -> None:
    index = METRIC_KEYS + ["error"]
    for method in methods:
        mat = pd.DataFrame(index=index, columns=run_cols, dtype=object)
        for col, summary in zip(run_cols, summaries):
            if method not in summary:
                mat.loc["error", col] = "method_missing"
                for k in METRIC_KEYS:
                    mat.loc[k, col] = np.nan
                continue
            block = summary[method]
            if isinstance(block, dict) and "error" in block:
                mat.loc["error", col] = str(block["error"])
                for k in METRIC_KEYS:
                    mat.loc[k, col] = np.nan
                continue
            for k in METRIC_KEYS:
                mat.loc[k, col] = block.get(k, np.nan)
            mat.loc["error", col] = ""
        mat.index.name = "metric"
        out_path = eval_dir / f"{method}.csv"
        mat.to_csv(out_path, float_format="%.10g")


def _mean_std_per_method_metric(
    summaries: List[Dict[str, Any]], methods: List[str]
) -> tuple[np.ndarray, np.ndarray]:
    """``means``, ``stds`` shaped (len(METRIC_KEYS), len(methods)) across runs."""
    means = np.full((len(METRIC_KEYS), len(methods)), np.nan, dtype=float)
    stds = np.full_like(means, np.nan)
    for j, method in enumerate(methods):
        per_key: Dict[str, List[float]] = {k: [] for k in METRIC_KEYS}
        for summary in summaries:
            if method not in summary:
                continue
            block = summary[method]
            if not isinstance(block, dict) or "error" in block:
                continue
            for k in METRIC_KEYS:
                v = block.get(k)
                if v is not None and np.isfinite(v):
                    per_key[k].append(float(v))
        for i, k in enumerate(METRIC_KEYS):
            arr = np.asarray(per_key[k], dtype=float)
            if arr.size == 0:
                continue
            means[i, j] = float(np.mean(arr))
            stds[i, j] = float(np.std(arr, ddof=0)) if arr.size > 1 else 0.0
    return means, stds


def _plot_metric_bar_charts(
    eval_dir: Path,
    methods: List[str],
    summaries: List[Dict[str, Any]],
) -> None:
    """One PNG per metric: x = methods, bar height = mean over runs, error = std."""
    means, stds = _mean_std_per_method_metric(summaries, methods)
    n_methods = len(methods)
    out_dir = eval_dir / "metric_bars"
    out_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(n_methods, dtype=float)
    width = min(0.82, 7.0 / max(n_methods, 1))
    cmap = plt.get_cmap("tab10")
    edge = "#222222"

    for i, metric in enumerate(METRIC_KEYS):
        mrow = np.array(means[i], dtype=float)
        srow = np.array(stds[i], dtype=float)
        srow[~np.isfinite(srow)] = 0.0
        fig_w = max(9.0, 0.52 * n_methods + 2.5)
        fig, ax = plt.subplots(figsize=(fig_w, 5.0))
        for j in range(n_methods):
            if not np.isfinite(mrow[j]):
                continue
            ax.bar(
                x[j],
                mrow[j],
                width=width,
                yerr=srow[j],
                color=cmap(j % 10),
                edgecolor=edge,
                linewidth=0.55,
                alpha=0.88,
                capsize=2.5,
                ecolor="black",
                error_kw={"elinewidth": 0.85, "capthick": 0.85, "alpha": 0.9},
            )
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=32, ha="right", fontsize=8)
        ax.set_ylabel("Mean ± std (across runs)", fontsize=10)
        ax.set_xlabel("Method", fontsize=10)
        ax.set_title(metric, fontsize=11, pad=10)
        ax.set_facecolor("#e5e5e5")
        fig.patch.set_facecolor("#e5e5e5")
        ax.grid(True, axis="y", color="white", linestyle="-", linewidth=1.0, alpha=1.0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        safe = metric.replace("/", "_").replace(" ", "_")
        fig.savefig(
            out_dir / f"{safe}.png",
            dpi=150,
            bbox_inches="tight",
            facecolor=fig.patch.get_facecolor(),
        )
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--n-runs",
        type=int,
        default=10,
        help="Number of datasets (run_1 … run_N). Default: %(default)s.",
    )
    p.add_argument(
        "--duration-s",
        type=int,
        default=3600,
        help="Simulation duration per dataset (seconds). Default: %(default)s.",
    )
    p.add_argument(
        "--data-seed",
        type=int,
        default=42,
        help="Base seed for dataset RNG; run i uses data_seed + i.",
    )
    p.add_argument(
        "--eval-seed",
        type=int,
        default=0,
        help="Base seed for evaluation; run i uses eval_seed + i.",
    )
    p.add_argument(
        "--methods",
        type=str,
        default=DEFAULT_METHODS,
        help="Comma-separated method names (same as path_estimation CLI).",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device for NN/GNN (e.g. cuda, cpu).",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip evaluation figures (metrics still computed).",
    )
    p.add_argument(
        "--no-bar-plot",
        action="store_true",
        help="Skip metric_bars/*.png (one mean±std bar chart per metric).",
    )
    args = p.parse_args()

    # Writes under ./method_eval/ in the current working directory (not inside site-packages).
    eval_dir = Path.cwd() / "method_eval"
    n = max(1, args.n_runs)
    methods = _parse_methods(args.methods)
    run_cols = [f"run_{i}" for i in range(1, n + 1)]
    summaries: List[Dict[str, Any]] = []

    for i in range(1, n + 1):
        run_name = f"run_{i}"
        run_dir = eval_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(int(args.data_seed) + i)
        generate_dataset(
            rng=rng,
            output_dir=run_dir,
            duration_s=args.duration_s,
            dataset_id=run_name,
        )
        obs = run_dir / f"{run_name}_observations.csv"
        tru = run_dir / f"{run_name}_true_path.csv"
        summary = run_evaluation(
            observations_csv=obs,
            true_path_csv=tru,
            output_dir=run_dir,
            methods=methods,
            plot=not args.no_plots,
            plot_map=False,
            device=args.device,
            seed=int(args.eval_seed) + i,
        )
        summaries.append(summary)

    _write_method_csvs(eval_dir, summaries, methods, run_cols)

    if not args.no_bar_plot:
        _plot_metric_bar_charts(eval_dir, methods, summaries)
        print(f"Wrote {len(METRIC_KEYS)} bar charts under {eval_dir / 'metric_bars'}")

    print(f"Wrote {len(methods)} CSV files under {eval_dir}")
    print(f"Per-run outputs: {eval_dir}/run_<i>/ (metrics.json, figures/)")


