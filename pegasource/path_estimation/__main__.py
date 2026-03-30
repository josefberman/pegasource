"""CLI: ``python -m pegasource.path_estimation ...``."""

from __future__ import annotations

import argparse
from pathlib import Path

from .evaluate import print_summary, run_evaluation


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate path reconstruction methods against a true_path CSV.",
    )
    p.add_argument(
        "--observations",
        type=Path,
        required=True,
        help="Path to *_observations.csv",
    )
    p.add_argument(
        "--true-path",
        type=Path,
        required=True,
        help="Path to *_true_path.csv (1 Hz ground truth).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("path_estimation_runs/run1"),
        help="Directory for metrics.json and figures/.",
    )
    p.add_argument(
        "--methods",
        type=str,
        default="dijkstra,astar,hmm,kf,ekf,ukf,particle,lstm,transformer,gnn",
        help="Comma-separated method names.",
    )
    p.add_argument("--no-plots", action="store_true", help="Skip figure generation.")
    p.add_argument(
        "--map-plots",
        action="store_true",
        help="Also write Web Mercator basemap PNGs (slower; may fetch tiles).",
    )
    p.add_argument("--device", type=str, default=None, help="Torch device, e.g. cuda or cpu.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = run_evaluation(
        args.observations,
        args.true_path,
        args.output_dir,
        methods=methods,
        plot=not args.no_plots,
        plot_map=bool(args.map_plots) and not args.no_plots,
        device=args.device,
        seed=args.seed,
    )
    print_summary(summary)


if __name__ == "__main__":
    main()
