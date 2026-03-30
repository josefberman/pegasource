"""Run estimators, compute metrics, save JSON summary and figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .filters import (
    estimate_ekf_fused,
    estimate_kf_gps,
    estimate_particle_filter,
    estimate_ukf_fused,
)
from .graph_stitch import estimate_graph_stitch
from .gnn.estimate import estimate_gnn
from .graph_utils import get_projected_graph
from .hmm_map_match import estimate_hmm_map_match
from .io import (
    align_times_to_true,
    load_observations_csv,
    load_true_path_csv,
    stub_true_path_from_observations,
)
from .metrics import compute_all_metrics
from .nn.lstm_model import predict_lstm_at_times, train_lstm
from .nn.transformer_model import predict_transformer_at_times, train_transformer
from .types import EstimationResult
from .viz import plot_estimation_enu, plot_estimation_map

EstimatorFn = Callable[..., EstimationResult]

# Supervised / needs real ``true_df`` (training labels or GNN node targets).
METHODS_REQUIRING_GROUND_TRUTH: frozenset[str] = frozenset({"lstm", "transformer", "gnn"})

METHOD_REGISTRY: Dict[str, EstimatorFn] = {
    "dijkstra": lambda o, t, G, r: estimate_graph_stitch(o, t, G, r, mode="dijkstra"),
    "astar": lambda o, t, G, r: estimate_graph_stitch(o, t, G, r, mode="astar"),
    "hmm": estimate_hmm_map_match,
    "kf": estimate_kf_gps,
    "ekf": estimate_ekf_fused,
    "ukf": estimate_ukf_fused,
    "particle": estimate_particle_filter,
    "gnn": estimate_gnn,
}


def _estimate_lstm(
    obs_df: pd.DataFrame,
    true_df: pd.DataFrame,
    G,
    rng: np.random.Generator,
    device: Optional[str] = None,
) -> EstimationResult:
    dev = torch_device(device)
    model, ds = train_lstm(obs_df, true_df, dev)
    times_s, xy = predict_lstm_at_times(model, obs_df, true_df, dev, ds)
    return EstimationResult(
        times_s=times_s,
        east_m=xy[:, 0],
        north_m=xy[:, 1],
        meta={"method": "lstm"},
    )


def _estimate_transformer(
    obs_df: pd.DataFrame,
    true_df: pd.DataFrame,
    G,
    rng: np.random.Generator,
    device: Optional[str] = None,
) -> EstimationResult:
    dev = torch_device(device)
    model, ds = train_transformer(obs_df, true_df, dev)
    times_s, xy = predict_transformer_at_times(model, obs_df, true_df, dev, ds)
    return EstimationResult(
        times_s=times_s,
        east_m=xy[:, 0],
        north_m=xy[:, 1],
        meta={"method": "transformer"},
    )


def torch_device(name: Optional[str] = None):
    import torch as _torch

    if name:
        return _torch.device(name)
    return _torch.device("cuda" if _torch.cuda.is_available() else "cpu")


def _run_methods(
    obs_df: pd.DataFrame,
    true_df: pd.DataFrame,
    G: Any,
    methods: List[str],
    *,
    device: Optional[str] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """Run each method; values are :class:`EstimationResult` or ``{"error": ...}``."""
    rng = np.random.default_rng(seed)
    dev = torch_device(device)
    out: Dict[str, Any] = {}
    for name in methods:
        name = name.strip().lower()
        try:
            if name == "lstm":
                res = _estimate_lstm(obs_df, true_df, G, rng, device=device)
            elif name == "transformer":
                res = _estimate_transformer(obs_df, true_df, G, rng, device=device)
            elif name == "gnn":
                res = estimate_gnn(obs_df, true_df, G, rng, device=dev)
            else:
                fn = METHOD_REGISTRY.get(name)
                if fn is None:
                    raise KeyError(f"Unknown method: {name}")
                res = fn(obs_df, true_df, G, rng)
        except Exception as exc:
            out[name] = {"error": str(exc)}
            continue
        out[name] = res
    return out


def _run_evaluation_core(
    obs_df: pd.DataFrame,
    true_df: pd.DataFrame,
    G: Any,
    methods: List[str],
    *,
    output_dir: Optional[Path] = None,
    plot: bool = False,
    plot_map: bool = False,
    device: Optional[str] = None,
    seed: int = 0,
) -> Dict[str, Dict]:
    """Run estimators; compute metrics. Optionally write ``metrics.json`` and figures."""
    _, true_xy = align_times_to_true(true_df)
    raw = _run_methods(obs_df, true_df, G, methods, device=device, seed=seed)
    out: Dict[str, Dict] = {}
    fig_dir = (output_dir / "figures") if output_dir is not None else None
    if fig_dir is not None:
        fig_dir.mkdir(parents=True, exist_ok=True)

    for name, val in raw.items():
        if isinstance(val, dict) and "error" in val:
            out[name] = val
            continue
        res = val
        if not isinstance(res, EstimationResult):
            out[name] = {"error": "unexpected result type"}
            continue
        est_xy = np.column_stack((res.east_m, res.north_m))
        m = compute_all_metrics(true_xy, est_xy)
        m["meta"] = res.meta
        out[name] = m

        if plot and fig_dir is not None:
            plot_estimation_enu(
                true_df,
                res,
                obs_df,
                fig_dir / f"{name}_path_enu.png",
                title=f"{name.upper()} — estimated vs true (ENU)",
                show_observations=True,
                show_true_path=True,
            )
        if plot_map and fig_dir is not None:
            plot_estimation_map(
                true_df,
                res,
                fig_dir / f"{name}_path_map.png",
                title=f"{name.upper()} — map",
            )

    if output_dir is not None:
        summary_path = output_dir / "metrics.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=float)

    return out


def estimate_paths_only(
    observations_csv: Path,
    road_graph: Any,
    methods: List[str],
    *,
    output_hz: float = 1.0,
    output_dir: Optional[Path] = None,
    plot: bool = False,
    plot_map: bool = False,
    device: Optional[str] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """Estimate paths from observations only (no ground-truth CSV).

    Builds an internal time grid from the observation span at ``output_hz`` (default 1 Hz).
    Methods that require supervised training (**lstm**, **transformer**, **gnn**) are not
    supported here; use :func:`evaluate_path_estimation` with a true path file.

    Args:
        observations_csv: Path to ``*_observations.csv``.
        road_graph: Road network graph for map-based methods (ignored by pure filters).
        methods: e.g. ``["kf", "ukf", "dijkstra"]``.
        output_hz: Sample rate for the output trajectory timeline (Hz).
        output_dir: If set and ``plot`` is True, writes ``figures/<method>_path_enu.png``.
        plot: ENU figure with **estimate + observations only** (no true path line).
        plot_map: Not supported without real lon/lat ground truth; raises if True.
        device: Torch device (unused unless extended).
        seed: RNG seed.

    Returns:
        Per method: :class:`EstimationResult`, or ``{"error": "..."}`` on failure.
    """
    names = [m.strip().lower() for m in methods]
    forbidden = sorted({m for m in names if m in METHODS_REQUIRING_GROUND_TRUTH})
    if forbidden:
        raise ValueError(
            "These methods require ground truth for training: "
            f"{forbidden}. Use evaluate_path_estimation(..., true_path_csv=...)."
        )
    if plot_map:
        raise ValueError(
            "plot_map needs a real true path with lon/lat. Use evaluate_path_estimation()."
        )
    obs_df = load_observations_csv(observations_csv)
    true_df = stub_true_path_from_observations(obs_df, hz=output_hz)
    raw = _run_methods(obs_df, true_df, road_graph, names, device=device, seed=seed)
    fig_dir = (output_dir / "figures") if output_dir is not None else None
    if fig_dir is not None:
        fig_dir.mkdir(parents=True, exist_ok=True)
    if plot and fig_dir is not None:
        for name, val in raw.items():
            if isinstance(val, EstimationResult):
                plot_estimation_enu(
                    true_df,
                    val,
                    obs_df,
                    fig_dir / f"{name}_path_enu.png",
                    title=f"{name.upper()} — estimated path (ENU, no ground truth)",
                    show_observations=True,
                    show_true_path=False,
                )
    return raw


def evaluate_path_estimation(
    observations_csv: Path,
    true_path_csv: Path,
    road_graph: Any,
    methods: List[str],
    *,
    output_dir: Optional[Path] = None,
    plot: bool = False,
    plot_map: bool = False,
    device: Optional[str] = None,
    seed: int = 0,
) -> Dict[str, Dict]:
    """Run path estimation for CSV paths and a caller-supplied road graph.

    Loads observations and ground truth from disk, then runs each method in
    ``methods`` using ``road_graph`` for graph-based estimators (Dijkstra, A*,
    HMM, GNN, …). Filter methods (KF, EKF, …) ignore the graph.

    Args:
        observations_csv: Path to ``*_observations.csv`` (mixed GPS / circle / cell).
        true_path_csv: Path to ``*_true_path.csv`` (1 Hz ground truth).
        road_graph: Projected road network (e.g. OSM ``MultiDiGraph`` with ``x``, ``y``,
            ``crs`` as expected by ``graph_stitch`` / GNN).
        methods: Method names (e.g. ``["kf", "dijkstra", "gnn"]``). Unknown names raise.
        output_dir: If set, writes ``metrics.json`` and, when plotting, ``figures/``.
        plot: Write ENU overlay PNGs under ``output_dir/figures`` when ``output_dir`` is set.
        plot_map: Write Web Mercator map PNGs when ``output_dir`` is set.
        device: Torch device string for LSTM / Transformer / GNN.
        seed: RNG seed for stochastic estimators.

    Returns:
        Per-method dicts: metric scores and ``meta``, or ``{"error": "..."}`` on failure.
    """
    obs_df = load_observations_csv(observations_csv)
    true_df = load_true_path_csv(true_path_csv)
    names = [m.strip().lower() for m in methods]
    return _run_evaluation_core(
        obs_df,
        true_df,
        road_graph,
        names,
        output_dir=output_dir,
        plot=plot,
        plot_map=plot_map,
        device=device,
        seed=seed,
    )


def run_evaluation(
    observations_csv: Path,
    true_path_csv: Path,
    output_dir: Path,
    methods: Optional[List[str]] = None,
    *,
    plot: bool = True,
    plot_map: bool = False,
    device: Optional[str] = None,
    seed: int = 0,
) -> Dict[str, Dict]:
    """Run selected methods; write ``metrics.json`` and figures under ``output_dir``."""
    obs_df = load_observations_csv(observations_csv)
    true_df = load_true_path_csv(true_path_csv)
    G = get_projected_graph()
    if methods is None:
        methods = list(METHOD_REGISTRY.keys())
    names = [m.strip().lower() for m in methods]
    return _run_evaluation_core(
        obs_df,
        true_df,
        G,
        names,
        output_dir=output_dir,
        plot=plot,
        plot_map=plot_map,
        device=device,
        seed=seed,
    )


def print_summary(summary: Dict[str, Dict]) -> None:
    for k, v in summary.items():
        if "error" in v:
            print(f"{k}: ERROR — {v['error']}")
        else:
            rmse = v.get("rmse_m", float("nan"))
            print(f"{k}: RMSE={rmse:.3f} m  MAE={v.get('mae_m', float('nan')):.3f} m")
