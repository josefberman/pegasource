"""High-level evaluation API: run estimators, compare to ground truth, save artifacts.

Individual estimators are exposed as functions in :mod:`pegasource.path_estimation.method_estimators`
(``estimate_hmm``, ``estimate_kf``, …) so each path loads only its own dependencies.
Observations-only entry points use ``estimate_*_obs_only`` (e.g. :func:`~pegasource.path_estimation.method_estimators.estimate_kf_obs_only`)
or :func:`estimate_paths_only` for multiple methods.

**Method registry** (``METHOD_REGISTRY``) mirrors :data:`~pegasource.path_estimation.method_estimators.METHOD_NAME_TO_FUNC`.

**Typical entry points**

- :func:`run_evaluation` — loads CSVs and the default projected OSM graph, writes
  ``metrics.json`` and figures.
- :func:`evaluate_path_estimation` — same pipeline but you pass the ``road_graph``.
- :func:`estimate_paths_only` — no true path file; cannot run supervised methods.

Output dicts map method name → either metric scores (see :func:`pegasource.path_estimation.metrics.compute_all_metrics`)
plus ``meta``, or ``{"error": "..."}`` if the estimator raised.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .io import (
    align_times_to_true,
    load_observations_csv,
    load_true_path_csv,
)
from .metrics import compute_all_metrics
from .method_estimators import METHOD_NAME_TO_FUNC, METHOD_NAME_TO_OBS_ONLY_FUNC
from .types import EstimationResult

EstimatorFn = Callable[..., EstimationResult]

# Supervised / needs real ``true_df`` (training labels or GNN node targets).
METHODS_REQUIRING_GROUND_TRUTH: frozenset[str] = frozenset({"lstm", "transformer", "gnn"})

METHOD_REGISTRY: Dict[str, EstimatorFn] = dict(METHOD_NAME_TO_FUNC)


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
    neural = frozenset({"lstm", "transformer", "gnn"})
    out: Dict[str, Any] = {}
    for name in methods:
        name = name.strip().lower()
        try:
            fn = METHOD_REGISTRY.get(name)
            if fn is None:
                raise KeyError(f"Unknown method: {name}")
            if name in neural:
                res = fn(obs_df, true_df, G, rng, device=device)
            else:
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
    from .viz import plot_estimation_enu, plot_estimation_map

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
    seed: int = 0,
) -> Dict[str, Any]:
    """Estimate paths from observations only (no ground-truth CSV).

    Builds a **stub** ground-truth frame with :func:`pegasource.path_estimation.io.stub_true_path_from_observations`
    so filter/graph code paths see a consistent time axis; **no real positions** exist,
    so metrics against ``true_x``/``true_y`` are meaningless.  Do **not** use for
    ``lstm``, ``transformer``, or ``gnn``.

    Parameters
    ----------
    observations_csv : pathlib.Path
        ``*_observations.csv`` with ``timestamp_s`` and source columns (see IO loaders).
    road_graph : networkx.MultiDiGraph or compatible
        Projected street graph for map-matching methods; ignored by pure filters.
    methods : list of str
        Lower-case names (e.g. ``["kf","dijkstra","hmm"]``).  Raises if any of
        ``lstm``, ``transformer``, ``gnn`` appear.
    output_hz : float, default 1.0
        Stub timeline sampling rate (Hz).
    output_dir : pathlib.Path, optional
        If set and ``plot`` is True, creates ``output_dir/figures/<method>_path_enu.png``.
    plot : bool, default False
        If True and ``output_dir`` is set, writes ENU overlays **without** a true path polyline.
    plot_map : bool, default False
        Must stay False (no real lon/lat); raises ``ValueError`` otherwise.
    seed : int, default 0
        RNG seed for stochastic methods.

    Returns
    -------
    dict
        Keys are method names. Values are :class:`~pegasource.path_estimation.types.EstimationResult`
        or ``{"error": "<message>"}``.

    Raises
    ------
    ValueError
        If ``plot_map`` is True, or if supervised method names are requested.
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
    from .viz import plot_estimation_enu

    obs_df = load_observations_csv(observations_csv)
    rng = np.random.default_rng(seed)
    raw: Dict[str, Any] = {}
    for name in names:
        try:
            fn = METHOD_NAME_TO_OBS_ONLY_FUNC.get(name)
            if fn is None:
                raise KeyError(f"Unknown method: {name}")
            raw[name] = fn(obs_df, road_graph, rng, output_hz=output_hz)
        except Exception as exc:
            raw[name] = {"error": str(exc)}
    fig_dir = (output_dir / "figures") if output_dir is not None else None
    if fig_dir is not None:
        fig_dir.mkdir(parents=True, exist_ok=True)
    if plot and fig_dir is not None:
        from .io import stub_true_path_from_observations

        true_df = stub_true_path_from_observations(obs_df, hz=output_hz)
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
    """Load CSVs and run estimators with a **caller-supplied** projected road graph.

    Graph-based methods (``dijkstra``, ``astar``, ``hmm``, ``gnn``) use ``road_graph``;
    filters (``kf``, ``ekf``, ``ukf``, ``particle``) ignore it.  Neural sequence models
    train against ``true_path_csv``.

    Parameters
    ----------
    observations_csv : pathlib.Path
        Event table: must include ``timestamp_s`` and columns per ``source_type``.
    true_path_csv : pathlib.Path
        1 Hz (or regular) ground truth: ``timestamp_s``, ``true_x``, ``true_y``, etc.
    road_graph : networkx.MultiDiGraph
        Projected OSM-style graph (``x``, ``y`` node attrs, ``crs`` on ``G.graph``).
    methods : list of str
        Subset of registered method names; unknown names cause errors inside :func:`_run_methods`.
    output_dir : pathlib.Path, optional
        If set, writes ``metrics.json`` and, when ``plot``/``plot_map``, PNGs under ``figures/``.
    plot : bool, default False
        Save ENU matplotlib overlays (estimate vs true + observations).
    plot_map : bool, default False
        Save Web-Mercator basemap figures (may download tiles; slower).
    device : str, optional
        Torch device for ``lstm``, ``transformer``, ``gnn``.
    seed : int, default 0
        Base seed for RNGs inside stochastic estimators.

    Returns
    -------
    dict[str, dict]
        Each key is a method name.  On success, value is a flat dict of metrics from
        :func:`pegasource.path_estimation.metrics.compute_all_metrics` plus a ``meta`` field;
        on failure, ``{"error": "<message>"}``.
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
    """Convenience wrapper: load CSVs, obtain default graph, evaluate, write ``output_dir``.

    Calls :func:`pegasource.path_estimation.graph_utils.get_projected_graph` for the same
    walkable OSM graph used by synthetic data generation (cached under the package).

    Parameters
    ----------
    observations_csv : pathlib.Path
        See :func:`evaluate_path_estimation`.
    true_path_csv : pathlib.Path
        See :func:`evaluate_path_estimation`.
    output_dir : pathlib.Path
        Created if needed; receives ``metrics.json`` and ``figures/`` when plotting.
    methods : list of str, optional
        If ``None``, uses ``list(METHOD_REGISTRY.keys())`` (dijkstra, astar, hmm, kf, ekf,
        ukf, particle, gnn).  Pass explicit names to add ``lstm`` or ``transformer``.
    plot : bool, default True
        Write ENU figures to ``output_dir/figures``.
    plot_map : bool, default False
        If True, also writes map tiles figures (requires network for contextily).
    device : str, optional
        Torch device for neural estimators.
    seed : int, default 0
        RNG seed forwarded to estimators.

    Returns
    -------
    dict[str, dict]
        Same structure as :func:`evaluate_path_estimation`.

    See Also
    --------
    evaluate_path_estimation : supply your own ``road_graph``.
    """
    from .graph_utils import get_projected_graph

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
    """Print one line per method: RMSE/MAE or an error string (CLI helper).

    Parameters
    ----------
    summary : dict[str, dict]
        Mapping produced by :func:`run_evaluation` / :func:`evaluate_path_estimation`
        (or compatible metric dicts).

    Returns
    -------
    None
        Writes to stdout only.
    """
    for k, v in summary.items():
        if "error" in v:
            print(f"{k}: ERROR — {v['error']}")
        else:
            rmse = v.get("rmse_m", float("nan"))
            print(f"{k}: RMSE={rmse:.3f} m  MAE={v.get('mae_m', float('nan')):.3f} m")
