"""Trajectory reconstruction from asynchronous observations (local ENU meters).

Re-exports :func:`run_evaluation`, :func:`evaluate_path_estimation`, and
:func:`estimate_paths_only` from :mod:`pegasource.path_estimation.evaluate`, and
per-method estimators from :mod:`pegasource.path_estimation.method_estimators`
(same pattern as :mod:`pegasource.pcap` / :mod:`pegasource.geo`).

Because Python loads this file when importing any submodule (e.g.
``import pegasource.path_estimation.metrics``), loading the package pulls in
``evaluate`` and ``method_estimators``.  Importing the package does **not** load
PyTorch or filterpy; those load only when you call an estimator that needs them
(e.g. :func:`estimate_lstm`) or run evaluation with those methods.

Quick start::

    from pegasource.path_estimation import run_evaluation
    from pathlib import Path

    run_evaluation(
        Path("obs.csv"),
        Path("true.csv"),
        Path("out"),
        methods=["kf", "dijkstra"],
    )

Examples
--------
>>> from pegasource.path_estimation.metrics import compute_all_metrics
>>> import numpy as np
>>> t = np.zeros((10, 2))
>>> out = compute_all_metrics(t, t + 0.1)
>>> assert "rmse_m" in out
"""

from .evaluate import (
    estimate_paths_only,
    evaluate_path_estimation,
    run_evaluation,
)
from .method_estimators import (
    METHOD_NAME_TO_FUNC,
    METHOD_NAME_TO_OBS_ONLY_FUNC,
    estimate_astar,
    estimate_astar_obs_only,
    estimate_dijkstra,
    estimate_dijkstra_obs_only,
    estimate_ekf,
    estimate_ekf_obs_only,
    estimate_gnn,
    estimate_gnn_obs_only,
    estimate_hmm,
    estimate_hmm_obs_only,
    estimate_kf,
    estimate_kf_obs_only,
    estimate_lstm,
    estimate_lstm_obs_only,
    estimate_particle,
    estimate_particle_obs_only,
    estimate_transformer,
    estimate_transformer_obs_only,
    estimate_ukf,
    estimate_ukf_obs_only,
)

__all__ = [
    "METHOD_NAME_TO_FUNC",
    "METHOD_NAME_TO_OBS_ONLY_FUNC",
    "estimate_astar",
    "estimate_astar_obs_only",
    "estimate_dijkstra",
    "estimate_dijkstra_obs_only",
    "estimate_ekf",
    "estimate_ekf_obs_only",
    "estimate_gnn",
    "estimate_gnn_obs_only",
    "estimate_hmm",
    "estimate_hmm_obs_only",
    "estimate_kf",
    "estimate_kf_obs_only",
    "estimate_lstm",
    "estimate_lstm_obs_only",
    "estimate_particle",
    "estimate_particle_obs_only",
    "estimate_paths_only",
    "estimate_transformer",
    "estimate_transformer_obs_only",
    "estimate_ukf",
    "estimate_ukf_obs_only",
    "evaluate_path_estimation",
    "run_evaluation",
]
