"""Trajectory reconstruction from asynchronous observations (local ENU meters).

Re-exports :func:`run_evaluation`, :func:`evaluate_path_estimation`, and
:func:`estimate_paths_only` from :mod:`pegasource.path_estimation.evaluate`, same pattern
as :mod:`pegasource.pcap` / :mod:`pegasource.geo`.  Requires the optional **path_estimation**
dependencies (filterpy, torch, …); install with ``pip install -e ".[path_estimation]"``.

Because Python loads this file when importing any submodule (e.g.
``import pegasource.path_estimation.metrics``), loading the package also pulls in
``evaluate`` and those dependencies.  Previously, lazy :func:`__getattr__` deferred that
until you accessed ``run_evaluation``; explicit imports match the rest of pegasource at
the cost of a heavier first import.

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

__all__ = [
    "estimate_paths_only",
    "evaluate_path_estimation",
    "run_evaluation",
]
