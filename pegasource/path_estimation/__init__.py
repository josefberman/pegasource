"""Trajectory reconstruction from asynchronous observations (local ENU meters).

**Lazy exports** (importing these loads :mod:`pegasource.path_estimation.evaluate` and
pulls in **filterpy**, **torch**, etc. — use ``pip install -e ".[path_estimation]"``):

- `estimate_paths_only` — no ground-truth CSV; stub timeline only
- `evaluate_path_estimation` — CSV paths + caller-supplied road graph
- `run_evaluation` — same as ``evaluate_path_estimation`` but loads the default OSM walk graph

**Always-light submodules** (safe without the extra):

- :mod:`pegasource.path_estimation.metrics` — RMSE, Fréchet, DTW, …
- :mod:`pegasource.path_estimation.io` — CSV loaders (used by evaluation pipelines)

Examples
--------
>>> from pegasource.path_estimation.metrics import compute_all_metrics
>>> import numpy as np
>>> t = np.zeros((10, 2))
>>> out = compute_all_metrics(t, t + 0.1)
>>> assert "rmse_m" in out
"""

from __future__ import annotations

from typing import Any

__all__ = ["estimate_paths_only", "evaluate_path_estimation", "run_evaluation"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import evaluate

        return getattr(evaluate, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
