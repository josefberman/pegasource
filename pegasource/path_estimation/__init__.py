"""Path estimation from mixed GPS / circle / cellular observations.

Heavy dependencies (filterpy, torch, …) load when you import
``evaluate_path_estimation`` / ``run_evaluation`` or ``pegasource.path_estimation.evaluate``.
Submodules such as :mod:`pegasource.path_estimation.metrics` work with the base install.
"""

from __future__ import annotations

from typing import Any

__all__ = ["estimate_paths_only", "evaluate_path_estimation", "run_evaluation"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import evaluate

        return getattr(evaluate, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
