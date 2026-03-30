"""Shared result type for path estimators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class EstimationResult:
    """Estimated trajectory on a 1 Hz (or aligned) time grid.

    Optional fields support uncertainty visualization for probabilistic methods.
    """

    times_s: np.ndarray
    east_m: np.ndarray
    north_m: np.ndarray
    std_east_m: Optional[np.ndarray] = None
    std_north_m: Optional[np.ndarray] = None
    cov_enu: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.times_s = np.asarray(self.times_s, dtype=float).reshape(-1)
        self.east_m = np.asarray(self.east_m, dtype=float).reshape(-1)
        self.north_m = np.asarray(self.north_m, dtype=float).reshape(-1)
        n = len(self.times_s)
        if len(self.east_m) != n or len(self.north_m) != n:
            raise ValueError("times_s, east_m, north_m must have the same length.")
        if self.std_east_m is not None:
            self.std_east_m = np.asarray(self.std_east_m, dtype=float).reshape(-1)
            if len(self.std_east_m) != n:
                raise ValueError("std_east_m length mismatch.")
        if self.std_north_m is not None:
            self.std_north_m = np.asarray(self.std_north_m, dtype=float).reshape(-1)
            if len(self.std_north_m) != n:
                raise ValueError("std_north_m length mismatch.")
        if self.cov_enu is not None:
            self.cov_enu = np.asarray(self.cov_enu, dtype=float)
            if self.cov_enu.shape != (n, 2, 2):
                raise ValueError("cov_enu must have shape (N, 2, 2).")
