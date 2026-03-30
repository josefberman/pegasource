"""Shared result type for path estimators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class EstimationResult:
    """Estimated trajectory on a common time axis (typically 1 Hz).

    Coordinates are **east** / **north** meters in the same local ENU frame as the
    synthetic datasets and CSV loaders.  Filters may populate optional standard
    deviations or per-step covariance for uncertainty bands.

    Attributes
    ----------
    times_s : numpy.ndarray
        Monotonic timestamps in seconds, shape ``(N,)``.
    east_m : numpy.ndarray
        Easting samples, shape ``(N,)``.
    north_m : numpy.ndarray
        Northing samples, shape ``(N,)``.
    std_east_m : numpy.ndarray, optional
        Marginal std dev east (same length as ``times_s``) if available.
    std_north_m : numpy.ndarray, optional
        Marginal std dev north.
    cov_enu : numpy.ndarray, optional
        Shape ``(N, 2, 2)`` per-step covariance in ENU if available.
    meta : dict
        Free-form metadata (e.g. ``{"method": "kf"}``).
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
