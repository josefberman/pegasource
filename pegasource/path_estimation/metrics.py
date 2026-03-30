"""Trajectory comparison metrics (true vs estimated polylines in ENU meters).

All distances are in **meters** assuming ``true_xy`` and ``est_xy`` share the same
planar ENU frame and row-wise alignment (same length / time ordering).  Expensive
Fréchet and DTW are run on subsampled pairs when sequences are long — see
``max_points_frechet_dtw`` in :func:`compute_all_metrics`.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from scipy.spatial.distance import directed_hausdorff


def _subsample_pair(
    a: np.ndarray, b: np.ndarray, max_n: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Subsample both trajectories to at most ``max_n`` points (same indices)."""
    n = min(len(a), len(b))
    if n <= max_n:
        return a[:n], b[:n]
    idx = np.linspace(0, n - 1, max_n).astype(int)
    return a[idx], b[idx]


def rmse_euclidean(true_xy: np.ndarray, est_xy: np.ndarray) -> float:
    """Root-mean-square of per-point Euclidean distances between aligned vertices.

    Parameters
    ----------
    true_xy, est_xy : numpy.ndarray
        Shape ``(n, 2)``; row ``i`` is compared to row ``i``.

    Returns
    -------
    float
        RMSE in meters.
    """
    d = np.sqrt(np.sum((true_xy - est_xy) ** 2, axis=1))
    return float(np.sqrt(np.mean(d**2)))


def mae_euclidean(true_xy: np.ndarray, est_xy: np.ndarray) -> float:
    d = np.sqrt(np.sum((true_xy - est_xy) ** 2, axis=1))
    return float(np.mean(d))


def mae_axes(true_xy: np.ndarray, est_xy: np.ndarray) -> Tuple[float, float]:
    dx = np.abs(true_xy[:, 0] - est_xy[:, 0])
    dy = np.abs(true_xy[:, 1] - est_xy[:, 1])
    return float(np.mean(dx)), float(np.mean(dy))


def path_length(xy: np.ndarray) -> float:
    if len(xy) < 2:
        return 0.0
    return float(np.sum(np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))))


def hausdorff_max(true_xy: np.ndarray, est_xy: np.ndarray, max_n: int = 400) -> float:
    """Symmetric Hausdorff using ``directed_hausdorff`` (max of both directions)."""
    p, q = _subsample_pair(true_xy, est_xy, max_n)
    d1 = directed_hausdorff(p, q)[0]
    d2 = directed_hausdorff(q, p)[0]
    return float(max(d1, d2))


def discrete_frechet(P: np.ndarray, Q: np.ndarray) -> float:
    """Discrete Fréchet distance (dynamic programming). O(len(P)*len(Q))."""
    p, q = len(P), len(Q)
    ca = np.full((p, q), -1.0, dtype=float)

    def dist(i: int, j: int) -> float:
        return float(np.linalg.norm(P[i] - Q[j]))

    def c(i: int, j: int) -> float:
        if ca[i, j] > -0.5:
            return ca[i, j]
        if i == 0 and j == 0:
            ca[i, j] = dist(0, 0)
        elif i > 0 and j == 0:
            ca[i, j] = max(c(i - 1, 0), dist(i, 0))
        elif i == 0 and j > 0:
            ca[i, j] = max(c(0, j - 1), dist(0, j))
        elif i > 0 and j > 0:
            ca[i, j] = max(
                min(c(i - 1, j), c(i - 1, j - 1), c(i, j - 1)),
                dist(i, j),
            )
        else:
            ca[i, j] = float("inf")
        return ca[i, j]

    return c(p - 1, q - 1)


def dtw_distance(P: np.ndarray, Q: np.ndarray) -> float:
    """Dynamic time warping (Euclidean), O(len(P)*len(Q))."""
    p, q = len(P), len(Q)
    inf = float("inf")
    dtw = np.full((p + 1, q + 1), inf)
    dtw[0, 0] = 0.0
    for i in range(1, p + 1):
        for j in range(1, q + 1):
            cost = float(np.linalg.norm(P[i - 1] - Q[j - 1]))
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return float(dtw[p, q])


def endpoint_error(true_xy: np.ndarray, est_xy: np.ndarray) -> float:
    """Distance between start and end points (combined: start + end)."""
    e0 = float(np.linalg.norm(true_xy[0] - est_xy[0]))
    e1 = float(np.linalg.norm(true_xy[-1] - est_xy[-1]))
    return e0 + e1


def compute_all_metrics(
    true_xy: np.ndarray,
    est_xy: np.ndarray,
    *,
    max_points_frechet_dtw: int = 200,
) -> Dict[str, Any]:
    """Compute a standard bundle of trajectory error metrics.

    Parameters
    ----------
    true_xy : numpy.ndarray
        Ground-truth polyline, shape ``(n, 2)`` (east, north).
    est_xy : numpy.ndarray
        Estimated polyline, same shape as ``true_xy`` (row-aligned samples).
    max_points_frechet_dtw : int, default 200
        Both Fréchet and DTW are evaluated on uniform subsamples of at most this many
        points per curve (for speed on long tracks). Hausdorff uses its own cap (400).

    Returns
    -------
    dict
        Keys include:

        ``rmse_m``, ``mae_m``, ``mae_east_m``, ``mae_north_m``,
        ``hausdorff_m``, ``path_length_true_m``, ``path_length_est_m``,
        ``length_ratio``, ``endpoint_error_m``,
        ``discrete_frechet_m``, ``dtw_m``.

        Values are floats in meters or dimensionless ratio for ``length_ratio``.

    Notes
    -----
    Assumes both paths use the same sampling grid or comparable point counts; this
    function does **not** perform temporal resampling.
    """
    out: Dict[str, Any] = {}
    out["rmse_m"] = rmse_euclidean(true_xy, est_xy)
    out["mae_m"] = mae_euclidean(true_xy, est_xy)
    mx, my = mae_axes(true_xy, est_xy)
    out["mae_east_m"] = mx
    out["mae_north_m"] = my
    out["hausdorff_m"] = hausdorff_max(true_xy, est_xy)
    out["path_length_true_m"] = path_length(true_xy)
    out["path_length_est_m"] = path_length(est_xy)
    den = max(out["path_length_true_m"], 1e-6)
    out["length_ratio"] = float(out["path_length_est_m"] / den)
    out["endpoint_error_m"] = endpoint_error(true_xy, est_xy)

    pf, qf = _subsample_pair(true_xy, est_xy, max_points_frechet_dtw)
    out["discrete_frechet_m"] = discrete_frechet(pf, qf)
    pd, qd = _subsample_pair(true_xy, est_xy, max_points_frechet_dtw)
    out["dtw_m"] = dtw_distance(pd, qd)
    return out
