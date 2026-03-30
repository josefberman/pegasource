"""CSV I/O and observation geometry for the synthetic / evaluation pipelines.

Observation files are sorted by ``timestamp_s``.  Ground-truth files provide
``true_x`` / ``true_y`` in local ENU meters.  Helper functions map each observation
row to a proxy ``(east, north)`` for graph snapping and filtering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def load_observations_csv(path: Path) -> pd.DataFrame:
    """Load an observations CSV and sort by ``timestamp_s``.

    Parameters
    ----------
    path : pathlib.Path
        Must contain a ``timestamp_s`` column (seconds).

    Returns
    -------
    pandas.DataFrame
        Sorted ascending by time; index reset.

    Raises
    ------
    ValueError
        If ``timestamp_s`` is missing.
    """
    df = pd.read_csv(path)
    if "timestamp_s" not in df.columns:
        raise ValueError(f"Missing timestamp_s in {path}")
    return df.sort_values("timestamp_s").reset_index(drop=True)


def load_true_path_csv(path: Path) -> pd.DataFrame:
    """Load a ground-truth path CSV (regular samples in ENU meters).

    Parameters
    ----------
    path : pathlib.Path
        Must include ``timestamp_s``, ``true_x``, and ``true_y``.

    Returns
    -------
    pandas.DataFrame
        Sorted by ``timestamp_s``.

    Raises
    ------
    ValueError
        If required columns are absent.
    """
    df = pd.read_csv(path)
    for col in ("timestamp_s", "true_x", "true_y"):
        if col not in df.columns:
            raise ValueError(f"Missing {col} in {path}")
    return df.sort_values("timestamp_s").reset_index(drop=True)


def stub_true_path_from_observations(obs_df: pd.DataFrame, *, hz: float = 1.0) -> pd.DataFrame:
    """Build a minimal ``true_path``-shaped frame for **output time grid only** (no real ground truth).

    Timestamps span ``[min(observation time), max(observation time)]`` at ``hz`` Hz.
    ``true_x`` / ``true_y`` are zero placeholders; most estimators only use the time axis.
    Do **not** use this for metrics or for supervised methods (LSTM, Transformer, GNN).

    Parameters
    ----------
    obs_df : pandas.DataFrame
        Sorted or unsorted observations; must contain ``timestamp_s``.
    hz : float, default 1.0
        Output sampling rate in Hz.

    Returns
    -------
    pandas.DataFrame
        Columns include ``timestamp_s``, ``true_x``, ``true_y``, NaN ``lon``/``lat`` placeholders,
        and ``dataset_id`` set to ``\"_stub_no_truth\"``.
    """
    obs_df = obs_df.sort_values("timestamp_s").reset_index(drop=True)
    t = obs_df["timestamp_s"].to_numpy(dtype=float)
    if len(t) == 0:
        raise ValueError("No observations.")
    t0, t1 = float(np.min(t)), float(np.max(t))
    step = 1.0 / float(hz)
    times = np.arange(t0, t1 + 1e-9, step)
    n = len(times)
    return pd.DataFrame(
        {
            "timestamp_s": times,
            "true_x": np.zeros(n, dtype=float),
            "true_y": np.zeros(n, dtype=float),
            "lon": np.full(n, np.nan),
            "lat": np.full(n, np.nan),
            "reference_origin_lat": np.nan,
            "reference_origin_lon": np.nan,
            "dataset_id": "_stub_no_truth",
        }
    )


def observation_enu_xy(row: pd.Series) -> Tuple[float, float]:
    """Map one observation row to a representative ``(east_m, north_m)`` point.

    Parameters
    ----------
    row : pandas.Series
        Must contain ``source_type`` and columns for that source:

        - ``"gps"`` → ``gps_x``, ``gps_y``
        - ``"circle"`` → ``circle_x``, ``circle_y``
        - ``"cell_sector"`` → tower position plus annulus/sector mid-angle

    Returns
    -------
    tuple[float, float]
        Easting and northing in the same ENU frame as ``true_x`` / ``true_y``.

    Raises
    ------
    ValueError
        If ``source_type`` is unknown.
    """
    src = row["source_type"]
    if src == "gps":
        return float(row["gps_x"]), float(row["gps_y"])
    if src == "circle":
        return float(row["circle_x"]), float(row["circle_y"])
    if src == "cell_sector":
        tx, ty = float(row["cell_tower_x"]), float(row["cell_tower_y"])
        rmid = 0.5 * (float(row["cell_r_min"]) + float(row["cell_r_max"]))
        th = 0.5 * (float(row["cell_theta_start"]) + float(row["cell_theta_end"]))
        return tx + rmid * np.cos(th), ty + rmid * np.sin(th)
    raise ValueError(f"Unknown source_type: {src}")


def build_event_points(obs_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack timestamps and ENU coordinates for all observation rows.

    Parameters
    ----------
    obs_df : pandas.DataFrame
        Sorted observations (see :func:`load_observations_csv`).

    Returns
    -------
    timestamp_s : numpy.ndarray
        Shape ``(n,)``.
    east_m : numpy.ndarray
        Shape ``(n,)``.
    north_m : numpy.ndarray
        Shape ``(n,)``.
    """
    ts = obs_df["timestamp_s"].to_numpy(dtype=float)
    xy = np.array([observation_enu_xy(obs_df.iloc[i]) for i in range(len(obs_df))])
    return ts, xy[:, 0], xy[:, 1]


def align_times_to_true(
    true_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract time axis and stacked true positions from a ground-truth frame.

    Parameters
    ----------
    true_df : pandas.DataFrame
        Must contain ``timestamp_s``, ``true_x``, ``true_y``.

    Returns
    -------
    times_s : numpy.ndarray
        Shape ``(N,)``.
    true_xy : numpy.ndarray
        Shape ``(N, 2)`` with columns east, north.
    """
    t = true_df["timestamp_s"].to_numpy(dtype=float)
    xy = np.column_stack(
        (true_df["true_x"].to_numpy(float), true_df["true_y"].to_numpy(float))
    )
    return t, xy
