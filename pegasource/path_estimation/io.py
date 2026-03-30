"""Load observation and ground-truth CSVs; derive per-event ENU points."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def load_observations_csv(path: Path) -> pd.DataFrame:
    """Load ``*_observations.csv`` sorted by time."""
    df = pd.read_csv(path)
    if "timestamp_s" not in df.columns:
        raise ValueError(f"Missing timestamp_s in {path}")
    return df.sort_values("timestamp_s").reset_index(drop=True)


def load_true_path_csv(path: Path) -> pd.DataFrame:
    """Load ``*_true_path.csv`` (1 Hz ground truth)."""
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

    Args:
        obs_df: Sorted or unsorted observations (must contain ``timestamp_s``).
        hz: Output sampling rate in Hz (default 1).

    Returns:
        DataFrame with ``timestamp_s``, ``true_x``, ``true_y``, and optional ``lon``/``lat`` (NaN).
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
    """Return a single (east, north) proxy for an observation row (meters)."""
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
    """Arrays of (timestamp, east, north) for each row."""
    ts = obs_df["timestamp_s"].to_numpy(dtype=float)
    xy = np.array([observation_enu_xy(obs_df.iloc[i]) for i in range(len(obs_df))])
    return ts, xy[:, 0], xy[:, 1]


def align_times_to_true(
    true_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``times_s``, ``true_xy`` shape (N, 2) from true path dataframe."""
    t = true_df["timestamp_s"].to_numpy(dtype=float)
    xy = np.column_stack(
        (true_df["true_x"].to_numpy(float), true_df["true_y"].to_numpy(float))
    )
    return t, xy
