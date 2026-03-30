"""Tensor dataset: observation sequence -> 1 Hz positions (supervised)."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..io import align_times_to_true, observation_enu_xy


def build_feature_matrix(obs_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Per-event features (fixed 12-D) and timestamps."""
    obs_df = obs_df.sort_values("timestamp_s").reset_index(drop=True)
    t = obs_df["timestamp_s"].to_numpy(float)
    t_max = float(np.max(t)) if len(t) else 1.0
    feats: list[list[float]] = []
    for _, row in obs_df.iterrows():
        tn = float(row["timestamp_s"]) / max(t_max, 1.0)
        src = row["source_type"]
        oh = [0.0, 0.0, 0.0]
        pad = [0.0] * 8
        if src == "gps":
            oh = [1.0, 0.0, 0.0]
            pad[0] = float(row["gps_x"])
            pad[1] = float(row["gps_y"])
        elif src == "circle":
            oh = [0.0, 1.0, 0.0]
            pad[0] = float(row["circle_x"])
            pad[1] = float(row["circle_y"])
            pad[2] = float(row["circle_r"])
        else:
            oh = [0.0, 0.0, 1.0]
            pad[0] = float(row["cell_tower_x"])
            pad[1] = float(row["cell_tower_y"])
            pad[2] = float(row["cell_r_min"])
            pad[3] = float(row["cell_r_max"])
            pad[4] = float(row["cell_theta_start"])
            pad[5] = float(row["cell_theta_end"])
        feats.append([tn] + oh + pad)
    return np.asarray(feats, dtype=np.float32), t


def interpolate_truth_to_events(
    t_ev: np.ndarray, true_df: pd.DataFrame
) -> np.ndarray:
    """True (east, north) linearly interpolated at event times."""
    tt, _ = align_times_to_true(true_df)
    ex = np.interp(t_ev, tt, true_df["true_x"].to_numpy(float))
    ny = np.interp(t_ev, tt, true_df["true_y"].to_numpy(float))
    return np.column_stack([ex, ny])


def obs_proxy_xy(obs_df: pd.DataFrame) -> np.ndarray:
    """Per-row observation proxy (same frame as true path)."""
    obs_df = obs_df.sort_values("timestamp_s").reset_index(drop=True)
    return np.array(
        [observation_enu_xy(obs_df.iloc[i]) for i in range(len(obs_df))],
        dtype=np.float32,
    )


def norm_scale_from_true(true_df: pd.DataFrame) -> Tuple[np.ndarray, float]:
    """Mean (2,) and scalar scale for normalizing coordinates."""
    tx = true_df["true_x"].to_numpy(float)
    ty = true_df["true_y"].to_numpy(float)
    mean = np.array([float(np.mean(tx)), float(np.mean(ty))], dtype=np.float32)
    span = max(float(np.ptp(tx)), float(np.ptp(ty)), 80.0)
    return mean, float(span)


def normalize_xy(xy: np.ndarray, mean: np.ndarray, scale: float) -> np.ndarray:
    return (xy - mean[None, :]) / max(scale, 1e-3)


def denormalize_xy(xy: np.ndarray, mean: np.ndarray, scale: float) -> np.ndarray:
    return xy * max(scale, 1e-3) + mean[None, :]


def normalize_feature_matrix(
    feats: np.ndarray, mean: np.ndarray, scale: float
) -> np.ndarray:
    """Normalize east/north slots in payload (cols 4–5 always x,y-ish)."""
    out = feats.copy()
    out[:, 4] = (out[:, 4] - mean[0]) / max(scale, 1e-3)
    out[:, 5] = (out[:, 5] - mean[1]) / max(scale, 1e-3)
    return out


class TrajectoryDataset(Dataset):
    """Observation sequence → residual targets (true − obs proxy), normalized."""

    def __init__(self, obs_df: pd.DataFrame, true_df: pd.DataFrame) -> None:
        self.feats, self.times = build_feature_matrix(obs_df)
        self.mean_xy, self.scale = norm_scale_from_true(true_df)
        self.feats = normalize_feature_matrix(self.feats, self.mean_xy, self.scale)
        truth_ev = interpolate_truth_to_events(self.times, true_df)
        proxy = obs_proxy_xy(obs_df)
        residual = truth_ev - proxy
        self.targets = normalize_xy(
            residual.astype(np.float32), np.array([0.0, 0.0], dtype=np.float32), self.scale
        )
        self.proxy_ev = proxy

    def __len__(self) -> int:
        return max(1, len(self.feats))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.feats).float()
        y = torch.from_numpy(self.targets.astype(np.float32))
        return x, y
