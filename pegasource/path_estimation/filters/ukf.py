"""Unscented Kalman filter (CV) with GPS + weak linear circle/cell position updates."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF

from ..io import align_times_to_true
from ..types import EstimationResult


def estimate_ukf_fused(
    obs_df: pd.DataFrame,
    true_df: pd.DataFrame,
    G,
    rng: np.random.Generator,
    *,
    sigma_acc: float = 0.25,
    sigma_gps: float = 6.0,
    sigma_circle_xy: float = 35.0,
    sigma_cell_xy: float = 120.0,
) -> EstimationResult:
    """UKF with same linear measurement model as fused KF; R varies by source."""
    times_s, _ = align_times_to_true(true_df)
    events = obs_df.sort_values("timestamp_s").reset_index(drop=True)
    if events.empty:
        raise ValueError("No observations.")

    def fx(x: np.ndarray, dt: float) -> np.ndarray:
        F = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )
        return F @ x

    def hx(x: np.ndarray) -> np.ndarray:
        return np.array([x[0], x[1]], dtype=float)

    def Qm(dt: float) -> np.ndarray:
        q = sigma_acc**2
        return np.diag([0.25 * q * dt**4] * 2 + [q * dt**2] * 2)

    points = MerweScaledSigmaPoints(n=4, alpha=0.25, beta=2.0, kappa=0.0)
    ukf = UKF(dim_x=4, dim_z=2, dt=1.0, fx=fx, hx=hx, points=points)
    ukf.Q = Qm(1.0)
    ukf.R = np.eye(2) * (sigma_gps**2)

    t_ev = events["timestamp_s"].to_numpy(float)
    row0 = events.iloc[0]
    if row0["source_type"] == "gps":
        ukf.x = np.array(
            [float(row0["gps_x"]), float(row0["gps_y"]), 0.0, 0.0], dtype=float
        )
    elif row0["source_type"] == "circle":
        ukf.x = np.array(
            [float(row0["circle_x"]), float(row0["circle_y"]), 0.0, 0.0], dtype=float
        )
    else:
        tx, ty = float(row0["cell_tower_x"]), float(row0["cell_tower_y"])
        rm = 0.5 * (float(row0["cell_r_min"]) + float(row0["cell_r_max"]))
        th = 0.5 * (
            float(row0["cell_theta_start"]) + float(row0["cell_theta_end"])
        )
        ukf.x = np.array(
            [tx + rm * math.cos(th), ty + rm * math.sin(th), 0.0, 0.0], dtype=float
        )
    ukf.P = np.eye(4) * 150.0

    traj_t: list[float] = []
    traj_xy: list[np.ndarray] = []
    traj_std: list[np.ndarray] = []

    t_prev = float(t_ev[0])
    traj_t.append(t_prev)
    traj_xy.append(ukf.x[:2].copy())
    traj_std.append(np.sqrt(np.maximum(np.diag(ukf.P[:2, :2]), 0.0)))

    for k in range(1, len(t_ev)):
        t = float(t_ev[k])
        dt = max(t - t_prev, 1e-3)
        ukf.Q = Qm(dt)
        ukf.predict(dt=dt)
        row = events.iloc[k]
        src = row["source_type"]
        if src == "gps":
            ukf.R = np.eye(2) * (sigma_gps**2)
            z = np.array([float(row["gps_x"]), float(row["gps_y"])], dtype=float)
        elif src == "circle":
            ukf.R = np.eye(2) * (sigma_circle_xy**2)
            z = np.array([float(row["circle_x"]), float(row["circle_y"])], dtype=float)
        else:
            ukf.R = np.eye(2) * (sigma_cell_xy**2)
            tx, ty = float(row["cell_tower_x"]), float(row["cell_tower_y"])
            rm = 0.5 * (float(row["cell_r_min"]) + float(row["cell_r_max"]))
            th = 0.5 * (
                float(row["cell_theta_start"]) + float(row["cell_theta_end"])
            )
            z = np.array([tx + rm * math.cos(th), ty + rm * math.sin(th)], dtype=float)
        ukf.update(z)

        std_pair = np.sqrt(np.maximum(np.diag(ukf.P[:2, :2]), 0.0))
        traj_t.append(t)
        traj_xy.append(ukf.x[:2].copy())
        traj_std.append(std_pair)
        t_prev = t

    t_keys = np.asarray(traj_t, dtype=float)
    xs = np.vstack(traj_xy)
    east = np.interp(times_s, t_keys, xs[:, 0])
    north = np.interp(times_s, t_keys, xs[:, 1])
    std_arr = np.vstack(traj_std)
    std_e = np.interp(times_s, t_keys, std_arr[:, 0])
    std_n = np.interp(times_s, t_keys, std_arr[:, 1])

    return EstimationResult(
        times_s=times_s,
        east_m=east,
        north_m=north,
        std_east_m=std_e,
        std_north_m=std_n,
        meta={"method": "ukf_fused_linear"},
    )
