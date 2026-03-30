"""Extended Kalman filter (CV model) with GPS, circle radius, and cell radial cues."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..io import align_times_to_true
from ..types import EstimationResult


def estimate_ekf_fused(
    obs_df: pd.DataFrame,
    true_df: pd.DataFrame,
    G,
    rng: np.random.Generator,
    *,
    sigma_acc: float = 0.35,
    sigma_gps: float = 6.0,
    sigma_circle_r: float = 8.0,
    sigma_cell_r: float = 25.0,
) -> EstimationResult:
    """4D CV; GPS 2D; circle scalar distance to center; cell radial distance to band center."""
    times_s, _ = align_times_to_true(true_df)
    events = obs_df.sort_values("timestamp_s").reset_index(drop=True)
    if events.empty:
        raise ValueError("No observations.")

    t_ev = events["timestamp_s"].to_numpy(float)
    row0 = events.iloc[0]
    if row0["source_type"] == "gps":
        x = np.array(
            [float(row0["gps_x"]), float(row0["gps_y"]), 0.0, 0.0], dtype=float
        )
    elif row0["source_type"] == "circle":
        x = np.array(
            [float(row0["circle_x"]), float(row0["circle_y"]), 0.0, 0.0], dtype=float
        )
    else:
        tx, ty = float(row0["cell_tower_x"]), float(row0["cell_tower_y"])
        rm = 0.5 * (float(row0["cell_r_min"]) + float(row0["cell_r_max"]))
        th = 0.5 * (
            float(row0["cell_theta_start"]) + float(row0["cell_theta_end"])
        )
        x = np.array([tx + rm * math.cos(th), ty + rm * math.sin(th), 0.0, 0.0])

    P = np.eye(4) * 200.0

    def F(dt: float) -> np.ndarray:
        return np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )

    def Q(dt: float) -> np.ndarray:
        q = sigma_acc**2
        return np.diag([0.25 * q * dt**4] * 2 + [q * dt**2] * 2)

    H_gps = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    R_gps = np.eye(2) * (sigma_gps**2)

    traj_t: list[float] = []
    traj_xy: list[np.ndarray] = []
    traj_std: list[np.ndarray] = []
    t_prev = float(t_ev[0])
    traj_t.append(t_prev)
    traj_xy.append(x[:2].copy())
    traj_std.append(np.sqrt(np.maximum(np.diag(P[:2, :2]), 0.0)))

    for k in range(1, len(t_ev)):
        t = float(t_ev[k])
        dt = max(t - t_prev, 1e-3)
        Fm = F(dt)
        x = Fm @ x
        P = Fm @ P @ Fm.T + Q(dt)

        row = events.iloc[k]
        src = row["source_type"]
        if src == "gps":
            z = np.array([float(row["gps_x"]), float(row["gps_y"])], dtype=float)
            S = H_gps @ P @ H_gps.T + R_gps
            K = P @ H_gps.T @ np.linalg.inv(S)
            x = x + K @ (z - H_gps @ x)
            P = (np.eye(4) - K @ H_gps) @ P
        elif src == "circle":
            cx, cy = float(row["circle_x"]), float(row["circle_y"])
            r_obs = float(row["circle_r"])
            px, py = x[0], x[1]
            dx, dy = px - cx, py - cy
            d = math.hypot(dx, dy)
            if d < 1e-6:
                d = 1e-6
            z_pred = d
            H1 = np.array([[dx / d, dy / d, 0.0, 0.0]], dtype=float)
            R1 = np.array([[sigma_circle_r**2]])
            S = H1 @ P @ H1.T + R1
            K = P @ H1.T @ np.linalg.inv(S)
            innov = r_obs - z_pred
            x = x + (K.flatten() * innov)
            P = (np.eye(4) - K @ H1) @ P
        else:
            tx, ty = float(row["cell_tower_x"]), float(row["cell_tower_y"])
            rmid = 0.5 * (float(row["cell_r_min"]) + float(row["cell_r_max"]))
            px, py = x[0], x[1]
            dx, dy = px - tx, py - ty
            d = math.hypot(dx, dy)
            if d < 1e-6:
                d = 1e-6
            H1 = np.array([[dx / d, dy / d, 0.0, 0.0]], dtype=float)
            R1 = np.array([[sigma_cell_r**2]])
            S = H1 @ P @ H1.T + R1
            K = P @ H1.T @ np.linalg.inv(S)
            innov = rmid - d
            x = x + (K.flatten() * innov)
            P = (np.eye(4) - K @ H1) @ P

        std_pair = np.sqrt(np.maximum(np.diag(P[:2, :2]), 0.0))
        traj_t.append(t)
        traj_xy.append(x[:2].copy())
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
        meta={"method": "ekf_fused"},
    )
