"""Bootstrap particle filter with soft Gaussian-like observation likelihoods."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..io import align_times_to_true, observation_enu_xy
from ..types import EstimationResult


def estimate_particle_filter(
    obs_df: pd.DataFrame,
    true_df: pd.DataFrame,
    G,
    rng: np.random.Generator,
    *,
    n_particles: int = 1200,
    sigma_vel: float = 0.35,
    sigma_gps: float = 6.0,
    sigma_circle: float = 12.0,
    sigma_cell_rad: float = 45.0,
    sigma_cell_ang: float = 0.55,
) -> EstimationResult:
    """Random-walk velocity; soft Gaussian weights (no hard zero on cell sector)."""
    times_s, _ = align_times_to_true(true_df)
    events = obs_df.sort_values("timestamp_s").reset_index(drop=True)
    if events.empty:
        raise ValueError("No observations.")

    t_ev = events["timestamp_s"].to_numpy(float)
    P = max(50, n_particles)
    x = rng.normal(0.0, 40.0, size=(P, 2))
    row0 = events.iloc[0]
    ox, oy = observation_enu_xy(row0)
    x[:, 0] += ox
    x[:, 1] += oy
    v = rng.normal(0.0, 0.15, size=(P, 2))
    w = np.ones(P) / P

    traj_mu: list[np.ndarray] = []
    traj_std: list[np.ndarray] = []
    traj_t: list[float] = []

    def predict(dt: float) -> None:
        nonlocal x, v
        v += rng.normal(0.0, sigma_vel * math.sqrt(dt), size=v.shape)
        x += v * dt

    def weight_gps(px: np.ndarray, py: np.ndarray, zx: float, zy: float) -> np.ndarray:
        d2 = (px - zx) ** 2 + (py - zy) ** 2
        return np.exp(-0.5 * d2 / (sigma_gps**2))

    def weight_circle(
        px: np.ndarray, py: np.ndarray, cx: float, cy: float, r: float
    ) -> np.ndarray:
        d = np.hypot(px - cx, py - cy)
        return np.exp(-0.5 * (d - r) ** 2 / (sigma_circle**2))

    def weight_cell_soft(
        px: np.ndarray,
        py: np.ndarray,
        tx: float,
        ty: float,
        rmin: float,
        rmax: float,
        th0: float,
        th1: float,
    ) -> np.ndarray:
        dx = px - tx
        dy = py - ty
        dist = np.hypot(dx, dy)
        ang = np.arctan2(dy, dx)
        rmid = 0.5 * (rmin + rmax)
        dr = dist - rmid
        # Prefer distance inside [rmin, rmax]: soft hinge
        if rmax > rmin:
            half = 0.5 * (rmax - rmin)
            dr_edge = np.maximum(np.abs(dr) - half, 0.0)
        else:
            dr_edge = np.abs(dist - rmid)
        # Angular distance to nearest sector boundary (wrap-aware)
        twopi = 2.0 * math.pi
        thm = (th0 + th1) * 0.5
        dang = ang - thm
        dang = (dang + math.pi) % twopi - math.pi
        span = abs(th1 - th0)
        if span > math.pi:
            span = twopi - span
        dang = np.clip(np.abs(dang) - 0.5 * span, 0.0, math.pi)
        lr = np.exp(-0.5 * (dr_edge / sigma_cell_rad) ** 2)
        la = np.exp(-0.5 * (dang / sigma_cell_ang) ** 2)
        return np.clip(lr * la, 1e-8, 1.0)

    t_prev = float(t_ev[0])
    traj_t.append(t_prev)
    traj_mu.append(np.average(x, axis=0, weights=w))
    traj_std.append(np.sqrt(np.average((x - traj_mu[-1]) ** 2, axis=0, weights=w)))

    for k in range(1, len(t_ev)):
        t = float(t_ev[k])
        dt = max(t - t_prev, 1e-3)
        predict(dt)
        row = events.iloc[k]
        src = row["source_type"]
        if src == "gps":
            w *= weight_gps(
                x[:, 0], x[:, 1], float(row["gps_x"]), float(row["gps_y"])
            )
        elif src == "circle":
            w *= weight_circle(
                x[:, 0],
                x[:, 1],
                float(row["circle_x"]),
                float(row["circle_y"]),
                float(row["circle_r"]),
            )
        else:
            w *= weight_cell_soft(
                x[:, 0],
                x[:, 1],
                float(row["cell_tower_x"]),
                float(row["cell_tower_y"]),
                float(row["cell_r_min"]),
                float(row["cell_r_max"]),
                float(row["cell_theta_start"]),
                float(row["cell_theta_end"]),
            )

        w = np.maximum(w, 1e-300)
        s = float(np.sum(w))
        if s <= 0.0 or not np.isfinite(s):
            w = np.ones(P) / P
        else:
            w /= s
        ess = 1.0 / (np.sum(w**2) + 1e-300)
        if ess < P / 5.0:
            idx = rng.choice(P, size=P, p=w)
            x = x[idx]
            v = v[idx]
            w = np.ones(P) / P

        traj_t.append(t)
        mu = np.average(x, axis=0, weights=w)
        traj_mu.append(mu)
        traj_std.append(np.sqrt(np.average((x - mu) ** 2, axis=0, weights=w)))
        t_prev = t

    t_keys = np.array(traj_t, dtype=float)
    xs = np.array(traj_mu)
    east = np.interp(times_s, t_keys, xs[:, 0])
    north = np.interp(times_s, t_keys, xs[:, 1])
    std_arr = np.array(traj_std)
    std_e = np.interp(times_s, t_keys, std_arr[:, 0])
    std_n = np.interp(times_s, t_keys, std_arr[:, 1])

    return EstimationResult(
        times_s=times_s,
        east_m=east,
        north_m=north,
        std_east_m=std_e,
        std_north_m=std_n,
        meta={"method": "particle_filter", "n_particles": P},
    )
