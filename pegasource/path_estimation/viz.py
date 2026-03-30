"""Plot estimated vs true paths and optional uncertainty bands."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Wedge

from .types import EstimationResult
from .plotting_utils import (
    COLOR_CELL_SECTOR,
    COLOR_GPS,
    COLOR_UNCERTAINTY,
    _enu_to_mercator_xy,
    _extent_with_padding,
    _try_add_basemap,
    _wedge_degrees,
)


def plot_estimation_enu(
    true_df: pd.DataFrame,
    result: EstimationResult,
    obs_df: Optional[pd.DataFrame],
    output_path: Path,
    *,
    title: str,
    show_observations: bool = True,
    show_true_path: bool = True,
) -> None:
    """Local ENU plot: true path, estimate, optional observations (no σ-ellipses)."""
    fig, ax = plt.subplots(figsize=(10, 7))
    if show_true_path:
        tx = true_df["true_x"].to_numpy(float)
        ty = true_df["true_y"].to_numpy(float)
        ax.plot(tx, ty, color="#333333", linewidth=3.5, label="True path", zorder=3)
    ex = result.east_m
    ny = result.north_m
    ax.plot(ex, ny, color="#0066cc", linewidth=2.2, label="Estimated", zorder=4)

    if show_observations and obs_df is not None:
        gps = obs_df[obs_df["source_type"] == "gps"]
        if not gps.empty:
            ax.scatter(
                gps["gps_x"],
                gps["gps_y"],
                s=10,
                c=COLOR_GPS,
                label="GPS",
                zorder=5,
            )
        cir = obs_df[obs_df["source_type"] == "circle"]
        for _, row in cir.iterrows():
            r = float(row["circle_r"])
            if r > 0 and not np.isnan(r):
                ax.add_patch(
                    Circle(
                        (float(row["circle_x"]), float(row["circle_y"])),
                        r,
                        facecolor=COLOR_UNCERTAINTY,
                        edgecolor="none",
                        alpha=0.25,
                        zorder=1,
                    )
                )
        cel = obs_df[obs_df["source_type"] == "cell_sector"]
        for _, row in cel.iterrows():
            tx_, ty_ = float(row["cell_tower_x"]), float(row["cell_tower_y"])
            r_in, r_out = float(row["cell_r_min"]), float(row["cell_r_max"])
            th0, th1 = float(row["cell_theta_start"]), float(row["cell_theta_end"])
            td1, td2 = _wedge_degrees(th0, th1)
            w = r_out - r_in
            if w > 0 and r_out > 0:
                ax.add_patch(
                    Wedge(
                        (tx_, ty_),
                        r_out,
                        td1,
                        td2,
                        width=w,
                        facecolor=COLOR_CELL_SECTOR,
                        edgecolor="none",
                        alpha=0.25,
                        zorder=1,
                    )
                )

    ax.set_title(title)
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.axis("equal")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_estimation_map(
    true_df: pd.DataFrame,
    result: EstimationResult,
    output_path: Path,
    *,
    title: str,
) -> None:
    """Web Mercator map with basemap when available."""
    fig, ax = plt.subplots(figsize=(10, 8))
    lon_t, lat_t = (
        true_df["lon"].to_numpy(float),
        true_df["lat"].to_numpy(float),
    )
    mx_t, my_t = _enu_to_mercator_xy(
        true_df["true_x"].to_numpy(float), true_df["true_y"].to_numpy(float)
    )
    mx_e, my_e = _enu_to_mercator_xy(result.east_m, result.north_m)
    ax.plot(mx_t, my_t, color="#333333", linewidth=3.0, label="True", zorder=3)
    ax.plot(mx_e, my_e, color="#0066cc", linewidth=2.0, label="Estimated", zorder=4)
    allx = np.concatenate([mx_t, mx_e])
    ally = np.concatenate([my_t, my_e])
    xmin, xmax, ymin, ymax = _extent_with_padding(allx, ally)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    _try_add_basemap(ax, alpha=0.5)
    ax.set_xlabel("Easting (Web Mercator, m)")
    ax.set_ylabel("Northing (Web Mercator, m)")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
