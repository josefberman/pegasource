"""Reusable plotting helpers for synthetic path/observation visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Wedge

from .geo_reference import local_enu_meters_to_lon_lat

try:
    from pyproj import Transformer
except ImportError:  # pragma: no cover
    Transformer = None  # type: ignore

COLOR_GPS = "#79c300"
COLOR_UNCERTAINTY = "#61007d"
COLOR_CELL_SECTOR = "#cd001a"
TRUE_PATH_LW = 4.5


def _lon_lat_to_web_mercator(
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project WGS84 lon/lat (degrees) to Web Mercator (EPSG:3857) meters."""
    if Transformer is None:
        raise ImportError("plotting on maps requires the 'pyproj' package.")
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x, y = tr.transform(
        np.asarray(lon_deg, dtype=float),
        np.asarray(lat_deg, dtype=float),
    )
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def _extent_with_padding(
    xs: np.ndarray,
    ys: np.ndarray,
    pad_frac: float = 0.08,
) -> Tuple[float, float, float, float]:
    """Return ``(xmin, xmax, ymin, ymax)`` with relative padding."""
    if xs.size == 0 or ys.size == 0:
        return -1000.0, 1000.0, -1000.0, 1000.0
    xmin, xmax = float(np.min(xs)), float(np.max(xs))
    ymin, ymax = float(np.min(ys)), float(np.max(ys))
    dx = max(xmax - xmin, 1.0)
    dy = max(ymax - ymin, 1.0)
    return (
        xmin - pad_frac * dx,
        xmax + pad_frac * dx,
        ymin - pad_frac * dy,
        ymax + pad_frac * dy,
    )


def _sector_ring_enu(
    tx: float,
    ty: float,
    r_in: float,
    r_out: float,
    th0: float,
    th1: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Closed polygon for an annulus sector in local ENU (0 rad = +x, CCW)."""
    if r_out <= 0 or r_in < 0 or r_in >= r_out:
        return np.array([]), np.array([])
    if th1 < th0:
        th1 = th1 + 2.0 * np.pi
    n = max(8, int(48 * (th1 - th0) / (2 * np.pi)))
    n = min(n, 96)
    outer = np.linspace(th0, th1, n)
    inner = np.linspace(th1, th0, n)
    ox = tx + r_out * np.cos(outer)
    oy = ty + r_out * np.sin(outer)
    ix = tx + r_in * np.cos(inner)
    iy = ty + r_in * np.sin(inner)
    return np.concatenate([ox, ix]), np.concatenate([oy, iy])


def _enu_to_mercator_xy(east_m: np.ndarray, north_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lon, lat = local_enu_meters_to_lon_lat(
        np.asarray(east_m, dtype=float),
        np.asarray(north_m, dtype=float),
    )
    return _lon_lat_to_web_mercator(lon, lat)


def _wedge_degrees(theta_start_rad: float, theta_end_rad: float) -> Tuple[float, float]:
    t1 = float(np.degrees(theta_start_rad))
    t2 = float(np.degrees(theta_end_rad))
    if t2 < t1:
        t2 += 360.0
    return t1, t2


def _try_add_basemap(ax: plt.Axes, *, alpha: float = 0.5) -> bool:
    """Overlay OpenStreetMap tiles if ``contextily`` is available (semi-transparent)."""
    try:
        import contextily as ctx
    except ImportError:
        return False
    try:
        ctx.add_basemap(
            ax,
            crs="EPSG:3857",
            source=ctx.providers.OpenStreetMap.Mapnik,
            alpha=alpha,
        )
        return True
    except Exception:
        return False


def plot_true_path(df: pd.DataFrame, output_path: Path, show: bool = False) -> None:
    """Plot only the ground-truth trajectory in local meters.

    Args:
        df: DataFrame with ``true_x`` and ``true_y`` (e.g. dense 1 Hz path or samples).
        output_path: Destination image path.
        show: If True, display the figure interactively.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(
        df["true_x"],
        df["true_y"],
        color="black",
        linewidth=TRUE_PATH_LW,
        label="True path",
    )
    ax.set_title("Ground Truth Path")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis("equal")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    if show:
        plt.show()
    plt.close(fig)


def plot_observations_only(df: pd.DataFrame, output_path: Path, show: bool = False) -> None:
    """Plot observations in local meters: GPS points, uncertainty disks, cell annulus sectors.

    Args:
        df: Dataset DataFrame containing source-specific observation columns.
        output_path: Destination image path.
        show: If True, display the figure interactively.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    gps = df[df["source_type"] == "gps"]
    cir = df[df["source_type"] == "circle"]
    cel = df[df["source_type"] == "cell_sector"]

    for _, row in cir.iterrows():
        r = float(row["circle_r"])
        if r <= 0 or np.isnan(r):
            continue
        disk = Circle(
            (float(row["circle_x"]), float(row["circle_y"])),
            r,
            facecolor=COLOR_UNCERTAINTY,
            edgecolor="none",
            alpha=0.5,
            zorder=2,
        )
        ax.add_patch(disk)
    if not cir.empty:
        ax.scatter([], [], c=COLOR_UNCERTAINTY, alpha=0.5, s=40, label="Uncertainty radius")

    for _, row in cel.iterrows():
        tx, ty = float(row["cell_tower_x"]), float(row["cell_tower_y"])
        r_in, r_out = float(row["cell_r_min"]), float(row["cell_r_max"])
        th0, th1 = float(row["cell_theta_start"]), float(row["cell_theta_end"])
        td1, td2 = _wedge_degrees(th0, th1)
        w = r_out - r_in
        if w <= 0 or r_out <= 0:
            continue
        wedge = Wedge(
            (tx, ty),
            r_out,
            td1,
            td2,
            width=w,
            facecolor=COLOR_CELL_SECTOR,
            edgecolor="none",
            alpha=0.5,
            zorder=2,
        )
        ax.add_patch(wedge)
    if not cel.empty:
        ax.scatter([], [], c=COLOR_CELL_SECTOR, alpha=0.5, s=40, label="Cell sector")

    if not gps.empty:
        ax.scatter(
            gps["gps_x"],
            gps["gps_y"],
            s=14,
            alpha=1.0,
            c=COLOR_GPS,
            label="GPS",
            zorder=4,
        )

    ax.set_title("Observations Only (No True Path)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis("equal")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    if show:
        plt.show()
    plt.close(fig)


def plot_true_path_on_map(
    track_df: pd.DataFrame,
    output_path: Path,
    *,
    show_basemap: bool = True,
    show_path: bool = True,
    title: str = "Ground truth path (London, OSM)",
    show: bool = False,
) -> None:
    """Plot the dense true path on a Web Mercator map (optionally with OSM tiles).

    Args:
        track_df: DataFrame with columns ``lon``, ``lat`` (WGS84 degrees), e.g. from
            ``*_true_path_track.csv``.
        output_path: Destination image path.
        show_basemap: If True, try to draw OpenStreetMap tiles under the path.
        show_path: If True, draw the polyline.
        title: Figure title.
        show: If True, display the figure interactively.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    lon = track_df["lon"].to_numpy(dtype=float)
    lat = track_df["lat"].to_numpy(dtype=float)
    xm, ym = _lon_lat_to_web_mercator(lon, lat)
    if show_path:
        ax.plot(
            xm,
            ym,
            color="black",
            linewidth=TRUE_PATH_LW,
            label="True path",
            zorder=3,
        )
    xmin, xmax, ymin, ymax = _extent_with_padding(xm, ym)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    if show_basemap:
        if not _try_add_basemap(ax):
            ax.set_facecolor("#e8e4dc")
    ax.set_xlabel("Easting (Web Mercator, m)")
    ax.set_ylabel("Northing (Web Mercator, m)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def plot_observations_on_map(
    df: pd.DataFrame,
    output_path: Path,
    *,
    show_basemap: bool = True,
    show_gps: bool = True,
    show_circle: bool = True,
    show_cell_tower: bool = True,
    title: str = "Observations (London, OSM)",
    show: bool = False,
) -> None:
    """Plot observation layers on a map: GPS points, uncertainty disks, annulus sectors.

    Args:
        df: Event DataFrame with local-meter and WGS84 columns per ``source_type``.
        output_path: Destination image path.
        show_basemap: If True, try OpenStreetMap tiles.
        show_gps: Layer toggle for GPS fixes.
        show_circle: Layer toggle for uncertainty radius disks.
        show_cell_tower: Layer toggle for cell annulus-sector regions.
        title: Figure title.
        show: If True, display interactively.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []

    gps = df[df["source_type"] == "gps"] if show_gps else pd.DataFrame()
    cir = df[df["source_type"] == "circle"] if show_circle else pd.DataFrame()
    cel = df[df["source_type"] == "cell_sector"] if show_cell_tower else pd.DataFrame()

    for _, row in cir.iterrows():
        r = float(row["circle_r"])
        if r <= 0 or np.isnan(r):
            continue
        cx, cy = float(row["circle_x"]), float(row["circle_y"])
        ang = np.linspace(0.0, 2.0 * np.pi, 65)
        px = cx + r * np.cos(ang)
        py = cy + r * np.sin(ang)
        xm, ym = _enu_to_mercator_xy(px, py)
        ax.fill(
            xm,
            ym,
            facecolor=COLOR_UNCERTAINTY,
            edgecolor="none",
            alpha=0.5,
            zorder=4,
        )
        xs.append(xm)
        ys.append(ym)
    if not cir.empty:
        ax.scatter([], [], c=COLOR_UNCERTAINTY, alpha=0.5, s=40, label="Uncertainty radius")

    for _, row in cel.iterrows():
        tx, ty = float(row["cell_tower_x"]), float(row["cell_tower_y"])
        r_in, r_out = float(row["cell_r_min"]), float(row["cell_r_max"])
        th0, th1 = float(row["cell_theta_start"]), float(row["cell_theta_end"])
        px, py = _sector_ring_enu(tx, ty, r_in, r_out, th0, th1)
        if px.size == 0:
            continue
        xm, ym = _enu_to_mercator_xy(px, py)
        ax.fill(
            xm,
            ym,
            facecolor=COLOR_CELL_SECTOR,
            edgecolor="none",
            alpha=0.5,
            zorder=4,
        )
        xs.append(xm)
        ys.append(ym)
    if not cel.empty:
        ax.scatter([], [], c=COLOR_CELL_SECTOR, alpha=0.5, s=40, label="Cell sector")

    if not gps.empty:
        gx, gy = _lon_lat_to_web_mercator(
            gps["gps_lon"].to_numpy(dtype=float),
            gps["gps_lat"].to_numpy(dtype=float),
        )
        ax.scatter(
            gx,
            gy,
            s=16,
            alpha=1.0,
            c=COLOR_GPS,
            label="GPS",
            zorder=5,
        )
        xs.append(gx)
        ys.append(gy)

    if xs:
        all_x = np.concatenate(xs)
        all_y = np.concatenate(ys)
        xmin, xmax, ymin, ymax = _extent_with_padding(all_x, all_y)
    else:
        xmin, xmax, ymin, ymax = -1000.0, 1000.0, -1000.0, 1000.0
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    if show_basemap:
        if not _try_add_basemap(ax):
            ax.set_facecolor("#e8e4dc")
    ax.set_xlabel("Easting (Web Mercator, m)")
    ax.set_ylabel("Northing (Web Mercator, m)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def plot_map_with_layers(
    track_df: pd.DataFrame,
    events_df: pd.DataFrame,
    output_path: Path,
    *,
    show_basemap: bool = True,
    show_true_path: bool = True,
    show_gps: bool = True,
    show_circle: bool = True,
    show_cell_tower: bool = True,
    title: str = "Path and observations (London, OSM)",
    show: bool = False,
) -> None:
    """Combined map: optional OSM basemap, true path, and observation layers.

    Args:
        track_df: Dense path with ``lon``, ``lat`` (WGS84).
        events_df: Per-event rows with geo columns for observations.
        output_path: Destination image path.
        show_basemap: If True, try OpenStreetMap tiles.
        show_true_path: Layer toggle for the ground-truth polyline.
        show_gps: Layer toggle for GPS.
        show_circle: Layer toggle for uncertainty radius disks.
        show_cell_tower: Layer toggle for cell annulus-sector regions.
        title: Figure title.
        show: If True, display interactively.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []

    if show_true_path and not track_df.empty:
        lon = track_df["lon"].to_numpy(dtype=float)
        lat = track_df["lat"].to_numpy(dtype=float)
        xm, ym = _lon_lat_to_web_mercator(lon, lat)
        ax.plot(
            xm,
            ym,
            color="black",
            linewidth=TRUE_PATH_LW,
            label="True path",
            zorder=3,
        )
        xs.append(xm)
        ys.append(ym)

    gps = events_df[events_df["source_type"] == "gps"] if show_gps else pd.DataFrame()
    cir = events_df[events_df["source_type"] == "circle"] if show_circle else pd.DataFrame()
    cel = events_df[events_df["source_type"] == "cell_sector"] if show_cell_tower else pd.DataFrame()

    for _, row in cir.iterrows():
        r = float(row["circle_r"])
        if r <= 0 or np.isnan(r):
            continue
        cx, cy = float(row["circle_x"]), float(row["circle_y"])
        ang = np.linspace(0.0, 2.0 * np.pi, 65)
        px = cx + r * np.cos(ang)
        py = cy + r * np.sin(ang)
        xmg, ymg = _enu_to_mercator_xy(px, py)
        ax.fill(
            xmg,
            ymg,
            facecolor=COLOR_UNCERTAINTY,
            edgecolor="none",
            alpha=0.5,
            zorder=4,
        )
        xs.append(xmg)
        ys.append(ymg)
    if not cir.empty:
        ax.scatter([], [], c=COLOR_UNCERTAINTY, alpha=0.5, s=40, label="Uncertainty radius")

    for _, row in cel.iterrows():
        tx, ty = float(row["cell_tower_x"]), float(row["cell_tower_y"])
        r_in, r_out = float(row["cell_r_min"]), float(row["cell_r_max"])
        th0, th1 = float(row["cell_theta_start"]), float(row["cell_theta_end"])
        px, py = _sector_ring_enu(tx, ty, r_in, r_out, th0, th1)
        if px.size == 0:
            continue
        xmg, ymg = _enu_to_mercator_xy(px, py)
        ax.fill(
            xmg,
            ymg,
            facecolor=COLOR_CELL_SECTOR,
            edgecolor="none",
            alpha=0.5,
            zorder=4,
        )
        xs.append(xmg)
        ys.append(ymg)
    if not cel.empty:
        ax.scatter([], [], c=COLOR_CELL_SECTOR, alpha=0.5, s=40, label="Cell sector")

    if not gps.empty:
        gx, gy = _lon_lat_to_web_mercator(
            gps["gps_lon"].to_numpy(dtype=float),
            gps["gps_lat"].to_numpy(dtype=float),
        )
        ax.scatter(
            gx,
            gy,
            s=16,
            alpha=1.0,
            c=COLOR_GPS,
            label="GPS",
            zorder=5,
        )
        xs.append(gx)
        ys.append(gy)

    if xs:
        all_x = np.concatenate(xs)
        all_y = np.concatenate(ys)
        xmin, xmax, ymin, ymax = _extent_with_padding(all_x, all_y)
    else:
        xmin, xmax, ymin, ymax = -1000.0, 1000.0, -1000.0, 1000.0
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    if show_basemap:
        if not _try_add_basemap(ax):
            ax.set_facecolor("#e8e4dc")
    ax.set_xlabel("Easting (Web Mercator, m)")
    ax.set_ylabel("Northing (Web Mercator, m)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)
