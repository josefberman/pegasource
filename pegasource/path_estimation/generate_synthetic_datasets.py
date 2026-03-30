#!/usr/bin/env python3
"""Generate a single synthetic trajectory dataset with asynchronous mixed-source observations.

Local motion uses east/north meters anchored at a London reference; CSV columns
include WGS84 ``*_lon`` / ``*_lat`` for mapping. Ground-truth routes follow
OpenStreetMap pedestrian edges in central London when OSMnx is available; cell
towers remain synthetic in the local meter frame (not real UK sites).
"""

import argparse
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
from .geo_reference import (
    LONDON_REFERENCE_LAT_DEG,
    LONDON_REFERENCE_LON_DEG,
    enu_scalar_to_lon_lat,
    local_enu_meters_to_lon_lat,
)
from .plotting_utils import (
    plot_map_with_layers,
    plot_observations_only,
    plot_observations_on_map,
    plot_true_path,
    plot_true_path_on_map,
)


@dataclass(frozen=True)
class SourceCadence:
    """Sampling behavior for one observation source.

    Use ``mode="bursts"`` for GPS and circular-uncertainty (~1 Hz bursts in each burst
    window, long gaps between bursts; ``burst_count`` is set from simulation duration).
    Use ``mode="uniform_intervals"`` for cellular (inter-arrival uniform in [min, max]).

    Attributes:
        mode: ``bursts`` or ``uniform_intervals``.
        burst_count: Number of active bursts (bursts mode only).
        burst_duration_s: Length of each burst in seconds (bursts mode only).
        burst_rate_hz: Nominal sampling rate inside a burst (bursts mode only).
        burst_jitter_ratio: Relative jitter on intra-burst step timing (bursts mode only).
        burst_dropout_probability: Probability to skip one intra-burst sample (bursts mode only).
        interval_min_s: Minimum inter-arrival seconds (uniform_intervals mode only).
        interval_max_s: Maximum inter-arrival seconds (uniform_intervals mode only).
        interval_jitter_ratio: Relative jitter on each inter-arrival (uniform_intervals mode only).
    """

    mode: Literal["bursts", "uniform_intervals"]
    burst_count: int = 0
    burst_duration_s: float = 0.0
    burst_rate_hz: float = 1.0
    burst_jitter_ratio: float = 0.12
    burst_dropout_probability: float = 0.08
    interval_min_s: float = 60.0
    interval_max_s: float = 300.0
    interval_jitter_ratio: float = 0.05


def normalize_angle_rad(angle: float) -> float:
    """Normalize an angle to the [0, 2pi) range.

    Args:
        angle: Angle in radians.

    Returns:
        Normalized angle in radians within [0, 2pi).
    """
    return (angle + 2.0 * math.pi) % (2.0 * math.pi)


def angle_in_sector(angle: float, start: float, end: float) -> bool:
    """Check whether an angle lies inside a potentially wrapped sector.

    Args:
        angle: Test angle in radians.
        start: Sector start angle in radians.
        end: Sector end angle in radians.

    Returns:
        True if `angle` is inside the sector, including wraparound sectors.
    """
    a = normalize_angle_rad(angle)
    s = normalize_angle_rad(start)
    e = normalize_angle_rad(end)
    if s <= e:
        return s <= a <= e
    return a >= s or a <= e


# Walking speed (m/s) for street-like polylines.
_SIMPLE_WALK_SPEED_MPS = 1.35
_COMPLEX_WALK_SPEED_MPS = 1.25

# One "city block" cycle: segment lengths (m) and turn after each segment (rad).
# Mostly right-angle corners; resembles walking a few blocks.
_SIMPLE_BLOCK_L = np.array(
    [92.0, 55.0, 78.0, 48.0, 65.0, 58.0, 84.0, 44.0, 71.0, 52.0, 88.0, 46.0], dtype=float
)
_SIMPLE_BLOCK_TURN = np.radians(
    np.array([87.0, -93.0, 90.0, -88.0, 91.0, -90.0, 89.0, -91.0, 88.0, -89.0, 90.0], dtype=float)
)

# Tighter, more irregular corners (alleys, diagonal crossings, short blocks).
_COMPLEX_BLOCK_L = np.array(
    [
        32.0,
        24.0,
        38.0,
        19.0,
        41.0,
        27.0,
        33.0,
        22.0,
        36.0,
        21.0,
        39.0,
        26.0,
        31.0,
        23.0,
        35.0,
        28.0,
        30.0,
        25.0,
        34.0,
        29.0,
        37.0,
        20.0,
        40.0,
        24.0,
        32.0,
        26.0,
        33.0,
        22.0,
        38.0,
        27.0,
    ],
    dtype=float,
)
_COMPLEX_BLOCK_TURN = np.radians(
    np.array(
        [
            90.0,
            -90.0,
            45.0,
            -135.0,
            90.0,
            -90.0,
            90.0,
            -45.0,
            135.0,
            -90.0,
            90.0,
            -90.0,
            90.0,
            -90.0,
            45.0,
            -90.0,
            90.0,
            -135.0,
            45.0,
            90.0,
            -90.0,
            90.0,
            -90.0,
            90.0,
            -90.0,
            45.0,
            -135.0,
            90.0,
            -90.0,
        ],
        dtype=float,
    )
)


def _trim_segments_to_arc_length(
    segment_lengths: np.ndarray,
    turn_after: np.ndarray,
    arc_length_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Repeat a block pattern until cumulative length covers ``arc_length_max``."""
    if segment_lengths.size == 0:
        raise ValueError("segment_lengths must be non-empty.")
    if turn_after.size != segment_lengths.size - 1:
        raise ValueError("turn_after must have length len(segment_lengths) - 1.")

    cycle = float(np.sum(segment_lengths))
    n_rep = max(2, int(np.ceil(arc_length_max / cycle)) + 2)
    L_full = np.tile(segment_lengths, n_rep)
    turn_full = np.tile(turn_after, n_rep)
    cumsum = np.cumsum(L_full)
    m = int(np.searchsorted(cumsum, arc_length_max, side="right"))
    m = max(1, min(m, len(L_full)))
    L = L_full[:m]
    turns = turn_full[: m - 1]
    return L, turns


def _polyline_xy_at_arclength(
    s: np.ndarray,
    segment_lengths: np.ndarray,
    turn_after: np.ndarray,
) -> np.ndarray:
    """Positions along a 2D polyline at arc lengths ``s`` (meters).

    Args:
        s: Arc length along path from origin, shape (N,).
        segment_lengths: Length of each straight segment (m).
        turn_after: Yaw added after each segment, length len(segment_lengths)-1.

    Returns:
        Array of shape (N, 2) with x/y in meters.
    """
    L = segment_lengths.astype(float)
    nseg = len(L)
    headings = np.zeros(nseg, dtype=float)
    for i in range(1, nseg):
        headings[i] = headings[i - 1] + turn_after[i - 1]

    cumlen = np.concatenate([[0.0], np.cumsum(L)])
    starts = np.zeros((nseg + 1, 2), dtype=float)
    for i in range(nseg):
        c, si = math.cos(headings[i]), math.sin(headings[i])
        starts[i + 1, 0] = starts[i, 0] + L[i] * c
        starts[i + 1, 1] = starts[i, 1] + L[i] * si

    idx = np.searchsorted(cumlen, s, side="right") - 1
    idx = np.clip(idx, 0, nseg - 1)
    offset = s - cumlen[idx]
    ux = np.cos(headings[idx])
    uy = np.sin(headings[idx])
    x = starts[idx, 0] + offset * ux
    y = starts[idx, 1] + offset * uy
    return np.column_stack((x, y))


def _rotate_xy(xy: np.ndarray, theta_rad: float) -> np.ndarray:
    """Rotate points by ``theta_rad`` around origin."""
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    r = np.array([[c, -s], [s, c]], dtype=float)
    return (r @ xy.T).T


def _street_like_path(
    t: np.ndarray,
    speed_mps: float,
    segment_lengths: np.ndarray,
    turn_after: np.ndarray,
    global_rotation_rad: float,
) -> np.ndarray:
    """Ground-truth path as a repeating street grid / irregular-walk polyline."""
    s_max = float(np.max(speed_mps * t))
    L, turns = _trim_segments_to_arc_length(segment_lengths, turn_after, s_max)
    s = speed_mps * t
    xy = _polyline_xy_at_arclength(s, L, turns)
    return _rotate_xy(xy, global_rotation_rad)


def simple_path(t: np.ndarray) -> np.ndarray:
    """Build a ground-truth trajectory with long blocks and mostly 90° street turns.

    Args:
        t: Time vector in seconds.

    Returns:
        Array of shape (N, 2) containing x/y positions.
    """
    return _street_like_path(
        t,
        _SIMPLE_WALK_SPEED_MPS,
        _SIMPLE_BLOCK_L,
        _SIMPLE_BLOCK_TURN,
        global_rotation_rad=0.12,
    )


def complex_path(t: np.ndarray) -> np.ndarray:
    """Build a ground-truth trajectory with frequent corners (alleys, short blocks, mixed angles).

    Args:
        t: Time vector in seconds.

    Returns:
        Array of shape (N, 2) containing x/y positions.
    """
    return _street_like_path(
        t,
        _COMPLEX_WALK_SPEED_MPS,
        _COMPLEX_BLOCK_L,
        _COMPLEX_BLOCK_TURN,
        global_rotation_rad=-0.18,
    )


def build_reference_timeline(duration_s: int, reference_hz: float) -> np.ndarray:
    """Create a high-resolution timeline for ground-truth interpolation.

    Args:
        duration_s: Total simulation duration in seconds.
        reference_hz: Reference sampling frequency in Hz.

    Returns:
        Monotonic array of timestamps from 0 to duration (inclusive).
    """
    step = 1.0 / reference_hz
    return np.arange(0.0, duration_s + 1e-9, step)


# OSM route scoring: prefer more corner-rich walks when using ``london_street_path``.
DEFAULT_PATH_KIND = "complex"

# GPS / circle: scale burst count with duration (same density as the old fixed 3 bursts per 3600 s).
_BURST_COUNT_PER_HOUR = 3.0


def make_cadences(duration_s: int) -> Dict[str, SourceCadence]:
    """Return per-source cadence (bursty GPS/circle, uniform-interval cellular).

    GPS and circle: independent ~8 s bursts at ~1 Hz; the number of bursts scales with
    ``duration_s`` (default density: three bursts per hour of simulation). Cellular: uniform 60–300 s
    inter-arrival across the full duration (already scales with length).

    Args:
        duration_s: Simulation horizon in seconds (determines burst_count for GPS/circle).

    Returns:
        Mapping from source type to cadence configuration.
    """
    burst_count = max(1, int(round(float(duration_s) * _BURST_COUNT_PER_HOUR / 3600.0)))
    burst_cfg = SourceCadence(
        mode="bursts",
        burst_count=burst_count,
        burst_duration_s=8.0,
        burst_rate_hz=1.0,
        burst_jitter_ratio=0.12,
        burst_dropout_probability=0.08,
    )
    cell_cfg = SourceCadence(
        mode="uniform_intervals",
        interval_min_s=60.0,
        interval_max_s=300.0,
        interval_jitter_ratio=0.05,
    )
    return {
        "gps": burst_cfg,
        "circle": burst_cfg,
        "cell_sector": cell_cfg,
    }


def sample_bursty_times(
    rng: np.random.Generator,
    duration_s: int,
    burst_count: int,
    burst_duration_s: float,
    burst_rate_hz: float,
    jitter_ratio: float,
    dropout_probability: float,
) -> np.ndarray:
    """Sample timestamps in a few non-overlapping bursts at roughly ``burst_rate_hz``.

    Args:
        rng: Random number generator.
        duration_s: Total simulation duration in seconds.
        burst_count: Number of bursts.
        burst_duration_s: Length of each burst in seconds.
        burst_rate_hz: Nominal sampling rate inside bursts.
        jitter_ratio: Relative jitter on intra-burst step timing.
        dropout_probability: Probability to skip one sample within a burst.

    Returns:
        Sorted timestamps within ``[0, duration_s]``.
    """
    if burst_count <= 0 or duration_s <= 0:
        return np.array([], dtype=float)

    burst_len = float(burst_duration_s)
    total_burst_len = burst_count * burst_len
    slack = float(duration_s) - total_burst_len
    if slack < 0.0:
        burst_len = float(duration_s) / float(burst_count)
        slack = 0.0

    n_gaps = burst_count + 1
    gaps = rng.dirichlet(np.ones(n_gaps)).astype(float) * slack
    starts: List[float] = []
    pos = float(gaps[0])
    for i in range(burst_count):
        starts.append(pos)
        pos += burst_len
        if i < burst_count - 1:
            pos += float(gaps[i + 1])

    dt = 1.0 / burst_rate_hz
    times: List[float] = []
    for start in starts:
        end = min(start + burst_len, float(duration_s))
        t = start
        while t < end - 1e-9:
            if rng.random() < dropout_probability:
                t += dt
                continue
            off = max(0.0, dt * abs(rng.normal(0.0, jitter_ratio)))
            ts = min(t + off, float(duration_s))
            if ts <= float(duration_s):
                times.append(ts)
            t += dt

    return np.sort(np.array(times, dtype=float))


def sample_uniform_interval_times(
    rng: np.random.Generator,
    duration_s: int,
    interval_min_s: float,
    interval_max_s: float,
    jitter_ratio: float,
) -> np.ndarray:
    """Sample timestamps with inter-arrival uniform in ``[interval_min_s, interval_max_s]``.

    Events are placed from early in the run through the end of ``duration_s``: the
    schedule is extended so the final gap before ``duration_s`` is at most
    ``interval_max_s`` when possible, then a last observation at ``duration_s`` is
    added only if that gap still lies in ``[interval_min_s, interval_max_s]``.

    Args:
        rng: Random number generator.
        duration_s: Total simulation duration in seconds.
        interval_min_s: Minimum inter-arrival seconds.
        interval_max_s: Maximum inter-arrival seconds.
        jitter_ratio: Relative jitter applied to each inter-arrival.

    Returns:
        Sorted timestamps within ``[0, duration_s]``.
    """
    if duration_s <= 0:
        return np.array([], dtype=float)

    lo = float(interval_min_s)
    hi = float(interval_max_s)
    if hi < lo:
        lo, hi = hi, lo

    times: List[float] = []
    t = float(rng.uniform(0.0, min(lo, float(duration_s))))
    while t <= duration_s:
        times.append(t)
        interval = float(rng.uniform(lo, hi))
        interval *= 1.0 + rng.normal(0.0, jitter_ratio)
        interval = max(lo * 0.5, interval)
        if t + interval > duration_s:
            break
        t += interval

    if not times:
        times.append(float(rng.uniform(0.0, min(lo, float(duration_s)))))

    def _jittered_step(a: float, b: float) -> float:
        span = float(rng.uniform(a, b))
        span *= 1.0 + rng.normal(0.0, jitter_ratio)
        return max(lo * 0.5, span)

    # No gap larger than ``hi`` before the end time (so cells cover the full path).
    while duration_s - times[-1] > hi + 1e-9:
        remain = duration_s - times[-1]
        upper = min(hi, remain - lo)
        if upper < lo - 1e-9:
            break
        step = _jittered_step(lo, upper)
        nxt = times[-1] + step
        if nxt >= duration_s - 1e-9:
            break
        times.append(nxt)

    gap = duration_s - times[-1]
    if gap >= lo - 1e-9 and gap <= hi + 1e-9:
        times.append(float(duration_s))

    out = np.sort(np.unique(np.array(times, dtype=float)))
    return out[out <= float(duration_s) + 1e-9]


def sample_source_times(
    rng: np.random.Generator,
    duration_s: int,
    cadence: SourceCadence,
) -> np.ndarray:
    """Sample asynchronous event timestamps for one source.

    Args:
        rng: Random number generator.
        duration_s: Total simulation duration in seconds.
        cadence: Cadence model (bursts or uniform-interval).

    Returns:
        Sorted array of event timestamps for the source.
    """
    if cadence.mode == "bursts":
        return sample_bursty_times(
            rng,
            duration_s,
            cadence.burst_count,
            cadence.burst_duration_s,
            cadence.burst_rate_hz,
            cadence.burst_jitter_ratio,
            cadence.burst_dropout_probability,
        )
    if cadence.mode == "uniform_intervals":
        return sample_uniform_interval_times(
            rng,
            duration_s,
            cadence.interval_min_s,
            cadence.interval_max_s,
            cadence.interval_jitter_ratio,
        )
    raise ValueError(f"Unknown cadence mode: {cadence.mode!r}")


def nearest_tower(point: np.ndarray, towers: np.ndarray) -> np.ndarray:
    """Find the nearest tower to a 2D point.

    Args:
        point: Query coordinate as [x, y].
        towers: Tower coordinates as shape (M, 2).

    Returns:
        Coordinate of the nearest tower as [x, y].
    """
    diffs = towers - point
    d2 = np.sum(diffs * diffs, axis=1)
    return towers[int(np.argmin(d2))]


def gps_observation(rng: np.random.Generator, true_xy: np.ndarray, sigma_m: float) -> Tuple[float, float]:
    """Generate a GPS-like point from true position plus Gaussian noise.

    Args:
        rng: Random number generator.
        true_xy: Ground-truth coordinate [x, y].
        sigma_m: Standard deviation of noise in meters.

    Returns:
        Noisy observed coordinate as (x, y).
    """
    noise = rng.normal(0.0, sigma_m, size=2)
    p = true_xy + noise
    return float(p[0]), float(p[1])


def circle_observation(
    rng: np.random.Generator,
    true_xy: np.ndarray,
    obs_sigma_m: float,
    radius_padding_m: Tuple[float, float],
) -> Tuple[float, float, float]:
    """Generate circular-uncertainty observation around true position.

    Args:
        rng: Random number generator.
        true_xy: Ground-truth coordinate [x, y].
        obs_sigma_m: Noise scale used for the observed center.
        radius_padding_m: Extra radius range added so true point is enclosed.

    Returns:
        Tuple of (observed_x, observed_y, radius_m).
    """
    offset = rng.normal(0.0, obs_sigma_m, size=2)
    obs = true_xy + offset
    d = float(np.linalg.norm(true_xy - obs))
    radius = d + float(rng.uniform(*radius_padding_m))
    return float(obs[0]), float(obs[1]), radius


def cell_sector_observation(
    rng: np.random.Generator,
    true_xy: np.ndarray,
    tower_xy: np.ndarray,
    radial_padding_m: Tuple[float, float],
    sector_width_deg: Tuple[float, float],
) -> Tuple[float, float, float, float, float, float]:
    """Generate annulus-sector observation around a serving tower.

    Args:
        rng: Random number generator.
        true_xy: Ground-truth coordinate [x, y].
        tower_xy: Serving tower coordinate [x, y].
        radial_padding_m: Inward/outward radial uncertainty range in meters.
        sector_width_deg: Random sector width range in degrees.

    Returns:
        Tuple of (tower_x, tower_y, r_min, r_max, theta_start, theta_end).
    """
    dx = float(true_xy[0] - tower_xy[0])
    dy = float(true_xy[1] - tower_xy[1])
    dist = math.hypot(dx, dy)
    angle = math.atan2(dy, dx)

    pad_in = float(rng.uniform(*radial_padding_m))
    pad_out = float(rng.uniform(*radial_padding_m))
    r_min = max(0.0, dist - pad_in)
    r_max = dist + pad_out

    width = math.radians(float(rng.uniform(*sector_width_deg)))
    center = angle + rng.normal(0.0, width * 0.08)
    theta_start = center - width / 2.0
    theta_end = center + width / 2.0
    if not angle_in_sector(angle, theta_start, theta_end):
        theta_start = angle - width / 2.0
        theta_end = angle + width / 2.0

    return (
        float(tower_xy[0]),
        float(tower_xy[1]),
        float(r_min),
        float(r_max),
        float(theta_start),
        float(theta_end),
    )


def build_events(
    rng: np.random.Generator,
    t_ref: np.ndarray,
    true_xy_ref: np.ndarray,
    cadences: Dict[str, SourceCadence],
    towers: np.ndarray,
) -> pd.DataFrame:
    """Build source-switched observation events on one merged timeline.

    Args:
        rng: Random number generator.
        t_ref: Reference timeline used for true-position lookup.
        true_xy_ref: Ground-truth coordinates aligned to `t_ref`.
        cadences: Per-source cadence configuration.
        towers: Tower coordinate array used for cellular observations.

    Returns:
        DataFrame with one row per event, local meter columns, and WGS84
        ``true_lon``/``true_lat`` plus source-specific ``*_lon``/``*_lat`` where
        applicable.
    """
    max_t = int(round(float(t_ref[-1])))
    source_times = {
        src: sample_source_times(rng, max_t, cadence) for src, cadence in cadences.items()
    }

    events = []
    for src, times in source_times.items():
        for ts in times:
            events.append((float(ts), src))
    events.sort(key=lambda x: x[0])

    used_ts = []
    used_src = []
    for ts, src in events:
        if used_ts and abs(ts - used_ts[-1]) < 1e-3:
            if rng.random() < 0.5:
                used_src[-1] = src
            continue
        used_ts.append(ts)
        used_src.append(src)

    rows: List[Dict[str, float]] = []
    for ts, src in zip(used_ts, used_src):
        idx = int(np.searchsorted(t_ref, ts))
        idx = min(max(idx, 0), len(t_ref) - 1)
        true_xy = true_xy_ref[idx]
        true_lon, true_lat = enu_scalar_to_lon_lat(float(true_xy[0]), float(true_xy[1]))
        row: Dict[str, float] = {
            "timestamp_s": float(ts),
            "source_type": src,
            "true_x": float(true_xy[0]),
            "true_y": float(true_xy[1]),
            "true_lon": true_lon,
            "true_lat": true_lat,
            "gps_x": np.nan,
            "gps_y": np.nan,
            "gps_lon": np.nan,
            "gps_lat": np.nan,
            "circle_x": np.nan,
            "circle_y": np.nan,
            "circle_r": np.nan,
            "circle_lon": np.nan,
            "circle_lat": np.nan,
            "cell_tower_x": np.nan,
            "cell_tower_y": np.nan,
            "cell_tower_lon": np.nan,
            "cell_tower_lat": np.nan,
            "cell_r_min": np.nan,
            "cell_r_max": np.nan,
            "cell_theta_start": np.nan,
            "cell_theta_end": np.nan,
        }

        if src == "gps":
            gx, gy = gps_observation(rng, true_xy, sigma_m=6.0)
            row["gps_x"] = gx
            row["gps_y"] = gy
            glon, glat = enu_scalar_to_lon_lat(gx, gy)
            row["gps_lon"] = glon
            row["gps_lat"] = glat
        elif src == "circle":
            ox, oy, rr = circle_observation(
                rng,
                true_xy,
                obs_sigma_m=14.0,
                radius_padding_m=(4.0, 20.0),
            )
            row["circle_x"] = ox
            row["circle_y"] = oy
            row["circle_r"] = rr
            olon, olat = enu_scalar_to_lon_lat(ox, oy)
            row["circle_lon"] = olon
            row["circle_lat"] = olat
        else:
            tower_xy = nearest_tower(true_xy, towers)
            tx, ty, rmin, rmax, ths, the = cell_sector_observation(
                rng,
                true_xy,
                tower_xy,
                radial_padding_m=(8.0, 35.0),
                sector_width_deg=(30.0, 95.0),
            )
            row["cell_tower_x"] = tx
            row["cell_tower_y"] = ty
            row["cell_r_min"] = rmin
            row["cell_r_max"] = rmax
            row["cell_theta_start"] = ths
            row["cell_theta_end"] = the
            tlon_t, tlat_t = enu_scalar_to_lon_lat(tx, ty)
            row["cell_tower_lon"] = tlon_t
            row["cell_tower_lat"] = tlat_t

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("timestamp_s").reset_index(drop=True)
    return df


def validate_events(df: pd.DataFrame) -> None:
    """Validate timeline and source-specific observation constraints.

    Args:
        df: Event DataFrame returned by `build_events`.

    Raises:
        ValueError: If ordering, payload sparsity, or geometric constraints fail.
    """
    if not np.all(np.diff(df["timestamp_s"].to_numpy()) > 0):
        raise ValueError("Timestamps are not strictly increasing.")

    src_cols = {
        "gps": ["gps_x", "gps_y", "gps_lon", "gps_lat"],
        "circle": ["circle_x", "circle_y", "circle_r", "circle_lon", "circle_lat"],
        "cell_sector": [
            "cell_tower_x",
            "cell_tower_y",
            "cell_tower_lon",
            "cell_tower_lat",
            "cell_r_min",
            "cell_r_max",
            "cell_theta_start",
            "cell_theta_end",
        ],
    }
    for _, row in df.iterrows():
        if pd.isna(row["true_lon"]) or pd.isna(row["true_lat"]):
            raise ValueError("Missing true_lon / true_lat.")
        s = row["source_type"]
        for src, cols in src_cols.items():
            vals = row[cols]
            if src == s and vals.isna().any():
                raise ValueError(f"Missing required values for {src}.")
            if src != s and vals.notna().any():
                raise ValueError(f"Unexpected populated values for {src}.")

    cell = df[df["source_type"] == "cell_sector"].copy()
    if not cell.empty:
        if (cell["cell_r_min"] < 0.0).any():
            raise ValueError("Negative cell_r_min found.")
        if (cell["cell_r_max"] < cell["cell_r_min"]).any():
            raise ValueError("cell_r_max < cell_r_min found.")

        dx = cell["true_x"].to_numpy() - cell["cell_tower_x"].to_numpy()
        dy = cell["true_y"].to_numpy() - cell["cell_tower_y"].to_numpy()
        dist = np.sqrt(dx * dx + dy * dy)
        if np.any(dist < cell["cell_r_min"].to_numpy()) or np.any(
            dist > cell["cell_r_max"].to_numpy()
        ):
            raise ValueError("True point not in cell radial interval.")

        bearings = np.arctan2(dy, dx)
        for b, s, e in zip(
            bearings,
            cell["cell_theta_start"].to_numpy(),
            cell["cell_theta_end"].to_numpy(),
        ):
            if not angle_in_sector(float(b), float(s), float(e)):
                raise ValueError("True bearing outside cell sector angle.")


def true_positions_for_reference_times(
    t_ref: np.ndarray,
    rng: np.random.Generator,
    path_kind: str = DEFAULT_PATH_KIND,
) -> np.ndarray:
    """Ground-truth positions (east/north meters): real London OSM walk or fallback polylines.

    Args:
        t_ref: Time samples in seconds.
        rng: Random generator (route choice for OSM).
        path_kind: ``simple`` or ``complex`` (affects OSM route scoring and fallback path).

    Returns:
        Array of shape ``(len(t_ref), 2)`` with columns east/north meters.
    """
    walk_speed = (
        _SIMPLE_WALK_SPEED_MPS if path_kind == "simple" else _COMPLEX_WALK_SPEED_MPS
    )
    try:
        from .london_street_path import positions_enu_along_osm_walk

        arr = positions_enu_along_osm_walk(
            t_ref, rng, path_kind, walk_speed_mps=walk_speed
        )
        if arr is not None:
            return arr
    except Exception as exc:
        warnings.warn(
            f"London OSM street path unavailable ({exc}); using synthetic polyline.",
            UserWarning,
            stacklevel=2,
        )

    if path_kind == "simple":
        return simple_path(t_ref)
    return complex_path(t_ref)


def make_towers() -> np.ndarray:
    """Create fixed synthetic tower locations covering long street-like paths.

    Returns:
        Array of shape (M, 2) containing tower x/y coordinates.
    """
    towers: List[List[float]] = []
    for x in range(-1800, 1900, 450):
        for y in range(-1800, 1900, 450):
            towers.append([float(x), float(y)])
    for x in [-900.0, 0.0, 900.0]:
        for y in [-1350.0, -450.0, 450.0, 1350.0]:
            towers.append([x, y])
    return np.array(towers, dtype=float)


def generate_dataset(
    rng: np.random.Generator,
    output_dir: Path,
    duration_s: int,
    dataset_id: str = "dataset",
) -> pd.DataFrame:
    """Generate, validate, plot, and write one dataset (observations + true path + figures).

    Writes ``*_observations.csv`` (mixed-source events) and ``*_true_path.csv``
    (ground truth at 1 Hz: ``timestamp_s``, ``true_x``, ``true_y``, ``lon``, ``lat``, metadata).

    Args:
        rng: Random number generator.
        output_dir: Directory where CSV and PNG files are written.
        duration_s: Total simulation duration in seconds.
        dataset_id: Output prefix for filenames and ``dataset_id`` column.

    Returns:
        Observations DataFrame (same as written to ``*_observations.csv``).
    """
    reference_hz = 10.0
    t_ref = build_reference_timeline(duration_s, reference_hz)
    true_xy_ref = true_positions_for_reference_times(t_ref, rng)

    cadences = make_cadences(duration_s)
    towers = make_towers()
    df = build_events(rng, t_ref, true_xy_ref, cadences, towers)
    df["dataset_id"] = dataset_id
    df["reference_origin_lat"] = LONDON_REFERENCE_LAT_DEG
    df["reference_origin_lon"] = LONDON_REFERENCE_LON_DEG

    validate_events(df)

    obs_path = output_dir / f"{dataset_id}_observations.csv"
    df.to_csv(obs_path, index=False)

    step_1hz = max(1, int(round(reference_hz / 1.0)))
    t_1hz = t_ref[::step_1hz]
    xy_1hz = true_xy_ref[::step_1hz]
    lon_1hz, lat_1hz = local_enu_meters_to_lon_lat(xy_1hz[:, 0], xy_1hz[:, 1])
    true_path_df = pd.DataFrame(
        {
            "timestamp_s": t_1hz,
            "true_x": xy_1hz[:, 0],
            "true_y": xy_1hz[:, 1],
            "lon": lon_1hz,
            "lat": lat_1hz,
            "reference_origin_lat": LONDON_REFERENCE_LAT_DEG,
            "reference_origin_lon": LONDON_REFERENCE_LON_DEG,
            "dataset_id": dataset_id,
        }
    )
    true_path_csv = output_dir / f"{dataset_id}_true_path.csv"
    true_path_df.to_csv(true_path_csv, index=False)

    track_df = true_path_df[["timestamp_s", "lon", "lat"]].copy()

    plot_true_path(true_path_df, output_dir / f"{dataset_id}_true_path.png", show=False)
    plot_observations_only(df, output_dir / f"{dataset_id}_observations_only.png", show=False)

    plot_true_path_on_map(
        track_df,
        output_dir / f"{dataset_id}_map_true_path.png",
        show_basemap=True,
        show_path=True,
        show=False,
    )
    plot_observations_on_map(
        df,
        output_dir / f"{dataset_id}_map_observations.png",
        show_basemap=True,
        show_gps=True,
        show_circle=True,
        show_cell_tower=True,
        show=False,
    )
    plot_map_with_layers(
        track_df,
        df,
        output_dir / f"{dataset_id}_map_layers.png",
        show_basemap=True,
        show_true_path=True,
        show_gps=True,
        show_circle=True,
        show_cell_tower=True,
        show=False,
    )
    return df


def main() -> None:
    """CLI: generate a single trajectory dataset (observations + true path + plots)."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate one synthetic trajectory dataset with asynchronous mixed-source "
            "observations (gps, circle, cell_sector) and a London OSM ground-truth path."
        ),
        epilog=(
            "Examples:\n"
            "  python generate_synthetic_datasets.py\n"
            "  python generate_synthetic_datasets.py --seed 7 --duration-s 1800\n"
            "  python generate_synthetic_datasets.py --output-dir ./data --dataset-id run_01"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible dataset generation. Default: %(default)s.",
    )
    parser.add_argument(
        "--duration-s",
        type=int,
        default=900,
        help="Simulation duration in seconds. Default: %(default)s.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Output directory for generated CSV and PNG files. Default: %(default)s.",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="dataset",
        help="Prefix for output CSV/PNG filenames and dataset_id column. Default: %(default)s.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    df = generate_dataset(
        rng=rng,
        output_dir=args.output_dir,
        duration_s=args.duration_s,
        dataset_id=args.dataset_id,
    )
    counts = df["source_type"].value_counts().to_dict()
    print(f"Generated dataset '{args.dataset_id}': {len(df)} events | source_counts={counts}")
    print(f"Output folder: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
