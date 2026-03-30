"""Geographic reference for synthetic paths (local ENU meters vs WGS84).

Ground-truth motion is simulated in a local east/north meter frame anchored at a
fixed point in London. Cell towers remain synthetic in that same meter frame
(not real UK tower locations).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# Central London (Trafalgar Square area): anchor for local ENU = (0, 0) meters.
LONDON_REFERENCE_LAT_DEG = 51.5080
LONDON_REFERENCE_LON_DEG = -0.1281


def local_enu_meters_to_lon_lat(
    east_m: np.ndarray,
    north_m: np.ndarray,
    origin_lat_deg: float = LONDON_REFERENCE_LAT_DEG,
    origin_lon_deg: float = LONDON_REFERENCE_LON_DEG,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert local east/north meters from the reference origin to WGS84 lon/lat.

    Uses a local tangent plane: x = east, y = north. Suitable for paths of a
    few kilometres.

    Args:
        east_m: Easting offset in meters (same shape as ``north_m``).
        north_m: Northing offset in meters.
        origin_lat_deg: Reference latitude in degrees.
        origin_lon_deg: Reference longitude in degrees.

    Returns:
        Tuple ``(longitude_deg, latitude_deg)`` in degrees (WGS84).
    """
    r_earth = 6378137.0
    lat0 = np.radians(origin_lat_deg)
    lon0 = np.radians(origin_lon_deg)
    lat_rad = lat0 + np.asarray(north_m, dtype=float) / r_earth
    lon_rad = lon0 + np.asarray(east_m, dtype=float) / (r_earth * np.cos(lat0))
    return np.degrees(lon_rad), np.degrees(lat_rad)


def enu_scalar_to_lon_lat(east_m: float, north_m: float) -> Tuple[float, float]:
    """Scalar variant of :func:`local_enu_meters_to_lon_lat`."""
    lon, lat = local_enu_meters_to_lon_lat(
        np.array([east_m], dtype=float),
        np.array([north_m], dtype=float),
    )
    return float(lon[0]), float(lat[0])


def lon_lat_to_local_enu_meters(
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    origin_lat_deg: float = LONDON_REFERENCE_LAT_DEG,
    origin_lon_deg: float = LONDON_REFERENCE_LON_DEG,
) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`local_enu_meters_to_lon_lat` (WGS84 to local east/north meters).

    Args:
        lon_deg: Longitude in degrees.
        lat_deg: Latitude in degrees.
        origin_lat_deg: Reference latitude in degrees.
        origin_lon_deg: Reference longitude in degrees.

    Returns:
        Tuple ``(east_m, north_m)`` in meters relative to the reference origin.
    """
    r_earth = 6378137.0
    lat0 = np.radians(origin_lat_deg)
    lon0 = np.radians(origin_lon_deg)
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    north_m = (lat - lat0) * r_earth
    east_m = (lon - lon0) * r_earth * np.cos(lat0)
    return east_m, north_m
