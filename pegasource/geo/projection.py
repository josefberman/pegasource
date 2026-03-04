"""
Coordinate reference system transformations.

Supported systems:
* WGS-84 (EPSG:4326) — GPS latitude/longitude
* ITM — Israeli Transverse Mercator (EPSG:2039)
* UTM — Universal Transverse Mercator (any zone)
"""

from __future__ import annotations

import math

from pyproj import Transformer, CRS  # type: ignore

# ---------------------------------------------------------------------------
# Pre-built transformers (thread-safe in pyproj ≥ 3.x)
# ---------------------------------------------------------------------------
_WGS84 = CRS("EPSG:4326")
_ITM = CRS("EPSG:2039")

_to_itm: Transformer | None = None
_from_itm: Transformer | None = None


def _get_to_itm() -> Transformer:
    global _to_itm
    if _to_itm is None:
        _to_itm = Transformer.from_crs(_WGS84, _ITM, always_xy=True)
    return _to_itm


def _get_from_itm() -> Transformer:
    global _from_itm
    if _from_itm is None:
        _from_itm = Transformer.from_crs(_ITM, _WGS84, always_xy=True)
    return _from_itm


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def wgs84_to_itm(lat: float, lon: float) -> tuple[float, float]:
    """Convert WGS-84 geographic coordinates to Israeli Transverse Mercator.

    Parameters
    ----------
    lat : float
        Latitude in decimal degrees (positive = north).
    lon : float
        Longitude in decimal degrees (positive = east).

    Returns
    -------
    tuple[float, float]
        ``(easting_m, northing_m)`` in metres (EPSG:2039).

    Examples
    --------
    >>> e, n = wgs84_to_itm(31.7683, 35.2137)   # Jerusalem
    >>> round(e), round(n)
    (220463, 631757)
    """
    return _get_to_itm().transform(lon, lat)


def itm_to_wgs84(easting: float, northing: float) -> tuple[float, float]:
    """Convert Israeli Transverse Mercator coordinates back to WGS-84.

    Parameters
    ----------
    easting : float
        Easting in metres (EPSG:2039).
    northing : float
        Northing in metres (EPSG:2039).

    Returns
    -------
    tuple[float, float]
        ``(latitude, longitude)`` in decimal degrees.
    """
    lon, lat = _get_from_itm().transform(easting, northing)
    return lat, lon


def wgs84_to_utm(
    lat: float,
    lon: float,
    zone: int | None = None,
) -> tuple[float, float, int]:
    """Convert WGS-84 coordinates to UTM.

    Parameters
    ----------
    lat : float
        Latitude in decimal degrees.
    lon : float
        Longitude in decimal degrees.
    zone : int or None
        UTM zone number (1–60). If ``None``, auto-detected from longitude.

    Returns
    -------
    tuple[float, float, int]
        ``(easting_m, northing_m, zone_number)``.
    """
    if zone is None:
        zone = int((lon + 180) / 6) + 1
    hemisphere = "north" if lat >= 0 else "south"
    epsg = 32600 + zone if hemisphere == "north" else 32700 + zone
    t = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    e, n = t.transform(lon, lat)
    return e, n, zone


def meters_offset(
    lat: float,
    lon: float,
    dx_m: float,
    dy_m: float,
) -> tuple[float, float]:
    """Move a point by *dx_m* metres east and *dy_m* metres north.

    Uses a simple equirectangular approximation — accurate to within a
    few centimetres for offsets < 100 km at Israeli latitudes.

    Parameters
    ----------
    lat : float
        Origin latitude in decimal degrees.
    lon : float
        Origin longitude in decimal degrees.
    dx_m : float
        Eastward offset in metres (negative = west).
    dy_m : float
        Northward offset in metres (negative = south).

    Returns
    -------
    tuple[float, float]
        New ``(latitude, longitude)`` in decimal degrees.
    """
    # Earth radius at given latitude (average)
    r_at_lat = 6_371_008.8
    lat_new = lat + math.degrees(dy_m / r_at_lat)
    lon_new = lon + math.degrees(dx_m / (r_at_lat * math.cos(math.radians(lat))))
    return lat_new, lon_new
