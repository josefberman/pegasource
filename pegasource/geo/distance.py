"""
Geographic distance and bearing functions.

All functions use the WGS-84 ellipsoid unless stated otherwise.
"""

from __future__ import annotations

import math

# WGS-84 parameters
_R_MEAN = 6_371_008.8  # mean Earth radius in metres
_A = 6_378_137.0       # semi-major axis (m)
_F = 1 / 298.257223563 # flattening
_B = _A * (1 - _F)     # semi-minor axis (m)


def haversine(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Great-circle distance between two points (Haversine formula).

    Parameters
    ----------
    lat1, lon1 : float
        Latitude/longitude of point 1 in decimal degrees.
    lat2, lon2 : float
        Latitude/longitude of point 2 in decimal degrees.

    Returns
    -------
    float
        Distance in metres.

    Examples
    --------
    >>> round(haversine(31.7683, 35.2137, 32.0853, 34.7818), 0)
    54031.0
    """
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return 2 * _R_MEAN * math.asin(math.sqrt(a))


def vincenty(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    max_iterations: int = 200,
    tol: float = 1e-12,
) -> float:
    """Ellipsoidal distance between two points (Vincenty's formula).

    More accurate than Haversine, especially at antipodal points.

    Parameters
    ----------
    lat1, lon1 : float
        Latitude/longitude of point 1 in decimal degrees.
    lat2, lon2 : float
        Latitude/longitude of point 2 in decimal degrees.
    max_iterations : int
        Maximum number of iterations for convergence.
    tol : float
        Convergence tolerance.

    Returns
    -------
    float
        Distance in metres.

    Raises
    ------
    ValueError
        If the formula fails to converge (near-antipodal points).
    """
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    L = math.radians(lon2 - lon1)

    U1 = math.atan((1 - _F) * math.tan(φ1))
    U2 = math.atan((1 - _F) * math.tan(φ2))
    sinU1, cosU1 = math.sin(U1), math.cos(U1)
    sinU2, cosU2 = math.sin(U2), math.cos(U2)

    λ = L
    for _ in range(max_iterations):
        sinλ, cosλ = math.sin(λ), math.cos(λ)
        sinσ = math.sqrt(
            (cosU2 * sinλ) ** 2 + (cosU1 * sinU2 - sinU1 * cosU2 * cosλ) ** 2
        )
        if sinσ == 0:
            return 0.0  # coincident points
        cosσ = sinU1 * sinU2 + cosU1 * cosU2 * cosλ
        σ = math.atan2(sinσ, cosσ)
        sinα = cosU1 * cosU2 * sinλ / sinσ
        cos2α = 1 - sinα ** 2
        if cos2α == 0:
            cos2σm = 0.0
        else:
            cos2σm = cosσ - 2 * sinU1 * sinU2 / cos2α
        C = _F / 16 * cos2α * (4 + _F * (4 - 3 * cos2α))
        λ_new = L + (1 - C) * _F * sinα * (
            σ + C * sinσ * (cos2σm + C * cosσ * (-1 + 2 * cos2σm ** 2))
        )
        if abs(λ_new - λ) < tol:
            λ = λ_new
            break
        λ = λ_new
    else:
        raise ValueError("Vincenty formula failed to converge. Points may be antipodal.")

    u2 = cos2α * (_A ** 2 - _B ** 2) / _B ** 2
    A_coef = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B_coef = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
    Δσ = B_coef * sinσ * (
        cos2σm
        + B_coef / 4 * (
            cosσ * (-1 + 2 * cos2σm ** 2)
            - B_coef / 6 * cos2σm * (-3 + 4 * sinσ ** 2) * (-3 + 4 * cos2σm ** 2)
        )
    )
    return _B * A_coef * (σ - Δσ)


def bearing(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Initial bearing from point 1 to point 2.

    Parameters
    ----------
    lat1, lon1 : float
        Latitude/longitude of origin in decimal degrees.
    lat2, lon2 : float
        Latitude/longitude of destination in decimal degrees.

    Returns
    -------
    float
        Bearing in degrees east of north [0, 360).
    """
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dλ = math.radians(lon2 - lon1)
    x = math.sin(dλ) * math.cos(φ2)
    y = math.cos(φ1) * math.sin(φ2) - math.sin(φ1) * math.cos(φ2) * math.cos(dλ)
    θ = math.degrees(math.atan2(x, y))
    return (θ + 360) % 360
