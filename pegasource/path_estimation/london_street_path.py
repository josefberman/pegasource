"""Real walking routes on OpenStreetMap pedestrian network (central London).

Uses OSMnx to download/cache a walk graph, samples shortest paths between random
nodes until length is sufficient, then walks along edge geometries at a fixed
speed. Positions are returned in the same local ENU meter frame as
:mod:`pegasource.path_estimation.geo_reference` (east/north relative to the London anchor).

If OSMnx is unavailable or routing fails, callers should fall back to synthetic
polylines.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .geo_reference import lon_lat_to_local_enu_meters

# Central London — Westminster / City / South Bank (walkable, dense streets)
_BBOX_WEST = -0.135
_BBOX_EAST = -0.085
_BBOX_SOUTH = 51.485
_BBOX_NORTH = 51.515

_CACHE_DIR = Path(__file__).resolve().parent / "cache"
_GRAPHML_NAME = "london_walk_bbox.graphml"


def _cumdist_xy(xy: np.ndarray) -> np.ndarray:
    """Cumulative distance along a vertex chain (first point at 0)."""
    if len(xy) < 2:
        return np.zeros(len(xy), dtype=float)
    seg = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))
    return np.concatenate([[0.0], np.cumsum(seg)])


def _truncate_polyline(xy: np.ndarray, max_len_m: float) -> np.ndarray:
    """Truncate ``xy`` so total arc length is at most ``max_len_m``."""
    c = _cumdist_xy(xy)
    if c[-1] <= max_len_m + 1e-6:
        return xy
    j = int(np.searchsorted(c, max_len_m, side="right") - 1)
    j = max(0, min(j, len(xy) - 2))
    denom = c[j + 1] - c[j]
    if denom < 1e-9:
        return xy[: j + 1]
    a = (max_len_m - c[j]) / denom
    last = (1.0 - a) * xy[j] + a * xy[j + 1]
    return np.vstack([xy[: j + 1], last])


def _positions_at_distances(xy: np.ndarray, cumdist: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Interpolate positions at arc lengths ``s`` (meters) along ``xy``."""
    s = np.asarray(s, dtype=float)
    s = np.clip(s, 0.0, cumdist[-1])
    j = np.searchsorted(cumdist, s, side="right") - 1
    j = np.clip(j, 0, len(xy) - 2)
    denom = cumdist[j + 1] - cumdist[j]
    denom = np.where(denom < 1e-9, 1.0, denom)
    a = (s - cumdist[j]) / denom
    return (1.0 - a)[:, None] * xy[j] + a[:, None] * xy[j + 1]


def _polyline_xy_from_route(G, route: List) -> np.ndarray:
    """Build dense (x, y) in projected CRS from a node route."""
    coords: List[Tuple[float, float]] = []
    for u, v in zip(route[:-1], route[1:]):
        if not G.has_edge(u, v):
            continue
        ed = G[u][v]
        best = min(ed.values(), key=lambda d: d.get("length", float("inf")))
        geom = best.get("geometry")
        xu, yu = G.nodes[u]["x"], G.nodes[u]["y"]
        xv, yv = G.nodes[v]["x"], G.nodes[v]["y"]
        if geom is not None:
            seg = list(geom.coords)
        else:
            seg = [(xu, yu), (xv, yv)]
        if not coords:
            coords.extend(seg)
        else:
            if len(seg) and seg[0] == coords[-1]:
                coords.extend(seg[1:])
            else:
                coords.extend(seg)
    if len(coords) < 2:
        return np.zeros((0, 2), dtype=float)
    arr = np.asarray(coords, dtype=float)
    # Remove consecutive duplicates
    keep = np.ones(len(arr), dtype=bool)
    keep[1:] = np.any(np.abs(np.diff(arr, axis=0)) > 1e-6, axis=1)
    return arr[keep]


def _proj_xy_to_enu_columns(xs: np.ndarray, ys: np.ndarray, crs) -> np.ndarray:
    """Projected CRS sample points -> (east_m, north_m) columns."""
    from pyproj import Transformer

    tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(np.asarray(xs), np.asarray(ys))
    east, north = lon_lat_to_local_enu_meters(np.asarray(lon), np.asarray(lat))
    return np.column_stack((east, north))


def load_walk_graph():
    """Load or download the OSM walk graph used for synthetic paths (projected CRS).

    Same graph as used internally for routing; safe to reuse for map-matching and
    graph-based estimators.
    """
    return _load_walk_graph()


def _load_walk_graph():
    """Load or download OSM walk graph; return projected MultiDiGraph."""
    import networkx as nx
    import osmnx as ox

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _CACHE_DIR / _GRAPHML_NAME

    if cache_file.exists():
        G = ox.load_graphml(cache_file)
    else:
        G = ox.graph_from_bbox(
            bbox=(_BBOX_WEST, _BBOX_SOUTH, _BBOX_EAST, _BBOX_NORTH),
            network_type="walk",
            simplify=True,
        )
        ox.save_graphml(G, cache_file)

    Gp = ox.project_graph(G.copy())
    if Gp.number_of_nodes() < 10:
        raise RuntimeError("OSM walk graph has too few nodes.")
    return Gp


def _longest_shortest_path_from_node(
    G,
    current: int,
    rng: np.random.Generator,
    path_kind: str,
    remaining_m: float,
) -> Optional[Tuple[List, float]]:
    """Pick the best shortest-path segment from ``current`` using one Dijkstra sweep."""
    import networkx as nx

    # Radius scales with how much distance we still need; cap to keep each sweep bounded.
    cutoff = min(35000.0, max(remaining_m * 1.35, 400.0))
    try:
        dist, paths = nx.single_source_dijkstra(
            G, current, cutoff=cutoff, weight="length"
        )
    except Exception:
        return None

    scored: List[Tuple[List, float, int]] = []
    for dest, plen in dist.items():
        if dest == current or plen < 3.0:
            continue
        path = paths.get(dest)
        if not path or len(path) < 2:
            continue
        scored.append((path, plen, len(path)))

    if not scored:
        return None

    if path_kind == "complex":
        scored.sort(key=lambda x: (-x[2], -x[1]))
    else:
        scored.sort(key=lambda x: (-x[1], -x[2]))

    best_path, best_len, _ = scored[0]
    return best_path, float(best_len)


def _merge_route_polylines(
    existing: np.ndarray,
    seg: np.ndarray,
) -> np.ndarray:
    """Append ``seg`` to ``existing``, dropping a duplicate junction vertex if needed."""
    if len(seg) < 2:
        return existing
    if len(existing) == 0:
        return seg
    if np.allclose(existing[-1], seg[0], atol=0.25):
        return np.vstack([existing, seg[1:]])
    return np.vstack([existing, seg])


def _chain_street_polyline(
    G,
    target_length_m: float,
    rng: np.random.Generator,
    path_kind: str,
) -> Optional[np.ndarray]:
    """Concatenate many shortest-path segments along real edges until length is reached.

    A single shortest path inside a city bbox is usually only a few km; long
    simulations need many chained segments so total arc length can match
    ``duration * walk_speed`` without falling back to synthetic parametric paths.
    """
    import networkx as nx

    nodes = list(G.nodes)
    current = int(rng.choice(nodes))
    merged = np.zeros((0, 2), dtype=float)
    # Worst case many short segments; keep upper bound generous for long targets.
    max_segments = min(1000, max(120, int(target_length_m / 120.0) + 80))

    for _ in range(max_segments):
        c = _cumdist_xy(merged)
        total = float(c[-1]) if len(c) else 0.0
        if total >= target_length_m * 0.999:
            break

        remaining = max(0.0, target_length_m - total)
        picked = _longest_shortest_path_from_node(
            G, current, rng, path_kind, remaining_m=remaining
        )
        if picked is None:
            break
        path, _plen = picked
        seg = _polyline_xy_from_route(G, path)
        if len(seg) < 2:
            current = path[-1]
            continue
        merged = _merge_route_polylines(merged, seg)
        current = path[-1]

        c2 = _cumdist_xy(merged)
        if len(c2) and float(c2[-1]) - total < 0.5:
            # no geometric progress; try another start once
            current = int(rng.choice(nodes))

    if len(merged) < 2:
        return None
    merged = _truncate_polyline(merged, target_length_m)
    cfinal = _cumdist_xy(merged)
    if float(cfinal[-1]) < min(50.0, target_length_m * 0.05):
        return None
    return merged


def positions_enu_along_osm_walk(
    times_s: np.ndarray,
    rng: np.random.Generator,
    path_kind: str,
    walk_speed_mps: float = 1.35,
) -> Optional[np.ndarray]:
    """Sample positions along a real London walking route (local ENU meters).

    Args:
        times_s: Timestamps (seconds) at which to evaluate position.
        rng: Random generator (controls start/end pair).
        path_kind: ``simple`` prefers fewer corners; ``complex`` prefers more corners.
        walk_speed_mps: Constant speed along the path.

    Returns:
        Array of shape ``(len(times_s), 2)`` with columns ``(east_m, north_m)``, or
        ``None`` if routing failed.
    """
    times_s = np.asarray(times_s, dtype=float)
    duration = float(np.max(times_s)) if times_s.size else 0.0
    target_len_m = duration * walk_speed_mps * 1.02

    try:
        G = _load_walk_graph()
    except Exception as exc:
        warnings.warn(f"Could not load OSM walk graph: {exc}", UserWarning)
        return None

    xy_proj = _chain_street_polyline(G, target_len_m, rng, path_kind)
    if xy_proj is None:
        warnings.warn(
            "Could not build a long enough London street polyline; use synthetic path.",
            UserWarning,
        )
        return None

    crs = G.graph.get("crs")
    if crs is None:
        return None

    cum_proj = _cumdist_xy(xy_proj)
    s_query = np.clip(times_s * walk_speed_mps, 0.0, cum_proj[-1])
    xy_samp = _positions_at_distances(xy_proj, cum_proj, s_query)
    return _proj_xy_to_enu_columns(xy_samp[:, 0], xy_samp[:, 1], crs)
