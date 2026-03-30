"""OSM graph loading, snapping, and polyline utilities."""

from __future__ import annotations

from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
from pyproj import Transformer

from .geo_reference import local_enu_meters_to_lon_lat
from .london_street_path import _proj_xy_to_enu_columns, load_walk_graph


def get_projected_graph():
    """Return the projected walk graph (same as synthetic data generation)."""
    return load_walk_graph()


def enu_to_proj_xy(G: nx.MultiDiGraph, east_m: float, north_m: float) -> Tuple[float, float]:
    """Local ENU meters -> graph projected ``x, y`` (for snapping)."""
    lon, lat = local_enu_meters_to_lon_lat(
        np.array([east_m], dtype=float),
        np.array([north_m], dtype=float),
    )
    crs = G.graph.get("crs")
    if crs is None:
        raise RuntimeError("Graph missing CRS.")
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = tr.transform(lon, lat)
    return float(x[0]), float(y[0])


def proj_polyline_to_enu(G: nx.MultiDiGraph, xy_proj: np.ndarray) -> np.ndarray:
    """Polyline in projected CRS -> (N, 2) ENU east/north."""
    if len(xy_proj) == 0:
        return np.zeros((0, 2), dtype=float)
    crs = G.graph.get("crs")
    if crs is None:
        raise RuntimeError("Graph missing CRS.")
    out = _proj_xy_to_enu_columns(xy_proj[:, 0], xy_proj[:, 1], crs)
    return np.asarray(out, dtype=float)


def node_xy(G: nx.MultiDiGraph, n: int) -> np.ndarray:
    """Node position as ``[x, y]`` in projected CRS (meters)."""
    d = G.nodes[n]
    return np.array([float(d["x"]), float(d["y"])], dtype=float)


def node_to_enu(G: nx.MultiDiGraph, n: int) -> np.ndarray:
    """Single node as ENU ``[east, north]``."""
    p = node_xy(G, n)
    return proj_polyline_to_enu(G, p.reshape(1, 2))[0]


def nearest_graph_node(G: nx.MultiDiGraph, x: float, y: float) -> int:
    """Nearest graph node by Euclidean distance in **projected** graph coordinates."""
    best: Optional[int] = None
    best_d2 = float("inf")
    for n in G.nodes:
        p = node_xy(G, n)
        d2 = float(np.sum((p - np.array([x, y])) ** 2))
        if d2 < best_d2:
            best_d2 = d2
            best = n
    if best is None:
        raise RuntimeError("Empty graph.")
    return int(best)


def k_nearest_nodes(G: nx.MultiDiGraph, x: float, y: float, k: int) -> List[int]:
    """``k`` nearest nodes by Euclidean distance in **projected** coordinates."""
    pts = [(n, float(np.sum((node_xy(G, n) - np.array([x, y])) ** 2))) for n in G.nodes]
    pts.sort(key=lambda t: t[1])
    return [int(p[0]) for p in pts[:k]]


def k_nearest_nodes_enu(
    G: nx.MultiDiGraph, east_m: float, north_m: float, k: int
) -> List[int]:
    """``k`` nearest graph nodes to an ENU point."""
    px, py = enu_to_proj_xy(G, east_m, north_m)
    return k_nearest_nodes(G, px, py, k)


def _polyline_from_route(G: nx.MultiDiGraph, route: List) -> np.ndarray:
    """Build dense (x,y) in projected CRS from a node route (same logic as london_street_path)."""
    from .london_street_path import _polyline_xy_from_route

    return _polyline_xy_from_route(G, route)


def cumdist_xy(xy: np.ndarray) -> np.ndarray:
    """Cumulative distance along vertex chain."""
    if len(xy) < 2:
        return np.zeros(len(xy), dtype=float)
    seg = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))
    return np.concatenate([[0.0], np.cumsum(seg)])


def positions_at_distances(
    xy: np.ndarray, cumdist: np.ndarray, s: np.ndarray
) -> np.ndarray:
    """Interpolate positions at arc lengths ``s`` along ``xy``."""
    s = np.asarray(s, dtype=float)
    s = np.clip(s, 0.0, cumdist[-1])
    j = np.searchsorted(cumdist, s, side="right") - 1
    j = np.clip(j, 0, len(xy) - 2)
    denom = cumdist[j + 1] - cumdist[j]
    denom = np.where(denom < 1e-9, 1.0, denom)
    a = (s - cumdist[j]) / denom
    return (1.0 - a)[:, None] * xy[j] + a[:, None] * xy[j + 1]


def resample_uniform_time(
    xy: np.ndarray,
    times_out: np.ndarray,
    t_start: float,
    t_end: float,
) -> np.ndarray:
    """Map polyline to ``times_out`` assuming uniform speed from ``t_start`` to ``t_end``."""
    c = cumdist_xy(xy)
    if c[-1] < 1e-6:
        return np.tile(xy[:1], (len(times_out), 1))
    span = max(t_end - t_start, 1e-9)
    s = (np.clip(times_out, t_start, t_end) - t_start) / span * c[-1]
    return positions_at_distances(xy, c, s)


def shortest_path_polyline(
    G: nx.MultiDiGraph, u: int, v: int
) -> Optional[np.ndarray]:
    """Shortest-path polyline between nodes ``u`` and ``v`` (projected x,y)."""
    try:
        route = nx.shortest_path(G, u, v, weight="length")
    except nx.NetworkXNoPath:
        return None
    pl = _polyline_from_route(G, route)
    return pl if len(pl) >= 2 else None


def astar_path_polyline(
    G: nx.MultiDiGraph, u: int, v: int
) -> Optional[np.ndarray]:
    """A* shortest path (same optimum as Dijkstra with Euclidean heuristic)."""
    try:
        route = nx.astar_path(
            G,
            u,
            v,
            heuristic=lambda a, b: float(np.linalg.norm(node_xy(G, a) - node_xy(G, b))),
            weight="length",
        )
    except nx.NetworkXNoPath:
        return None
    pl = _polyline_from_route(G, route)
    return pl if len(pl) >= 2 else None


def merge_polylines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Concatenate polylines, dropping duplicate junction."""
    if len(a) == 0:
        return b.copy()
    if len(b) == 0:
        return a.copy()
    if np.allclose(a[-1], b[0], atol=0.25):
        return np.vstack([a, b[1:]])
    return np.vstack([a, b])
