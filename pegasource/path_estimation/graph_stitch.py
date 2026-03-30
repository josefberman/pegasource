"""Segment-wise shortest-path stitching on the OSM graph (Dijkstra / A*)."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from .graph_utils import (
    astar_path_polyline,
    enu_to_proj_xy,
    merge_polylines,
    nearest_graph_node,
    proj_polyline_to_enu,
    resample_uniform_time,
    shortest_path_polyline,
)
from .io import align_times_to_true, build_event_points
from .types import EstimationResult


def estimate_graph_stitch(
    obs_df: pd.DataFrame,
    true_df: pd.DataFrame,
    G,
    rng: np.random.Generator,
    *,
    mode: Literal["dijkstra", "astar"] = "dijkstra",
) -> EstimationResult:
    """Snap event points to nodes; stitch shortest paths; resample uniformly in time."""
    times_s, true_xy = align_times_to_true(true_df)
    t_obs, xe, ye = build_event_points(obs_df)
    if len(t_obs) < 1:
        raise ValueError("No observations for graph stitch.")

    nodes = []
    for i in range(len(t_obs)):
        px, py = enu_to_proj_xy(G, float(xe[i]), float(ye[i]))
        nodes.append(nearest_graph_node(G, px, py))
    path_fn = shortest_path_polyline if mode == "dijkstra" else astar_path_polyline

    merged = np.zeros((0, 2), dtype=float)
    for i in range(len(nodes) - 1):
        u, v = nodes[i], nodes[i + 1]
        if u == v:
            continue
        seg = path_fn(G, u, v)
        if seg is None or len(seg) < 2:
            continue
        merged = merge_polylines(merged, seg)

    if len(merged) < 2:
        from .graph_utils import node_xy

        p0 = node_xy(G, nodes[0])
        merged = np.array([p0, p0 + 1e-3], dtype=float)

    t0, t1 = float(times_s[0]), float(times_s[-1])
    merged_enu = proj_polyline_to_enu(G, merged)

    xy_at_times = resample_uniform_time(merged_enu, times_s, t0, t1)
    return EstimationResult(
        times_s=times_s,
        east_m=xy_at_times[:, 0],
        north_m=xy_at_times[:, 1],
        meta={"method": f"graph_stitch_{mode}", "n_events": len(t_obs)},
    )
