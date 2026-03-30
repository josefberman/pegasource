"""HMM-style map matching: Viterbi over candidate graph nodes per observation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx

from .graph_utils import (
    enu_to_proj_xy,
    k_nearest_nodes_enu,
    merge_polylines,
    node_to_enu,
    proj_polyline_to_enu,
    resample_uniform_time,
    shortest_path_polyline,
)
from .io import align_times_to_true, build_event_points
from .types import EstimationResult


def _path_length_m(G: nx.MultiDiGraph, u: int, v: int) -> float:
    try:
        return float(nx.shortest_path_length(G, u, v, weight="length"))
    except nx.NetworkXNoPath:
        return float("inf")


def estimate_hmm_map_match(
    obs_df: pd.DataFrame,
    true_df: pd.DataFrame,
    G: nx.MultiDiGraph,
    rng: np.random.Generator,
    *,
    k_candidates: int = 8,
    sigma_obs_m: float = 12.0,
    walk_speed_mps: float = 1.35,
) -> EstimationResult:
    """Viterbi on k-nearest nodes per observation; stitch shortest paths between winners."""
    times_s, _ = align_times_to_true(true_df)
    t_obs, xe, ye = build_event_points(obs_df)
    T = len(t_obs)
    if T < 1:
        raise ValueError("No observations.")

    # Candidate nodes per time (same k indices across steps)
    cand: list[list[int]] = []
    for i in range(T):
        cand.append(
            k_nearest_nodes_enu(G, float(xe[i]), float(ye[i]), k_candidates)
        )

    emit_cost = np.full((T, k_candidates), np.inf, dtype=float)
    for t in range(T):
        obs = np.array([xe[t], ye[t]], dtype=float)
        for s in range(len(cand[t])):
            n = cand[t][s]
            mu = node_to_enu(G, n)
            emit_cost[t, s] = 0.5 * float(np.sum((obs - mu) ** 2)) / (sigma_obs_m**2)

    # DP: cost[t][s] = min cost to reach (t, s)
    cost = np.full((T, k_candidates), np.inf, dtype=float)
    back = np.full((T, k_candidates), -1, dtype=int)
    cost[0, :] = emit_cost[0, :]

    for t in range(1, T):
        dt = max(float(t_obs[t] - t_obs[t - 1]), 1e-3)
        expected = walk_speed_mps * dt
        for s in range(k_candidates):
            v = cand[t][s]
            best = np.inf
            best_p = -1
            for sp in range(k_candidates):
                u = cand[t - 1][sp]
                if not np.isfinite(cost[t - 1, sp]):
                    continue
                plen = _path_length_m(G, u, v)
                if not np.isfinite(plen):
                    continue
                trans = 0.5 * (plen - expected) ** 2 / max(50.0, expected) ** 2
                val = cost[t - 1, sp] + trans + emit_cost[t, s]
                if val < best:
                    best = val
                    best_p = sp
            cost[t, s] = best
            back[t, s] = best_p

    # Backtrack
    s_end = int(np.argmin(cost[T - 1, :]))
    seq = [0] * T
    seq[T - 1] = s_end
    for t in range(T - 1, 0, -1):
        seq[t - 1] = back[t, seq[t]]
    nodes_seq = [cand[t][seq[t]] for t in range(T)]

    merged = np.zeros((0, 2), dtype=float)
    for t in range(T - 1):
        u, v = nodes_seq[t], nodes_seq[t + 1]
        if u == v:
            continue
        seg = shortest_path_polyline(G, u, v)
        if seg is None or len(seg) < 2:
            continue
        merged = merge_polylines(merged, seg)

    if len(merged) < 2:
        p = node_to_enu(G, nodes_seq[0])
        merged_enu = np.array([p, p + 1e-3], dtype=float)
    else:
        merged_enu = proj_polyline_to_enu(G, merged)

    t0, t1 = float(times_s[0]), float(times_s[-1])
    xy_at = resample_uniform_time(merged_enu, times_s, t0, t1)
    return EstimationResult(
        times_s=times_s,
        east_m=xy_at[:, 0],
        north_m=xy_at[:, 1],
        meta={"method": "hmm_map_match", "k_candidates": k_candidates},
    )
