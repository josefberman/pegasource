"""Tests for pegasource.path_estimation (metrics + graph helpers)."""

import networkx as nx
import numpy as np

from pegasource.path_estimation.graph_utils import astar_path_polyline, shortest_path_polyline
from pegasource.path_estimation.metrics import compute_all_metrics, discrete_frechet, rmse_euclidean


def test_rmse_zero():
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    m = rmse_euclidean(xy, xy)
    assert m < 1e-9


def test_frechet_identical():
    p = np.array([[0.0, 0.0], [1.0, 0.0]])
    assert discrete_frechet(p, p) < 1e-9


def test_compute_all_metrics_runs():
    t = np.array([[0, 0], [1, 0], [2, 0]], dtype=float)
    e = t + 0.1
    out = compute_all_metrics(t, e, max_points_frechet_dtw=50)
    assert "rmse_m" in out and out["rmse_m"] > 0


def test_line_graph_shortest():
    G = nx.MultiDiGraph()
    for i in range(4):
        G.add_edge(i, i + 1, length=1.0)
    for i in range(5):
        G.nodes[i]["x"] = float(i)
        G.nodes[i]["y"] = 0.0

    a = shortest_path_polyline(G, 0, 4)
    b = astar_path_polyline(G, 0, 4)
    assert a is not None and b is not None
    assert np.allclose(a, b)
