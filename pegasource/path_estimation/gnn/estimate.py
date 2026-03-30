"""GCN node classifier: nodes near true path vs not; decode via graph stitch on GNN-guided snaps."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops

from ..graph_utils import (
    enu_to_proj_xy,
    k_nearest_nodes_enu,
    merge_polylines,
    nearest_graph_node,
    node_to_enu,
    proj_polyline_to_enu,
    resample_uniform_time,
    shortest_path_polyline,
)
from ..io import align_times_to_true, build_event_points
from ..types import EstimationResult


def _subgraph_near_observations(G, obs_df: pd.DataFrame, radius_m: float = 900.0):
    """Restrict to nodes within ``radius_m`` of observation centroid (speed)."""
    _, ox, oy = build_event_points(obs_df)
    cx, cy = float(np.mean(ox)), float(np.mean(oy))
    keep = []
    for n in G.nodes:
        e = node_to_enu(G, n)
        if float(np.hypot(e[0] - cx, e[1] - cy)) <= radius_m:
            keep.append(n)
    if len(keep) < 30:
        return G
    return G.subgraph(keep).copy()


class NodeGCN(nn.Module):
    def __init__(self, in_dim: int = 4, hidden: int = 32) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv1(x, edge_index))
        return self.conv2(h, edge_index).squeeze(-1)


def _graph_to_data(G, obs_df: pd.DataFrame, true_df: pd.DataFrame, device: torch.device) -> tuple[Data, np.ndarray]:
    """Build PyG Data and node label array (1 = near true path).

    ``G`` should already be the working subgraph (see :func:`_subgraph_near_observations`).
    """
    nodes = list(G.nodes)
    nmap = {n: i for i, n in enumerate(nodes)}
    edges = []
    for u, v, _ in G.edges(keys=True):
        if u in nmap and v in nmap:
            edges.append([nmap[u], nmap[v]])
            edges.append([nmap[v], nmap[u]])
    if not edges:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index, _ = add_self_loops(edge_index, num_nodes=len(nodes))

    tx = true_df["true_x"].to_numpy(float)
    ty = true_df["true_y"].to_numpy(float)
    obs_t, ox, oy = build_event_points(obs_df)

    feats: list[list[float]] = []
    labels: list[float] = []
    for n in nodes:
        en = node_to_enu(G, n)
        d_true = np.min(np.hypot(tx - en[0], ty - en[1]))
        d_obs = np.min(np.hypot(ox - en[0], oy - en[1])) if len(ox) else 0.0
        feats.append([en[0], en[1], d_true, d_obs])
        labels.append(1.0 if d_true < 35.0 else 0.0)

    x = torch.tensor(feats, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.float32, device=device)
    ei = edge_index.to(device=device, dtype=torch.long)
    data = Data(x=x, edge_index=ei, y=y)
    return data, np.array(labels)


def train_gcn(
    G, obs_df: pd.DataFrame, true_df: pd.DataFrame, device: torch.device, epochs: int = 5
) -> NodeGCN:
    data, _ = _graph_to_data(G, obs_df, true_df, device)
    model = NodeGCN(in_dim=data.x.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = loss_fn(logits, data.y)
        loss.backward()
        opt.step()
    model.eval()
    return model


def estimate_gnn(
    obs_df: pd.DataFrame,
    true_df: pd.DataFrame,
    G,
    rng: np.random.Generator,
    device: torch.device | None = None,
) -> EstimationResult:
    """Train GCN, snap observations to best-scoring neighbor among k-NN, then stitch."""
    if device is None:
        device = torch.device("cpu")
    G_work = _subgraph_near_observations(G, obs_df)
    model = train_gcn(G_work, obs_df, true_df, device)
    data, _ = _graph_to_data(G_work, obs_df, true_df, device)
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        scores = torch.sigmoid(logits).cpu().numpy()
    nodes = list(G_work.nodes)
    nmap = {n: i for i, n in enumerate(nodes)}

    times_s, _ = align_times_to_true(true_df)
    t_obs, xe, ye = build_event_points(obs_df)
    chosen: list[int] = []
    for i in range(len(t_obs)):
        kcand = k_nearest_nodes_enu(G_work, float(xe[i]), float(ye[i]), 12)
        best = max(kcand, key=lambda n: scores[nmap[n]])
        chosen.append(best)

    merged = np.zeros((0, 2), dtype=float)
    for i in range(len(chosen) - 1):
        u, v = chosen[i], chosen[i + 1]
        if u == v:
            continue
        seg = shortest_path_polyline(G_work, u, v)
        if seg is None or len(seg) < 2:
            continue
        merged = merge_polylines(merged, seg)

    if len(merged) < 2:
        px, py = enu_to_proj_xy(G_work, float(xe[0]), float(ye[0]))
        n0 = nearest_graph_node(G_work, px, py)
        p = node_to_enu(G_work, n0)
        merged_enu = np.array([p, p + 1e-3], dtype=float)
    else:
        merged_enu = proj_polyline_to_enu(G_work, merged)

    t0, t1 = float(times_s[0]), float(times_s[-1])
    xy_at = resample_uniform_time(merged_enu, times_s, t0, t1)
    return EstimationResult(
        times_s=times_s,
        east_m=xy_at[:, 0],
        north_m=xy_at[:, 1],
        meta={"method": "gnn_gcn_stitch"},
    )
