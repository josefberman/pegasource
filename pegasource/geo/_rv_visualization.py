"""
Visualisation helpers for overlaying a NetworkX graph on top of the
original density histogram.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_graph_overlay(
    density_map: np.ndarray,
    graph: nx.Graph,
    *,
    ax: plt.Axes | None = None,
    cmap: str = "hot",
    edge_cmap: str = "winter",
    node_color: str = "#00e5ff",
    edge_color: str | None = None,
    color_edges_by_weight: bool = True,
    node_size: int = 40,
    edge_width: float = 2.0,
    title: str = "Road graph overlay",
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Axes:
    """Draw the density map with the extracted graph overlaid.

    Parameters
    ----------
    density_map : np.ndarray
        Original 2D density histogram.
    graph : nx.Graph
        Graph returned by :func:`build_graph`.
    ax : matplotlib Axes, optional
        Axes to draw on; a new figure is created when ``None``.
    cmap : str
        Colour-map for the density background.
    edge_cmap : str
        Colour-map used when *color_edges_by_weight* is True.
    node_color : str
        Colour for graph nodes.
    edge_color : str or None
        Fixed colour for all edges.  When ``None`` and
        *color_edges_by_weight* is True, edges are coloured by their
        ``weight`` attribute using *edge_cmap*.
    color_edges_by_weight : bool
        If True (default) and *edge_color* is None, colour each edge
        according to its ``weight`` and add a colour-bar.
    node_size : int
        Marker size for nodes.
    edge_width : float
        Line width for edges.
    title : str
        Plot title.
    save_path : str or Path or None
        If given, save the figure to this path.
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.figure

    # Background: density heatmap
    ax.imshow(density_map, cmap=cmap, origin="upper")

    # ── Prepare edge colours ─────────────────────────────────────────
    use_weight_colors = color_edges_by_weight and edge_color is None
    if use_weight_colors:
        weights = [d["weight"] for _, _, d in graph.edges(data=True)]
        if weights:
            norm = mcolors.Normalize(vmin=min(weights), vmax=max(weights))
            scalar_map = cm.ScalarMappable(norm=norm, cmap=edge_cmap)
            scalar_map.set_array([])
        else:
            use_weight_colors = False

    fallback_color = edge_color or "#76ff03"

    # ── Draw edges as polylines (follow the actual pixel path) ───────
    for u, v, data in graph.edges(data=True):
        if use_weight_colors:
            ec = scalar_map.to_rgba(data["weight"])
        else:
            ec = fallback_color

        path = data.get("path")
        if path:
            ys = [p[0] for p in path]  # row = y
            xs = [p[1] for p in path]  # col = x
            ax.plot(xs, ys, color=ec, linewidth=edge_width, alpha=0.9)
        else:
            pos_u = graph.nodes[u]["pos"]
            pos_v = graph.nodes[v]["pos"]
            ax.plot(
                [pos_u[0], pos_v[0]],
                [pos_u[1], pos_v[1]],
                color=ec,
                linewidth=edge_width,
                alpha=0.9,
            )

    # ── Colour-bar ───────────────────────────────────────────────────
    if use_weight_colors:
        cbar = fig.colorbar(scalar_map, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Edge weight (mean density)", fontsize=11)

    # ── Draw nodes ───────────────────────────────────────────────────
    pos = nx.get_node_attributes(graph, "pos")
    if pos:
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        ax.scatter(
            xs, ys,
            s=node_size, c=node_color, zorder=5,
            edgecolors="black", linewidths=0.5,
        )

    ax.set_title(title, fontsize=14)
    ax.axis("off")

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return ax
