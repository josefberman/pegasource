"""
Graph construction from a skeletonised road map.

The pipeline:
    1. Label each skeleton pixel by its 8-connected neighbour count.
    2. Mark *nodes* — pixels with ≠ 2 neighbours (junctions & endpoints).
    3. Trace skeleton paths between every pair of adjacent nodes.
    4. Build and return a weighted ``networkx.Graph``.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import networkx as nx
import numpy as np
from scipy import ndimage

from ._rv_preprocessing import binarize, skeletonize_map

# 8-connectivity structuring element
_STRUCT_8 = np.ones((3, 3), dtype=int)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _neighbour_count(skeleton: np.ndarray) -> np.ndarray:
    """Return an array where each skeleton pixel holds its 8-neighbour count."""
    skel = skeleton.astype(np.uint8)
    count = ndimage.convolve(skel, _STRUCT_8, mode="constant", cval=0)
    # subtract the pixel itself
    count = count - skel
    # zero out non-skeleton pixels
    count[~skeleton] = 0
    return count


def find_nodes(skeleton: np.ndarray) -> list[tuple[int, int]]:
    """Identify junction and endpoint pixels on *skeleton*.

    A pixel is a *node* if it has **≠ 2** skeleton neighbours:
    * 1 neighbour → endpoint
    * ≥ 3 neighbours → junction / intersection
    * 0 neighbours → isolated pixel (also treated as a node)

    Returns a list of ``(row, col)`` coordinates.
    """
    nbr = _neighbour_count(skeleton)
    # Nodes: skeleton pixels that are NOT simple chain links (≠ 2 neighbours)
    node_mask = skeleton & (nbr != 2)
    rows, cols = np.nonzero(node_mask)
    return list(zip(rows.tolist(), cols.tolist()))


def _neighbours_of(r: int, c: int, shape: tuple[int, int]):
    """Yield valid 8-connected neighbours of (r, c)."""
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < shape[0] and 0 <= nc < shape[1]:
                yield nr, nc


def trace_edges(
    skeleton: np.ndarray,
    nodes: list[tuple[int, int]],
    density_map: np.ndarray,
) -> list[dict[str, Any]]:
    """Walk the skeleton from every node and trace edges to adjacent nodes.

    Each edge record contains:
    * ``u``, ``v`` — node coordinates ``(row, col)``
    * ``weight`` — mean density along the path
    * ``max_density`` — maximum density along the path
    * ``length`` — number of pixels in the path
    * ``path`` — ordered list of ``(row, col)`` for every pixel

    Duplicate edges (u–v and v–u) are naturally deduplicated because we
    mark visited pixels during traversal.
    """
    node_set = set(nodes)
    shape = skeleton.shape
    visited_edges: set[frozenset] = set()
    edges: list[dict[str, Any]] = []

    # For every node, try to walk each of its skeleton neighbours
    for node in nodes:
        r0, c0 = node
        for nr, nc in _neighbours_of(r0, c0, shape):
            if not skeleton[nr, nc]:
                continue

            # Walk along the skeleton until we hit another node
            path = [node, (nr, nc)]
            prev = node
            cur = (nr, nc)

            while cur not in node_set:
                # Find the next skeleton pixel (not the one we came from)
                moved = False
                for nnr, nnc in _neighbours_of(cur[0], cur[1], shape):
                    if (nnr, nnc) != prev and skeleton[nnr, nnc]:
                        prev = cur
                        cur = (nnr, nnc)
                        path.append(cur)
                        moved = True
                        break
                if not moved:
                    # Dead-end that isn't marked as a node (shouldn't happen
                    # normally, but be safe)
                    break

            # `cur` should now be another node (or we hit a dead end)
            if cur not in node_set:
                continue

            edge_key = frozenset((node, cur))
            if edge_key in visited_edges:
                continue
            visited_edges.add(edge_key)

            densities = np.array([density_map[r, c] for r, c in path])
            edges.append(
                {
                    "u": node,
                    "v": cur,
                    "weight": float(densities.mean()),
                    "max_density": float(densities.max()),
                    "length": len(path),
                    "path": path,
                }
            )

    return edges


def _cluster_junction_nodes(
    G: nx.Graph,
    density_map: np.ndarray,
    merge_distance: int,
) -> nx.Graph:
    """Collapse clusters of spatially-close nodes into single representatives.

    Uses hierarchical clustering on the **Euclidean** pixel coordinates so
    that a cluster can never span more than *merge_distance* pixels,
    regardless of how many short edges chain together in the graph.
    """
    from scipy.cluster.hierarchy import fcluster, linkage

    nodes = list(G.nodes())
    if len(nodes) <= 1:
        return G

    coords = np.array(nodes, dtype=float)  # (N, 2) — row, col

    # Single-linkage hierarchical clustering with Euclidean distance
    Z = linkage(coords, method="single", metric="euclidean")
    labels = fcluster(Z, t=merge_distance, criterion="distance")

    # Group nodes by cluster label
    clusters: dict[int, list[tuple]] = {}
    for node, label in zip(nodes, labels):
        clusters.setdefault(label, []).append(node)

    # Map every node to its cluster representative (closest to centroid)
    node_to_rep: dict[tuple, tuple] = {}
    for members in clusters.values():
        cr = int(round(np.mean([m[0] for m in members])))
        cc = int(round(np.mean([m[1] for m in members])))
        best = min(members, key=lambda m: (m[0] - cr) ** 2 + (m[1] - cc) ** 2)
        for m in members:
            node_to_rep[m] = best

    # Build new graph
    H = nx.Graph()
    reps = set(node_to_rep.values())
    for rep in reps:
        H.add_node(rep, pos=(rep[1], rep[0]))

    # Re-add edges, skipping intra-cluster ones; keep longest path for dupes
    for u, v, data in G.edges(data=True):
        ru, rv = node_to_rep[u], node_to_rep[v]
        if ru == rv:
            continue

        if H.has_edge(ru, rv):
            if data["length"] > H.edges[ru, rv]["length"]:
                H.edges[ru, rv].update(data)
        else:
            H.add_edge(ru, rv, **data)

    return H


def _merge_paths(path_ab: list, path_bc: list) -> list:
    """Join two pixel paths that share an endpoint (the middle node)."""
    if path_ab[-1] == path_bc[0]:
        return path_ab + path_bc[1:]
    if path_ab[-1] == path_bc[-1]:
        return path_ab + path_bc[-2::-1]
    if path_ab[0] == path_bc[0]:
        return path_ab[::-1] + path_bc[1:]
    if path_ab[0] == path_bc[-1]:
        return path_ab[::-1] + path_bc[-2::-1]
    # Fallback: concatenate
    return path_ab + path_bc


def _contract_degree2_chains(
    G: nx.Graph,
    density_map: np.ndarray,
) -> nx.Graph:
    """Remove degree-2 pass-through nodes, merging their two edges into one."""
    changed = True
    while changed:
        changed = False
        for node in list(G.nodes()):
            if node not in G or G.degree(node) != 2:
                continue
            neighbours = list(G.neighbors(node))
            if len(neighbours) != 2:
                continue
            a, b = neighbours

            path_a = G.edges[node, a].get("path", [])
            path_b = G.edges[node, b].get("path", [])
            new_path = _merge_paths(path_a, path_b)
            densities = np.array([density_map[r, c] for r, c in new_path])

            G.remove_node(node)
            G.add_edge(
                a, b,
                weight=float(densities.mean()),
                max_density=float(densities.max()),
                length=len(new_path),
                path=new_path,
            )
            # Update pos for surviving nodes
            if a not in nx.get_node_attributes(G, "pos"):
                G.nodes[a]["pos"] = (a[1], a[0])
            if b not in nx.get_node_attributes(G, "pos"):
                G.nodes[b]["pos"] = (b[1], b[0])
            changed = True

    return G


def simplify_graph(
    G: nx.Graph,
    density_map: np.ndarray,
    merge_distance: int = 5,
) -> nx.Graph:
    """Simplify the graph in two phases:

    1. **Cluster** spatially-close nodes (junctions, nearby branch points)
       into single representative nodes.
    2. **Contract** remaining degree-2 chain nodes into longer edges.

    Parameters
    ----------
    G : nx.Graph
        Raw graph from edge tracing.
    density_map : np.ndarray
        Original density map for recomputing weights.
    merge_distance : int
        Maximum edge length (pixels) for clustering nearby nodes.

    Returns
    -------
    nx.Graph
    """
    G = _cluster_junction_nodes(G, density_map, merge_distance)
    G = _contract_degree2_chains(G, density_map)
    return G


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_graph(
    density_map: np.ndarray,
    threshold: float | None = None,
    dilate_radius: int = 0,
    prune_length: int = 0,
    merge_distance: int = 5,
) -> nx.Graph:
    """Convert a 2D density histogram into a weighted undirected graph.

    Parameters
    ----------
    density_map : np.ndarray
        2D numpy array of non-negative density values.
    threshold : float or None
        Density value above which a pixel is considered part of a road.
        ``None`` → automatic (Otsu).
    dilate_radius : int
        Optional dilation before skeletonization (helps connect fragmented
        roads).
    prune_length : int
        After merging, drop dead-end spur edges (where at least one
        endpoint has degree 1) with ``length <= prune_length``.
    merge_distance : int
        Iteratively merge degree-2 nodes connected by edges shorter than
        this many pixels.  Set to 0 to disable.

    Returns
    -------
    nx.Graph
        Undirected graph whose nodes carry ``pos = (col, row)`` (x, y for
        plotting) and whose edges carry ``weight`` (mean density),
        ``max_density``, ``length``, and ``path``.
    """
    # 1. Preprocess
    mask = binarize(density_map, threshold=threshold)
    skeleton = skeletonize_map(mask, dilate_radius=dilate_radius)

    # 2. Detect nodes
    nodes = find_nodes(skeleton)

    # Handle edge case: skeleton exists but no nodes were found
    # (e.g. a single closed loop with no junctions)
    if not nodes and skeleton.any():
        r, c = np.argwhere(skeleton)[0]
        nodes = [(int(r), int(c))]

    # 3. Trace edges — keep ALL edges initially
    raw_edges = trace_edges(skeleton, nodes, density_map)

    # 4. Build NetworkX graph (no pruning yet)
    G = nx.Graph()
    for node in nodes:
        G.add_node(node, pos=(node[1], node[0]))

    for edge in raw_edges:
        G.add_edge(
            edge["u"],
            edge["v"],
            weight=edge["weight"],
            max_density=edge["max_density"],
            length=edge["length"],
            path=edge["path"],
        )

    # Remove isolated nodes (no edges at all)
    isolated = [n for n in G.nodes() if G.degree(n) == 0]
    G.remove_nodes_from(isolated)

    # 5. Simplify: cluster nearby nodes + contract degree-2 chains
    if merge_distance > 0:
        G = simplify_graph(G, density_map, merge_distance=merge_distance)

    # 6. Prune dead-end spurs AFTER simplification
    #    Only remove short edges where at least one endpoint is a dead-end
    #    (degree 1), so we never break through-roads
    if prune_length > 0:
        changed = True
        while changed:
            changed = False
            for u, v, data in list(G.edges(data=True)):
                if u not in G or v not in G:
                    continue
                if data["length"] > prune_length:
                    continue
                du, dv = G.degree(u), G.degree(v)
                if du == 1 or dv == 1:
                    G.remove_edge(u, v)
                    changed = True
            # Clean up newly isolated nodes
            isolated = [n for n in G.nodes() if G.degree(n) == 0]
            G.remove_nodes_from(isolated)

    return G


def compute_road_coverage(
    full_graph: nx.Graph,
    partial_graph: nx.Graph,
    tolerance: int = 2,
) -> dict:
    """Compute what fraction of a full road network is present in a partial one.

    For each edge in *full_graph*, the function checks how many of its
    path pixels lie within *tolerance* pixels of any path pixel in
    *partial_graph*.  The overall coverage is the length-weighted
    fraction of matched road.

    Parameters
    ----------
    full_graph : nx.Graph
        The reference graph (built from the full density map).
        Edges must carry a ``path`` attribute (list of ``(row, col)``).
    partial_graph : nx.Graph
        A graph built from a partial density map (some roads removed).
        Edges must carry a ``path`` attribute.
    tolerance : int
        A full-graph pixel is considered "covered" if any partial-graph
        pixel is within this many pixels (Chebyshev / L∞ distance).
        Use 0 for exact pixel match, 1–3 for fuzzy matching that
        tolerates slight skeleton shifts between the two maps.

    Returns
    -------
    dict
        ``"coverage"``  — float in [0, 1], overall fraction of road
        length present in the partial graph.

        ``"edges"``     — list of dicts, one per edge, each containing:
        ``"u"``, ``"v"``, ``"length"``, ``"covered_pixels"``,
        ``"edge_coverage"`` (fraction for that edge).
    """
    # Build a set of all path pixels from the partial graph
    # When tolerance > 0, dilate the set by adding neighbouring pixels
    partial_pixels: set[tuple[int, int]] = set()
    for _, _, data in partial_graph.edges(data=True):
        for r, c in data.get("path", []):
            for dr in range(-tolerance, tolerance + 1):
                for dc in range(-tolerance, tolerance + 1):
                    partial_pixels.add((r + dr, c + dc))

    total_pixels = 0
    covered_pixels = 0
    edge_details = []

    for u, v, data in full_graph.edges(data=True):
        path = data.get("path", [])
        n_total = len(path)
        if n_total == 0:
            continue

        n_covered = sum(1 for p in path if p in partial_pixels)

        total_pixels += n_total
        covered_pixels += n_covered

        edge_details.append({
            "u": u,
            "v": v,
            "length": n_total,
            "covered_pixels": n_covered,
            "edge_coverage": n_covered / n_total if n_total > 0 else 0.0,
        })

    overall = covered_pixels / total_pixels if total_pixels > 0 else 0.0

    return {
        "coverage": overall,
        "edges": edge_details,
    }

