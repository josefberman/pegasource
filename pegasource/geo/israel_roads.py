"""
Israel/Palestine road network as an offline NetworkX graph.

The graph is built from OpenStreetMap data (Geofabrik extract) and stored
as a compressed GraphML file inside the package's data directory.

The bundled graph covers:
  • Israel proper
  • Golan Heights
  • West Bank
  • Gaza Strip

Usage::

    from pegasource.geo.israel_roads import load_israel_graph, shortest_path, subgraph_bbox

    G = load_israel_graph()
    route = shortest_path(G, (31.7683, 35.2137), (32.0853, 34.7818))  # Jeru → TLV
"""

from __future__ import annotations

import gzip
import os
import pickle
import sys
from pathlib import Path
from typing import Sequence

import networkx as nx

# ---------------------------------------------------------------------------
# Data location
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).parent.parent / "data"
_GRAPH_PATH = _DATA_DIR / "israel_roads.pkl.gz"
_GRAPHML_PATH = _DATA_DIR / "israel_roads.graphml.gz"

# Cached singleton
_GRAPH_CACHE: nx.MultiDiGraph | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_israel_graph(
    cache_dir: str | Path | None = None,
    road_types: list[str] | None = None,
    force_reload: bool = False,
) -> nx.MultiDiGraph:
    """Load the bundled Israel road network as a NetworkX MultiDiGraph.

    The graph is loaded from a pre-processed file bundled with pegasource.
    Use :func:`_cli_download` (or ``pegasource-download-roads`` command) to
    download and pre-process the latest OSM data.

    Parameters
    ----------
    cache_dir : str or Path or None
        Alternative directory to search for the graph file.
    road_types : list[str] or None
        If provided, keep only edges whose ``highway`` attribute is in this
        list (e.g. ``["motorway", "trunk", "primary"]``).
    force_reload : bool
        Ignore the in-memory singleton and reload from disk.

    Returns
    -------
    nx.MultiDiGraph
        Nodes have ``lat``, ``lon`` attributes.
        Edges have ``highway``, ``name``, ``length_m``, ``oneway``, ``speed_kph``.

    Raises
    ------
    FileNotFoundError
        If no graph file is found. Run ``pegasource-download-roads`` first.
    """
    global _GRAPH_CACHE
    if _GRAPH_CACHE is not None and not force_reload:
        return _filter_road_types(_GRAPH_CACHE, road_types)

    search_dirs = [Path(cache_dir)] if cache_dir else []
    search_dirs.append(_DATA_DIR)

    for d in search_dirs:
        for name in ("israel_roads.pkl.gz", "israel_roads.graphml.gz"):
            p = d / name
            if p.exists():
                _GRAPH_CACHE = _load_graph(p)
                return _filter_road_types(_GRAPH_CACHE, road_types)

    raise FileNotFoundError(
        "Israel road graph not found. Please run:\n\n"
        "    pegasource-download-roads\n\n"
        "or:\n\n"
        "    python -m pegasource.geo.israel_roads\n\n"
        "to download and pre-process the OpenStreetMap data (~90 MB download)."
    )


def shortest_path(
    G: nx.MultiDiGraph,
    origin: tuple[float, float],
    destination: tuple[float, float],
    weight: str = "length_m",
) -> list[tuple[float, float]]:
    """Find the shortest road path between two geographic coordinates.

    Parameters
    ----------
    G : nx.MultiDiGraph
        Graph returned by :func:`load_israel_graph`.
    origin : tuple[float, float]
        ``(lat, lon)`` of the start point.
    destination : tuple[float, float]
        ``(lat, lon)`` of the end point.
    weight : str
        Edge attribute to minimise (``"length_m"`` for distance,
        ``"travel_time_s"`` for time).

    Returns
    -------
    list[tuple[float, float]]
        Ordered list of ``(lat, lon)`` waypoints along the route.
    """
    origin_node = _nearest_node(G, origin[0], origin[1])
    dest_node = _nearest_node(G, destination[0], destination[1])

    node_path = nx.shortest_path(G, origin_node, dest_node, weight=weight)
    return [(G.nodes[n]["lat"], G.nodes[n]["lon"]) for n in node_path]


def subgraph_bbox(
    G: nx.MultiDiGraph,
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
) -> nx.MultiDiGraph:
    """Extract the subgraph within a bounding box.

    Parameters
    ----------
    G : nx.MultiDiGraph
    min_lat, min_lon, max_lat, max_lon : float
        Bounding box in WGS-84 decimal degrees.

    Returns
    -------
    nx.MultiDiGraph
        Induced subgraph (view) — shares edge/node data with *G*.
    """
    nodes_in_bbox = [
        n
        for n, d in G.nodes(data=True)
        if min_lat <= d.get("lat", float("nan")) <= max_lat
        and min_lon <= d.get("lon", float("nan")) <= max_lon
    ]
    return G.subgraph(nodes_in_bbox)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nearest_node(G: nx.MultiDiGraph, lat: float, lon: float) -> int:
    """Return the node ID closest to (lat, lon) using Euclidean approximation."""
    best_node = None
    best_dist = float("inf")
    for n, d in G.nodes(data=True):
        dlat = d.get("lat", 0) - lat
        dlon = d.get("lon", 0) - lon
        dist = dlat * dlat + dlon * dlon
        if dist < best_dist:
            best_dist = dist
            best_node = n
    return best_node


def _filter_road_types(
    G: nx.MultiDiGraph,
    road_types: list[str] | None,
) -> nx.MultiDiGraph:
    if road_types is None:
        return G
    edges_to_keep = [
        (u, v, k)
        for u, v, k, d in G.edges(keys=True, data=True)
        if d.get("highway") in road_types
    ]
    return G.edge_subgraph(edges_to_keep)


def _load_graph(path: Path) -> nx.MultiDiGraph:
    """Load a graph from a pickle.gz or graphml.gz file."""
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as fh:
            raw = fh.read()
        if path.name.endswith(".pkl.gz"):
            return pickle.loads(raw)
        else:
            # GraphML
            import io
            return nx.read_graphml(io.BytesIO(raw))
    return nx.read_graphml(path)


# ---------------------------------------------------------------------------
# CLI / download entrypoint
# ---------------------------------------------------------------------------

def _cli_download() -> None:
    """Download and pre-process the Israel OSM road network.

    Invoked by the ``pegasource-download-roads`` console script.
    Requires ``osmium`` (``pip install osmium``) and an internet connection.
    """
    import urllib.request
    import tempfile

    OSM_URL = (
        "https://download.geofabrik.de/asia/israel-and-palestine-latest.osm.pbf"
    )

    print("pegasource — Israel road network downloader")
    print("=" * 60)
    print(f"Source : {OSM_URL}")
    print(f"Target : {_GRAPH_PATH}")
    print()

    try:
        import osmium  # type: ignore
    except ImportError:
        print("ERROR: osmium is required. Install with:\n  pip install osmium")
        sys.exit(1)

    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Download to temp file with progress
    with tempfile.NamedTemporaryFile(suffix=".osm.pbf", delete=False) as tmp:
        tmp_path = tmp.name

    print("Downloading OSM data (this may take a few minutes)...")
    _download_with_progress(OSM_URL, tmp_path)

    print("\nParsing OSM data and building graph...")
    G = _build_graph_from_pbf(tmp_path)

    print(f"Graph has {G.number_of_nodes():,} nodes and {G.number_of_edges():,} edges.")

    print(f"Saving graph to {_GRAPH_PATH} ...")
    with gzip.open(_GRAPH_PATH, "wb") as fh:
        pickle.dump(G, fh, protocol=pickle.HIGHEST_PROTOCOL)

    os.unlink(tmp_path)
    print("Done! You can now use load_israel_graph() offline.")


def _download_with_progress(url: str, dest: str) -> None:
    import urllib.request

    def reporthook(count, block_size, total_size):
        if total_size > 0:
            pct = count * block_size * 100 / total_size
            mb = count * block_size / 1e6
            total_mb = total_size / 1e6
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            print(f"\r  [{bar}] {pct:.1f}% — {mb:.1f}/{total_mb:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=reporthook)


def _build_graph_from_pbf(pbf_path: str) -> nx.MultiDiGraph:
    """Parse an OSM PBF file and return a road MultiDiGraph."""
    import osmium  # type: ignore

    _ROAD_TYPES = {
        "motorway", "motorway_link",
        "trunk", "trunk_link",
        "primary", "primary_link",
        "secondary", "secondary_link",
        "tertiary", "tertiary_link",
        "unclassified", "residential",
        "living_street", "road",
        "service",
    }

    # Speed defaults by highway type (km/h)
    _SPEEDS: dict[str, float] = {
        "motorway": 110, "motorway_link": 80,
        "trunk": 90, "trunk_link": 70,
        "primary": 80, "primary_link": 60,
        "secondary": 70, "secondary_link": 50,
        "tertiary": 60, "tertiary_link": 40,
        "unclassified": 50, "residential": 30,
        "living_street": 20, "road": 50, "service": 20,
    }

    class NodeCollector(osmium.SimpleHandler):  # type: ignore
        def __init__(self):
            super().__init__()
            self.nodes: dict[int, tuple[float, float]] = {}
            self.way_nodes: set[int] = set()

        def way(self, w):
            hw = w.tags.get("highway", "")
            if hw in _ROAD_TYPES:
                for n in w.nodes:
                    self.way_nodes.add(n.ref)

        def node(self, n):
            if n.id in self.way_nodes:
                self.nodes[n.id] = (n.location.lat, n.location.lon)

    class WayCollector(osmium.SimpleHandler):  # type: ignore
        def __init__(self, node_coords):
            super().__init__()
            self.node_coords = node_coords
            self.ways: list[dict] = []

        def way(self, w):
            hw = w.tags.get("highway", "")
            if hw not in _ROAD_TYPES:
                return
            node_refs = [n.ref for n in w.nodes if n.ref in self.node_coords]
            if len(node_refs) < 2:
                return
            self.ways.append({
                "nodes": node_refs,
                "highway": hw,
                "name": w.tags.get("name", w.tags.get("name:en", "")),
                "oneway": w.tags.get("oneway", "no") in ("yes", "1", "true"),
                "maxspeed": w.tags.get("maxspeed", ""),
            })

    # Two-pass: collect all node IDs used by roads, then their coords
    print("  Pass 1: collecting road way-node IDs...")
    nc = NodeCollector()
    nc.apply_file(pbf_path, locations=True)

    print(f"  Found {len(nc.nodes):,} road nodes.")
    print("  Pass 2: collecting ways...")
    wc = WayCollector(nc.nodes)
    wc.apply_file(pbf_path)
    print(f"  Found {len(wc.ways):,} road ways.")

    # Build graph
    import re as _re
    G = nx.MultiDiGraph()

    for nid, (lat, lon) in nc.nodes.items():
        G.add_node(nid, lat=lat, lon=lon)

    for way in wc.ways:
        hw = way["highway"]
        spd = _SPEEDS.get(hw, 50)
        raw_spd = way["maxspeed"]
        if raw_spd:
            m = _re.match(r"(\d+)", raw_spd)
            if m:
                spd = float(m.group(1))
                if "mph" in raw_spd:
                    spd *= 1.60934

        node_refs = way["nodes"]
        for u, v in zip(node_refs, node_refs[1:]):
            if u not in G or v not in G:
                continue
            lat1, lon1 = nc.nodes[u]
            lat2, lon2 = nc.nodes[v]
            # Haversine inline
            import math
            dφ = math.radians(lat2 - lat1)
            dλ = math.radians(lon2 - lon1)
            a = math.sin(dφ / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dλ / 2) ** 2
            length_m = 2 * 6_371_008.8 * math.asin(math.sqrt(a))
            travel_time_s = length_m / (spd / 3.6)

            attrs = {
                "highway": hw,
                "name": way["name"],
                "length_m": length_m,
                "speed_kph": spd,
                "travel_time_s": travel_time_s,
                "oneway": way["oneway"],
            }
            G.add_edge(u, v, **attrs)
            if not way["oneway"]:
                G.add_edge(v, u, **attrs)

    return G


if __name__ == "__main__":
    _cli_download()
