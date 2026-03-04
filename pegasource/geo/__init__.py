"""
pegasource.geo — Geographic utilities: distance, projection, road vectorizer, Israel roads.

Quick start::

    from pegasource.geo import haversine, wgs84_to_itm, load_israel_graph

    dist_m = haversine(31.7683, 35.2137, 32.0853, 34.7818)   # Jerusalem → TLV
    e, n   = wgs84_to_itm(31.7683, 35.2137)
    G      = load_israel_graph()
"""

from .distance import haversine, vincenty, bearing
from .projection import wgs84_to_itm, itm_to_wgs84, wgs84_to_utm, meters_offset
from .vectorizer import build_graph, compute_road_coverage, plot_graph_overlay
from .israel_roads import load_israel_graph, shortest_path, subgraph_bbox

__all__ = [
    # distance
    "haversine",
    "vincenty",
    "bearing",
    # projection
    "wgs84_to_itm",
    "itm_to_wgs84",
    "wgs84_to_utm",
    "meters_offset",
    # vectorizer
    "build_graph",
    "compute_road_coverage",
    "plot_graph_overlay",
    # israel roads
    "load_israel_graph",
    "shortest_path",
    "subgraph_bbox",
]
