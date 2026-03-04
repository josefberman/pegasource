"""
Road vectorizer: convert 2D density histograms into weighted NetworkX graphs.

This is a direct integration of https://github.com/josefberman/RoadVectorizer
into pegasource, exposing the same public API:

    build_graph(density_map, **kwargs) → nx.Graph
    compute_road_coverage(full_graph, partial_graph, tolerance=2) → dict
    plot_graph_overlay(density_map, graph, **kwargs) → Axes
"""

# Re-export all three public symbols from the vendored submodules so that
# users can do:
#   from pegasource.geo import build_graph
# or:
#   from pegasource.geo.vectorizer import build_graph

from ._rv_graph_builder import build_graph, compute_road_coverage
from ._rv_visualization import plot_graph_overlay

__all__ = ["build_graph", "compute_road_coverage", "plot_graph_overlay"]
