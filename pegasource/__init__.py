"""
pegasource — Offline-capable Python toolkit.

Modules
-------
pegasource.pcap                  PCAP reader, statistics, and anomaly/pattern detection
pegasource.geo                   Geographic utilities, coordinate transforms, road graphs
pegasource.timeseries            Simple automatic time-series forecasting
pegasource.dataset_clustering    Hardware inventory embedding + clustering (optional deps)
pegasource.path_estimation       GPS/cellular path reconstruction (filters, graph, NN; optional torch stack)
"""

__version__ = "0.1.0"
__all__ = ["pcap", "geo", "timeseries", "dataset_clustering", "path_estimation"]
