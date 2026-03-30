# Pegasource Comprehensive Guide

Welcome to the comprehensive guide for **Pegasource** — an offline-capable Python toolkit for PCAP analysis, geographic functions, time-series forecasting, optional **hardware-inventory clustering**, and **path estimation** from mixed sensor observations.

This guide provides an in-depth look at every module, its purpose, and examples of how to use its functions.

---

## Table of Contents

1. [Installation](#installation)
2. [Module: PCAP (`pegasource.pcap`)](pcap.md)
   - [Reading & Summarizing](pcap.md#pegasource.pcap.read_pcap)
   - [Statistics](pcap.md#pegasource.pcap.protocol_distribution)
   - [Pattern & Anomaly Detection](pcap.md#pegasource.pcap.detect_port_scan)
   - [Reporting](pcap.md#pegasource.pcap.generate_report)
3. [Module: Geography (`pegasource.geo`)](geo.md)
   - [Distance Calculations](geo.md#pegasource.geo.haversine)
   - [Coordinate Projections](geo.md#pegasource.geo.wgs84_to_itm)
   - [Road Vectorizer](geo.md#pegasource.geo.build_graph)
   - [Israel Road Network](geo.md#pegasource.geo.load_israel_graph)
4. [Module: Time-Series (`pegasource.timeseries`)](timeseries.md)
   - [Auto Forecaster (SARIMAX & Fallbacks)](timeseries.md#pegasource.timeseries.AutoForecaster)
   - [Utilities](timeseries.md#pegasource.timeseries.detect_seasonality)
5. [Module: Dataset clustering (`pegasource.dataset_clustering`)](dataset_clustering.md)
   - [Core pipeline & embeddings](dataset_clustering.md#core-pipeline-python-api)
   - [CLI & interactive server](dataset_clustering.md#command-line-batch-run)
   - [iFixit catalog & static viz](dataset_clustering.md#interactive-visualization-server)
6. [Module: Path estimation (`pegasource.path_estimation`)](path_estimation.md)
   - [Metrics](path_estimation.md#trajectory-metrics-no-ground-truth-pipeline)
   - [Evaluation with ground truth](path_estimation.md#evaluation-with-observations-and-true-path)
   - [Observations only](path_estimation.md#observations-only-no-ground-truth-file)

---

## Installation

As Pegasource is designed with an offline-first approach, it relies heavily on local computations rather than external APIs for core functions (except for an initial optional download of OSM road data).

```bash
# Basic installation
pip install pegasource

# Development installation (tests + docs)
pip install -e ".[dev]"

# Optional feature sets
pip install -e ".[clustering]"       # dataset clustering: embeddings, FastAPI viz, Excel export
pip install -e ".[path_estimation]"  # path estimation: torch, filterpy, osmnx, GNN, …
```

> **Note**: For live packet capture via `scapy`, root privileges may be required on some systems. Reading existing `.pcap` or `.pcapng` files works without root.

---

## Module: PCAP (`pegasource.pcap`)

The `pcap` module is dedicated to reading network capture files and extracting meaningful insights, statistics, and anomalies.

### Reading & Summarizing

Extract packets and format them into readable tabular data.

```python
from pegasource.pcap import read_pcap, packet_summary

# Read a PCAP file containing scapy packets
packets = read_pcap("capture.pcap")

# Convert the raw packets into a pandas DataFrame for analysis
df = packet_summary(packets)
print(df.head())
```

### Statistics

Generate insights into network protocols and traffic sources.

```python
from pegasource.pcap import protocol_distribution, top_talkers, conversation_table

# Get the count of occurrences per protocol
proto_counts = protocol_distribution(packets)

# Find IP addresses making the most requests
talkers = top_talkers(packets, top_n=10)

# Create a tabular summary of IP-to-IP conversations
conversations = conversation_table(packets)
```

### Pattern & Anomaly Detection

Built-in logic to flag suspicious network behavior.

```python
from pegasource.pcap import detect_port_scan, detect_beaconing, detect_dns_anomalies

# Port Scan: Look for IPs attempting connections on multiple ports quickly
scans = detect_port_scan(packets, threshold=20)

# Beaconing: Detect repeated, periodic outbound connections (e.g. C2 servers)
beacons = detect_beaconing(packets, min_occurrences=5)

# DNS Anomalies: Identify high volumes of NXDOMAINs, unusually long TxT records, etc.
dns_anomalies = detect_dns_anomalies(packets)
```

### Reporting

Automatically bundle packet analysis into a single JSON report.

```python
from pegasource.pcap import generate_report

# Run all analyses and dump to JSON
report = generate_report(packets, output_path="report.json")
```

---

## Module: Geography (`pegasource.geo`)

A toolkit for spatial analysis, graph-based routing, and coordinate projection tailored for both general use and specific Israeli geographical contexts.

### Distance Calculations

Compute the distance between WGS84 coordinates.

```python
from pegasource.geo import haversine, vincenty, bearing

lat1, lon1 = 31.7683, 35.2137  # Jerusalem
lat2, lon2 = 32.0853, 34.7818  # Tel Aviv

# Great-circle distance
dist_hav = haversine(lat1, lon1, lat2, lon2)
print(f"Haversine: {dist_hav / 1000:.2f} km")

# Ellipsoid distance (more accurate)
dist_vin = vincenty(lat1, lon1, lat2, lon2)
print(f"Vincenty: {dist_vin / 1000:.2f} km")

# Initial bearing from Point A to Point B
b = bearing(lat1, lon1, lat2, lon2)
print(f"Bearing: {b:.1f} degrees")
```

### Coordinate Projections

Perform localized transformations and meter-based offsets.

```python
from pegasource.geo import wgs84_to_itm, itm_to_wgs84, wgs84_to_utm, meters_offset

# Convert to Israeli Transverse Mercator (EPSG:2039)
easting, northing = wgs84_to_itm(31.7683, 35.2137)

# Convert back to Latitude/Longitude (WGS84)
lat, lon = itm_to_wgs84(easting, northing)

# Get the UTM zone coordinates automatically
utm_x, utm_y, zone = wgs84_to_utm(31.7683, 35.2137)

# Move 500 meters North and 200 meters East from a starting lat/lon
new_lat, new_lon = meters_offset(31.7683, 35.2137, d_north=500, d_east=200)
```

### Road Vectorizer

Reconstruct road graphs from 2D movement density maps. Ideal for reconstructing offline routes without access to traditional mapping APIs.

```python
import numpy as np
from pegasource.geo import build_graph, compute_road_coverage, plot_graph_overlay

density_map = np.load("density_map.npy")

# Convert the histogram to a NetworkX Graph
G = build_graph(density_map, threshold=0.3, prune_length=5)

# Compare an estimated partial graph against a known full graph
coverage = compute_road_coverage(full_graph=G, partial_graph=G_estimated, tolerance=2)

# Visualize the graph over the original density image
plot_graph_overlay(density_map, G)
```

### Israel Road Network

Offline routing utilizing a pre-processed mapping graph for Israel and surroundings. Before using this for the first time, you must download the offline graph.

```bash
# Run this once on your machine (~90 MB)
pegasource-download-roads
```

Then use the API:

```python
from pegasource.geo import load_israel_graph, shortest_path, subgraph_bbox

# Load the offline routing graph
G = load_israel_graph()

# Compute shortest path
start_coord = (31.7683, 35.2137)
end_coord = (32.0853, 34.7818)
route = shortest_path(G, start_coord, end_coord)
print(f"Calculated route with {len(route)} nodes.")

# Extract a small bounding box graph for localized processing
local_graph = subgraph_bbox(G, min_lat=31.7, min_lon=35.1, max_lat=31.8, max_lon=35.3)
```

---

## Module: Time-Series (`pegasource.timeseries`)

Automated, statistically robust time-series forecasting supporting built-in fallback models when standard approaches fail to converge.

### Auto Forecaster

The `AutoForecaster` will detect seasonality, automatically search for the best SARIMAX parameters, and default to a Linear Trend model if SARIMAX encounters issues.

```python
import numpy as np
import pandas as pd
from pegasource.timeseries import AutoForecaster

# Generate synthetic time-series data
data = np.sin(np.linspace(0, 8 * np.pi, 96)) + np.random.randn(96) * 0.1

fc = AutoForecaster()
fc.fit(data)

# Predict the next 12 steps
predictions = fc.predict(steps=12)

# Output AIC and chosen model parameters
print(fc.diagnostics())

# Visualize the model's performance
fc.plot(steps=12)
```

**Using Exogenous Variables (External Drivers):**

```python
exog_data = pd.DataFrame({"temperature": [20, 22, 21, 19, 18], "holiday": [0, 0, 1, 0, 0]})
exog_future = pd.DataFrame({"temperature": [17, 16], "holiday": [0, 1]})

fc.fit(data, exog=exog_data)
predictions = fc.predict(steps=2, exog=exog_future)
```

### Utilities

Helper functions for standalone time-series operations.

```python
from pegasource.timeseries import detect_seasonality, train_test_split_ts, rmse

# Automatically find dominant seasonality period via ACF
period = detect_seasonality(data)

# temporal split (no shuffling)
train, test = train_test_split_ts(data, test_size=0.2)

# Calculate Root Mean Squared Error
error = rmse(y_true, y_pred)
```

---

## Module: Dataset clustering (`pegasource.dataset_clustering`)

Cluster dirty tabular inventory by **embedding** each row’s text and **agglomerative clustering** in cosine space. Optional **FastAPI** server with a browser UI for threshold tuning, uploads, and **Excel** export. Full details: [dataset_clustering.md](dataset_clustering.md).

### Core pipeline

```python
from pathlib import Path
from pegasource.dataset_clustering import (
    load_data,
    build_text_representations,
    generate_embeddings,
    cluster_embeddings,
)

df = load_data(str(Path("inventory.csv")))
texts = build_text_representations(df)
emb = generate_embeddings(
    texts,
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    batch_size=512,
    device="cpu",
)
labels = cluster_embeddings(emb, threshold=0.3)
df["cluster_id"] = labels
```

Install **`[clustering]`** for **sentence-transformers** and the HTTP stack. Very large tables use a **two-phase** (KMeans pre-grouping + agglomerative) strategy automatically.

### CLI and server

```bash
# Batch CSV → clustered CSV
python -m pegasource.dataset_clustering.cluster_hardware --input data.csv --output out.csv --threshold 0.3

# Interactive UI (requires [clustering])
pegasource-cluster-viz --port 8001
```

Cache **iFixit** device names for richer cluster labels: `python -m pegasource.dataset_clustering.fetch_ifixit_devices`.

---

## Module: Path estimation (`pegasource.path_estimation`)

Reconstruct paths from **observations** and optional **ground-truth** trajectories: graph map-matching (Dijkstra, A*, HMM), filters (KF, EKF, UKF, particle), and optional LSTM / Transformer / GNN (supervised). Full details: [path_estimation.md](path_estimation.md).

### Metrics (lightweight import)

```python
import numpy as np
from pegasource.path_estimation.metrics import compute_all_metrics

true_xy = np.array([[0.0, 0.0], [10.0, 0.0]])
est_xy = true_xy + 0.2
print(compute_all_metrics(true_xy, est_xy))
```

### Evaluation with a true path file

```python
from pathlib import Path
from pegasource.path_estimation.evaluate import run_evaluation

run_evaluation(
    observations_csv=Path("run_1_observations.csv"),
    true_path_csv=Path("run_1_true_path.csv"),
    output_dir=Path("out"),
    methods=["dijkstra", "hmm", "kf"],
    plot=True,
)
```

### Observations only

Use **`estimate_paths_only`** when no ground-truth CSV exists (methods **lstm** / **transformer** / **gnn** are disabled; use `run_evaluation` with labels for those).

```python
from pathlib import Path
from pegasource.path_estimation import estimate_paths_only
from pegasource.path_estimation.graph_utils import get_projected_graph

results = estimate_paths_only(
    Path("observations.csv"),
    get_projected_graph(),
    methods=["dijkstra", "kf", "ekf"],
    output_dir=Path("out_no_gt"),
    plot=True,
)
```

Install **`[path_estimation]`** for the full estimator stack.

---

*This guide aims to cover all standard use-cases. For lower-level functionality, consult the docstrings within the respective python modules.*
