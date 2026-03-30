# Pegasource

> **Offline-capable Python toolkit** — PCAP analysis, geographic functions, time-series forecasting, optional hardware-inventory clustering, and path estimation (GPS / map-matching / filters / NN).

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Modules

| Module | Description |
|--------|-------------|
| `pegasource.pcap` | PCAP reader, statistics, anomaly & pattern detection |
| `pegasource.geo` | Distance, coordinate transforms, road vectorizer, Israel road network |
| `pegasource.timeseries` | Automatic time-series forecasting (SARIMAX + fallback) |
| `pegasource.dataset_clustering` | Dirty hardware CSV → embeddings, hierarchical clustering, optional FastAPI + browser UI |
| `pegasource.path_estimation` | Trajectory reconstruction: graph map-matching, KF/EKF/UKF/particle, LSTM/Transformer/GNN (optional `[path_estimation]`) |

---

## Installation

```bash
pip install -e ".[dev]"              # development install (tests + docs)
pip install -e ".[clustering]"       # adds embedding server deps (sentence-transformers, FastAPI, …)
pip install -e ".[path_estimation]"  # torch, torch-geometric, filterpy, osmnx, contextily, …
# or
pip install pegasource               # once published
```

Install **`[clustering]`** when you use `pegasource.dataset_clustering` for full pipelines (embeddings, HTTP server, Excel export). Core imports such as `load_data` and `cluster_embeddings` need scikit-learn and pandas (already in the base package); `generate_embeddings` needs **sentence-transformers** (included in `[clustering]`).

Install **`[path_estimation]`** for `run_evaluation`, neural estimators, and Kalman/particle filters. Submodules such as `pegasource.path_estimation.metrics` work with the base install; `from pegasource.path_estimation import run_evaluation` loads the full stack.

> **scapy** requires root privileges to capture live traffic, but reading PCAP files works without root.

---

## Quick Start

### PCAP Analysis

```python
from pegasource.pcap import read_pcap, generate_report

packets = read_pcap("capture.pcap")
report  = generate_report(packets, output_path="report.json")

# Individual detectors
from pegasource.pcap import detect_port_scan, detect_beaconing, detect_dns_anomalies

scans    = detect_port_scan(packets, threshold=20)
beacons  = detect_beaconing(packets, min_occurrences=5)
dns_anom = detect_dns_anomalies(packets)
```

### Geographic Functions

```python
from pegasource.geo import haversine, wgs84_to_itm, load_israel_graph, shortest_path

# Distances
dist_m = haversine(31.7683, 35.2137, 32.0853, 34.7818)   # Jerusalem → TLV ≈ 54 km
print(f"Distance: {dist_m / 1000:.1f} km")

# Coordinate conversion
easting, northing = wgs84_to_itm(31.7683, 35.2137)   # WGS84 → ITM (EPSG:2039)

# Israel road network
G = load_israel_graph()                               # loads pre-processed graph
route = shortest_path(G, (31.7683, 35.2137), (32.0853, 34.7818))
print("First waypoint:", route[0])
```

#### Downloading the road graph (one-time, ~90 MB)

```bash
pegasource-download-roads
# or
python -m pegasource.geo.israel_roads
```

This downloads the Geofabrik `israel-and-palestine-latest.osm.pbf` and saves a
pre-processed `israel_roads.pkl.gz` to `pegasource/data/`.

### Road Vectorizer (from density maps)

```python
import numpy as np
from pegasource.geo import build_graph, plot_graph_overlay

density = np.load("my_density_map.npy")
G = build_graph(density, threshold=0.3, prune_length=5)
plot_graph_overlay(density, G)
```

API mirrors [josefberman/RoadVectorizer](https://github.com/josefberman/RoadVectorizer):

| Function | Description |
|----------|-------------|
| `build_graph(density_map, **kwargs)` | Convert 2D density histogram → `nx.Graph` |
| `compute_road_coverage(full_graph, partial_graph, tolerance=2)` | Coverage fraction |
| `plot_graph_overlay(density_map, graph, **kwargs)` | Matplotlib overlay |

### Time-Series Prediction

```python
import numpy as np
from pegasource.timeseries import AutoForecaster

# Univariate
y = np.sin(np.linspace(0, 8 * np.pi, 96)) + np.random.randn(96) * 0.1
fc = AutoForecaster()
fc.fit(y)
pred = fc.predict(steps=12)
print(fc.diagnostics())
fc.plot(steps=12)

# With exogenous variables
import pandas as pd
exog = pd.DataFrame({"temperature": ..., "holiday": ...})    # shape (n, k)
fc2 = AutoForecaster()
fc2.fit(y, exog=exog)
pred2 = fc2.predict(steps=6, exog=exog_future)
```

`AutoForecaster` automatically:
1. Detects the dominant seasonal period via ACF
2. Tries multiple SARIMAX configurations and selects by AIC
3. Falls back to OLS linear trend + seasonal dummies if SARIMAX fails

### Dataset clustering (hardware inventory)

Cluster messy tabular inventory (e.g. hardware types/models) using sentence embeddings and agglomerative clustering. A bundled sample CSV lives under `pegasource/dataset_clustering/data/`.

```python
from pegasource.dataset_clustering import (
    load_data,
    build_text_representations,
    generate_embeddings,
    cluster_embeddings,
)

df = load_data("path/to/data.csv")           # use any CSV; CLI defaults to bundled sample if omitted
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

**CLI — batch clustering to CSV**

```bash
python -m pegasource.dataset_clustering.cluster_hardware \
  --input path/to.csv --output clustered.csv --threshold 0.3
```

**Interactive visualization server** (requires `[clustering]`):

```bash
pegasource-cluster-viz --port 8001
# or
python -m pegasource.dataset_clustering.server --port 8001
```

Serves the bundled `cluster_viz` frontend, loads the default dataset (or `--data`), and computes embeddings in the background. For richer device matching in the UI, download the iFixit device list once:

```bash
python -m pegasource.dataset_clustering.fetch_ifixit_devices
```

**Regenerate bundled sample data** (optional, long run):

```bash
python -m pegasource.dataset_clustering.dataset_generator
```

**Static viz JSON** from an existing `clustered_output.csv`:

```bash
python -m pegasource.dataset_clustering.prepare_viz_data
```

### Path estimation (trajectories)

Evaluate reconstruction methods on observation CSVs vs a 1 Hz ground-truth path (requires **`[path_estimation]`** for full method set):

```bash
pip install -e ".[path_estimation]"
pegasource-path-estimation --observations run_1_observations.csv --true-path run_1_true_path.csv --output-dir out/
# or
python -m pegasource.path_estimation --observations ... --true-path ... --output-dir out/
```

```python
from pegasource.path_estimation.evaluate import run_evaluation
from pathlib import Path

summary = run_evaluation(
    observations_csv=Path("run_1_observations.csv"),
    true_path_csv=Path("run_1_true_path.csv"),
    output_dir=Path("out"),
    methods=["dijkstra", "hmm", "kf"],
)
```

**Observations only (no ground-truth CSV)** — use `estimate_paths_only` when you do not have a true path file. It builds an internal time grid from the observation span and returns per-method `EstimationResult` values (or `{"error": ...}`). Methods **lstm**, **transformer**, and **gnn** need labels; use `run_evaluation` / `evaluate_path_estimation` with a true path instead.

```python
from pathlib import Path

from pegasource.path_estimation import estimate_paths_only
from pegasource.path_estimation.graph_utils import get_projected_graph

obs = Path("run_1_observations.csv")
G = get_projected_graph()  # same OSM walk graph as evaluation uses

results = estimate_paths_only(
    obs,
    G,
    methods=["dijkstra", "hmm", "kf", "ekf"],
    output_hz=1.0,
    output_dir=Path("out_no_gt"),
    plot=True,   # writes out_no_gt/figures/<method>_path_enu.png
)
# results["kf"] is an EstimationResult with times_s, east_m, north_m (or {"error": "..."})
```

Synthetic data generation and batch method comparison:

```bash
python -m pegasource.path_estimation.generate_synthetic_datasets --help
python -m pegasource.path_estimation.run_method_evaluation --help   # writes ./method_eval/
```

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Package Structure

```
pegasource/
├── pcap/
│   ├── reader.py        # read_pcap, packet_summary
│   ├── stats.py         # protocol_distribution, top_talkers, conversation_table
│   ├── patterns.py      # port scan, beaconing, DNS anomalies, …
│   └── report.py        # generate_report
├── geo/
│   ├── distance.py      # haversine, vincenty, bearing
│   ├── projection.py    # wgs84_to_itm, itm_to_wgs84, wgs84_to_utm, meters_offset
│   ├── vectorizer.py    # build_graph, compute_road_coverage, plot_graph_overlay
│   ├── israel_roads.py  # load_israel_graph, shortest_path, subgraph_bbox
│   └── _rv_*.py         # vendored RoadVectorizer source (josefberman)
├── timeseries/
│   ├── auto.py          # AutoForecaster
│   ├── models.py        # SARIMAXModel, LinearTrendModel
│   └── utils.py         # detect_seasonality, train_test_split_ts, rmse
├── dataset_clustering/  # optional [clustering] extra for full stack
│   ├── cluster_hardware.py   # CLI + core clustering API
│   ├── server.py             # FastAPI + static cluster_viz UI
│   ├── custom_devices.py     # supplemental device names for matching
│   ├── fetch_ifixit_devices.py
│   ├── prepare_viz_data.py
│   ├── dataset_generator.py
│   ├── data/                 # sample CSVs
│   └── cluster_viz/          # HTML/JS frontend
├── path_estimation/          # optional [path_estimation] for evaluate + NN/torch
│   ├── evaluate.py         # run_evaluation, metrics, figures
│   ├── metrics.py, io.py, graph_utils.py, …
│   ├── filters/, nn/, gnn/
│   ├── london_street_path.py, geo_reference.py, plotting_utils.py
│   └── sample_*.csv, cache/  # bundled samples + OSM graph cache
└── data/
    └── israel_roads.pkl.gz   # pre-processed OSM graph (after download)
```

---

## Dependencies

- **scapy** — PCAP parsing
- **numpy, scipy, scikit-image** — numerical + image processing
- **networkx** — road graphs
- **pyproj** — coordinate transforms
- **shapely** — geometric operations
- **statsmodels** — SARIMAX
- **scikit-learn** — feature engineering, clustering in `dataset_clustering`
- **pandas** — data manipulation
- **matplotlib** — visualisation

Optional **`[clustering]`**: **sentence-transformers**, **FastAPI**, **uvicorn**, **python-multipart**, **openpyxl**, **requests**.

Optional **`[path_estimation]`**: **filterpy**, **torch**, **torch-geometric**, **contextily**, **osmnx**, **tqdm** (plus base **networkx**, **pyproj**, **matplotlib**, **pandas**, **scipy** already in the package).

---

## License

MIT © Josef Berman

Road Vectorizer code adapted from [josefberman/RoadVectorizer](https://github.com/josefberman/RoadVectorizer) (MIT).  
Road data © OpenStreetMap contributors (ODbL).
