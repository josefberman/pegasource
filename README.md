# Pegasource

> **Offline-capable Python toolkit** — PCAP analysis, geographic functions, and automatic time-series forecasting.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Modules

| Module | Description |
|--------|-------------|
| `pegasource.pcap` | PCAP reader, statistics, anomaly & pattern detection |
| `pegasource.geo` | Distance, coordinate transforms, road vectorizer, Israel road network |
| `pegasource.timeseries` | Automatic time-series forecasting (SARIMAX + fallback) |

---

## Installation

```bash
pip install -e ".[dev]"      # development install
# or
pip install pegasource        # once published
```

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
- **scikit-learn** — feature engineering
- **pandas** — data manipulation
- **matplotlib** — visualisation

---

## License

MIT © Josef Berman

Road Vectorizer code adapted from [josefberman/RoadVectorizer](https://github.com/josefberman/RoadVectorizer) (MIT).  
Road data © OpenStreetMap contributors (ODbL).
