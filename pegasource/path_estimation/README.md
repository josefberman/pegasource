# Path estimation

Reconstructs a trajectory (local ENU meters) from mixed asynchronous observations (`gps`, `circle`, `cell_sector`) and compares against `*_true_path.csv` (1 Hz).

## Sample CSVs

Paired toy data in this directory (61 s at 1 Hz, eight mixed observations):

- `sample_true_path.csv`
- `sample_observations.csv`

```bash
python -m path_estimation \
  --observations path_estimation/sample_observations.csv \
  --true-path path_estimation/sample_true_path.csv \
  --output-dir path_estimation_runs/sample \
  --methods kf \
  --no-plots
```

## Python API

- **`evaluate_path_estimation(observations_csv, true_path_csv, road_graph, methods, ...)`** — metrics + optional plots; needs a real `*_true_path.csv`.
- **`estimate_paths_only(observations_csv, road_graph, methods, ...)`** — returns `EstimationResult` per method; **no** truth file. Uses an internal time grid from the observation span (`output_hz`, default 1 Hz). Not available for **`lstm`**, **`transformer`**, or **`gnn`** (supervised / needs labels). **`plot_map`** is disallowed (needs real lon/lat).
- **`stub_true_path_from_observations`** (`io.py`) — time-grid helper used internally for the no-truth path.

## CLI

```bash
python -m path_estimation \
  --observations data/dataset_observations.csv \
  --true-path data/dataset_true_path.csv \
  --output-dir path_estimation_runs/run1 \
  --seed 42 \
  --device cpu
```

- `--methods`: comma-separated list (default: all implemented methods).
- `--no-plots`: skip PNG figures.
- `--map-plots`: also write Web Mercator basemap figures (may fetch tiles).

Outputs:

- `metrics.json` — per-method scores (RMSE, MAE, Hausdorff, discrete Fréchet, DTW, length ratio, endpoint error, …).
- `figures/<method>_path_enu.png` — ENU overlay (true vs estimated; optional observation overlays; no uncertainty ellipses).

## Methods

| ID | Description |
|----|-------------|
| `dijkstra` | Snap observations to OSM nodes; stitch shortest paths; uniform speed along polyline. |
| `astar` | Same as Dijkstra with `networkx.astar_path` (heuristic). |
| `hmm` | Viterbi–style map match over k-nearest nodes per observation. |
| `kf` | 4D constant-velocity Kalman filter; fused linear updates (GPS tight; circle/cell as weaker position cues). |
| `ekf` | EKF with GPS + circle + cell geometry. |
| `ukf` | Unscented Kalman filter via `filterpy`; fused measurement schedule. |
| `particle` | Bootstrap particle filter with mixed likelihoods. |
| `lstm` | LSTM on observation sequence; supervised fit on the same run. |
| `transformer` | Small Transformer encoder. |
| `gnn` | GCN node classifier on OSM subgraph; guided snaps + stitch. |

## Metrics (see `metrics.py`)

- Pointwise: RMSE, MAE (Euclidean and per-axis).
- Sequence: Hausdorff (subsampled), discrete Fréchet, DTW (subsampled).
- Global: path length ratio, combined endpoint error.

## Dependencies

See project `requirements.txt` (`scipy`, `filterpy`, `torch`, `torch-geometric`, …).

## Tests

```bash
python -m pytest tests/test_path_metrics.py tests/test_graph_shortest.py -q
```
