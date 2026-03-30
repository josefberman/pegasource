# Path estimation

Reconstructs a trajectory (local ENU meters) from mixed asynchronous observations (`gps`, `circle`, `cell_sector`) and compares against `*_true_path.csv` (1 Hz).

## Sample CSVs

Paired toy data in this directory (61 s at 1 Hz, eight mixed observations):

- `sample_true_path.csv`
- `sample_observations.csv`

```python
from pathlib import Path

from pegasource.path_estimation.evaluate import run_evaluation

run_evaluation(
    observations_csv=Path("pegasource/path_estimation/sample_observations.csv"),
    true_path_csv=Path("pegasource/path_estimation/sample_true_path.csv"),
    output_dir=Path("path_estimation_runs/sample"),
    methods=["kf"],
    plot=False,
)
```

## Python API

- **`run_evaluation(observations_csv, true_path_csv, output_dir, methods, ...)`** — loads the default OSM graph, writes ``metrics.json`` and ``figures/`` under ``output_dir``.
- **`evaluate_path_estimation(observations_csv, true_path_csv, road_graph, methods, ...)`** — metrics + optional plots; needs a real `*_true_path.csv`; you supply ``road_graph``.
- **`estimate_paths_only(observations_csv, road_graph, methods, ...)`** — returns `EstimationResult` per method; **no** truth file. Uses an internal time grid from the observation span (`output_hz`, default 1 Hz). Not available for **`lstm`**, **`transformer`**, or **`gnn`**. **`plot_map`** is disallowed (needs real lon/lat).
- **`stub_true_path_from_observations`** (`io.py`) — time-grid helper used internally for the no-truth path.

**Synthetic data:** `generate_dataset` in `generate_synthetic_datasets.py`, or `main()` there if you want to drive the same argparse-defined options from code.

**Batch method comparison:** `main()` in `run_method_evaluation.py` (writes under ``./method_eval/`` when invoked from your script).

Outputs:

- `metrics.json` — per-method scores (RMSE, MAE, Hausdorff, discrete Fréchet, DTW, length ratio, endpoint error, …).
- `figures/<method>_path_enu.png` — ENU overlay (true vs estimated; optional observation overlays).

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

Install optional extra: ``pip install -e ".[path_estimation]"`` (see project ``pyproject.toml``).

## Tests

```bash
pytest tests/test_path_estimation.py -q
```
