"""One public function per path estimator; heavy deps load only when that function runs.

Call :func:`estimate_hmm`, :func:`estimate_kf`, etc., when you want a single method without
importing PyTorch or other stacks used by LSTM / Transformer / GNN.

For **observations-only** runs (no real ground truth), use :func:`estimate_kf_obs_only`,
:func:`estimate_dijkstra_obs_only`, etc.  They build a stub timeline via
:func:`pegasource.path_estimation.io.stub_true_path_from_observations` (same as
:func:`pegasource.path_estimation.evaluate.estimate_paths_only`).  Supervised methods
(:func:`estimate_gnn_obs_only`, :func:`estimate_lstm_obs_only`,
:func:`estimate_transformer_obs_only`) raise ``ValueError``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from .types import EstimationResult


def estimate_dijkstra(
    obs_df: pd.DataFrame,
    true_df: pd.DataFrame,
    G: Any,
    rng: np.random.Generator,
) -> EstimationResult:
    """Graph map-matching via observation snaps and Dijkstra shortest-path stitching.

    Loads only graph-stitching code (``graph_stitch``, ``graph_utils``). No PyTorch.

    Parameters
    ----------
    obs_df : pandas.DataFrame
        Sorted observations with ``timestamp_s`` and per-row geometry (see ``io``).
    true_df : pandas.DataFrame
        Ground-truth timeline for resampling (``timestamp_s``, ``true_x``, ``true_y``).
    G : networkx.MultiDiGraph
        Projected road graph with ``x``, ``y``, ``crs``.
    rng : numpy.random.Generator
        Unused for Dijkstra; kept for a uniform estimator signature.

    Returns
    -------
    EstimationResult
        Estimated east/north on the true timeline.
    """
    from .graph_stitch import estimate_graph_stitch

    return estimate_graph_stitch(obs_df, true_df, G, rng, mode="dijkstra")


def estimate_astar(
    obs_df: pd.DataFrame,
    true_df: pd.DataFrame,
    G: Any,
    rng: np.random.Generator,
) -> EstimationResult:
    """Same as :func:`estimate_dijkstra` but uses A* for each segment.

    No PyTorch.

    Parameters
    ----------
    obs_df, true_df, G, rng
        See :func:`estimate_dijkstra`.

    Returns
    -------
    EstimationResult
    """
    from .graph_stitch import estimate_graph_stitch

    return estimate_graph_stitch(obs_df, true_df, G, rng, mode="astar")


def estimate_hmm(
    obs_df: pd.DataFrame,
    true_df: pd.DataFrame,
    G: Any,
    rng: np.random.Generator,
) -> EstimationResult:
    """HMM / Viterbi–style map matching over candidate graph nodes per observation.

    Loads ``hmm_map_match`` and graph utilities only. No PyTorch.

    Parameters
    ----------
    obs_df, true_df, G, rng
        See :func:`estimate_dijkstra`.

    Returns
    -------
    EstimationResult
    """
    from .hmm_map_match import estimate_hmm_map_match

    return estimate_hmm_map_match(obs_df, true_df, G, rng)


def estimate_kf(
    obs_df: pd.DataFrame,
    true_df: pd.DataFrame,
    G: Any,
    rng: np.random.Generator,
) -> EstimationResult:
    """Constant-velocity Kalman filter with GPS and weak circle/cell cues.

    Loads ``filterpy``-based KF code only (not torch).

    Parameters
    ----------
    obs_df, true_df, G, rng
        ``G`` is ignored but kept for a uniform signature.

    Returns
    -------
    EstimationResult
    """
    from .filters.kf import estimate_kf_gps

    return estimate_kf_gps(obs_df, true_df, G, rng)


def estimate_ekf(
    obs_df: pd.DataFrame,
    true_df: pd.DataFrame,
    G: Any,
    rng: np.random.Generator,
) -> EstimationResult:
    """Extended Kalman filter for fused observations.

    Loads EKF implementation only (``filterpy``; no torch).

    Parameters
    ----------
    obs_df, true_df, G, rng
        See :func:`estimate_kf`.

    Returns
    -------
    EstimationResult
    """
    from .filters.ekf import estimate_ekf_fused

    return estimate_ekf_fused(obs_df, true_df, G, rng)


def estimate_ukf(
    obs_df: pd.DataFrame,
    true_df: pd.DataFrame,
    G: Any,
    rng: np.random.Generator,
) -> EstimationResult:
    """Unscented Kalman filter for fused observations.

    Loads UKF implementation only (``filterpy``; no torch).

    Parameters
    ----------
    obs_df, true_df, G, rng
        See :func:`estimate_kf`.

    Returns
    -------
    EstimationResult
    """
    from .filters.ukf import estimate_ukf_fused

    return estimate_ukf_fused(obs_df, true_df, G, rng)


def estimate_particle(
    obs_df: pd.DataFrame,
    true_df: pd.DataFrame,
    G: Any,
    rng: np.random.Generator,
) -> EstimationResult:
    """Bootstrap particle filter with mixed measurement likelihoods.

    Loads particle filter code only (numpy/scipy stack; no torch).

    Parameters
    ----------
    obs_df, true_df, G, rng
        See :func:`estimate_kf`.

    Returns
    -------
    EstimationResult
    """
    from .filters.particle import estimate_particle_filter

    return estimate_particle_filter(obs_df, true_df, G, rng)


def estimate_gnn(
    obs_df: pd.DataFrame,
    true_df: pd.DataFrame,
    G: Any,
    rng: np.random.Generator,
    *,
    device: Optional[str] = None,
) -> EstimationResult:
    """GCN-based map matching (subgraph, node logits, stitched decode).

    Imports **torch**, **torch-geometric**, and ``gnn.estimate`` when called.

    Parameters
    ----------
    obs_df, true_df, G, rng
        See :func:`estimate_dijkstra`.
    device : str, optional
        Torch device; default is CUDA if available else CPU.

    Returns
    -------
    EstimationResult
    """
    from .gnn.estimate import estimate_gnn as _estimate_gnn_impl

    dev = _torch_device(device)
    return _estimate_gnn_impl(obs_df, true_df, G, rng, device=dev)


def estimate_lstm(
    obs_df: pd.DataFrame,
    true_df: pd.DataFrame,
    G: Any,
    rng: np.random.Generator,
    *,
    device: Optional[str] = None,
) -> EstimationResult:
    """LSTM sequence model trained on the same run (supervised).

    Imports **torch** and ``nn.lstm_model`` when called.

    Parameters
    ----------
    obs_df, true_df, G, rng
        See :func:`estimate_dijkstra`.
    device : str, optional
        ``\"cpu\"``, ``\"cuda\"``, etc.

    Returns
    -------
    EstimationResult
    """
    from .nn.lstm_model import predict_lstm_at_times, train_lstm

    dev = _torch_device(device)
    model, ds = train_lstm(obs_df, true_df, dev)
    times_s, xy = predict_lstm_at_times(model, obs_df, true_df, dev, ds)
    return EstimationResult(
        times_s=times_s,
        east_m=xy[:, 0],
        north_m=xy[:, 1],
        meta={"method": "lstm"},
    )


def estimate_transformer(
    obs_df: pd.DataFrame,
    true_df: pd.DataFrame,
    G: Any,
    rng: np.random.Generator,
    *,
    device: Optional[str] = None,
) -> EstimationResult:
    """Small Transformer encoder for per-event residuals (supervised).

    Imports **torch** and ``nn.transformer_model`` when called.

    Parameters
    ----------
    obs_df, true_df, G, rng
        See :func:`estimate_dijkstra`.
    device : str, optional
        Torch device string.

    Returns
    -------
    EstimationResult
    """
    from .nn.transformer_model import predict_transformer_at_times, train_transformer

    dev = _torch_device(device)
    model, ds = train_transformer(obs_df, true_df, dev)
    times_s, xy = predict_transformer_at_times(model, obs_df, true_df, dev, ds)
    return EstimationResult(
        times_s=times_s,
        east_m=xy[:, 0],
        north_m=xy[:, 1],
        meta={"method": "transformer"},
    )


def _torch_device(name: Optional[str] = None):
    import torch as _torch

    if name:
        return _torch.device(name)
    return _torch.device("cuda" if _torch.cuda.is_available() else "cpu")


def _stub_true_and_call(
    obs_df: pd.DataFrame,
    G: Any,
    rng: np.random.Generator,
    *,
    output_hz: float,
    estimate_fn: Callable[..., EstimationResult],
    estimate_kwargs: Optional[dict[str, Any]] = None,
) -> EstimationResult:
    from .io import stub_true_path_from_observations

    true_df = stub_true_path_from_observations(obs_df, hz=output_hz)
    ek = estimate_kwargs or {}
    return estimate_fn(obs_df, true_df, G, rng, **ek)


def estimate_dijkstra_obs_only(
    obs_df: pd.DataFrame,
    G: Any,
    rng: np.random.Generator,
    *,
    output_hz: float = 1.0,
) -> EstimationResult:
    """Dijkstra stitching with a stub timeline from observations only (no real ground truth).

    Builds ``true_df`` with :func:`~pegasource.path_estimation.io.stub_true_path_from_observations`.
    Metrics against ``true_x``/``true_y`` are meaningless.  Same dependency rules as
    :func:`estimate_dijkstra`.

    Parameters
    ----------
    obs_df : pandas.DataFrame
        Sorted observations (see IO loaders).
    G : networkx.MultiDiGraph
        Projected road graph.
    rng : numpy.random.Generator
        RNG for stochastic pieces of the pipeline.
    output_hz : float, default 1.0
        Sampling rate (Hz) for the stub ground-truth timeline.

    Returns
    -------
    EstimationResult
    """
    return _stub_true_and_call(
        obs_df, G, rng, output_hz=output_hz, estimate_fn=estimate_dijkstra,
    )


def estimate_astar_obs_only(
    obs_df: pd.DataFrame,
    G: Any,
    rng: np.random.Generator,
    *,
    output_hz: float = 1.0,
) -> EstimationResult:
    """A* stitching with a stub timeline from observations only.

    See :func:`estimate_dijkstra_obs_only`.

    Parameters
    ----------
    obs_df, G, rng
        See :func:`estimate_dijkstra_obs_only`.
    output_hz : float, default 1.0
        See :func:`estimate_dijkstra_obs_only`.

    Returns
    -------
    EstimationResult
    """
    return _stub_true_and_call(
        obs_df, G, rng, output_hz=output_hz, estimate_fn=estimate_astar,
    )


def estimate_hmm_obs_only(
    obs_df: pd.DataFrame,
    G: Any,
    rng: np.random.Generator,
    *,
    output_hz: float = 1.0,
) -> EstimationResult:
    """HMM map matching with a stub timeline from observations only.

    See :func:`estimate_dijkstra_obs_only`.

    Parameters
    ----------
    obs_df, G, rng
        See :func:`estimate_dijkstra_obs_only`.
    output_hz : float, default 1.0
        See :func:`estimate_dijkstra_obs_only`.

    Returns
    -------
    EstimationResult
    """
    return _stub_true_and_call(
        obs_df, G, rng, output_hz=output_hz, estimate_fn=estimate_hmm,
    )


def estimate_kf_obs_only(
    obs_df: pd.DataFrame,
    G: Any,
    rng: np.random.Generator,
    *,
    output_hz: float = 1.0,
) -> EstimationResult:
    """Kalman filter with a stub timeline from observations only.

    See :func:`estimate_dijkstra_obs_only`.

    Parameters
    ----------
    obs_df, G, rng
        See :func:`estimate_dijkstra_obs_only`.
    output_hz : float, default 1.0
        See :func:`estimate_dijkstra_obs_only`.

    Returns
    -------
    EstimationResult
    """
    return _stub_true_and_call(
        obs_df, G, rng, output_hz=output_hz, estimate_fn=estimate_kf,
    )


def estimate_ekf_obs_only(
    obs_df: pd.DataFrame,
    G: Any,
    rng: np.random.Generator,
    *,
    output_hz: float = 1.0,
) -> EstimationResult:
    """EKF with a stub timeline from observations only.

    See :func:`estimate_dijkstra_obs_only`.

    Parameters
    ----------
    obs_df, G, rng
        See :func:`estimate_dijkstra_obs_only`.
    output_hz : float, default 1.0
        See :func:`estimate_dijkstra_obs_only`.

    Returns
    -------
    EstimationResult
    """
    return _stub_true_and_call(
        obs_df, G, rng, output_hz=output_hz, estimate_fn=estimate_ekf,
    )


def estimate_ukf_obs_only(
    obs_df: pd.DataFrame,
    G: Any,
    rng: np.random.Generator,
    *,
    output_hz: float = 1.0,
) -> EstimationResult:
    """UKF with a stub timeline from observations only.

    See :func:`estimate_dijkstra_obs_only`.

    Parameters
    ----------
    obs_df, G, rng
        See :func:`estimate_dijkstra_obs_only`.
    output_hz : float, default 1.0
        See :func:`estimate_dijkstra_obs_only`.

    Returns
    -------
    EstimationResult
    """
    return _stub_true_and_call(
        obs_df, G, rng, output_hz=output_hz, estimate_fn=estimate_ukf,
    )


def estimate_particle_obs_only(
    obs_df: pd.DataFrame,
    G: Any,
    rng: np.random.Generator,
    *,
    output_hz: float = 1.0,
) -> EstimationResult:
    """Particle filter with a stub timeline from observations only.

    See :func:`estimate_dijkstra_obs_only`.

    Parameters
    ----------
    obs_df, G, rng
        See :func:`estimate_dijkstra_obs_only`.
    output_hz : float, default 1.0
        See :func:`estimate_dijkstra_obs_only`.

    Returns
    -------
    EstimationResult
    """
    return _stub_true_and_call(
        obs_df, G, rng, output_hz=output_hz, estimate_fn=estimate_particle,
    )


def estimate_gnn_obs_only(
    obs_df: pd.DataFrame,
    G: Any,
    rng: np.random.Generator,
    *,
    output_hz: float = 1.0,
    device: Optional[str] = None,
) -> EstimationResult:
    """Not available without ground truth (GNN is supervised on node labels).

    Parameters
    ----------
    obs_df, G, rng
        Unused; signature matches other ``*_obs_only`` helpers.
    output_hz : float, default 1.0
        Unused.
    device : str, optional
        Unused.

    Raises
    ------
    ValueError
        Always — use :func:`estimate_gnn` with a real ``true_df`` from
        :func:`~pegasource.path_estimation.io.load_true_path_csv`.

    Returns
    -------
    EstimationResult
        Never.
    """
    del obs_df, G, rng, output_hz, device
    raise ValueError(
        "gnn requires ground truth for training. Use estimate_gnn(...) with "
        "load_true_path_csv(...) or evaluate_path_estimation(..., true_path_csv=...)."
    )


def estimate_lstm_obs_only(
    obs_df: pd.DataFrame,
    G: Any,
    rng: np.random.Generator,
    *,
    output_hz: float = 1.0,
    device: Optional[str] = None,
) -> EstimationResult:
    """Not available without ground truth (LSTM is trained on the run).

    Parameters
    ----------
    obs_df, G, rng
        Unused.
    output_hz : float, default 1.0
        Unused.
    device : str, optional
        Unused.

    Raises
    ------
    ValueError
        Always — use :func:`estimate_lstm` with a real ``true_df``.

    Returns
    -------
    EstimationResult
        Never.
    """
    del obs_df, G, rng, output_hz, device
    raise ValueError(
        "lstm requires ground truth for training. Use estimate_lstm(...) with "
        "load_true_path_csv(...) or evaluate_path_estimation(..., true_path_csv=...)."
    )


def estimate_transformer_obs_only(
    obs_df: pd.DataFrame,
    G: Any,
    rng: np.random.Generator,
    *,
    output_hz: float = 1.0,
    device: Optional[str] = None,
) -> EstimationResult:
    """Not available without ground truth (Transformer is trained on the run).

    Parameters
    ----------
    obs_df, G, rng
        Unused.
    output_hz : float, default 1.0
        Unused.
    device : str, optional
        Unused.

    Raises
    ------
    ValueError
        Always — use :func:`estimate_transformer` with a real ``true_df``.

    Returns
    -------
    EstimationResult
        Never.
    """
    del obs_df, G, rng, output_hz, device
    raise ValueError(
        "transformer requires ground truth for training. Use estimate_transformer(...) with "
        "load_true_path_csv(...) or evaluate_path_estimation(..., true_path_csv=...)."
    )


METHOD_NAME_TO_FUNC = {
    "dijkstra": estimate_dijkstra,
    "astar": estimate_astar,
    "hmm": estimate_hmm,
    "kf": estimate_kf,
    "ekf": estimate_ekf,
    "ukf": estimate_ukf,
    "particle": estimate_particle,
    "gnn": estimate_gnn,
    "lstm": estimate_lstm,
    "transformer": estimate_transformer,
}

METHOD_NAME_TO_OBS_ONLY_FUNC = {
    "dijkstra": estimate_dijkstra_obs_only,
    "astar": estimate_astar_obs_only,
    "hmm": estimate_hmm_obs_only,
    "kf": estimate_kf_obs_only,
    "ekf": estimate_ekf_obs_only,
    "ukf": estimate_ukf_obs_only,
    "particle": estimate_particle_obs_only,
    "gnn": estimate_gnn_obs_only,
    "lstm": estimate_lstm_obs_only,
    "transformer": estimate_transformer_obs_only,
}

__all__ = [
    "METHOD_NAME_TO_FUNC",
    "METHOD_NAME_TO_OBS_ONLY_FUNC",
    "estimate_astar",
    "estimate_astar_obs_only",
    "estimate_dijkstra",
    "estimate_dijkstra_obs_only",
    "estimate_ekf",
    "estimate_ekf_obs_only",
    "estimate_gnn",
    "estimate_gnn_obs_only",
    "estimate_hmm",
    "estimate_hmm_obs_only",
    "estimate_kf",
    "estimate_kf_obs_only",
    "estimate_lstm",
    "estimate_lstm_obs_only",
    "estimate_particle",
    "estimate_particle_obs_only",
    "estimate_transformer",
    "estimate_transformer_obs_only",
    "estimate_ukf",
    "estimate_ukf_obs_only",
]
