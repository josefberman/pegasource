#!/usr/bin/env python3
r"""Cluster dirty hardware records using embeddings + agglomerative clustering.

This module is the core library behind ``python -m pegasource.dataset_clustering.cluster_hardware``
and the symbols re-exported from :mod:`pegasource.dataset_clustering`.

**Scaling:** For :math:`n >` ``DIRECT_CLUSTERING_LIMIT``, use :func:`cluster_twophase`
(via :func:`cluster_embeddings`) so memory stays bounded.

See Also
--------
pegasource.dataset_clustering.server : interactive FastAPI + static UI
"""

import argparse
import os
import time
import sys

from ._paths import PACKAGE_DIR

import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans


#: Maximum row count for single-pass agglomerative clustering. Above this,
#: :func:`cluster_embeddings` calls :func:`cluster_twophase` automatically.
DIRECT_CLUSTERING_LIMIT = 15_000

_DEFAULT_INPUT = PACKAGE_DIR / "data" / "dirty_hardware_data_40k.csv"
_DEFAULT_OUTPUT = PACKAGE_DIR / "data" / "clustered_output.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cluster dirty hardware CSV records using embeddings + agglomerative clustering."
    )
    parser.add_argument(
        "--input", default=str(_DEFAULT_INPUT),
        help=f"Path to the input CSV file (default: bundled sample under {_DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output", default=str(_DEFAULT_OUTPUT),
        help=f"Path to the output CSV file (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.3,
        help="Distance threshold for clustering. Lower = tighter/more clusters, higher = looser/fewer clusters. "
             "Range 0.0–2.0 for cosine distance (default: 0.3)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=512,
        help="Batch size for embedding generation (default: 512)"
    )
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Use only the first N rows for quick testing (default: use all rows)"
    )
    parser.add_argument(
        "--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Path to local model folder or HuggingFace model ID (default: paraphrase-multilingual-MiniLM-L12-v2)"
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device for embedding model: 'cpu' or 'cuda' (default: cpu, avoids CUDA index errors with some models)"
    )
    parser.add_argument(
        "--pre-clusters", type=int, default=None,
        help="Number of KMeans pre-clusters for large datasets. Auto-calculated if not set."
    )
    return parser.parse_args()


def load_data(path, sample_size=None):
    """Load a CSV with all columns as strings (missing values become empty strings).

    Parameters
    ----------
    path : str or os.PathLike
        Input ``.csv`` path.
    sample_size : int, optional
        If set, only the first ``sample_size`` rows are kept (for quick tests).

    Returns
    -------
    pandas.DataFrame
        All columns have ``dtype`` str; ``NaN`` replaced with ``""``.

    Notes
    -----
    Prints progress to stdout (row × column counts).
    """
    print(f"📂 Loading data from {path}...")
    df = pd.read_csv(path, dtype=str).fillna("")
    if sample_size is not None:
        df = df.head(sample_size)
    print(f"   Loaded {len(df):,} rows × {len(df.columns)} columns")
    return df


def build_text_representations(df):
    """Join every column of each row into one text line for embedding.

    Columns are separated by ``" | "`` (space-pipe-space), preserving column order.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table (typically all strings).

    Returns
    -------
    list of str
        Length ``len(df)``; one concatenated string per row.

    Notes
    -----
    Prints the first row’s text to stdout as a sanity check.
    """
    print("📝 Building text representations...")
    texts = df.apply(lambda row: " | ".join(row.values), axis=1).tolist()
    print(f"   Example: {texts[0]!r}")
    return texts


def generate_embeddings(texts, model_name, batch_size, device="cpu"):
    """Encode texts with SentenceTransformer (L2-normalized for cosine distance).

    Requires the optional **sentence-transformers** dependency
    (``pip install -e ".[clustering]"``).

    Parameters
    ----------
    texts : list of str
        One sentence/line per inventory row.
    model_name : str
        Hugging Face model id (e.g. ``sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2``)
        or path to a local model directory.
    batch_size : int
        ``model.encode`` batch size (GPU memory vs throughput).
    device : str, default ``"cpu"``
        ``"cpu"``, ``"cuda"``, or another torch device string.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_rows, dim)``; rows are unit vectors if the backend normalizes.

    Notes
    -----
    Local paths are resolved with :func:`os.path.abspath` when the path exists on disk.
    """
    from sentence_transformers import SentenceTransformer

    # Resolve local paths to absolute (avoids cwd issues)
    if "/" not in model_name or os.path.exists(model_name):
        model_path = os.path.abspath(model_name)
    else:
        model_path = model_name

    print(f"🤖 Loading model '{model_path}' on {device}...")
    model = SentenceTransformer(model_path, device=device)

    print(f"⚡ Generating embeddings for {len(texts):,} texts (batch_size={batch_size})...")
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # Pre-normalize for cosine similarity
    )
    elapsed = time.time() - t0
    print(f"   Done in {elapsed:.1f}s — shape: {embeddings.shape}")
    return embeddings


def cluster_direct(embeddings, threshold):
    """Agglomerative clustering with cosine distance and average linkage.

    Parameters
    ----------
    embeddings : numpy.ndarray
        Shape ``(n, dim)``; rows are typically L2-normalized.
    threshold : float
        ``distance_threshold`` in ``AgglomerativeClustering`` (cosine metric).  Smaller
        values yield **more** clusters (tighter merges); larger values yield **fewer**
        clusters. Typical range ~0.2–0.5 for normalized vectors.

    Returns
    -------
    numpy.ndarray
        Integer labels, shape ``(n,)``, not necessarily starting at 0 or contiguous.
    """
    print(f"🔗 Clustering {len(embeddings):,} rows with distance_threshold={threshold}...")
    t0 = time.time()
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(embeddings)
    elapsed = time.time() - t0
    n_clusters = len(set(labels))
    print(f"   Found {n_clusters:,} clusters in {elapsed:.1f}s")
    return labels


def cluster_twophase(embeddings, threshold, n_pre_clusters=None):
    """Scale to large *n*: KMeans pre-groups, then agglomerative within each group.

    Global cluster ids are formed by offsetting each group’s local labels so they do not
    collide across groups.

    Parameters
    ----------
    embeddings : numpy.ndarray
        Shape ``(n, dim)``.
    threshold : float
        Same meaning as in :func:`cluster_direct` (cosine / average linkage).
    n_pre_clusters : int, optional
        Number of KMeans buckets. If omitted, uses ``max(10, n // 5000)`` capped at ``n``.

    Returns
    -------
    numpy.ndarray
        Integer labels, shape ``(n,)``.
    """
    n = len(embeddings)
    if n_pre_clusters is None:
        # Aim for pre-groups of ~5000 rows each
        n_pre_clusters = max(10, n // 5000)
    n_pre_clusters = min(n_pre_clusters, n)

    print(f"🔗 Phase 1: Pre-grouping {n:,} rows into {n_pre_clusters} groups with KMeans...")
    t0 = time.time()
    kmeans = MiniBatchKMeans(
        n_clusters=n_pre_clusters,
        batch_size=min(4096, n),
        random_state=42,
        n_init=3,
    )
    pre_labels = kmeans.fit_predict(embeddings)
    elapsed = time.time() - t0
    print(f"   Pre-grouping done in {elapsed:.1f}s")

    print(f"🔗 Phase 2: Agglomerative clustering within each group (threshold={threshold})...")
    t0 = time.time()
    final_labels = np.zeros(n, dtype=int)
    global_cluster_id = 0

    for group_id in range(n_pre_clusters):
        mask = pre_labels == group_id
        group_embeddings = embeddings[mask]
        group_size = len(group_embeddings)

        if group_size <= 1:
            final_labels[mask] = global_cluster_id
            global_cluster_id += 1
            continue

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            metric="cosine",
            linkage="average",
        )
        group_labels = clustering.fit_predict(group_embeddings)
        n_group_clusters = len(set(group_labels))

        # Map local cluster IDs to global IDs
        final_labels[mask] = group_labels + global_cluster_id
        global_cluster_id += n_group_clusters

    elapsed = time.time() - t0
    n_clusters = len(set(final_labels))
    print(f"   Found {n_clusters:,} clusters in {elapsed:.1f}s")
    return final_labels


def cluster_embeddings(embeddings, threshold, n_pre_clusters=None):
    """Dispatch to :func:`cluster_direct` or :func:`cluster_twophase` by row count.

    If ``len(embeddings) <= DIRECT_CLUSTERING_LIMIT`` uses direct agglomerative
    clustering; otherwise uses the two-phase pipeline.

    Parameters
    ----------
    embeddings : numpy.ndarray
        Row embedding matrix.
    threshold : float
        Cosine distance threshold for agglomerative steps.
    n_pre_clusters : int, optional
        Forwarded to :func:`cluster_twophase` when the two-phase path is used.

    Returns
    -------
    numpy.ndarray
        Cluster label per row.
    """
    n = len(embeddings)
    if n <= DIRECT_CLUSTERING_LIMIT:
        return cluster_direct(embeddings, threshold)
    else:
        print(f"ℹ️  Dataset has {n:,} rows (>{DIRECT_CLUSTERING_LIMIT:,}), using two-phase clustering")
        return cluster_twophase(embeddings, threshold, n_pre_clusters)


def print_summary(df, max_clusters=30, samples_per_cluster=3):
    """Print cluster sizes and sample rows to stdout (requires ``cluster_id`` column).

    Parameters
    ----------
    df : pandas.DataFrame
        Must include a ``cluster_id`` column (e.g. after assigning labels).
    max_clusters : int, default 30
        Maximum number of largest clusters to list.
    samples_per_cluster : int, default 3
        Rows shown per cluster (excluding ``cluster_id`` in the display).

    Returns
    -------
    None
    """
    cluster_sizes = df.groupby("cluster_id").size().sort_values(ascending=False)
    n_clusters = len(cluster_sizes)

    print("\n" + "=" * 80)
    print(f"📊 CLUSTERING SUMMARY")
    print(f"   Total rows:     {len(df):,}")
    print(f"   Total clusters: {n_clusters:,}")
    print(f"   Largest:        {cluster_sizes.iloc[0]:,} rows")
    print(f"   Smallest:       {cluster_sizes.iloc[-1]:,} rows")
    print(f"   Median size:    {int(cluster_sizes.median()):,} rows")
    print("=" * 80)

    # Show the top clusters with sample rows
    show_n = min(max_clusters, n_clusters)
    print(f"\n🔍 Top {show_n} clusters by size:\n")

    for i, (cluster_id, size) in enumerate(cluster_sizes.head(show_n).items()):
        cluster_rows = df[df["cluster_id"] == cluster_id]
        # Show a few sample rows (excluding the cluster_id column for readability)
        display_cols = [c for c in df.columns if c != "cluster_id"]
        samples = cluster_rows[display_cols].head(samples_per_cluster)

        print(f"  Cluster {cluster_id} ({size:,} rows):")
        for _, row in samples.iterrows():
            print(f"    → {' | '.join(row.values)}")
        print()


def main():
    args = parse_args()

    print(f"\n{'=' * 80}")
    print(f"  Hardware Record Clustering")
    print(f"  Threshold: {args.threshold}  |  Model: {args.model}")
    print(f"{'=' * 80}\n")

    # 1. Load data
    df = load_data(args.input, args.sample_size)

    # 2. Build text representations
    texts = build_text_representations(df)

    # 3. Generate embeddings
    embeddings = generate_embeddings(texts, args.model, args.batch_size, device=args.device)

    # 4. Cluster
    labels = cluster_embeddings(embeddings, args.threshold, args.pre_clusters)

    # 5. Attach labels and sort by cluster
    df["cluster_id"] = labels
    df = df.sort_values("cluster_id").reset_index(drop=True)

    # 6. Save output
    df.to_csv(args.output, index=False)
    print(f"\n💾 Saved clustered output to {args.output}")

    # 7. Print summary
    print_summary(df)


if __name__ == "__main__":
    main()
