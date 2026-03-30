"""
Hardware inventory clustering: text embeddings + hierarchical clustering.

Reads a CSV of string cells, concatenates each row into one text line, encodes with a
`sentence-transformers <https://www.sbert.net/>`_ model (cosine-normalized vectors),
then clusters with scikit-learn: **agglomerative** clustering for smaller tables, or a
**two-phase** MiniBatchKMeans + agglomerative pipeline when row count exceeds
`DIRECT_CLUSTERING_LIMIT`.

**Public API** (import from ``pegasource.dataset_clustering``):

- `load_data` — load CSV as strings
- `build_text_representations` — one string per row
- `generate_embeddings` — requires optional ``[clustering]`` (sentence-transformers)
- `cluster_direct` / `cluster_twophase` / `cluster_embeddings` — label vectors
- `print_summary` — CLI-style cluster size table

**Optional tooling**: FastAPI server (``server``), iFixit catalog fetch, Excel export,
bundled ``cluster_viz`` UI. Install extras with ``pip install -e ".[clustering]"``.

Examples
--------
>>> # After: pip install -e ".[clustering]"
>>> from pegasource.dataset_clustering import (
...     load_data,
...     build_text_representations,
...     generate_embeddings,
...     cluster_embeddings,
... )
>>> # df = load_data("data.csv")
>>> # emb = generate_embeddings(build_text_representations(df), "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 256)
>>> # labels = cluster_embeddings(emb, 0.3)

CLI and server::

    python -m pegasource.dataset_clustering.cluster_hardware --help
    python -m pegasource.dataset_clustering.server
    pegasource-cluster-viz --port 8001
"""

from .cluster_hardware import (
    DIRECT_CLUSTERING_LIMIT,
    build_text_representations,
    cluster_direct,
    cluster_embeddings,
    cluster_twophase,
    generate_embeddings,
    load_data,
    print_summary,
)

__all__ = [
    "DIRECT_CLUSTERING_LIMIT",
    "build_text_representations",
    "cluster_direct",
    "cluster_embeddings",
    "cluster_twophase",
    "generate_embeddings",
    "load_data",
    "print_summary",
]
