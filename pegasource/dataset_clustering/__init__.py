"""
pegasource.dataset_clustering — Hardware inventory clustering via embeddings,
agglomerative clustering, optional FastAPI viz server, and iFixit device matching.

Quick start::

    from pegasource.dataset_clustering import (
        load_data,
        build_text_representations,
        generate_embeddings,
        cluster_embeddings,
    )

    df = load_data(path_to_csv)
    texts = build_text_representations(df)
    emb = generate_embeddings(texts, model_name, batch_size)
    labels = cluster_embeddings(emb, threshold=0.3)

Run the interactive server (requires optional deps)::

    python -m pegasource.dataset_clustering.server
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
