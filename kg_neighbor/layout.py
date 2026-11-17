"""UMAP layout from a given neighbor search (Section 4.2).

The neighbor step is the only part that differs between kNN and KG-Neighbor;
graph weighting (fuzzy simplicial set) and layout optimization are UMAP's own
building blocks, applied identically to every method.
"""

from __future__ import annotations

import numpy as np


def umap_layout(
    Z: np.ndarray,
    neighbors,
    *,
    seed: int = 0,
    min_dist: float = 0.1,
    n_epochs: int = 200,
    metric: str = "cosine",
) -> np.ndarray:
    """2D UMAP layout whose neighbor graph comes from ``neighbors``.

    Parameters
    ----------
    Z : (N, d) input embeddings, already passed to ``neighbors.fit``
    neighbors : fitted object with ``n_neighbors`` and a scikit-learn style
        ``kneighbors(X)`` (:class:`~kg_neighbor.neighbors.KGNeighbors` or
        :class:`sklearn.neighbors.NearestNeighbors`)
    seed, min_dist, n_epochs : paper settings are 0, 0.1, 200 (spectral init)

    Returns
    -------
    Y : (N, 2) float32
    """
    import scipy.sparse
    from sklearn.utils import check_random_state
    from umap.umap_ import (
        compute_membership_strengths,
        find_ab_params,
        simplicial_set_embedding,
        smooth_knn_dist,
    )

    rs = check_random_state(seed)
    n = len(Z)
    k = neighbors.n_neighbors

    # neighbor graph, self included (UMAP convention)
    knn_dists, knn_indices = neighbors.kneighbors(Z)
    knn_dists = knn_dists.astype(np.float32)
    knn_indices = knn_indices.astype(np.int32)

    # fuzzy simplicial set
    sigmas, rhos = smooth_knn_dist(knn_dists, float(k), local_connectivity=1.0)
    rows, cols, vals, _ = compute_membership_strengths(
        knn_indices, knn_dists, sigmas, rhos, return_dists=False
    )
    graph = scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    graph.eliminate_zeros()
    t = graph.transpose()
    graph = (graph + t - graph.multiply(t)).tocsr()  # fuzzy union
    graph.eliminate_zeros()
    graph = graph.tocoo()

    # layout optimization: spectral init + SGD
    a, b = find_ab_params(1.0, min_dist)
    Y, _ = simplicial_set_embedding(
        Z, graph, 2, 1.0, a, b, 1.0, 5, n_epochs, "spectral",
        rs, metric, {}, False, {}, False,
    )
    return np.asarray(Y, dtype=np.float32)
