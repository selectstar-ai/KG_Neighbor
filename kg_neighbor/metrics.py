"""Evaluation metrics (Section 4.1).

Class labels are used only here, as an evaluation reference.
"""

from __future__ import annotations

import numpy as np


def nmi_kmeans(Y: np.ndarray, labels: np.ndarray, *, seed: int = 0, n_init: int = 10) -> float:
    """NMI between k-means assignments on the layout (``K = C``) and the labels."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score

    n_clusters = len(np.unique(labels))
    pred = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=seed).fit_predict(Y)
    return float(normalized_mutual_info_score(labels, pred))


def lt_2d(Y: np.ndarray, labels: np.ndarray, *, k: int = 20) -> float:
    """Rank-weighted local label trustworthiness on the Euclidean kNN graph of ``Y``.

    Same formula as the MAPLE reference implementation:
    ``LT_i = 1 - 2 P_i / (k (k + 1) / 2)`` with
    ``P_i = Σ_p 1[label(n_p) != label(i)] · (k + 1 - p)``; 1 means all neighbors
    share the label, -1 means none do. Returns the mean over instances.
    """
    from sklearn.neighbors import NearestNeighbors

    labels = np.asarray(labels)
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(Y).kneighbors(Y)
    idx = idx[:, 1:]  # drop self
    ranks = np.arange(1, k + 1, dtype=np.float64)[None, :]
    impostor = labels[idx] != labels[:, None]
    P = (impostor * (k + 1 - ranks)).sum(axis=1)
    return float((1.0 - 2.0 * P / (k * (k + 1) / 2.0)).mean())


def neighbor_retention(idx: np.ndarray, idx_base: np.ndarray) -> float:
    """Mean fraction of the original top-k neighbors kept after distance adjustment.

    Parameters
    ----------
    idx : (N, k) neighbors after adjustment, self excluded
    idx_base : (N, k) kNN neighbors (``alpha = 0``), self excluded
    """
    idx, idx_base = np.asarray(idx), np.asarray(idx_base)
    if idx.shape != idx_base.shape:
        raise ValueError(f"shape mismatch: {idx.shape} vs {idx_base.shape}")
    k = idx.shape[1]
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(idx, idx_base)]))
