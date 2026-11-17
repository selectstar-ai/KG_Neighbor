"""Residual mismatch detection on a layout (Section 3.4, Eq. 7). No LLM involved."""

from __future__ import annotations

import numpy as np


def neighborhood_majority(Y: np.ndarray, labels: np.ndarray, k: int = 15) -> np.ndarray:
    """Neighborhood-majority class ``B_i`` over the Euclidean kNN of ``Y`` (self excluded).

    Ties go to the tied class with the smallest mean distance to instance ``i``;
    remaining ties go to the lowest class index.
    """
    from sklearn.neighbors import NearestNeighbors

    Y = np.asarray(Y, dtype=np.float64)
    labels = np.asarray(labels)
    kk = min(k, len(Y) - 1)
    dist, idx = NearestNeighbors(n_neighbors=kk + 1).fit(Y).kneighbors(Y)
    dist, idx = dist[:, 1:], idx[:, 1:]  # drop self

    B = np.empty(len(Y), dtype=labels.dtype)
    for i in range(len(Y)):
        neigh = labels[idx[i]]
        classes, counts = np.unique(neigh, return_counts=True)
        tied = classes[counts == counts.max()]
        if len(tied) == 1:
            B[i] = tied[0]
        else:
            B[i] = tied[int(np.argmin([dist[i][neigh == c].mean() for c in tied]))]
    return B


def detect_mismatches(Y: np.ndarray, labels: np.ndarray, k: int = 15) -> dict[str, np.ndarray]:
    """Instances whose assigned class ``A_i`` differs from ``B_i``.

    Returns
    -------
    dict with arrays of equal length M:
        ``"idx"`` row index, ``"A"`` assigned class, ``"B"`` neighborhood-majority class
    """
    labels = np.asarray(labels)
    B = neighborhood_majority(Y, labels, k=k)
    idx = np.flatnonzero(B != labels)
    return {"idx": idx, "A": labels[idx], "B": B[idx]}
