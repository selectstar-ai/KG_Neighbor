"""Knowledge-guided distance and neighbor selection (Section 3.3, Eqs. 4-6).

``KGNeighbors`` follows the interface of :class:`sklearn.neighbors.NearestNeighbors`
(hyperparameters in the constructor, data in :meth:`fit`), so it can replace
the kNN step of a neighborhood-based DR pipeline. With ``alpha = 0`` it
delegates to scikit-learn, so the kNN baseline is the library's own output.
"""

from __future__ import annotations

import numpy as np


def kg_guided_distance(Z: np.ndarray, S_kg: np.ndarray, alpha: float) -> np.ndarray:
    """``D_kgn = D_emb · (1 - α · S_kg)`` with cosine ``D_emb`` (Eqs. 4-6).

    Computed in place to bound memory ((N, N) float32 is 1.6 GB at N = 20,000).

    Parameters
    ----------
    Z : (N, d) input embeddings
    S_kg : (N, N) KG similarity in [0, 1]
    alpha : guidance strength in [0, 1]

    Returns
    -------
    D : (N, N) float32, symmetric, zero diagonal
    """
    from sklearn.metrics.pairwise import cosine_distances

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    S = np.asarray(S_kg, dtype=np.float32)
    if S.shape != (len(Z), len(Z)):
        raise ValueError(f"S_kg shape {S.shape} != (N, N) = ({len(Z)}, {len(Z)})")
    if S.min() < -1e-5 or S.max() > 1.0 + 1e-5:
        raise ValueError(f"S_kg must be in [0, 1], got [{S.min()}, {S.max()}]")
    S = np.clip(S, 0.0, 1.0)

    D = cosine_distances(Z).astype(np.float32)
    if alpha > 0.0:
        factor = np.multiply(S, -alpha, out=np.empty_like(S))
        np.add(factor, 1.0, out=factor)  # 1 - alpha * S_kg
        np.multiply(D, factor, out=D)
        del factor
    np.maximum(D, 0.0, out=D)
    np.fill_diagonal(D, 0.0)
    T = D.T.copy()  # remove floating-point asymmetry
    D += T
    D *= 0.5
    del T
    np.fill_diagonal(D, 0.0)
    return D.astype(np.float32)


class KGNeighbors:
    """Top-k neighbors under the knowledge-guided distance ``D_kgn``.

    Parameters
    ----------
    n_neighbors : k (paper: 15)
    alpha : guidance strength in [0, 1]; ``0`` reproduces kNN
    S_kg : (N, N) KG similarity, required when ``alpha > 0``
    metric : embedding distance; the paper uses ``"cosine"``

    Examples
    --------
    >>> nn = KGNeighbors(n_neighbors=15, alpha=0.3, S_kg=S_kg).fit(Z)
    >>> dist, idx = nn.kneighbors()          # self excluded, (N, k)
    """

    def __init__(
        self,
        n_neighbors: int = 15,
        *,
        alpha: float = 0.0,
        S_kg: np.ndarray | None = None,
        metric: str = "cosine",
    ):
        if alpha > 0.0 and S_kg is None:
            raise ValueError("alpha > 0 requires S_kg")
        if alpha > 0.0 and metric != "cosine":
            raise ValueError("the knowledge-guided distance is defined for metric='cosine'")
        self.n_neighbors = n_neighbors
        self.alpha = float(alpha)
        self.S_kg = S_kg
        self.metric = metric
        self._sk = None  # scikit-learn NearestNeighbors (alpha == 0)
        self._D: np.ndarray | None = None  # D_kgn (alpha > 0)

    def fit(self, Z: np.ndarray) -> KGNeighbors:
        """Prepare neighbor search over the N instances in ``Z``."""
        Z = np.asarray(Z)
        self._sk, self._D = None, None
        if self.alpha == 0.0:
            from sklearn.neighbors import NearestNeighbors

            self._sk = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric).fit(Z)
        else:
            self._D = kg_guided_distance(Z, self.S_kg, self.alpha)
        return self

    def kneighbors(
        self,
        X: np.ndarray | None = None,
        n_neighbors: int | None = None,
        return_distance: bool = True,
    ):
        """``(dist, idx)`` of shape (N, k), each row sorted by ascending distance.

        Same convention as scikit-learn: ``X=None`` excludes each instance from
        its own neighbors; passing the fitted data as ``X`` includes it (the
        convention UMAP uses). For ``alpha > 0`` only the fitted data can be
        queried, since ``S_kg`` is defined between fitted instances.
        """
        k = n_neighbors or self.n_neighbors
        if self._sk is not None:
            return self._sk.kneighbors(X, n_neighbors=k, return_distance=return_distance)
        if self._D is None:
            raise RuntimeError("call fit() first")
        n = self._D.shape[0]
        if X is not None and len(X) != n:
            raise ValueError("alpha > 0 supports querying the fitted data only")

        Dq = self._D.copy()
        if X is None:
            np.fill_diagonal(Dq, np.inf)  # exclude self
        idx = np.argpartition(Dq, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(n)[:, None]
        order = np.argsort(Dq[rows, idx], axis=1)
        ind = idx[rows, order]
        dist = Dq[rows, ind].astype(np.float32)
        return (dist, ind) if return_distance else ind
