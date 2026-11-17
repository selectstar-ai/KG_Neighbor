"""KG similarity between instance-specific triple sets (Section 3.3, Eqs. 1-3).

Every measure takes ``triples`` -- one set of ``(subject, predicate, object)``
tuples per instance -- and returns a dense ``(N, N)`` float32 matrix in
``[0, 1]``. Undefined values and the diagonal are set to 0. All measures use
exact triple matching.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.sparse import csr_matrix, diags

Triple = tuple[str, str, str]


def _triple_matrix(triples: Sequence[set[Triple]]) -> tuple[csr_matrix, np.ndarray]:
    """Binary ``(N, V)`` instance-by-triple matrix and document frequency ``df``."""
    vocab = sorted(set().union(*triples)) if len(triples) else []
    index = {t: j for j, t in enumerate(vocab)}
    rows, cols = [], []
    for i, T in enumerate(triples):
        for t in T:
            rows.append(i)
            cols.append(index[t])
    X = csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(triples), len(vocab)),
        dtype=np.float32,
    )
    df = np.zeros(len(vocab), dtype=np.float32)
    for c in cols:
        df[c] += 1
    return X, df


def jaccard(triples: Sequence[set[Triple]]) -> np.ndarray:
    """Jaccard similarity ``|T_i ∩ T_j| / |T_i ∪ T_j|`` (Eq. 1)."""
    X, _ = _triple_matrix(triples)
    inter = (X @ X.T).toarray().astype(np.float32)
    sizes = np.asarray(X.sum(axis=1)).flatten()
    union = sizes[:, None] + sizes[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        S = np.where(union > 0, inter / union, 0.0).astype(np.float32)
    np.fill_diagonal(S, 0.0)
    return S


def overlap(triples: Sequence[set[Triple]]) -> np.ndarray:
    """Overlap coefficient ``|T_i ∩ T_j| / min(|T_i|, |T_j|)`` (Eq. 1)."""
    X, _ = _triple_matrix(triples)
    inter = (X @ X.T).toarray().astype(np.float32)
    sizes = np.asarray(X.sum(axis=1)).flatten()
    min_sizes = np.minimum(sizes[:, None], sizes[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        S = np.where(min_sizes > 0, inter / min_sizes, 0.0).astype(np.float32)
    np.fill_diagonal(S, 0.0)
    return S


def idf_cosine(triples: Sequence[set[Triple]]) -> np.ndarray:
    """IDF-weighted cosine with ``v_i[t] = log(N / df(t)) · 1[t ∈ T_i]`` (Eq. 2)."""
    N = len(triples)
    X, df = _triple_matrix(triples)
    idf = np.log(np.where(df > 0, N / df, 1.0)).astype(np.float32)
    X = X.multiply(idf[None, :]).tocsr().astype(np.float32)
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).flatten())
    Xn = diags(1.0 / np.maximum(norms, 1e-12)) @ X
    S = (Xn @ Xn.T).toarray().astype(np.float32)
    np.fill_diagonal(S, 0.0)
    return S


def adamic_adar(triples: Sequence[set[Triple]]) -> np.ndarray:
    """Adamic-Adar ``Σ_{t ∈ T_i ∩ T_j} 1 / log df(t)``, max-normalized over pairs (Eq. 3).

    A triple with ``df = 1`` cannot be shared, so it contributes 0.
    """
    X, df = _triple_matrix(triples)
    weight = np.where(df > 1, 1.0 / np.log(np.maximum(df, 2)), 0.0).astype(np.float32)
    S = (X.multiply(weight[None, :]) @ X.T).toarray().astype(np.float32)
    np.fill_diagonal(S, 0.0)
    m = S.max()
    if m > 0:
        S /= m
    return S


MEASURES: dict[str, Callable[[Sequence[set[Triple]]], np.ndarray]] = {
    "jaccard": jaccard,
    "overlap": overlap,
    "idf_cosine": idf_cosine,
    "adamic_adar": adamic_adar,
}


def kg_similarity(triples: Sequence[set[Triple]], measure: str = "overlap") -> np.ndarray:
    """``S_kg`` for one of :data:`MEASURES`.

    Parameters
    ----------
    triples : sequence of sets of (subject, predicate, object), length N
    measure : {"jaccard", "overlap", "idf_cosine", "adamic_adar"}

    Returns
    -------
    S_kg : (N, N) float32 in [0, 1], zero diagonal
    """
    if measure not in MEASURES:
        raise ValueError(f"unknown measure {measure!r}; choose from {sorted(MEASURES)}")
    return MEASURES[measure](triples)
