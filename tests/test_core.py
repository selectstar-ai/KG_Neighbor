import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors

from kg_neighbor import (
    KGNeighbors,
    kg_guided_distance,
    kg_similarity,
    lt_2d,
    neighbor_retention,
    nmi_kmeans,
)

A, B, C, D = ("a", "r", "x"), ("b", "r", "y"), ("c", "r", "z"), ("d", "r", "w")
TRIPLES = [{A, B}, {B, C}, {D}, set()]


def test_jaccard_and_overlap_by_hand():
    J = kg_similarity(TRIPLES, "jaccard")
    O = kg_similarity(TRIPLES, "overlap")
    assert J[0, 1] == pytest.approx(1 / 3)
    assert O[0, 1] == pytest.approx(1 / 2)
    assert J[0, 2] == 0 and O[0, 2] == 0


@pytest.mark.parametrize("measure", ["jaccard", "overlap", "idf_cosine", "adamic_adar"])
def test_similarity_range_and_diagonal(measure):
    S = kg_similarity(TRIPLES, measure)
    assert S.shape == (4, 4)
    assert np.all(np.diag(S) == 0)
    assert np.all((S >= 0) & (S <= 1 + 1e-6))
    assert not np.isnan(S).any()  # empty triple set gives a zero row


def test_kg_guided_distance_formula():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(5, 3)).astype(np.float32)
    S = np.zeros((5, 5), dtype=np.float32)
    S[0, 1] = S[1, 0] = 0.5
    D0 = kg_guided_distance(Z, S, 0.0)
    D = kg_guided_distance(Z, S, 0.4)
    assert D[0, 1] == pytest.approx(D0[0, 1] * (1 - 0.4 * 0.5), rel=1e-5)
    assert D[0, 2] == pytest.approx(D0[0, 2], rel=1e-6)


def test_alpha_zero_is_sklearn():
    Z = np.random.default_rng(0).normal(size=(30, 4))
    ours = KGNeighbors(5).fit(Z).kneighbors()
    ref = NearestNeighbors(n_neighbors=5, metric="cosine").fit(Z).kneighbors()
    assert np.array_equal(ours[1], ref[1])


def test_zero_kg_similarity_keeps_neighbors():
    Z = np.random.default_rng(1).normal(size=(30, 4))
    _, idx0 = KGNeighbors(5).fit(Z).kneighbors()
    _, idx = KGNeighbors(5, alpha=0.7, S_kg=np.zeros((30, 30))).fit(Z).kneighbors()
    assert neighbor_retention(idx, idx0) == 1.0


def test_alpha_needs_kg():
    with pytest.raises(ValueError):
        KGNeighbors(5, alpha=0.3)


def test_separated_clusters_score_one():
    rng = np.random.default_rng(0)
    Y = np.vstack([rng.normal(0, 0.1, (20, 2)), rng.normal(10, 0.1, (20, 2))])
    labels = np.repeat([0, 1], 20)
    assert nmi_kmeans(Y, labels) == pytest.approx(1.0)
    assert lt_2d(Y, labels, k=10) == pytest.approx(1.0)


def test_umap_layout_shape():
    pytest.importorskip("umap")
    from kg_neighbor import umap_layout

    Z = np.random.default_rng(0).normal(size=(60, 8)).astype(np.float32)
    S = kg_similarity([{A}] * 30 + [{B}] * 30, "overlap")
    Y = umap_layout(Z, KGNeighbors(15, alpha=0.5, S_kg=S).fit(Z), n_epochs=20)
    assert Y.shape == (60, 2) and np.isfinite(Y).all()
