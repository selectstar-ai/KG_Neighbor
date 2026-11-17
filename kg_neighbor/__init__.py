"""KG-Neighbor: Knowledge Guided Neighborhood Construction for Embedding Visualization.

    S_kg = kg_similarity(ds.triples, "overlap")                  # Section 3.3, Eqs. 1-3
    nn = KGNeighbors(n_neighbors=15, alpha=0.3, S_kg=S_kg).fit(ds.Z)  # Eqs. 4-6
    Y = umap_layout(ds.Z, nn, seed=0)                            # Section 4.2
"""

from kg_neighbor.layout import umap_layout
from kg_neighbor.metrics import lt_2d, neighbor_retention, nmi_kmeans
from kg_neighbor.neighbors import KGNeighbors, kg_guided_distance
from kg_neighbor.similarity import MEASURES, kg_similarity
from kg_neighbor.utils.data import Dataset, load_dataset
from kg_neighbor.utils.sampling import sample_classes

__version__ = "1.0.0"

__all__ = [
    "MEASURES",
    "Dataset",
    "KGNeighbors",
    "kg_guided_distance",
    "kg_similarity",
    "load_dataset",
    "lt_2d",
    "neighbor_retention",
    "nmi_kmeans",
    "sample_classes",
    "umap_layout",
]
