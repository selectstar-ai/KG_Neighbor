"""Class subsampling (Section 4.1: 20 classes, random seed 0)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from kg_neighbor.utils.data import Dataset


def sample_classes(ds: Dataset, n_classes: int = 20, *, seed: int = 0) -> Dataset:
    """Keep ``n_classes`` randomly chosen classes and relabel them ``0..n_classes-1``.

    Classes are drawn with ``np.random.default_rng(seed).choice`` and kept in
    ascending order of their original index. ``all_class_names`` is unchanged.
    Returns ``ds`` itself when it has no more than ``n_classes`` classes.
    """
    n_orig = int(ds.labels.max()) + 1
    if n_classes >= n_orig:
        return ds
    picked = sorted(np.random.default_rng(seed).choice(n_orig, size=n_classes, replace=False).tolist())
    remap = {old: new for new, old in enumerate(picked)}
    sub = ds.subset(np.isin(ds.labels, picked))
    return replace(
        sub,
        labels=np.array([remap[int(c)] for c in sub.labels]),
        class_names=[ds.class_names[c] for c in picked],
    )
