"""Dataset container and file loading.

On-disk layout per dataset, inputs under ``<root>/<name>/``::

    meta.json                      {"class_names": [...], ...}
    texts/sentences.jsonl          {"id", "text", "class_id", "label_name", ["image_file"]}
    images/<image_file>            image datasets only
    kg/triples.tsv                 sample_id <TAB> subject <TAB> predicate <TAB> object

and embeddings under ``<results_root>/<name>/embeddings/<encoder>/``::

    embedding_vectors.npy (N, d), embedding_labels.npy (N,), embedding_ids.npy (N,)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

Triple = tuple[str, str, str]


@dataclass
class Dataset:
    """Everything the pipeline needs for one dataset--encoder condition.

    Rows of every per-instance field follow the order of ``Z``.
    """

    name: str
    Z: np.ndarray  # (N, d) input embeddings
    labels: np.ndarray  # (N,) class index into class_names
    sample_ids: list[str]
    class_names: list[str]  # classes present in this (possibly sampled) dataset
    all_class_names: list[str]  # every class of the original dataset (IDP candidate list)
    triples: list[set[Triple]] | None = None
    texts: list[str] | None = None
    image_paths: list[str] | None = None

    def __len__(self) -> int:
        return len(self.labels)

    def subset(self, mask: np.ndarray) -> Dataset:
        """Rows where ``mask`` is True; class fields are left unchanged."""
        mask = np.asarray(mask, dtype=bool)

        def pick(xs):
            return None if xs is None else [x for x, m in zip(xs, mask) if m]

        return replace(
            self,
            Z=self.Z[mask],
            labels=self.labels[mask],
            sample_ids=pick(self.sample_ids),
            triples=pick(self.triples),
            texts=pick(self.texts),
            image_paths=pick(self.image_paths),
        )


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_triples(path: str | Path) -> dict[str, set[Triple]]:
    """4-column ``triples.tsv`` -> {sample_id: set of (subject, predicate, object)}."""
    out: dict[str, set[Triple]] = {}
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                continue
            sid, s, p, o = parts
            out.setdefault(sid, set()).add((s, p, o))
    return out


def load_dataset(
    name: str,
    encoder: str,
    *,
    root: str | Path = "data",
    results_root: str | Path = "results",
    emb_dir: str | Path | None = None,
) -> Dataset:
    """Load embeddings, labels, class names, triples, texts and image paths.

    Parameters
    ----------
    name : dataset folder under ``root`` (e.g. ``"stackoverflow"``)
    encoder : embedding folder under ``<results_root>/<name>/embeddings/``
    emb_dir : explicit embedding folder, overriding the default location

    Raises
    ------
    ValueError
        if an embedded instance has no triples in ``kg/triples.tsv``.
    """
    ds_dir = Path(root) / name
    emb_dir = Path(emb_dir) if emb_dir else Path(results_root) / name / "embeddings" / encoder

    Z = np.load(emb_dir / "embedding_vectors.npy").astype(np.float32)
    labels = np.load(emb_dir / "embedding_labels.npy")
    meta = json.loads((ds_dir / "meta.json").read_text(encoding="utf-8"))
    class_names = list(meta["class_names"])

    rows = read_jsonl(ds_dir / "texts" / "sentences.jsonl")
    ids_path = emb_dir / "embedding_ids.npy"
    sample_ids = (
        [str(s) for s in np.load(ids_path, allow_pickle=True)]
        if ids_path.exists()
        else [r["id"] for r in rows]
    )
    if len(sample_ids) != len(Z):
        raise ValueError(f"{len(sample_ids)} sample ids for {len(Z)} embeddings")
    by_id = {r["id"]: r for r in rows}

    triples = None
    tsv = ds_dir / "kg" / "triples.tsv"
    if tsv.exists():
        T = read_triples(tsv)
        missing = [s for s in sample_ids if s not in T]
        if missing:
            raise ValueError(f"no triples for {len(missing)} instances, e.g. {missing[:3]}")
        triples = [T[s] for s in sample_ids]

    texts = [by_id[s]["text"] for s in sample_ids] if by_id else None
    image_paths = None
    if all("image_file" in by_id.get(s, {}) for s in sample_ids):
        image_paths = [str(ds_dir / "images" / by_id[s]["image_file"]) for s in sample_ids]

    return Dataset(
        name=name,
        Z=Z,
        labels=labels,
        sample_ids=sample_ids,
        class_names=class_names,
        all_class_names=class_names,
        triples=triples,
        texts=texts,
        image_paths=image_paths,
    )
