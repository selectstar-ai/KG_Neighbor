"""Raw sources -> on-disk layout read by :func:`kg_neighbor.utils.data.load_dataset`.

Instance ids, texts and class indices match the files used in the paper.

Nothing is downloaded here: the raw files are placed under ``data/<dataset>/raw/`` beforehand
(``SOURCES`` says where each one comes from), and a missing file stops with that hint.

- StackOverflow: titles and labels from the ``jacoxu/StackOverflow`` GitHub repository
- BANKING77: ``train.csv`` and ``test.csv`` from ``PolyAI-LDN/task-specific-datasets``
- CUB-200-2011 / Flowers 102: captions from Reed et al. (2016) and images from the original archives
- Flowers 102 class names: ``FLOWERS102_CLASSES`` below (the archives hold class numbers only)
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import tarfile
from pathlib import Path

SO_CLASSES = [
    "wordpress", "oracle", "svn", "apache", "excel", "matlab", "visual-studio", "cocoa",
    "osx", "bash", "spring", "hibernate", "scala", "sharepoint", "ajax", "qt", "drupal",
    "linq", "haskell", "magento",
]
# Oxford Flowers 102 class names in label order (index == class_id), as listed in the TensorFlow Datasets
# ``oxford_flowers102`` builder; the archives hold class numbers only. The trailing ``?`` of
# ``pink-yellow dahlia?`` is dropped.
FLOWERS102_CLASSES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea",
    "english marigold", "tiger lily", "moon orchid", "bird of paradise", "monkshood",
    "globe thistle", "snapdragon", "colt's foot", "king protea", "spear thistle", "yellow iris",
    "globe-flower", "purple coneflower", "peruvian lily", "balloon flower", "giant white arum lily",
    "fire lily", "pincushion flower", "fritillary", "red ginger", "grape hyacinth", "corn poppy",
    "prince of wales feathers", "stemless gentian", "artichoke", "sweet william", "carnation",
    "garden phlox", "love in the mist", "mexican aster", "alpine sea holly", "ruby-lipped cattleya",
    "cape flower", "great masterwort", "siam tulip", "lenten rose", "barbeton daisy", "daffodil",
    "sword lily", "poinsettia", "bolero deep blue", "wallflower", "marigold", "buttercup",
    "oxeye daisy", "common dandelion", "petunia", "wild pansy", "primula", "sunflower",
    "pelargonium", "bishop of llandaff", "gaura", "geranium", "orange dahlia", "pink-yellow dahlia",
    "cautleya spicata", "japanese anemone", "black-eyed susan", "silverbush", "californian poppy",
    "osteospermum", "spring crocus", "bearded iris", "windflower", "tree poppy", "gazania",
    "azalea", "water lily", "rose", "thorn apple", "morning glory", "passion flower", "lotus",
    "toad lily", "anthurium", "frangipani", "clematis", "hibiscus", "columbine", "desert-rose",
    "tree mallow", "magnolia", "cyclamen", "watercress", "canna lily", "hippeastrum", "bee balm",
    "ball moss", "foxglove", "bougainvillea", "camellia", "mallow", "mexican petunia", "bromelia",
    "blanket flower", "trumpet creeper", "blackberry lily",
]
REED_FILE = {"cub200": "cvpr2016_cub.tar.gz", "flowers102": "cvpr2016_flowers.tar.gz"}
IMAGE_FILE = {"cub200": "CUB_200_2011.tgz", "flowers102": "102flowers.tgz"}
SOURCES = {  # file under data/<dataset>/raw/ -> where to download it
    "title_StackOverflow.txt": "https://github.com/jacoxu/StackOverflow (rawText/)",
    "label_StackOverflow.txt": "https://github.com/jacoxu/StackOverflow (rawText/)",
    "train.csv": "https://github.com/PolyAI-LDN/task-specific-datasets (banking_data/)",
    "test.csv": "https://github.com/PolyAI-LDN/task-specific-datasets (banking_data/)",
    "cvpr2016_cub.tar.gz": "https://github.com/reedscot/cvpr2016 (birds data)",
    "cvpr2016_flowers.tar.gz": "https://github.com/reedscot/cvpr2016 (flowers data)",
    "CUB_200_2011.tgz": "https://www.vision.caltech.edu/datasets/cub_200_2011/",
    "102flowers.tgz": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/",
}
TRIPLE_RE = re.compile(r"^\((.+?), (.+?), (.+?)\)$")


def write_dataset(out: Path, rows: list[dict], class_names: list[str], source: str) -> None:
    """``texts/sentences.jsonl`` and ``meta.json`` under ``out``.

    Keys already in ``meta.json`` that are not written here are kept.
    """
    (out / "texts").mkdir(parents=True, exist_ok=True)
    with (out / "texts" / "sentences.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    path = out / "meta.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    meta = {**old, "source": source, "n_samples": len(rows), "n_classes": len(class_names),
            "class_names": class_names}
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _source(raw: Path, name: str) -> Path:
    """A raw file placed under ``raw`` beforehand; a missing one stops with where to get it."""
    path = raw / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: download {name} from {SOURCES[name]} and place it there")
    return path


# text datasets ---------------------------------------------------------------
def build_stackoverflow(out: Path) -> int:
    """20,000 titles; ids ``so_00000`` follow the line order of the source file."""
    titles = _source(out / "raw", "title_StackOverflow.txt")
    labels = _source(out / "raw", "label_StackOverflow.txt")
    texts = titles.read_text(encoding="utf-8").splitlines()
    ys = [int(x) - 1 for x in labels.read_text().splitlines()]  # 1-indexed in the source
    rows = [{"id": f"so_{i:05d}", "text": t, "class_id": y, "label_name": SO_CLASSES[y],
             "split": "train"} for i, (t, y) in enumerate(zip(texts, ys))]
    write_dataset(out, rows, SO_CLASSES,
                  "StackOverflow 20-class short text (Xu et al. 2015) — jacoxu/StackOverflow GitHub")
    return len(rows)


def build_banking77(out: Path) -> int:
    """13,083 queries (train then test); ids ``bank_00000`` follow that order.

    Classes are numbered in case-insensitive alphabetical order of their names, which gives the
    same rows and class indices as ``legacy-datasets/banking77`` on the Hugging Face Hub.
    """
    pairs = []
    for split in ("train", "test"):
        with _source(out / "raw", f"{split}.csv").open(encoding="utf-8", newline="") as f:
            pairs += [(r["text"], r["category"], split) for r in csv.DictReader(f)]
    names = sorted({c for _, c, _ in pairs}, key=str.lower)
    index = {c: i for i, c in enumerate(names)}
    rows = [{"id": f"bank_{i:05d}", "text": text, "class_id": index[c], "label_name": c, "split": split}
            for i, (text, c, split) in enumerate(pairs)]
    write_dataset(out, rows, names,
                  "BANKING77 (Casanueva et al. 2020) — PolyAI-LDN/task-specific-datasets (banking_data)")
    return len(rows)


# image datasets --------------------------------------------------------------
def build_reed(
    name: str, out: Path, *, archive: Path | None = None, class_names: list[str] | None = None
) -> int:
    """Captions of Reed et al. (2016): ``text`` joins the 10 captions of each image.

    Parameters
    ----------
    name : ``"cub200"`` or ``"flowers102"``
    archive : the Reed text archive at another path (default: ``raw/cvpr2016_*.tar.gz``)
    class_names : names for numbered class folders (Flowers 102: index == class_id)
    """
    raw = out / "raw"
    if archive is None:
        archive = _source(raw, REED_FILE[name])
    ext = raw / "reed"
    if not (ext / "text_c10").exists():
        with tarfile.open(archive) as t:
            members = [m for m in t.getmembers()
                       if "/text_c10/" in "/" + m.name or m.name.startswith("text_c10")]
            t.extractall(ext, members=members)
    tc = next(ext.rglob("text_c10"))

    class_dirs = sorted(d for d in tc.iterdir() if d.is_dir())
    numeric = all(re.fullmatch(r"class_\d+", d.name) for d in class_dirs)
    if numeric and (class_names is None or len(class_names) != len(class_dirs)):
        raise ValueError(f"{name}: numbered class folders need {len(class_dirs)} class_names")
    rows, names = [], []
    for cid, cdir in enumerate(class_dirs):
        if numeric:
            if int(cdir.name.split("_")[1]) - 1 != cid:
                raise ValueError(f"unexpected class folder order at {cdir.name}")
            label = class_names[cid]
        else:
            label = cdir.name  # e.g. "001.Black_footed_Albatross"
        names.append(label)
        for txt in sorted(cdir.glob("*.txt")):
            caps = [line.strip() for line in
                    txt.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
            if caps:
                rows.append({"id": txt.stem, "text": " ".join(caps), "captions": caps,
                             "class_id": cid, "label_name": label, "image_file": txt.stem + ".jpg"})
    source = {"cub200": "CUB-200-2011", "flowers102": "Oxford Flowers 102"}[name]
    write_dataset(out, rows, names, f"{source} + Reed et al. (2016) captions (reedscot/cvpr2016)")
    return len(rows)


def fetch_images(name: str, out: Path, *, archive: Path | None = None) -> int:
    """Extract the image archive into a flat ``images/`` folder and check every instance.

    ``archive`` is the image archive at another path (default: ``raw/CUB_200_2011.tgz`` or
    ``raw/102flowers.tgz``).
    """
    if archive is None:
        archive = _source(out / "raw", IMAGE_FILE[name])
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as t:
        for m in t:
            if m.isfile() and m.name.lower().endswith(".jpg"):
                dst = images / Path(m.name).name
                if not dst.exists():
                    with t.extractfile(m) as src, dst.open("wb") as f:
                        shutil.copyfileobj(src, f)
    rows = [json.loads(line) for line in (out / "texts" / "sentences.jsonl").open(encoding="utf-8")]
    missing = [r["image_file"] for r in rows if not (images / r["image_file"]).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images missing, e.g. {missing[:3]}")
    return len(rows)


# KG extraction output ----------------------------------------------------------
def collect_triples(out: Path) -> dict[str, int]:
    """Per-instance ``kg/**/<sample_id>.triples`` -> ``kg/triples.tsv`` in instance order.

    Each ``.triples`` line is ``(subject, predicate, object)``; ``#`` lines are comments.
    """
    files = {p.stem: p for p in (out / "kg").rglob("*.triples")}
    rows = [json.loads(line) for line in (out / "texts" / "sentences.jsonl").open(encoding="utf-8")]
    n = {"instances": 0, "triples": 0, "missing": 0, "malformed": 0}
    with (out / "kg" / "triples.tsv").open("w", encoding="utf-8") as f:
        for r in rows:
            p = files.get(r["id"])
            if p is None:
                n["missing"] += 1
                continue
            lines = [line.strip() for line in p.read_text(encoding="utf-8").splitlines()
                     if line.strip() and not line.lstrip().startswith("#")]
            n["instances"] += bool(lines)
            for line in lines:
                m = TRIPLE_RE.match(line)
                if not m:
                    n["malformed"] += 1
                    continue
                f.write(r["id"] + "\t" + "\t".join(m.groups()) + "\n")
                n["triples"] += 1
    return n
