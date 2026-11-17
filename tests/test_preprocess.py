import io
import json
import tarfile

import pytest

from kg_neighbor.utils.data import read_jsonl
from kg_neighbor.utils.preprocess import (FLOWERS102_CLASSES, build_banking77, build_reed, build_stackoverflow,
                                          collect_triples, fetch_images, write_dataset)


def test_missing_source_names_the_file_and_where_to_get_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="title_StackOverflow.txt.*github.com/jacoxu/StackOverflow"):
        build_stackoverflow(tmp_path)  # nothing is downloaded


def test_banking77_reads_csv_in_train_then_test_order(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "train.csv").write_text(
        'text,category\n"Top up, please",top_up\nno refund,Refund_not_showing_up\n', encoding="utf-8")
    (tmp_path / "raw" / "test.csv").write_text("text,category\nwhere is it,card_arrival\n", encoding="utf-8")
    assert build_banking77(tmp_path) == 3
    rows = read_jsonl(tmp_path / "texts" / "sentences.jsonl")
    assert rows[0] == {"id": "bank_00000", "text": "Top up, please", "class_id": 2, "label_name": "top_up",
                       "split": "train"}
    assert [(r["id"], r["split"]) for r in rows[1:]] == [("bank_00001", "train"), ("bank_00002", "test")]
    # class order ignores case, as in the dataset used in the paper
    names = json.loads((tmp_path / "meta.json").read_text())["class_names"]
    assert names == ["card_arrival", "Refund_not_showing_up", "top_up"]


def test_flowers102_class_names():
    assert len(FLOWERS102_CLASSES) == len(set(FLOWERS102_CLASSES)) == 102
    assert (FLOWERS102_CLASSES[0], FLOWERS102_CLASSES[59]) == ("pink primrose", "pink-yellow dahlia")


def test_write_dataset_keeps_other_meta_keys(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"class_names_source": "x", "class_names": ["a"]}))
    write_dataset(tmp_path, [{"id": "1", "text": "t", "class_id": 0, "label_name": "a"}], ["a"], "src")
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["class_names_source"] == "x"
    assert (meta["source"], meta["n_samples"], meta["class_names"]) == ("src", 1, ["a"])


def make_tar(path, files: dict[str, bytes]):
    with tarfile.open(path, "w:gz") as t:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))


def test_reed_numbered_classes(tmp_path):
    arc = tmp_path / "reed.tgz"
    make_tar(arc, {
        "cvpr2016_flowers/text_c10/class_00001/image_00002.txt": b"red petals\nround\n",
        "cvpr2016_flowers/text_c10/class_00002/image_00001.txt": b"yellow center\n",
    })
    out = tmp_path / "flowers102"
    assert build_reed("flowers102", out, archive=arc, class_names=["rose", "daisy"]) == 2
    rows = read_jsonl(out / "texts" / "sentences.jsonl")
    assert rows[0] == {"id": "image_00002", "text": "red petals round", "captions": ["red petals", "round"],
                       "class_id": 0, "label_name": "rose", "image_file": "image_00002.jpg"}
    assert json.loads((out / "meta.json").read_text())["class_names"] == ["rose", "daisy"]
    with pytest.raises(ValueError):
        build_reed("flowers102", tmp_path / "again", archive=arc)  # numbered folders need names


def test_images_and_triples(tmp_path):
    out = tmp_path / "cub200"
    arc = tmp_path / "reed.tgz"
    make_tar(arc, {"text_c10/001.Albatross/Albatross_0001.txt": b"a bird\n"})
    build_reed("cub200", out, archive=arc)
    img = tmp_path / "images.tgz"
    make_tar(img, {"CUB_200_2011/images/001.Albatross/Albatross_0001.jpg": b"jpg"})
    assert fetch_images("cub200", out, archive=img) == 1
    assert (out / "images" / "Albatross_0001.jpg").exists()

    (out / "kg" / "any_folder").mkdir(parents=True)
    (out / "kg" / "any_folder" / "Albatross_0001.triples").write_text(
        "# comment\n(Bird, has_color, White)\nnot a triple\n", encoding="utf-8")
    assert collect_triples(out) == {"instances": 1, "triples": 1, "missing": 0, "malformed": 1}
    assert (out / "kg" / "triples.tsv").read_text() == "Albatross_0001\tBird\thas_color\tWhite\n"
