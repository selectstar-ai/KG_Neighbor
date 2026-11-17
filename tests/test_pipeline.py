import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import run_pipeline as rp
from kg_neighbor import load_dataset


def test_outputs_go_under_results_with_the_default_kg():
    args = SimpleNamespace(data_root=Path("D"), results_root=Path("R"), emb_root=None)
    assert rp.emb_dir(args, "so", "bert") == Path("R/so/embeddings/bert")
    assert rp.layout_dir(args, "so", "bert") == Path("R/so/layouts/bert/default")
    assert rp.layout_file(Path("L"), "overlap_a0.1") == Path("L/layout_overlap_a0.1.npy")
    assert rp.metrics_path(args, "so", "bert") == Path("R/so/metrics/bert/metric_default.json")
    assert rp.plot_dir(args, "so", "bert") == Path("R/so/plot/bert/default")
    assert rp.analysis_dir(args, "so", "bert") == Path("R/so/analysis/bert/default")
    assert rp.questions_path(args, "so") == Path("R/so/idp/idp_questions.json")
    assert rp.judge_dir(args, "so", "gpt-5.4") == Path("R/so/idp/gpt-5.4")
    assert rp.idp_path(args, "so", "bert", "gpt-5.4") == Path("R/so/idp/gpt-5.4/bert/idp_default.json")


def test_load_env_keeps_set_and_skips_empty(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# keys\nKGN_A='a1'\nexport KGN_B=b1\nKGN_C=\nKGN_D=d1\n")
    for k in ("KGN_A", "KGN_B", "KGN_C"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("KGN_D", "shell")
    rp.load_env(env)
    import os
    assert (os.environ["KGN_A"], os.environ["KGN_B"], os.environ["KGN_D"]) == ("a1", "b1", "shell")
    assert "KGN_C" not in os.environ


def test_load_dataset_reads_embedding_files(tmp_path):
    data, emb = tmp_path / "data" / "so", tmp_path / "results" / "so" / "embeddings" / "bert"
    (data / "texts").mkdir(parents=True)
    emb.mkdir(parents=True)
    (data / "meta.json").write_text(json.dumps({"class_names": ["a", "b"]}))
    (data / "texts" / "sentences.jsonl").write_text(
        "".join(json.dumps({"id": i, "text": i}) + "\n" for i in ("x", "y")))
    np.save(emb / "embedding_vectors.npy", np.eye(2, dtype=np.float32))
    np.save(emb / "embedding_labels.npy", np.array([0, 1]))
    np.save(emb / "embedding_ids.npy", np.array(["x", "y"]))
    ds = load_dataset("so", "bert", root=tmp_path / "data", results_root=tmp_path / "results")
    assert ds.sample_ids == ["x", "y"] and ds.Z.shape == (2, 2) and ds.labels.tolist() == [0, 1]


class FakeJudge:
    """Stands in for a batch judge; ``jobs`` plays the API side and outlives one run."""

    def __init__(self, jobs, fail=None):
        self.jobs, self.fail = jobs, fail  # fail: "submit" | "wait" (the run breaks) | "job" (the job fails)

    def settings(self):
        return {"judge": "fake", "model": "fake"}

    @staticmethod
    def request_id(key):
        return key

    def build_request(self, question):
        return {"custom_id": question["key"]}

    def submit(self, requests):
        if self.fail == "submit":
            raise ConnectionError
        job = f"job{len(self.jobs) + 1}"
        self.jobs[job] = [r["custom_id"] for r in requests]
        return [job]

    def wait(self, job_ids):
        if self.fail == "wait":
            raise ConnectionError
        if self.fail == "job":
            raise RuntimeError("batch failed")
        return [{"custom_id": k} for j in job_ids for k in self.jobs[j]]

    @staticmethod
    def parse(rows):
        return {r["custom_id"]: {"rationale": "r", "valid": True} for r in rows}


def questions(*keys):
    return [{"key": k} for k in keys]


def read_meta(folder):
    return json.loads((folder / "idp_meta.json").read_text())


def test_judge_folder_never_asks_a_question_twice(tmp_path):
    jobs, keys = {}, ["a|x", "a|y", "b|x"]
    assert set(rp.run_judge(tmp_path, FakeJudge(jobs), questions("a|x", "a|y"), keys)) == {"a|x", "a|y"}
    assert set(rp.run_judge(tmp_path, FakeJudge(jobs), questions("a|x", "b|x"), keys)) == set(keys)
    assert list(jobs.values()) == [["a|x", "a|y"], ["b|x"]]
    meta = read_meta(tmp_path)
    assert meta["model"] == "fake" and meta["batches"]["batch_0002"] == {
        "job_ids": ["job2"], "n_requests": 1, "n_responses": 1, "status": "received"}
    for name in ("batch_0001", "batch_0002"):
        assert (tmp_path / "batches" / f"{name}.jsonl").exists() and (tmp_path / "responses" / f"{name}.jsonl").exists()


@pytest.mark.parametrize("fail, status", [("submit", "composed"), ("wait", "sent")])
def test_interrupted_batch_resumes_without_a_new_batch(tmp_path, fail, status):
    jobs = {}
    with pytest.raises(ConnectionError):
        rp.run_judge(tmp_path, FakeJudge(jobs, fail), questions("a|x"), ["a|x"])
    assert read_meta(tmp_path)["batches"]["batch_0001"]["status"] == status
    assert (tmp_path / "batches" / "batch_0001.jsonl").exists()  # saved before it is sent
    assert set(rp.run_judge(tmp_path, FakeJudge(jobs), questions("a|x"), ["a|x"])) == {"a|x"}
    assert list(jobs.values()) == [["a|x"]] and list(read_meta(tmp_path)["batches"]) == ["batch_0001"]


def test_failed_job_is_recorded_and_its_questions_go_into_a_new_batch(tmp_path):
    jobs = {}
    with pytest.raises(RuntimeError):
        rp.run_judge(tmp_path, FakeJudge(jobs, "job"), questions("a|x"), ["a|x"])
    rp.run_judge(tmp_path, FakeJudge(jobs), questions("a|x"), ["a|x"])
    assert [b["status"] for b in read_meta(tmp_path)["batches"].values()] == ["failed", "received"]


def test_other_judge_settings_stop_the_run(tmp_path):
    (tmp_path / "idp_meta.json").write_text(json.dumps({"judge": "fake", "model": "other", "batches": {}}))
    with pytest.raises(SystemExit):
        rp.run_judge(tmp_path, FakeJudge({}), questions("a|x"), ["a|x"])
