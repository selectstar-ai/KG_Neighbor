import numpy as np
import pytest

from kg_neighbor.utils.embed import length_batches
from run_pipeline import embed_chunks


def _fake_encoder(calls):
    def encode(texts):
        calls.append(len(texts))
        return np.array([[float(t[1:]), 1.0] for t in texts], dtype=np.float32)
    return encode


def test_embed_chunks_saves_parts_and_resumes(tmp_path):
    texts, ids = [f"t{i}" for i in range(5)], [str(i) for i in range(5)]
    calls = []
    Z = embed_chunks(_fake_encoder(calls), texts, ids, tmp_path, chunk_size=2, settings={"m": 1})
    assert calls == [2, 2, 1]
    assert Z[:, 0].tolist() == [0, 1, 2, 3, 4]
    calls.clear()
    Z2 = embed_chunks(_fake_encoder(calls), texts, ids, tmp_path, chunk_size=2, settings={"m": 1})
    assert calls == []  # every part already saved
    assert np.array_equal(Z, Z2)


def test_embed_chunks_refuses_parts_from_other_settings(tmp_path):
    texts, ids = ["t0", "t1"], ["0", "1"]
    embed_chunks(_fake_encoder([]), texts, ids, tmp_path, chunk_size=1, settings={"m": 1})
    with pytest.raises(SystemExit):
        embed_chunks(_fake_encoder([]), texts, ids, tmp_path, chunk_size=1, settings={"m": 2})


def test_length_batches_longest_first_and_batch_size():
    lengths = [5, 100, 20, 7, 100, 3]
    assert length_batches(lengths, batch_size=2) == [[1, 4], [2, 3], [0, 5]]


def test_length_batches_token_budget_with_truncation():
    # capped at 800: one 800-token text fills the budget, two 500s share one, the 10s share one
    lengths = [1000, 10, 10, 10, 10, 500, 500]
    batches = length_batches(lengths, batch_size=8, token_budget=1000, max_seq_length=800)
    assert batches == [[0], [5, 6], [1, 2, 3, 4]]
