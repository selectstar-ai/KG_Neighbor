import numpy as np
import pytest

from kg_neighbor import Dataset
from kg_neighbor.idp import IDP
from kg_neighbor.idp.judge import assemble, question_key
from kg_neighbor.idp.prompts import SYSTEM_PROMPT, user_prompt
from kg_neighbor.idp.states import BOUNDARY, LABEL, PLACEMENT, UNRESOLVED, diagnose


def toy():
    """Two clusters; instance 0 sits in class 1's cluster but is labeled 0."""
    rng = np.random.default_rng(0)
    Y = np.vstack([rng.normal(0, 0.1, (10, 2)), rng.normal(5, 0.1, (10, 2))])
    labels = np.repeat([0, 1], 10)
    Y[0] = [5.0, 5.0]
    ds = Dataset(
        name="toy",
        Z=Y,
        labels=labels,
        sample_ids=[f"s{i}" for i in range(20)],
        class_names=["cat", "dog"],
        all_class_names=["cat", "dog", "bird"],
        texts=[f"text {i}" for i in range(20)],
    )
    return ds, Y


def test_states():
    a = np.array([1, 0, 1, 0], dtype=bool)
    b = np.array([0, 1, 1, 0], dtype=bool)
    assert list(diagnose(a, b)) == [PLACEMENT, LABEL, BOUNDARY, UNRESOLVED]


def test_detect_and_questions():
    ds, Y = toy()
    idp = IDP(k=5)
    mm = idp.detect(ds, Y)
    assert list(mm["idx"]) == [0] and mm["A"][0] == 0 and mm["B"][0] == 1
    qs = idp.questions(ds, mm)
    assert [q["key"] for q in qs] == ["s0|cat", "s0|dog"]
    assert "cat, dog, bird" in qs[0]["prompt"]  # full class list is the candidate set


def test_missing_answer_is_false():
    ds, Y = toy()
    mm = IDP(k=5).detect(ds, Y)
    a, b = assemble(ds, mm, {question_key("s0", "dog"): {"valid": True}})
    assert not a[0] and b[0]


def test_run_with_fake_judge():
    class Judge:
        def run(self, questions):
            return {q["key"]: {"valid": q["class_name"] == "cat"} for q in questions}

    ds, Y = toy()
    result, new = IDP(k=5, judge=Judge()).run(ds, Y)
    assert result["n_mismatches"] == 1
    assert result["counts"][PLACEMENT] == 1
    assert len(new) == 2


def test_prompts():
    assert "among the best-fitting classes" in SYSTEM_PROMPT
    assert "'Black footed Albatross'" in user_prompt("001.Black_footed_Albatross", ["a"], has_image=True)
    with pytest.raises(ValueError):
        user_prompt("a", ["a"])
