"""IDP: Instance Diagnosis Profile for residual mismatches (Section 3.4).

detect (layout kNN majority) -> questions (two per mismatch) -> judge -> four states.
"""

from __future__ import annotations

import numpy as np

from kg_neighbor.idp.candidates import detect_mismatches
from kg_neighbor.idp.judge import assemble, make_questions
from kg_neighbor.idp.llm_batch import AnthropicBatch, IDPBatch, OpenAIBatch
from kg_neighbor.idp.states import STATES, diagnose, summarize
from kg_neighbor.utils.data import Dataset

__all__ = ["IDP", "IDPBatch", "OpenAIBatch", "AnthropicBatch", "STATES"]


class IDP:
    """Diagnose the residual mismatches of one layout.

    Parameters
    ----------
    k : neighbors for mismatch detection on the layout (paper: 15)
    judge : object with ``run(questions) -> {key: {"valid": bool, ...}}``,
        such as an :class:`IDPBatch` (:class:`OpenAIBatch`, :class:`AnthropicBatch`). Needed only by :meth:`run`.

    Examples
    --------
    >>> idp = IDP(k=15, judge=OpenAIBatch())
    >>> result, new_answers = idp.run(ds, Y, cache=answers_so_far)

    Without API access, use the three steps separately:

    >>> mm = idp.detect(ds, Y)
    >>> questions = idp.questions(ds, mm)          # send these to any judge
    >>> result = idp.diagnose(ds, mm, answers)
    """

    def __init__(self, k: int = 15, judge=None):
        self.k = k
        self.judge = judge

    def detect(self, ds: Dataset, Y: np.ndarray) -> dict[str, np.ndarray]:
        """Mismatches ``A_i != B_i`` on layout ``Y``."""
        return detect_mismatches(Y, ds.labels, k=self.k)

    def questions(self, ds: Dataset, mismatches: dict[str, np.ndarray]) -> list[dict]:
        """Unique judge questions, two per mismatch."""
        return make_questions(ds, mismatches)

    def diagnose(
        self, ds: Dataset, mismatches: dict[str, np.ndarray], answers: dict[str, dict]
    ) -> dict:
        """Detection rate, state counts, and the state of every mismatch."""
        a, b = assemble(ds, mismatches, answers)
        states = diagnose(a, b)
        out = summarize(states, n_total=len(ds))
        out["mismatches"] = [
            {"sample_id": ds.sample_ids[i], "assigned": ds.class_names[A],
             "majority": ds.class_names[B], "a": bool(x), "b": bool(y), "state": s}
            for i, A, B, x, y, s in zip(
                mismatches["idx"], mismatches["A"], mismatches["B"], a, b, states
            )
        ]
        return out

    def run(
        self, ds: Dataset, Y: np.ndarray, cache: dict[str, dict] | None = None
    ) -> tuple[dict, dict[str, dict]]:
        """detect -> questions -> judge (uncached only) -> diagnose.

        Returns the diagnosis and the newly judged answers, so the caller can
        add them to its cache.
        """
        if self.judge is None:
            raise ValueError("IDP.run needs a judge, e.g. IDP(judge=OpenAIBatch())")
        cache = cache or {}
        mm = self.detect(ds, Y)
        todo = [q for q in self.questions(ds, mm) if q["key"] not in cache]
        new = self.judge.run(todo)
        return self.diagnose(ds, mm, {**cache, **new}), new
