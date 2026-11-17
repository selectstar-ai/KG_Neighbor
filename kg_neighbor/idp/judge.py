"""Questions for the judge and assembly of its answers (Section 3.4, Eq. 8).

Each mismatch yields two independent questions: does the content support the
assigned class ``A_i``, and does it support the neighborhood-majority class
``B_i``? A question is keyed by ``"{sample_id}|{class_name}"``; it does not
depend on the layout, so one answer serves every layout and guidance strength.
"""

from __future__ import annotations

import numpy as np

from kg_neighbor.idp.prompts import user_prompt
from kg_neighbor.utils.data import Dataset


def question_key(sample_id: str, class_name: str) -> str:
    return f"{sample_id}|{class_name}"


def make_questions(ds: Dataset, mismatches: dict[str, np.ndarray]) -> list[dict]:
    """Unique questions for the mismatches.

    Image datasets are judged from the image, text datasets from the text.

    Returns
    -------
    list of dicts with ``key``, ``sample_id``, ``class_name``, ``prompt`` and,
    for image datasets, ``image_path``
    """
    use_image = ds.image_paths is not None
    if not use_image and ds.texts is None:
        raise ValueError("the dataset needs texts or image_paths for the judge")
    seen: dict[str, dict] = {}
    for i, a, b in zip(mismatches["idx"], mismatches["A"], mismatches["B"]):
        sid = ds.sample_ids[i]
        for c in (a, b):
            name = ds.class_names[c]
            key = question_key(sid, name)
            if key in seen:
                continue
            q = {
                "key": key,
                "sample_id": sid,
                "class_name": name,
                "prompt": user_prompt(
                    name,
                    ds.all_class_names,
                    text=None if use_image else ds.texts[i],
                    has_image=use_image,
                ),
            }
            if use_image:
                q["image_path"] = ds.image_paths[i]
            seen[key] = q
    return list(seen.values())


def assemble(
    ds: Dataset, mismatches: dict[str, np.ndarray], answers: dict[str, dict]
) -> tuple[np.ndarray, np.ndarray]:
    """Judgments ``(a, b)`` per mismatch from answers keyed by :func:`question_key`.

    ``answers[key]`` is the judge's structured output ``{"valid": bool, ...}``.
    A missing answer counts as ``False``.
    """

    def valid(i, c) -> bool:
        ans = answers.get(question_key(ds.sample_ids[i], ds.class_names[c]))
        return bool(ans["valid"]) if ans else False

    rows = list(zip(mismatches["idx"], mismatches["A"], mismatches["B"]))
    a = np.array([valid(i, c) for i, c, _ in rows], dtype=bool)
    b = np.array([valid(i, c) for i, _, c in rows], dtype=bool)
    return a, b
