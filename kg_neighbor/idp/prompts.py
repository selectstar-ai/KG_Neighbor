"""Judge prompts and response schema for the binary judgment J(x, c; C) (Eq. 8).

One request asks about one target class; the full class list is shown as the
candidate set. The layout is never shown.
"""

from __future__ import annotations

import re

FIT = "among the best-fitting classes (more than one class may qualify)"

SYSTEM_PROMPT = (
    "You are an annotator for a classification dataset. In this dataset, every "
    "instance was constructed to belong to exactly one class from the full "
    "label list provided in each request. Your judgment is not absolute but "
    "relative: considering the entire label list as the candidate set, "
    f"decide whether the target class is {FIT} for the instance. "
    "An instance may fit more than one class equally well; judge each "
    "target on its own merit. State your reasoning first, then give your verdict."
)

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"rationale": {"type": "string"}, "valid": {"type": "boolean"}},
    "required": ["rationale", "valid"],
    "additionalProperties": False,
}


def humanize(class_name: str) -> str:
    """Display form of a class name: ``"001.Black_footed_Albatross"`` -> ``"Black footed Albatross"``."""
    s = re.sub(r"^\d+\.", "", class_name)
    s = s.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", s).strip()


def user_prompt(
    target: str,
    class_names: list[str],
    *,
    text: str | None = None,
    has_image: bool = False,
) -> str:
    """User message asking whether ``target`` fits the instance.

    Parameters
    ----------
    target : class to judge (canonical name)
    class_names : full class list of the dataset (the candidate set)
    text : instance text (text datasets)
    has_image : the instance image is attached to the request (image datasets)
    """
    blocks = []
    if has_image:
        blocks.append("[Instance image] Judge from the attached image.")
    if text is not None:
        blocks.append(f"[Instance to classify]\n{text}")
    if not blocks:
        raise ValueError("a prompt needs the instance text or image")
    labels = ", ".join(humanize(c) for c in class_names)
    return (
        "[Full label list — the complete candidate set]\n"
        f"{labels}\n\n"
        + "\n\n".join(blocks)
        + "\n\n[Question]\n"
        f"Considering the full label list above as the candidate set, is "
        f"'{humanize(target)}' {FIT} for this instance? Answer yes or no."
    )
