"""Four diagnostic states from the two judgments (Section 3.4)."""

from __future__ import annotations

import numpy as np

PLACEMENT = "placement_issue"  # a = 1, b = 0
LABEL = "label_issue"  # a = 0, b = 1
BOUNDARY = "boundary_case"  # a = 1, b = 1
UNRESOLVED = "unresolved_case"  # a = 0, b = 0
STATES = (PLACEMENT, LABEL, BOUNDARY, UNRESOLVED)


def diagnose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """State per mismatch from ``a`` (assigned class supported) and ``b`` (majority class supported)."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    return np.select(
        [a & ~b, ~a & b, a & b],
        [PLACEMENT, LABEL, BOUNDARY],
        default=UNRESOLVED,
    ).astype(object)


def summarize(states: np.ndarray, n_total: int) -> dict:
    """Detection rate (mismatches / all instances) and the count of each state."""
    states = np.asarray(states, dtype=object)
    return {
        "n_instances": int(n_total),
        "n_mismatches": int(len(states)),
        "detection_rate": len(states) / n_total if n_total else 0.0,
        "counts": {s: int(np.sum(states == s)) for s in STATES},
    }
