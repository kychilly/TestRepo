"""Calibration helpers kept separate from the primary metric calculations."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

from .metrics import non_estimable


def binary_calibration(
    true: np.ndarray[Any, Any], scores: np.ndarray[Any, Any], min_samples: int = 10
) -> dict[str, Any]:
    """Return calibration slope/intercept or explicit non-estimability."""
    if len(true) < min_samples:
        return {
            "calibration_slope": non_estimable(
                "calibration_slope", "insufficient_sample_size_for_calibration"
            ),
            "calibration_intercept": non_estimable(
                "calibration_intercept", "insufficient_sample_size_for_calibration"
            ),
        }
    if len(np.unique(true)) < 2:
        return {
            "calibration_slope": non_estimable(
                "calibration_slope", "only_one_class_present"
            ),
            "calibration_intercept": non_estimable(
                "calibration_intercept", "only_one_class_present"
            ),
        }
    clipped = np.clip(scores, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    fitted = LogisticRegression(C=1e6, solver="lbfgs").fit(logits, true)
    return {
        "calibration_slope": float(fitted.coef_[0, 0]),
        "calibration_intercept": float(fitted.intercept_[0]),
    }
