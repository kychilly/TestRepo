"""Validated cell-state and patient-level metric calculations."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (  # type: ignore[import-untyped]
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

CELL_LABELS: tuple[str, ...] = ("AC", "MES", "NPC", "OPC")


class EvaluationError(ValueError):
    """Raised when validated prediction data cannot support an evaluation."""


def non_estimable(metric: str, reason: str) -> dict[str, Any]:
    """Return the explicit representation for a metric that cannot be estimated."""
    return {"status": "non_estimable", "reason": reason, "metric": metric}


def _require_all_cell_labels(
    true: np.ndarray[Any, Any], predicted: np.ndarray[Any, Any]
) -> None:
    observed = set(true.tolist()) | set(predicted.tolist())
    missing = sorted(set(CELL_LABELS) - observed)
    if missing:
        raise EvaluationError(f"Required cell-state labels are absent: {missing}")


def cell_metrics(
    true: np.ndarray[Any, Any],
    predicted: np.ndarray[Any, Any],
    probabilities: np.ndarray[Any, Any] | None = None,
) -> dict[str, Any]:
    """Calculate four-state metrics using fixed AC/MES/NPC/OPC ordering."""
    _require_all_cell_labels(true, predicted)
    matrix = confusion_matrix(true, predicted, labels=CELL_LABELS)
    precision = precision_score(
        true, predicted, labels=CELL_LABELS, average=None, zero_division=0
    )
    recall = recall_score(
        true, predicted, labels=CELL_LABELS, average=None, zero_division=0
    )
    f1 = f1_score(true, predicted, labels=CELL_LABELS, average=None, zero_division=0)
    result: dict[str, Any] = {
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
        "per_state_precision": {
            label: float(value) for label, value in zip(CELL_LABELS, precision)
        },
        "per_state_recall": {
            label: float(value) for label, value in zip(CELL_LABELS, recall)
        },
        "per_state_f1": {label: float(value) for label, value in zip(CELL_LABELS, f1)},
        "confusion_matrix": {"labels": list(CELL_LABELS), "values": matrix.tolist()},
    }
    if probabilities is not None:
        result["multiclass_log_loss"] = float(
            log_loss(true, probabilities, labels=list(CELL_LABELS))
        )
        one_hot = np.eye(len(CELL_LABELS))[
            np.asarray([CELL_LABELS.index(label) for label in true])
        ]
        result["multiclass_brier_score"] = float(
            np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))
        )
    return result


def binary_metrics(
    true: np.ndarray[Any, Any],
    scores: np.ndarray[Any, Any],
    metric_names: tuple[str, ...],
) -> dict[str, Any]:
    """Calculate patient-level binary metrics without deriving labels from cells."""
    if set(true.tolist()) - {0, 1}:
        raise EvaluationError("Binary true_label values must be exactly 0 or 1")
    result: dict[str, Any] = {}
    if len(np.unique(true)) < 2:
        for name in metric_names:
            result[name] = non_estimable(name, "only_one_class_present")
        return result
    predicted = (scores >= 0.5).astype(int)
    for name in metric_names:
        if name == "auroc":
            result[name] = float(roc_auc_score(true, scores))
        elif name == "auprc":
            result[name] = float(average_precision_score(true, scores))
        elif name == "balanced_accuracy":
            result[name] = float(balanced_accuracy_score(true, predicted))
        elif name == "sensitivity":
            result[name] = float(
                recall_score(true, predicted, pos_label=1, zero_division=0)
            )
        elif name == "specificity":
            result[name] = float(
                recall_score(true, predicted, pos_label=0, zero_division=0)
            )
        elif name == "brier_score":
            result[name] = float(np.mean((scores - true) ** 2))
        elif name in {"calibration_slope", "calibration_intercept"}:
            if len(true) < 10:
                result[name] = non_estimable(
                    name, "insufficient_sample_size_for_calibration"
                )
            else:
                clipped = np.clip(scores, 1e-6, 1 - 1e-6)
                design = np.log(clipped / (1 - clipped)).reshape(-1, 1)
                calibration = LogisticRegression(C=1e6, solver="lbfgs").fit(
                    design, true
                )
                result["calibration_slope"] = float(calibration.coef_[0, 0])
                result["calibration_intercept"] = float(calibration.intercept_[0])
        else:
            raise EvaluationError(f"Unrecognized binary metric: {name}")
    return result
