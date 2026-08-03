"""Patient-cluster bootstrap confidence intervals."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from .metrics import CELL_LABELS, EvaluationError, binary_metrics, cell_metrics


def patient_bootstrap(
    rows: pd.DataFrame,
    metric_names: tuple[str, ...],
    replicates: int,
    seed: int,
    task: str,
) -> pd.DataFrame:
    """Resample patients with replacement and retain one estimate per replicate."""
    if replicates < 1:
        raise EvaluationError("bootstrap replicates must be positive")
    patients = sorted(rows["patient_id"].astype(str).unique().tolist())
    if not patients:
        raise EvaluationError("Cannot bootstrap an empty prediction table")
    rng = np.random.default_rng(seed)
    true_column = "true_label" if "true_label" in rows.columns else "true_state"
    predicted_column = "predicted_label" if "predicted_label" in rows.columns else "predicted_state"
    records: list[dict[str, Any]] = []
    for replicate in range(replicates):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        sampled_rows = pd.concat(
            [rows.loc[rows["patient_id"].astype(str) == patient] for patient in sampled],
            ignore_index=True,
        )
        represented = set(sampled_rows[true_column].astype(str).tolist())
        required = set(CELL_LABELS) if task == "cell_state" else {"0", "1"}
        has_all = required.issubset(represented)
        for metric in metric_names:
            record: dict[str, Any] = {
                "replicate": replicate,
                "metric": metric,
                "sampled_patients": json.dumps(sampled.tolist()),
                "sampled_cell_count": int(len(sampled_rows)),
                "required_classes_represented": has_all,
                "valid": False,
                "estimate": None,
            }
            if not has_all:
                record["reason"] = "required_class_missing_in_bootstrap_sample"
            else:
                try:
                    if task == "cell_state":
                        values = cell_metrics(
                            sampled_rows[true_column].to_numpy(),
                            sampled_rows[predicted_column].to_numpy(),
                            sampled_rows[
                                [f"probability_{label}" for label in CELL_LABELS]
                            ].to_numpy(dtype=float),
                        )
                    else:
                        values = binary_metrics(
                            sampled_rows["true_label"].to_numpy(dtype=int),
                            sampled_rows["probability_positive"].to_numpy(dtype=float),
                            (metric,),
                        )
                    estimate = values.get(metric)
                    if isinstance(estimate, (float, int)) and np.isfinite(estimate):
                        record["valid"] = True
                        record["estimate"] = float(estimate)
                    else:
                        record["reason"] = (
                            estimate.get("reason", "metric_non_estimable")
                            if isinstance(estimate, dict)
                            else "metric_non_estimable"
                        )
                except (EvaluationError, ValueError) as exc:
                    record["reason"] = str(exc)
            records.append(record)
    return pd.DataFrame.from_records(records)


def summarize_bootstrap(distribution: pd.DataFrame) -> dict[str, Any]:
    """Summarize bootstrap estimates without converting non-estimable values to zero."""
    summaries: dict[str, Any] = {}
    for metric, group in distribution.groupby("metric", sort=True):
        valid = group.loc[group["valid"], "estimate"].to_numpy(dtype=float)
        summaries[metric] = {
            "point_estimate": None,
            "bootstrap_median": float(np.median(valid)) if len(valid) else None,
            "ci_2_5": float(np.percentile(valid, 2.5)) if len(valid) else None,
            "ci_97_5": float(np.percentile(valid, 97.5)) if len(valid) else None,
            "valid_replicates": int(len(valid)),
            "non_estimable_replicates": int(len(group) - len(valid)),
        }
    return summaries
