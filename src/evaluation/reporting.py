"""Prediction validation, evaluation orchestration, and auditable output files."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from .bootstrap import patient_bootstrap, summarize_bootstrap
from .calibration import binary_calibration
from .metrics import CELL_LABELS, EvaluationError, binary_metrics, cell_metrics
from gbm_study.leakage import LeakageError, normalize_split_keys

CELL_REQUIRED = {
    "run_id",
    "method",
    "fold",
    "seed",
    "patient_id",
    "cell_id",
    "true_state",
    "predicted_state",
    "split",
    "split_hash",
    "config_hash",
    "model_hash",
    *(f"probability_{label}" for label in CELL_LABELS),
}
BINARY_REQUIRED = {
    "run_id",
    "method",
    "fold",
    "seed",
    "patient_id",
    "true_label",
    "probability_positive",
    "split",
    "split_hash",
    "config_hash",
    "model_hash",
    "task",
}
VALID_SPLITS = {"train", "validation", "test"}
VALID_UNITS = {"cell_state", "patient_binary"}


def sha256_file(path: Path) -> str:
    """Hash an input file in bounded-memory chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    """Hash canonical JSON configuration content."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _read_table(path: Path) -> pd.DataFrame:
    """Read JSONL, CSV, or parquet predictions without importing model code."""
    try:
        if path.suffix == ".jsonl":
            return pd.read_json(path, lines=True)
        if path.suffix == ".csv":
            return pd.read_csv(path)
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
    except (OSError, ValueError, ImportError) as exc:
        raise EvaluationError(f"Cannot read prediction file {path}: {exc}") from exc
    raise EvaluationError("Prediction file must use .jsonl, .csv, or .parquet")


def _required_metadata(frame: pd.DataFrame, required: set[str]) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise EvaluationError(f"Prediction file is missing required columns: {missing}")
    if frame.empty:
        raise EvaluationError("Prediction file is empty")
    for column in ("run_id", "method", "split_hash", "config_hash", "model_hash"):
        if frame[column].isna().any() or frame[column].astype(str).eq("").any():
            raise EvaluationError(
                f"Prediction metadata column {column!r} contains missing values"
            )


def _validate_run_metadata(frame: pd.DataFrame, fold: int) -> None:
    for column in (
        "run_id",
        "method",
        "fold",
        "seed",
        "split_hash",
        "config_hash",
        "model_hash",
    ):
        if frame[column].nunique(dropna=False) != 1:
            raise EvaluationError(
                f"Prediction metadata column {column!r} is inconsistent"
            )
    if int(frame["fold"].iloc[0]) != fold:
        raise EvaluationError(f"Prediction fold does not match requested fold {fold}")


def validate_prediction_file(
    path: Path, split_path: Path, unit: str, fold: int
) -> tuple[pd.DataFrame, str]:
    """Validate predictions, patient membership, labels, probabilities, and hashes."""
    if unit not in VALID_UNITS:
        raise EvaluationError(f"Unrecognized metric units: {unit!r}")
    frame = _read_table(path)
    expected = CELL_REQUIRED if unit == "cell_state" else BINARY_REQUIRED
    _required_metadata(frame, expected)
    _validate_run_metadata(frame, fold)
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    if isinstance(split_payload, dict) and isinstance(split_payload.get("folds"), list):
        if fold < 0 or fold >= len(split_payload["folds"]):
            raise EvaluationError(f"Requested fold {fold} is absent from split file")
        split_payload = split_payload["folds"][fold]
    try:
        split_payload = normalize_split_keys(split_payload)
    except LeakageError as exc:
        raise EvaluationError(str(exc)) from exc
    split_sets = {name: set(values) for name, values in split_payload.items()}
    if any(
        len(values) == 0 or not all(isinstance(value, str) for value in values)
        for values in split_sets.values()
    ):
        raise EvaluationError("Split patient lists must be non-empty string lists")
    names = list(split_sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if split_sets[left] & split_sets[right]:
                raise EvaluationError(f"Patient overlap between {left} and {right}")
    split_hash = sha256_file(split_path)
    if (
        frame["split_hash"].astype(str).nunique() != 1
        or frame["split_hash"].iloc[0] != split_hash
    ):
        raise EvaluationError("Prediction split-hash mismatch")
    if not frame["split"].isin(VALID_SPLITS).all():
        raise EvaluationError("Prediction rows contain an unknown split")
    for split_name, group in frame.groupby("split"):
        if not set(group["patient_id"].astype(str)).issubset(split_sets[split_name]):
            raise EvaluationError(
                f"Prediction patients do not belong to declared {split_name} split"
            )
    if unit == "cell_state":
        _validate_cell(frame)
    else:
        _validate_binary(frame)
    return (
        frame.sort_values(
            ["patient_id"] + (["cell_id"] if unit == "cell_state" else [])
        ).reset_index(drop=True),
        split_hash,
    )


def _validate_cell(frame: pd.DataFrame) -> None:
    if set(frame["true_state"].astype(str)) - set(CELL_LABELS) or set(
        frame["predicted_state"].astype(str)
    ) - set(CELL_LABELS):
        raise EvaluationError("Unknown four-state label")
    if frame.duplicated(["run_id", "patient_id", "cell_id"]).any():
        raise EvaluationError("Duplicate prediction key")
    probabilities = frame[[f"probability_{label}" for label in CELL_LABELS]].to_numpy(
        dtype=float
    )
    if (
        not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise EvaluationError("Probability columns must be finite and within [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise EvaluationError("Probability rows must sum to one")


def _validate_binary(frame: pd.DataFrame) -> None:
    normalized_task = frame["task"].astype(str).str.lower()
    if frame.assign(task=normalized_task).duplicated(["task", "patient_id"]).any():
        raise EvaluationError("Duplicate patient-level prediction key")
    if not normalized_task.isin({"idh", "tp53"}).all():
        raise EvaluationError("Patient binary task must be IDH or TP53")
    labels = pd.to_numeric(frame["true_label"], errors="coerce")
    scores = pd.to_numeric(frame["probability_positive"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise EvaluationError("Patient true_label must be binary 0/1")
    if scores.isna().any() or ((scores < 0) | (scores > 1)).any():
        raise EvaluationError("probability_positive must be within [0, 1]")


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_versions() -> dict[str, str | None]:
    """Record versions relevant to metric computation."""
    names = ("numpy", "pandas", "scikit-learn", "scipy", "pyarrow", "jsonschema")
    return {name: _package_version(name) for name in names}


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def run_evaluation(
    prediction_path: Path, split_path: Path, config_path: Path, output: Path
) -> dict[str, Any]:
    """Run the only approved metrics pathway and write every required artifact."""
    config = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.suffix == ".json"
        else _yaml_config(config_path)
    )
    unit = config.get("metric_units")
    if not isinstance(unit, str):
        raise EvaluationError("Configuration metric_units must be a string")
    fold = int(config.get("fold", 0))
    frame, split_hash = validate_prediction_file(
        prediction_path, split_path, unit, fold
    )
    metric_config_hash = canonical_hash(config)
    if unit == "cell_state":
        probabilities = frame[
            [f"probability_{label}" for label in CELL_LABELS]
        ].to_numpy(dtype=float)
        point = cell_metrics(
            frame["true_state"].to_numpy(),
            frame["predicted_state"].to_numpy(),
            probabilities,
        )
        metric_names = tuple(
            config.get("bootstrap_metrics", ["macro_f1", "balanced_accuracy"])
        )
        distribution = patient_bootstrap(
            frame.rename(
                columns={
                    "true_state": "true_label",
                    "predicted_state": "predicted_label",
                }
            ),
            metric_names,
            int(config.get("bootstrap_replicates", 1000)),
            int(config.get("bootstrap_seed", 17)),
            unit,
        )
        patient_summary = (
            frame.assign(correct=frame["true_state"] == frame["predicted_state"])
            .groupby("patient_id", as_index=False)
            .agg(n_cells=("cell_id", "size"), accuracy=("correct", "mean"))
        )
        _atomic_json(
            output / "metrics.json",
            {
                "status": "completed",
                "metric_units": unit,
                "point_estimate": point,
                "bootstrap": _bootstrap_with_points(distribution, point),
            },
        )
        pd.DataFrame(
            point["confusion_matrix"]["values"], index=CELL_LABELS, columns=CELL_LABELS
        ).to_csv(output / "confusion_matrix.csv")
    else:
        metrics = tuple(
            config.get(
                "metrics",
                [
                    "auroc",
                    "auprc",
                    "balanced_accuracy",
                    "sensitivity",
                    "specificity",
                    "brier_score",
                    "calibration_slope",
                    "calibration_intercept",
                ],
            )
        )
        true = frame["true_label"].to_numpy(dtype=int)
        scores = frame["probability_positive"].to_numpy(dtype=float)
        point = binary_metrics(true, scores, metrics)
        point.update(
            binary_calibration(
                true, scores, int(config.get("calibration_min_samples", 10))
            )
        )
        distribution = patient_bootstrap(
            frame,
            metrics,
            int(config.get("bootstrap_replicates", 1000)),
            int(config.get("bootstrap_seed", 17)),
            unit,
        )
        patient_summary = frame[
            ["task", "patient_id", "true_label", "probability_positive", "split"]
        ].copy()
        _atomic_json(
            output / "metrics.json",
            {
                "status": "completed",
                "metric_units": unit,
                "point_estimate": point,
                "bootstrap": _bootstrap_with_points(distribution, point),
            },
        )
    _write_parquet(output / "bootstrap_distribution.parquet", distribution)
    _write_parquet(output / "per_patient_metrics.parquet", patient_summary)
    _atomic_json(output / "warnings.json", {"warnings": []})
    manifest = {
        "evaluator_git_commit": _git_commit(),
        "input_file_sha256": sha256_file(prediction_path),
        "split_file_sha256": sha256_file(split_path),
        "split_hash": split_hash,
        "metric_config_hash": metric_config_hash,
        "bootstrap_seed": int(config.get("bootstrap_seed", 17)),
        "bootstrap_replicates": int(config.get("bootstrap_replicates", 1000)),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "package_versions": package_versions(),
    }
    _atomic_json(output / "evaluation_manifest.json", manifest)
    return manifest


def _bootstrap_with_points(
    distribution: pd.DataFrame, point: dict[str, Any]
) -> dict[str, Any]:
    summaries = summarize_bootstrap(distribution)
    for metric, summary in summaries.items():
        value = point.get(metric)
        summary["point_estimate"] = (
            float(value) if isinstance(value, (float, int)) else None
        )
    return summaries


def _yaml_config(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-untyped]

        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise EvaluationError(f"Cannot read evaluation config {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvaluationError("Evaluation configuration must be a YAML object")
    return payload
