"""Shared data contracts, leakage checks, evaluation, and persistence."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from gbm_study.leakage import LeakageError, normalize_split_keys

STATE_LABELS: tuple[str, ...] = ("AC", "MES", "NPC", "OPC")


class BaselineError(ValueError):
    """Raised when a baseline data or scientific contract is violated."""


class MethodNotApplicable(BaselineError):
    """Raised when a method cannot be scientifically applied to this run."""


@dataclass(frozen=True)
class CellData:
    """Explicit intermediate representation accepted by every baseline."""

    X: NDArray[Any]
    patient_id: NDArray[Any]
    cell_id: NDArray[Any]
    state: NDArray[Any]
    gene_ids: tuple[str, ...]
    batch: NDArray[Any] | None = None

    def __post_init__(self) -> None:
        if self.X.ndim != 2:
            raise BaselineError("X must be a two-dimensional cell-by-gene matrix")
        n_cells = self.X.shape[0]
        arrays = {
            "patient_id": self.patient_id,
            "cell_id": self.cell_id,
            "state": self.state,
        }
        if any(array.ndim != 1 or len(array) != n_cells for array in arrays.values()):
            raise BaselineError(
                "patient_id, cell_id, and state must map one value to every cell"
            )
        if self.X.shape[1] != len(self.gene_ids):
            raise BaselineError("X columns must equal the number of gene_ids")
        if self.batch is not None and (
            self.batch.ndim != 1 or len(self.batch) != n_cells
        ):
            raise BaselineError("batch must map one value to every cell")
        if not np.isfinite(self.X).all():
            raise BaselineError("X contains NaN or infinite values")
        if len(set(self.cell_id.tolist())) != n_cells:
            raise BaselineError("every cell_id must identify exactly one cell")
        unknown_states = set(self.state.tolist()) - set(STATE_LABELS)
        if unknown_states:
            raise BaselineError(f"Unknown state labels: {sorted(unknown_states)}")

    def subset(self, indices: NDArray[Any]) -> "CellData":
        """Return a row subset without changing the frozen gene contract."""
        return CellData(
            self.X[indices],
            self.patient_id[indices],
            self.cell_id[indices],
            self.state[indices],
            self.gene_ids,
            None if self.batch is None else self.batch[indices],
        )


@dataclass(frozen=True)
class PatientSplits:
    """Disjoint patient sets for one requested fold."""

    train: frozenset[str]
    validation: frozenset[str]
    test: frozenset[str]
    split_hash: str
    fold: int

    def as_dict(self) -> dict[str, frozenset[str]]:
        return {"train": self.train, "validation": self.validation, "test": self.test}


def sha256_file(path: Path) -> str:
    """Return a file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_patient_splits(path: Path, fold: int) -> PatientSplits:
    """Load either a direct split object or a ``folds`` list from JSON."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineError(f"Cannot read split file {path}: {exc}") from exc
    payload: Any = raw
    if isinstance(raw, dict) and isinstance(raw.get("folds"), list):
        if fold < 0 or fold >= len(raw["folds"]):
            raise BaselineError(f"Requested fold {fold} is not present in {path}")
        payload = raw["folds"][fold]
    try:
        payload = normalize_split_keys(payload)
    except LeakageError as exc:
        raise BaselineError(str(exc)) from exc
    sets: dict[str, frozenset[str]] = {}
    for name in ("train", "validation", "test"):
        values = payload[name]
        if (
            not isinstance(values, list)
            or not values
            or not all(isinstance(value, str) and value for value in values)
        ):
            raise BaselineError(f"Split {name} must be a non-empty list of patient IDs")
        if len(set(values)) != len(values):
            raise BaselineError(f"Split {name} contains duplicate patient IDs")
        sets[name] = frozenset(values)
    assert_zero_patient_overlap(sets)
    return PatientSplits(
        sets["train"], sets["validation"], sets["test"], sha256_file(path), fold
    )


def assert_zero_patient_overlap(splits: dict[str, frozenset[str]]) -> None:
    """Reject any patient shared by two partitions."""
    names = list(splits)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = splits[left] & splits[right]
            if overlap:
                raise BaselineError(
                    f"Patient overlap between {left} and {right}: {sorted(overlap)}"
                )


def assign_cells(data: CellData, splits: PatientSplits) -> dict[str, NDArray[Any]]:
    """Assign every cell to exactly one requested patient split."""
    patient_values = data.patient_id.astype(str)
    known = set().union(*splits.as_dict().values())
    if not set(patient_values).issubset(known):
        missing = sorted(set(patient_values) - known)
        raise BaselineError(f"Cells contain patients absent from split: {missing}")
    assignments: dict[str, NDArray[Any]] = {}
    masks = {
        name: np.isin(patient_values, list(values))
        for name, values in splits.as_dict().items()
    }
    combined = sum(mask.astype(np.int8) for mask in masks.values())
    if not np.all(combined == 1):
        raise BaselineError("Every cell must map to exactly one patient split")
    for name, mask in masks.items():
        if not np.any(mask):
            raise BaselineError(f"No cells are available for split {name}")
        assignments[name] = np.flatnonzero(mask)
    return assignments


def config_hash(config: dict[str, Any]) -> str:
    """Hash canonical configuration content."""
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def model_hash(payload: dict[str, Any]) -> str:
    """Hash serializable fitted-model metadata."""
    return config_hash(payload)


class Baseline(ABC):
    """Common interface implemented by every baseline method."""

    method: ClassVar[str]

    @abstractmethod
    def fit(self, train_data: CellData, train_metadata: dict[str, Any]) -> "Baseline":
        """Fit learned transformations and classifier on training cells only."""

    @abstractmethod
    def predict(self, data: CellData, metadata: dict[str, Any]) -> NDArray[Any]:
        """Predict state labels."""

    @abstractmethod
    def predict_proba(self, data: CellData, metadata: dict[str, Any]) -> NDArray[Any]:
        """Predict probabilities in STATE_LABELS order."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the fitted baseline."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "Baseline":
        """Restore a fitted baseline."""

    @abstractmethod
    def get_run_metadata(self) -> dict[str, Any]:
        """Return method and fitted-model provenance."""


def evaluate_predictions(
    baseline: Baseline,
    data: CellData,
    assignments: dict[str, NDArray[Any]],
    run: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Produce cell-level predictions and patient-level summaries identically for all methods."""
    rows: list[dict[str, Any]] = []
    for split, indices in assignments.items():
        subset = data.subset(indices)
        probabilities = baseline.predict_proba(subset, {"split": split})
        predictions = baseline.predict(subset, {"split": split})
        if probabilities.shape != (len(indices), len(STATE_LABELS)):
            raise BaselineError("Prediction probabilities have an unstable shape")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
            raise BaselineError("Prediction probabilities must sum to one")
        for offset, index in enumerate(indices):
            rows.append(
                {
                    **run,
                    "patient_id": str(data.patient_id[index]),
                    "cell_id": str(data.cell_id[index]),
                    "true_state": str(data.state[index]),
                    "predicted_state": str(predictions[offset]),
                    **{
                        f"probability_{label}": float(probabilities[offset, position])
                        for position, label in enumerate(STATE_LABELS)
                    },
                    "split": split,
                }
            )
    patient_rows: list[dict[str, Any]] = []
    for patient_id in sorted(set(str(value) for value in data.patient_id)):
        patient_cells = [row for row in rows if row["patient_id"] == patient_id]
        split = patient_cells[0]["split"]
        patient_rows.append(
            {
                **run,
                "patient_id": patient_id,
                "split": split,
                "n_cells": len(patient_cells),
                "accuracy": float(
                    np.mean(
                        [
                            row["true_state"] == row["predicted_state"]
                            for row in patient_cells
                        ]
                    )
                ),
            }
        )
    return rows, patient_rows


def runtime_metadata() -> dict[str, str]:
    """Record basic compute provenance for a baseline run."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": commit,
    }
