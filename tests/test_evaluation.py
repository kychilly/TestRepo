"""Hand-computable tests for the single manuscript evaluation pathway."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from evaluation.bootstrap import patient_bootstrap
from evaluation.metrics import EvaluationError, binary_metrics, cell_metrics
from evaluation.reporting import run_evaluation, validate_prediction_file


def split_file(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "splits.json"
    path.write_text(
        json.dumps({"train": ["p1"], "validation": ["p2"], "test": ["p3", "p4"]}),
        encoding="utf-8",
    )
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def cell_frame(split_hash: str) -> pd.DataFrame:
    rows = []
    states = ["AC", "MES", "NPC", "OPC"]
    for patient, state, predicted in [
        ("p1", "AC", "AC"),
        ("p2", "MES", "AC"),
        ("p3", "NPC", "NPC"),
        ("p4", "OPC", "OPC"),
    ]:
        probabilities = {f"probability_{label}": 0.0 for label in states}
        probabilities[f"probability_{predicted}"] = 1.0
        rows.append(
            {
                "run_id": "r",
                "method": "m",
                "fold": 0,
                "seed": 17,
                "patient_id": patient,
                "cell_id": f"{patient}-c",
                "true_state": state,
                "predicted_state": predicted,
                **probabilities,
                "split": {
                    "p1": "train",
                    "p2": "validation",
                    "p3": "test",
                    "p4": "test",
                }[patient],
                "split_hash": split_hash,
                "config_hash": "config",
                "model_hash": "model",
            }
        )
    return pd.DataFrame(rows)


def test_exact_macro_f1_and_confusion_matrix() -> None:
    true = np.array(["AC", "MES", "NPC", "OPC"])
    predicted = np.array(["AC", "AC", "NPC", "OPC"])
    probabilities = np.eye(4)[[0, 0, 2, 3]]
    result = cell_metrics(true, predicted, probabilities)
    assert result["macro_f1"] == pytest.approx((2 / 3 + 0 + 1 + 1) / 4)
    assert result["confusion_matrix"]["values"] == [
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]


def test_patient_bootstrap_resamples_clusters() -> None:
    rows = pd.DataFrame(
        {
            "patient_id": ["p1"] * 1 + ["p2"] * 2 + ["p3"] * 3 + ["p4"] * 4,
            "true_label": [
                "AC",
                "MES",
                "MES",
                "NPC",
                "NPC",
                "NPC",
                "OPC",
                "OPC",
                "OPC",
                "OPC",
            ],
            "predicted_label": ["AC"] * 10,
            **{
                f"probability_{label}": [1.0 if label == "AC" else 0.0] * 10
                for label in ("AC", "MES", "NPC", "OPC")
            },
        }
    )
    distribution = patient_bootstrap(rows, ("macro_f1",), 20, 4, "cell_state")
    assert set(distribution["sampled_cell_count"]) <= {
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
    }
    assert all(
        len(json.loads(value)) == 4 for value in distribution["sampled_patients"]
    )


def test_class_missing_bootstrap_replicates_are_recorded() -> None:
    rows = cell_frame("hash")
    rows = pd.concat([rows] * 2, ignore_index=True)
    distribution = patient_bootstrap(rows, ("macro_f1",), 50, 1, "cell_state")
    assert distribution["required_classes_represented"].isin([True, False]).all()
    assert (~distribution["required_classes_represented"]).any()


def test_binary_auroc_non_estimability() -> None:
    result = binary_metrics(np.array([1, 1]), np.array([0.2, 0.8]), ("auroc",))
    assert result["auroc"] == {
        "status": "non_estimable",
        "reason": "only_one_class_present",
        "metric": "auroc",
    }


def test_validation_rejects_probability_and_hash_errors(tmp_path: Path) -> None:
    split, split_hash = split_file(tmp_path)
    frame = cell_frame(split_hash)
    frame.loc[0, "probability_AC"] = 1.1
    prediction = tmp_path / "predictions.jsonl"
    frame.to_json(prediction, orient="records", lines=True)
    with pytest.raises(EvaluationError, match=r"within \[0, 1\]"):
        validate_prediction_file(prediction, split, "cell_state", 0)
    frame = cell_frame("wrong")
    frame.to_json(prediction, orient="records", lines=True)
    with pytest.raises(EvaluationError, match="hash mismatch"):
        validate_prediction_file(prediction, split, "cell_state", 0)


def test_validation_rejects_mixed_run_metadata(tmp_path: Path) -> None:
    split, split_hash = split_file(tmp_path)
    frame = cell_frame(split_hash)
    frame.loc[1, "seed"] = 18
    prediction = tmp_path / "predictions.jsonl"
    frame.to_json(prediction, orient="records", lines=True)
    with pytest.raises(EvaluationError, match="seed.*inconsistent"):
        validate_prediction_file(prediction, split, "cell_state", 0)


def test_binary_duplicate_patient_is_case_insensitive(tmp_path: Path) -> None:
    split, split_hash = split_file(tmp_path)
    rows = pd.DataFrame(
        [
            {
                "run_id": "r",
                "method": "m",
                "fold": 0,
                "seed": 17,
                "patient_id": "p3",
                "task": "IDH",
                "true_label": 0,
                "probability_positive": 0.2,
                "split": "test",
                "split_hash": split_hash,
                "config_hash": "config",
                "model_hash": "model",
            },
            {
                "run_id": "r",
                "method": "m",
                "fold": 0,
                "seed": 17,
                "patient_id": "p3",
                "task": "idh",
                "true_label": 1,
                "probability_positive": 0.8,
                "split": "test",
                "split_hash": split_hash,
                "config_hash": "config",
                "model_hash": "model",
            },
        ]
    )
    prediction = tmp_path / "binary.jsonl"
    rows.to_json(prediction, orient="records", lines=True)
    with pytest.raises(EvaluationError, match="Duplicate patient-level"):
        validate_prediction_file(prediction, split, "patient_binary", 0)


def test_row_reordering_and_fixed_seed_are_stable(tmp_path: Path) -> None:
    split, split_hash = split_file(tmp_path)
    config = tmp_path / "evaluation.yaml"
    config.write_text(
        "metric_units: cell_state\nbootstrap_replicates: 20\nbootstrap_seed: 9\nfold: 0\n",
        encoding="utf-8",
    )
    first = cell_frame(split_hash)
    second = first.sample(frac=1.0, random_state=42)
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    first.to_json(first_path, orient="records", lines=True)
    second.to_json(second_path, orient="records", lines=True)
    output_one = tmp_path / "one"
    output_two = tmp_path / "two"
    run_evaluation(first_path, split, config, output_one)
    run_evaluation(second_path, split, config, output_two)
    metrics_one = json.loads((output_one / "metrics.json").read_text(encoding="utf-8"))
    metrics_two = json.loads((output_two / "metrics.json").read_text(encoding="utf-8"))
    assert metrics_one == metrics_two


def test_outputs_and_manifest_are_written(tmp_path: Path) -> None:
    split, split_hash = split_file(tmp_path)
    prediction = tmp_path / "predictions.jsonl"
    cell_frame(split_hash).to_json(prediction, orient="records", lines=True)
    config = tmp_path / "evaluation.yaml"
    config.write_text(
        "metric_units: cell_state\nbootstrap_replicates: 4\nbootstrap_seed: 2\nfold: 0\n",
        encoding="utf-8",
    )
    output = tmp_path / "out"
    run_evaluation(prediction, split, config, output)
    for name in (
        "metrics.json",
        "bootstrap_distribution.parquet",
        "confusion_matrix.csv",
        "per_patient_metrics.parquet",
        "warnings.json",
        "evaluation_manifest.json",
    ):
        assert (output / name).is_file()
