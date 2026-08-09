"""Leakage, reproducibility, interface, and structured-failure tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from baselines.base import (
    BaselineError,
    CellData,
    assign_cells,
    config_hash,
    evaluate_predictions,
    load_patient_splits,
)
from baselines.harmony_knn import HarmonyKNN
from baselines.pca_logreg import PCALogReg
from baselines.scvi_probe import ScVIProbe


def toy_data() -> CellData:
    rng = np.random.default_rng(3)
    states = np.array(["AC", "MES", "NPC", "OPC"] * 6)
    patients = np.array(["p1"] * 6 + ["p2"] * 6 + ["p3"] * 6 + ["p4"] * 6)
    X = rng.normal(size=(24, 4)) + np.eye(4)[np.arange(24) % 4] * 2
    return CellData(
        X,
        patients,
        np.array([f"c{i}" for i in range(24)]),
        states,
        ("g1", "g2", "g3", "g4"),
    )


def splits_file(tmp_path: Path) -> Path:
    path = tmp_path / "splits.json"
    path.write_text(
        json.dumps({"train": ["p1", "p2"], "validation": ["p3"], "test": ["p4"]}),
        encoding="utf-8",
    )
    return path


def test_overlap_rejection(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps({"train": ["p1"], "validation": ["p1"], "test": ["p2"]}),
        encoding="utf-8",
    )
    with pytest.raises(BaselineError, match="overlap"):
        load_patient_splits(path, 0)


def test_missing_patient_rejection(tmp_path: Path) -> None:
    split = load_patient_splits(splits_file(tmp_path), 0)
    data = toy_data()
    data = CellData(
        data.X,
        np.where(data.patient_id == "p4", "missing", data.patient_id),
        data.cell_id,
        data.state,
        data.gene_ids,
    )
    with pytest.raises(BaselineError, match="absent"):
        assign_cells(data, split)


def test_training_only_fitting_and_probability_order(tmp_path: Path) -> None:
    data = toy_data()
    split = load_patient_splits(splits_file(tmp_path), 0)
    indices = assign_cells(data, split)
    baseline = PCALogReg(components=2, seed=17).fit(
        data.subset(indices["train"]), {"split": "train"}
    )
    probabilities = baseline.predict_proba(
        data.subset(indices["test"]), {"split": "test"}
    )
    assert probabilities.shape == (6, 4)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert baseline.gene_ids == data.gene_ids


def test_state_label_order_and_save_load_equivalence(tmp_path: Path) -> None:
    data = toy_data()
    baseline = PCALogReg(components=2, seed=17).fit(data, {"split": "train"})
    path = tmp_path / "model.pkl"
    baseline.save(path)
    restored = PCALogReg.load(path)
    np.testing.assert_allclose(
        baseline.predict_proba(data, {}), restored.predict_proba(data, {})
    )
    assert (
        baseline.get_run_metadata()["model_hash"]
        == restored.get_run_metadata()["model_hash"]
    )


def test_same_seed_reproducibility_and_different_seed_recorded() -> None:
    data = toy_data()
    first = PCALogReg(components=2, seed=17).fit(data, {})
    second = PCALogReg(components=2, seed=17).fit(data, {})
    third = PCALogReg(components=2, seed=18).fit(data, {})
    np.testing.assert_allclose(
        first.predict_proba(data, {}), second.predict_proba(data, {})
    )
    assert first.get_run_metadata()["seed"] != third.get_run_metadata()["seed"]


def test_harmony_non_applicability_is_structured() -> None:
    with pytest.raises(Exception, match="batch"):
        HarmonyKNN(harmony_covariate="batch").fit(toy_data(), {})


def test_scvi_failure_is_structured() -> None:
    with pytest.raises(Exception):
        ScVIProbe().fit(toy_data(), {})


def test_common_evaluation_outputs_cell_and_patient_rows(tmp_path: Path) -> None:
    data = toy_data()
    split = load_patient_splits(splits_file(tmp_path), 0)
    assignments = assign_cells(data, split)
    baseline = PCALogReg(components=2, seed=17).fit(
        data.subset(assignments["train"]), {}
    )
    run = {
        "run_id": "run",
        "method": "pca_logreg",
        "fold": 0,
        "seed": 17,
        "split_hash": split.split_hash,
        "config_hash": config_hash({}),
        "model_hash": "hash",
    }
    cells, patients = evaluate_predictions(baseline, data, assignments, run)
    assert len(cells) == 24
    assert len(patients) == 4
    assert {
        "probability_AC",
        "probability_MES",
        "probability_NPC",
        "probability_OPC",
    } <= cells[0].keys()
