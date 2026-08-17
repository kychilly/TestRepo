from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from eval import main


def test_combined_eval_writes_cell_and_idh_tasks(tmp_path: Path) -> None:
    splits = tmp_path / "splits.json"
    splits.write_text(
        json.dumps({"train": ["p1"], "validation": ["p2"], "test": ["p3", "p4"]}),
        encoding="utf-8",
    )
    split_hash = hashlib.sha256(splits.read_bytes()).hexdigest()
    cell_rows = []
    for patient, state in (("p1", "AC"), ("p2", "MES"), ("p3", "NPC"), ("p4", "OPC")):
        row = {
            "run_id": "r",
            "method": "m",
            "fold": 0,
            "seed": 17,
            "patient_id": patient,
            "cell_id": f"{patient}-c",
            "true_state": state,
            "predicted_state": state,
            "split": {"p1": "train", "p2": "validation", "p3": "test", "p4": "test"}[
                patient
            ],
            "split_hash": split_hash,
            "config_hash": "config",
            "model_hash": "model",
        }
        row.update(
            {
                f"probability_{label}": float(label == state)
                for label in ("AC", "MES", "NPC", "OPC")
            }
        )
        cell_rows.append(row)
    cell_path = tmp_path / "cells.jsonl"
    pd.DataFrame(cell_rows).to_json(cell_path, orient="records", lines=True)
    idh_rows = []
    for patient, label, score in (("p3", 0, 0.1), ("p4", 1, 0.9)):
        idh_rows.append(
            {
                "run_id": "r",
                "method": "m",
                "fold": 0,
                "seed": 17,
                "patient_id": patient,
                "task": "IDH",
                "true_label": label,
                "probability_positive": score,
                "split": "test",
                "split_hash": split_hash,
                "config_hash": "config",
                "model_hash": "model",
            }
        )
    idh_path = tmp_path / "idh.jsonl"
    pd.DataFrame(idh_rows).to_json(idh_path, orient="records", lines=True)
    cell_config = tmp_path / "cell.yaml"
    cell_config.write_text(
        "metric_units: cell_state\nbootstrap_replicates: 5\nbootstrap_seed: 1\n",
        encoding="utf-8",
    )
    idh_config = tmp_path / "idh.yaml"
    idh_config.write_text(
        "metric_units: patient_binary\nbootstrap_replicates: 5\nbootstrap_seed: 1\nmetrics: [auroc]\n",
        encoding="utf-8",
    )
    output = tmp_path / "baseline_results"

    assert (
        main(
            [
                "--cell-predictions",
                str(cell_path),
                "--idh-predictions",
                str(idh_path),
                "--splits",
                str(splits),
                "--config",
                str(cell_config),
                "--idh-config",
                str(idh_config),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    result = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert result["tasks"]["cell_state"]["metrics"]["point_estimate"]["macro_f1"] == 1.0
    assert result["tasks"]["idh"]["metrics"]["point_estimate"]["auroc"] == 1.0
