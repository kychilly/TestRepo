#!/usr/bin/env python3
"""Audit Neftel cell-state and CGGA bulk datasets against separate contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import anndata as ad  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

CANONICAL_STATES = {"AC", "MES", "NPC", "OPC"}
STATE_ALIASES = {
    "AC-like": "AC",
    "MES-like": "MES",
    "NPC-like": "NPC",
    "OPC-like": "OPC",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matrix_quality(matrix: Any, rows: int, chunk_size: int = 256) -> dict[str, int]:
    nonfinite = 0
    affected_rows = 0
    for start in range(0, rows, chunk_size):
        block = matrix[start : start + chunk_size]
        block = block.toarray() if hasattr(block, "toarray") else np.asarray(block)
        bad = ~np.isfinite(block)
        nonfinite += int(bad.sum())
        affected_rows += int(np.any(bad, axis=1).sum())
    return {"nonfinite_values": nonfinite, "rows_with_nonfinite": affected_rows}


def audit_neftel(path: Path, split_path: Path) -> dict[str, Any]:
    data = ad.read_h5ad(path, backed="r")
    blockers: list[str] = []
    warnings: list[str] = []
    required = {"Sample", "derived_state"}
    missing = sorted(required - set(data.obs.columns))
    if missing:
        blockers.append("Missing Neftel observation columns: " + ", ".join(missing))
    patients = data.obs["Sample"].astype(str) if "Sample" in data.obs else None
    states = data.obs["derived_state"].astype(str) if "derived_state" in data.obs else None
    normalized = states.replace(STATE_ALIASES) if states is not None else None
    unknown_states = sorted(set(normalized) - CANONICAL_STATES) if normalized is not None else []
    if unknown_states:
        blockers.append(f"Non-canonical Neftel states remain: {unknown_states}")
    quality = matrix_quality(data.X, data.n_obs)
    if quality["nonfinite_values"]:
        blockers.append("Neftel expression matrix contains non-finite values")

    split = json.loads(split_path.read_text(encoding="utf-8"))
    split_counts: dict[str, dict[str, Any]] = {}
    assigned: set[str] = set()
    for name in ("train", "validation", "test"):
        ids = {str(value) for value in split.get(name, [])}
        assigned.update(ids)
        mask = patients.isin(ids) if patients is not None else np.zeros(data.n_obs, dtype=bool)
        partition_states = normalized[mask] if normalized is not None else []
        counts = {
            state: int(np.asarray(partition_states == state).sum())
            for state in sorted(CANONICAL_STATES)
        }
        split_counts[name] = {
            "patients": int(patients[mask].nunique()) if patients is not None else 0,
            "cells": int(np.asarray(mask).sum()),
            "states": counts,
        }
        if split_counts[name]["cells"] == 0:
            blockers.append(f"Neftel split {name} has no cells")
        absent_states = [state for state, count in counts.items() if count == 0]
        if absent_states:
            blockers.append(f"Neftel split {name} lacks states: {absent_states}")
    unassigned = sorted(set(patients) - assigned) if patients is not None else []
    if unassigned:
        blockers.append(f"Neftel patients absent from split: {unassigned}")
    test_counts = split_counts.get("test", {}).get("states", {})
    if test_counts and min(test_counts.values()) < 20:
        warnings.append("At least one Neftel test state has fewer than 20 cells")

    counts_layer = "counts" in data.layers
    if not counts_layer:
        warnings.append("No raw integer counts layer; scVI is not applicable")
    batch_values = (
        data.obs["CrossSection"].astype(str).nunique() if "CrossSection" in data.obs else 0
    )
    if batch_values < 2:
        warnings.append(
            "Configured CrossSection batch has fewer than two values; Harmony is not applicable"
        )
    return {
        "status": "accepted_with_limitations" if not blockers else "rejected",
        "path": str(path),
        "sha256": sha256_file(path),
        "shape": [data.n_obs, data.n_vars],
        "patients": int(patients.nunique()) if patients is not None else None,
        "split": str(split_path),
        "split_counts": split_counts,
        "unassigned_patients": unassigned,
        "matrix_quality": quality,
        "counts_layer": counts_layer,
        "cross_section_values": int(batch_values),
        "blockers": blockers,
        "warnings": warnings,
        "task_readiness": {
            "pca_logreg_cell_state": not blockers,
            "harmony_cell_state": not blockers and batch_values >= 2,
            "scvi_cell_state": not blockers and counts_layer,
        },
    }


def audit_cgga(path: Path) -> dict[str, Any]:
    data = ad.read_h5ad(path, backed="r")
    blockers: list[str] = []
    warnings: list[str] = []
    quality = matrix_quality(data.X, data.n_obs)
    if quality["nonfinite_values"]:
        blockers.append(
            f"CGGA expression has {quality['nonfinite_values']} non-finite values across "
            f"{quality['rows_with_nonfinite']} rows"
        )
    idh_candidates = ("IDH_mutation_status", "IDH_status", "idh_status")
    idh_column = next((column for column in idh_candidates if column in data.obs), None)
    if idh_column is None:
        blockers.append("CGGA H5AD has no authoritative IDH-status column")
    patient_column = "Sample" if "Sample" in data.obs else None
    if patient_column is None:
        blockers.append("CGGA H5AD has no Sample patient identifier")
        unique_patients = None
    else:
        unique_patients = int(data.obs[patient_column].astype(str).nunique())
        if unique_patients != data.n_obs:
            warnings.append("CGGA is expected to have one bulk expression row per patient")
    states = (
        data.obs["derived_state"].astype(str).value_counts().to_dict()
        if "derived_state" in data.obs
        else {}
    )
    if states:
        warnings.append(
            "CGGA derived_state is not cell-state ground truth and must not be evaluated as such"
        )
    return {
        "status": "accepted_for_patient_idh" if not blockers else "rejected",
        "path": str(path),
        "sha256": sha256_file(path),
        "shape": [data.n_obs, data.n_vars],
        "observation_unit": "bulk_patient",
        "patients": unique_patients,
        "idh_column": idh_column,
        "derived_state_counts": states,
        "matrix_quality": quality,
        "blockers": blockers,
        "warnings": warnings,
        "task_readiness": {
            "cell_state_external_test": False,
            "patient_idh_external_test": not blockers,
        },
    }


def audit_tcga(path: Path) -> dict[str, Any]:
    clinical = pd.read_csv(path, sep="\t")
    blockers = [
        "No TCGA expression matrix is present in the supplied dataset directory",
        "TCGA clinical TSV has no authoritative IDH-status column",
    ]
    patient_column = "Patient ID" if "Patient ID" in clinical.columns else None
    if patient_column is None:
        blockers.append("TCGA clinical TSV has no Patient ID column")
        patients = None
        duplicates = None
    else:
        patient_ids = clinical[patient_column].astype(str)
        patients = int(patient_ids.nunique())
        duplicates = int(patient_ids.duplicated().sum())
    return {
        "status": "rejected",
        "path": str(path),
        "sha256": sha256_file(path),
        "shape": [int(clinical.shape[0]), int(clinical.shape[1])],
        "patients": patients,
        "duplicate_patient_rows": duplicates,
        "blockers": blockers,
        "task_readiness": {
            "expression_modeling": False,
            "patient_idh_evaluation": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--neftel", type=Path, required=True)
    parser.add_argument("--cgga", type=Path, required=True)
    parser.add_argument("--tcga-clinical", type=Path, required=True)
    parser.add_argument("--neftel-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result: dict[str, Any] = {
        "neftel": audit_neftel(args.neftel, args.neftel_split),
        "cgga": audit_cgga(args.cgga),
        "tcga": audit_tcga(args.tcga_clinical),
    }
    result["status"] = (
        "accepted_with_limitations"
        if all(result[name]["status"] != "rejected" for name in ("neftel", "cgga", "tcga"))
        else "blocked"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "accepted_with_limitations" else 2


if __name__ == "__main__":
    raise SystemExit(main())
