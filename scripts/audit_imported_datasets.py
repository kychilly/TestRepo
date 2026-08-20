#!/usr/bin/env python3
"""Audit the newly imported TP53 archive without modifying any dataset files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import anndata as ad  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from gbm_study.plain_english import write_json_with_explanation


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matrix_check(matrix: Any, rows: int, cols: int, chunk: int = 128) -> dict[str, Any]:
    nonfinite = 0
    negative = 0
    integer_like = True
    minimum = float("inf")
    maximum = float("-inf")
    for start in range(0, rows, chunk):
        block = matrix[start : start + chunk]
        block = block.toarray() if hasattr(block, "toarray") else np.asarray(block)
        finite = np.isfinite(block)
        nonfinite += int((~finite).sum())
        if finite.any():
            values = block[finite]
            minimum = min(minimum, float(values.min()))
            maximum = max(maximum, float(values.max()))
            negative += int((values < 0).sum())
            integer_like = integer_like and bool(np.allclose(values, np.floor(values)))
    return {
        "shape": [rows, cols],
        "nonfinite_values": nonfinite,
        "negative_values": negative,
        "integer_like": integer_like,
        "minimum": None if minimum == float("inf") else minimum,
        "maximum": None if maximum == float("-inf") else maximum,
    }


def inspect_h5ad(path: Path, *, state_column: str = "derived_state") -> dict[str, Any]:
    data = ad.read_h5ad(path, backed="r")
    obs = {str(column): data.obs[column].astype(str).value_counts().head(20).to_dict() for column in data.obs.columns}
    genes = {str(gene).upper() for gene in data.var_names}
    required = {"TP53", "IDH1", "EGFR", "RPRM"}
    result: dict[str, Any] = {
        "path": str(path),
        "sha256": sha256(path),
        "shape": [int(data.n_obs), int(data.n_vars)],
        "layers": sorted(str(key) for key in data.layers.keys()),
        "obs_columns": sorted(str(key) for key in data.obs.columns),
        "required_genes_present": sorted(required & genes),
        "required_genes_missing": sorted(required - genes),
        "matrix_X": matrix_check(data.X, data.n_obs, data.n_vars),
        "state_counts": obs.get(state_column, {}),
    }
    if "counts" in data.layers:
        result["matrix_counts"] = matrix_check(data.layers["counts"], data.n_obs, data.n_vars)
    return result


def inspect_mutations(path: Path) -> dict[str, Any]:
    table = pd.read_csv(path)
    patient_column = "patient_id" if "patient_id" in table else "patient"
    gene_column = "gene_symbol" if "gene_symbol" in table else "gene"
    return {
        "path": str(path),
        "sha256": sha256(path),
        "rows": int(len(table)),
        "columns": [str(column) for column in table.columns],
        "patients": int(table[patient_column].nunique()) if patient_column in table else None,
        "genes": int(table[gene_column].nunique()) if gene_column in table else None,
        "alteration_type_counts": {str(k): int(v) for k, v in table.get("alteration_type", pd.Series(dtype=str)).value_counts().items()},
        "missing_values": {str(k): int(v) for k, v in table.isna().sum().items()},
        "provenance_columns_present": sorted(set(table.columns) & {"Transcript_ID", "HGVSp", "protein_change", "genome_build", "source", "source_file"}),
    }


def inspect_edges(path: Path) -> dict[str, Any]:
    table = pd.read_csv(path)
    source = next((c for c in ("source_gene", "source", "tf") if c in table), None)
    target = next((c for c in ("target_gene", "target", "gene") if c in table), None)
    unique = int(table[[source, target]].drop_duplicates().shape[0]) if source and target else None
    return {"path": str(path), "sha256": sha256(path), "rows": int(len(table)), "unique_edges": unique, "columns": [str(c) for c in table.columns]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/import_20260820/TP53 Dataset(preprocessed)"))
    parser.add_argument("--output", type=Path, default=Path("reports/readiness/imported_dataset_audit.json"))
    args = parser.parse_args(argv)
    root = args.root
    files = {
        "full_cohort": root / "processed/full_cohort.h5ad",
        "pilot": root / "pilot/pilot_subsample.h5ad",
        "tcga_pilot": root / "pilot/tcga_pilot_subsample.h5ad",
        "cgga_pilot": root / "pilot/cgga_pilot_subsample.h5ad",
        "mutations": root / "pilot/patient_gene_mutation_join.csv",
        "grn_train": root / "prior/grn_pilot_train_prior.csv",
        "grn_holdout": root / "prior/grn_pilot_adit_holdout_check.csv",
    }
    blockers: list[str] = []
    inspected: dict[str, Any] = {}
    for name in ("full_cohort", "pilot", "tcga_pilot", "cgga_pilot"):
        if not files[name].is_file():
            blockers.append(f"Missing {name}: {files[name]}")
            continue
        inspected[name] = inspect_h5ad(files[name])
        if inspected[name]["matrix_X"]["nonfinite_values"]:
            blockers.append(f"{name} X contains non-finite values")
    if files["mutations"].is_file():
        inspected["mutations"] = inspect_mutations(files["mutations"])
        required_provenance = {"Transcript_ID", "HGVSp", "genome_build", "source_file"}
        if not required_provenance.issubset(set(inspected["mutations"]["columns"])):
            blockers.append("Derived mutation join is missing transcript/protein/build/source provenance fields")
    else:
        blockers.append(f"Missing mutations: {files['mutations']}")
    for name in ("grn_train", "grn_holdout"):
        if files[name].is_file():
            inspected[name] = inspect_edges(files[name])
        else:
            blockers.append(f"Missing {name}: {files[name]}")
    full = inspected.get("full_cohort", {})
    if "counts" not in full.get("layers", []):
        blockers.append("Full cohort has no counts layer")
    pilot_states = inspected.get("pilot", {}).get("state_counts", {})
    if pilot_states.get("Unknown", 0) == sum(pilot_states.values()) and pilot_states:
        blockers.append("Pilot state labels are all Unknown")
    result = {
        "status": "completed" if not blockers else "completed_with_blockers",
        "scope": "read-only audit of the newly imported TP53 dataset folder",
        "root": str(root),
        "files": {name: str(path) for name, path in files.items()},
        "inspected": inspected,
        "blockers": blockers,
        "next_actions": [
            "Use the full cohort counts layer for scVI only after integer-like counts are confirmed.",
            "Use a labeled single-cell external cohort for AC/MES/NPC/OPC external testing; CGGA bulk cannot supply those labels.",
            "Use patient_gene_mutation_join.csv for mutation/CNA provenance; add a methylation source before claiming silencing labels.",
            "Repair or regenerate the CGGA pilot H5AD from the raw CGGA expression files before external modeling.",
        ],
    }
    write_json_with_explanation(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
