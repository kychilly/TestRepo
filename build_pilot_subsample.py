#!/usr/bin/env python3
"""Build a deterministic IDH-balanced pilot with a frozen HVG panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from gbm_study.plain_english import write_json_with_explanation


REQUIRED_GENES = ("TP53", "IDH1", "EGFR", "RPRM")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_idh(value: Any) -> str | None:
    text = str(value).strip().lower()
    if text in {"wt", "wildtype", "wild-type", "idh-wildtype"}:
        return "Wildtype"
    if text in {"mutant", "mut", "idh-mutant"}:
        return "Mutant"
    return None


def _gene_variance(matrix: Any) -> NDArray[np.float64]:
    if hasattr(matrix, "multiply"):
        mean = np.asarray(matrix.mean(axis=0)).ravel()
        squared_mean = np.asarray(matrix.multiply(matrix).mean(axis=0)).ravel()
        return np.asarray(np.maximum(squared_mean - mean**2, 0.0), dtype=np.float64)
    dense = np.asarray(matrix, dtype=np.float64)
    return np.asarray(np.var(dense, axis=0), dtype=np.float64)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/import_20260820/TP53 Dataset(preprocessed)/processed/analysis_ready_combined.h5ad"
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/pilot/pilot_balanced_idh_hvg.h5ad")
    )
    parser.add_argument("--patient-column", default="Sample")
    parser.add_argument("--idh-column", default="IDH_status")
    parser.add_argument("--target-patients", type=int, default=20)
    parser.add_argument("--hvg-count", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    if args.target_patients < 2 or args.hvg_count < len(REQUIRED_GENES):
        raise ValueError("Pilot requires at least two patients and four genes")

    import anndata as ad  # type: ignore[import-untyped]

    if hasattr(ad.settings, "allow_write_nullable_strings"):
        ad.settings.allow_write_nullable_strings = True
    data = ad.read_h5ad(args.input)
    if args.patient_column not in data.obs or args.idh_column not in data.obs:
        raise ValueError("Input H5AD lacks the configured patient or IDH column")
    obs_status = data.obs[args.idh_column].map(_canonical_idh)
    patient_status: dict[str, str] = {}
    for patient, status in zip(data.obs[args.patient_column].astype(str), obs_status, strict=True):
        if status is None:
            continue
        previous = patient_status.get(patient)
        if previous is not None and previous != status:
            raise ValueError(f"Patient {patient} has conflicting IDH labels")
        patient_status[patient] = status
    by_status = {
        status: sorted(patient for patient, value in patient_status.items() if value == status)
        for status in ("Wildtype", "Mutant")
    }
    if not all(by_status.values()):
        raise ValueError("Both Wildtype and Mutant patients are required")
    rng = np.random.default_rng(args.seed)
    for values in by_status.values():
        rng.shuffle(values)
    mutant_target = args.target_patients // 2
    wildtype_target = args.target_patients - mutant_target
    if len(by_status["Mutant"]) < mutant_target or len(by_status["Wildtype"]) < wildtype_target:
        raise ValueError("Not enough patients to create the requested balanced pilot")
    selected_patients = sorted(
        by_status["Mutant"][:mutant_target] + by_status["Wildtype"][:wildtype_target]
    )
    row_mask = data.obs[args.patient_column].astype(str).isin(selected_patients).to_numpy()
    selected = data[row_mask].copy()
    genes = np.asarray(selected.var_names.astype(str))
    missing_required = sorted(set(REQUIRED_GENES) - set(genes))
    if missing_required:
        raise ValueError(f"Required genes are absent: {missing_required}")
    variances = _gene_variance(selected.X)
    order = np.lexsort((genes, -variances))
    panel_size = min(args.hvg_count, len(genes))
    required_set = set(REQUIRED_GENES)
    chosen = [gene for gene in genes[order] if gene not in required_set][
        : panel_size - len(REQUIRED_GENES)
    ] + list(REQUIRED_GENES)
    variance_by_gene = dict(zip(genes.tolist(), variances.tolist(), strict=True))
    chosen = sorted(chosen, key=lambda gene: (-variance_by_gene[gene], gene))
    pilot = selected[:, chosen].copy()
    pilot.uns["pilot_selection"] = {
        "seed": args.seed,
        "target_patients": args.target_patients,
        "hvg_method": "highest variance within selected rows; required genes forced",
        "required_genes": list(REQUIRED_GENES),
        "source_sha256": sha256(args.input),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pilot.write_h5ad(args.output)
    status_counts = {
        status: sum(patient_status[patient] == status for patient in selected_patients)
        for status in ("Wildtype", "Mutant")
    }
    result = {
        "status": "completed",
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "shape": list(pilot.shape),
        "patients": len(selected_patients),
        "idh_patient_counts": status_counts,
        "hvg_count": int(pilot.n_vars),
        "required_genes_present": sorted(set(REQUIRED_GENES) & set(pilot.var_names)),
        "seed": args.seed,
        "next_actions": [
            "Use patient-level splits; never split rows or cells independently.",
            "Keep CGGA separate as the external cohort for final evaluation.",
        ],
    }
    write_json_with_explanation(args.output.with_suffix(".json"), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
