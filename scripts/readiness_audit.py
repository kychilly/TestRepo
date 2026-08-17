#!/usr/bin/env python3
"""Machine-check the real pilot prerequisites; never substitutes synthetic evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _split(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "folds" in payload:
        payload = payload["folds"][0]
    return {key: [str(x) for x in value] for key, value in payload.items() if isinstance(value, list)}


def audit(
    pilot: Path,
    split: Path,
    mutations: Path,
    grn_train: Path,
    grn_holdout: Path,
    output: Path,
) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}
    for label, path in (("pilot_h5ad", pilot), ("split", split), ("mutation_table", mutations),
                        ("grn_train", grn_train), ("grn_holdout", grn_holdout)):
        checks[label] = {"path": str(path), "exists": path.is_file()}
        if not path.is_file():
            blockers.append(f"Missing {label}: {path}")

    split_payload = _split(split) if split.is_file() else {}
    checks["split_counts"] = {key: len(value) for key, value in split_payload.items()}

    if pilot.is_file():
        try:
            import anndata as ad  # type: ignore[import-not-found]

            data = ad.read_h5ad(pilot, backed="r")
            genes = {str(x).upper() for x in data.var_names}
            required_genes = {"TP53", "IDH1", "EGFR", "RPRM"}
            obs_columns = {str(x) for x in data.obs.columns}
            checks["expression"] = {
                "shape": [int(data.n_obs), int(data.n_vars)],
                "layers": sorted(str(x) for x in data.layers.keys()),
                "obs_columns": sorted(obs_columns),
                "required_genes_present": sorted(required_genes & genes),
                "required_genes_missing": sorted(required_genes - genes),
            }
            if required_genes - genes:
                blockers.append("Pilot is missing required genes: " + ", ".join(sorted(required_genes - genes)))
            if "derived_state" not in obs_columns:
                blockers.append("Pilot has no derived_state labels for the requested state model")
            elif "IDH" not in " ".join(obs_columns).upper():
                blockers.append("Pilot has no explicit IDH-status column; derived cell states are not a substitute for both IDH statuses")
            if "Sample" not in obs_columns:
                blockers.append("Pilot has no Sample patient identifier column")
            if "counts" not in data.layers:
                blockers.append("Pilot has no layers['counts']; scVI baseline is not applicable")
            if "CrossSection" in obs_columns:
                values = data.obs["CrossSection"].astype(str).dropna().unique().tolist()
                checks["expression"]["cross_section_values"] = sorted(values)
                if len(values) < 2:
                    warnings.append("Pilot has fewer than two CrossSection batches; Harmony baseline is not applicable")
        except Exception as exc:  # dependency or malformed-file failure is actionable
            blockers.append(f"Could not inspect pilot H5AD: {exc}")

    if mutations.is_file():
        try:
            import pandas as pd  # type: ignore[import-not-found]

            table = pd.read_csv(mutations)
            required = {"patient_id", "gene_symbol", "variant_status", "impact"}
            missing = required - set(table.columns)
            checks["mutations"] = {
                "rows": int(len(table)),
                "columns": [str(x) for x in table.columns],
                "status_counts": {str(k): int(v) for k, v in table["variant_status"].value_counts().items()}
                if "variant_status" in table else {},
            }
            if missing:
                blockers.append("Mutation table is missing columns: " + ", ".join(sorted(missing)))
            for column in ("transcript_id", "protein_change", "genome_build", "source_file"):
                if column not in table.columns:
                    blockers.append(f"Mutation table lacks provenance/mapping field: {column}")
            if "silencing" not in set(table.get("variant_status", [])):
                blockers.append("Mutation table has no silencing calls; add methylation/CNV-derived silencing evidence or mark unavailable")
        except Exception as exc:
            blockers.append(f"Could not inspect mutation table: {exc}")

    checks["grn"] = {"train_exists": grn_train.is_file(), "heldout_exists": grn_holdout.is_file()}
    if grn_holdout.is_file():
        try:
            import pandas as pd  # type: ignore[import-not-found]

            holdout = pd.read_csv(grn_holdout)
            checks["grn"]["heldout_rows"] = int(len(holdout))
            if len(holdout) < 10:
                warnings.append("GRN held-out set is very small; AUROC is a sanity check, not a powered result")
        except Exception as exc:
            blockers.append(f"Could not inspect GRN holdout: {exc}")

    result = {
        "status": "ready" if not blockers else "blocked",
        "scientific_scope": "real_pilot_assets",
        "checks": checks,
        "blockers": blockers,
        "warnings": warnings,
        "next_actions": [
            "Provide a counts layer or explicitly record scVI as not applicable.",
            "Add both IDH statuses and retain patient-level split provenance.",
            "Replace the mutation table with transcript/protein/genome-build/source-aware calls.",
            "Provide a real checkpoint, vocabulary, gene metadata, and model-specific mask_score_provider on CUDA.",
            "Run scripts/run_pilot_scgpt.py, then scripts/run_stage34_validation.py with its real candidate output.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--mutations", type=Path, required=True)
    parser.add_argument("--grn-train", type=Path, required=True)
    parser.add_argument("--grn-holdout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.pilot, args.split, args.mutations, args.grn_train, args.grn_holdout, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
