#!/usr/bin/env python3
"""Rebuild a finite CGGA bulk H5AD from the raw CGGA ZIP members."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

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


def read_matrix(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        name = next(name for name in archive.namelist() if name.endswith(".txt"))
        return pd.read_csv(archive.open(name), sep="\t", low_memory=False).set_index("Gene_Name")


def read_clinical(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        name = next(name for name in archive.namelist() if name.endswith(".txt"))
        return pd.read_csv(archive.open(name), sep="\t", low_memory=False).set_index("CGGA_ID")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/import_20260820/TP53 Dataset(preprocessed)/raw/cgga"))
    parser.add_argument("--output", type=Path, default=Path("data/import_20260820/TP53 Dataset(preprocessed)/processed/cgga_bulk_clean.h5ad"))
    args = parser.parse_args(argv)
    root = args.raw_root
    expression_paths = sorted(root.glob("CGGA.mRNAseq_*.RSEM-genes*.zip"))
    clinical_paths = sorted(root.glob("CGGA.mRNAseq_*_clinical*.zip"))
    matrices = [read_matrix(path) for path in expression_paths]
    clinical = pd.concat([read_clinical(path) for path in clinical_paths], axis=0)
    matrix = pd.concat(matrices, axis=1, join="outer").fillna(0.0)
    matrix = matrix[~matrix.index.duplicated(keep="first")]
    matrix = matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    matrix = matrix.replace([np.inf, -np.inf], 0.0)
    obs = clinical.reindex(matrix.columns).copy()
    obs["Sample"] = obs.index.astype(str)
    obs["Cohort"] = "CGGA"
    obs["IDH_status"] = obs["IDH_mutation_status"].astype(str).replace({"Mutant": "Mutant", "Wildtype": "Wildtype"})
    obs["derived_state"] = "Unknown"
    if hasattr(ad.settings, "allow_write_nullable_strings"):
        ad.settings.allow_write_nullable_strings = True
    result_data = ad.AnnData(X=matrix.T.to_numpy(dtype=np.float32), obs=obs, var=pd.DataFrame(index=matrix.index.astype(str)))
    result_data.var_names_make_unique()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result_data.write_h5ad(args.output)
    result = {
        "status": "completed",
        "scope": "clean CGGA bulk expression rebuilt from raw RSEM ZIP members",
        "output": {"path": str(args.output), "sha256": sha256(args.output), "shape": list(result_data.shape)},
        "expression_sources": [{"path": str(path), "sha256": sha256(path)} for path in expression_paths],
        "clinical_sources": [{"path": str(path), "sha256": sha256(path)} for path in clinical_paths],
        "idh_counts": {str(k): int(v) for k, v in obs["IDH_status"].value_counts().items()},
        "finite_values": bool(np.isfinite(result_data.X).all()),
        "cell_state_truth": False,
        "next_actions": ["Use this file for external patient-level IDH evaluation only; it has no AC/MES/NPC/OPC labels."],
    }
    write_json_with_explanation(args.output.with_suffix(".json"), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
