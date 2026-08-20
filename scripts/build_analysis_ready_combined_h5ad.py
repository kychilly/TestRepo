#!/usr/bin/env python3
"""Combine labeled Neftel cells with rebuilt finite CGGA bulk rows."""

from __future__ import annotations

import argparse
import hashlib
import json
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--neftel", type=Path, default=Path("data/import_20260820/TP53 Dataset(preprocessed)/processed/neftel_analysis_cohort.h5ad"))
    parser.add_argument("--cgga", type=Path, default=Path("data/import_20260820/TP53 Dataset(preprocessed)/processed/cgga_bulk_clean.h5ad"))
    parser.add_argument("--output", type=Path, default=Path("data/import_20260820/TP53 Dataset(preprocessed)/processed/analysis_ready_combined.h5ad"))
    args = parser.parse_args(argv)
    if hasattr(ad.settings, "allow_write_nullable_strings"):
        ad.settings.allow_write_nullable_strings = True
    neftel = ad.read_h5ad(args.neftel)
    cgga = ad.read_h5ad(args.cgga)
    common = [gene for gene in neftel.var_names.astype(str) if gene in set(cgga.var_names.astype(str))]
    neftel = neftel[:, common].copy()
    cgga = cgga[:, common].copy()
    cgga.obs["Cohort"] = "CGGA"
    cgga.obs["derived_state"] = "Unknown"
    if "cgga_batch" in cgga.obs:
        cgga.obs["CrossSection"] = cgga.obs["cgga_batch"].astype(str).to_numpy()
    else:
        cgga.obs["CrossSection"] = "CGGA"
    combined = ad.concat([neftel, cgga], join="inner", merge="first", index_unique=None)
    combined.obs_names_make_unique()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.write_h5ad(args.output)
    result = {
        "status": "completed",
        "scope": "analysis-ready combined Neftel state plus clean CGGA bulk cohort",
        "output": {"path": str(args.output), "sha256": sha256(args.output), "shape": list(combined.shape)},
        "inputs": {"neftel": {"path": str(args.neftel), "sha256": sha256(args.neftel)}, "cgga": {"path": str(args.cgga), "sha256": sha256(args.cgga)}},
        "common_genes": len(common),
        "cohort_counts": {str(k): int(v) for k, v in combined.obs["Cohort"].value_counts().items()},
        "idh_counts": {str(k): int(v) for k, v in combined.obs["IDH_status"].astype(str).value_counts().items()},
        "finite_values": bool(np.isfinite(combined.X).all()),
        "cell_state_external_truth": False,
        "next_actions": ["Use this file for patient-level CGGA IDH evaluation; do not score CGGA cell states."],
    }
    write_json_with_explanation(args.output.with_suffix(".json"), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
