#!/usr/bin/env python3
"""Create the Neftel-only analysis H5AD from the verified combined cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad  # type: ignore[import-untyped]

from gbm_study.plain_english import write_json_with_explanation


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/import_20260820/TP53 Dataset(preprocessed)/processed/full_cohort_with_states.h5ad"))
    parser.add_argument("--output", type=Path, default=Path("data/import_20260820/TP53 Dataset(preprocessed)/processed/neftel_analysis_cohort.h5ad"))
    args = parser.parse_args(argv)
    if hasattr(ad.settings, "allow_write_nullable_strings"):
        ad.settings.allow_write_nullable_strings = True
    data = ad.read_h5ad(args.input)
    selected = data[data.obs["Cohort"].astype(str).eq("Neftel")].copy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.write_h5ad(args.output)
    result = {
        "status": "completed",
        "scope": "Neftel-only analysis input extracted from verified combined H5AD",
        "input": {"path": str(args.input), "sha256": sha256(args.input)},
        "output": {"path": str(args.output), "sha256": sha256(args.output), "shape": list(selected.shape)},
        "patients": int(selected.obs["Sample"].astype(str).nunique()),
        "state_counts": {str(k): int(v) for k, v in selected.obs["derived_state"].value_counts().items()},
        "next_actions": ["Use the matching 27-patient split for the three baseline runs."],
    }
    write_json_with_explanation(args.output.with_suffix(".json"), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
