#!/usr/bin/env python3
"""Attach verified Neftel state labels to the new full cohort by exact cell ID."""

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
    parser.add_argument("--full", type=Path, default=Path("data/import_20260820/TP53 Dataset(preprocessed)/processed/full_cohort.h5ad"))
    parser.add_argument("--states", type=Path, default=Path("data/neftel_qc-002.h5ad"))
    parser.add_argument("--output", type=Path, default=Path("data/import_20260820/TP53 Dataset(preprocessed)/processed/full_cohort_with_states.h5ad"))
    args = parser.parse_args(argv)
    full = ad.read_h5ad(args.full)
    states = ad.read_h5ad(args.states, backed="r")
    if hasattr(ad.settings, "allow_write_nullable_strings"):
        ad.settings.allow_write_nullable_strings = True
    if "state" not in states.obs:
        raise SystemExit("The older Neftel file has no state column")
    full_neftel = full[full.obs["Cohort"].astype(str).eq("Neftel")].copy()
    state_ids = set(states.obs_names.astype(str))
    full_ids = set(full_neftel.obs_names.astype(str))
    if state_ids != full_ids:
        raise SystemExit(f"Exact cell-ID join failed: old={len(state_ids)} new={len(full_ids)} overlap={len(state_ids & full_ids)}")
    state_map = states.obs["state"].astype(str).to_dict()
    full.obs["derived_state"] = [state_map.get(str(cell), "Unknown") for cell in full.obs_names]
    if "CrossSection" in states.obs:
        batch_map = states.obs["CrossSection"].astype(str).to_dict()
        full.obs["CrossSection"] = [batch_map.get(str(cell), "not_available") for cell in full.obs_names]
        full.obs["batch_source"] = ["data/neftel_qc-002.h5ad exact cell-ID join" if str(cell) in batch_map else "not_available" for cell in full.obs_names]
    full.obs["state_source"] = ["data/neftel_qc-002.h5ad exact cell-ID join" if str(cell) in state_map else "not_available" for cell in full.obs_names]
    full.obs["state_source_sha256"] = sha256(args.states)
    # Older anndata releases reject pandas nullable strings when writing H5AD.
    # Convert only the newly added metadata to ordinary Python strings.
    for column in ("derived_state", "CrossSection", "batch_source", "state_source", "state_source_sha256"):
        if column not in full.obs:
            continue
        full.obs[column] = full.obs[column].astype(object)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    full.write_h5ad(args.output)
    result = {
        "status": "completed",
        "scope": "new full cohort with verified Neftel state labels",
        "input_full": {"path": str(args.full), "sha256": sha256(args.full), "shape": list(full.shape)},
        "input_state_file": {"path": str(args.states), "sha256": sha256(args.states), "cells_joined": len(state_ids)},
        "output": {"path": str(args.output), "sha256": sha256(args.output), "shape": list(full.shape)},
        "state_counts": {str(k): int(v) for k, v in full.obs["derived_state"].value_counts().items()},
        "join": "exact obs_names for all 6,576 Neftel cells; CGGA rows remain Unknown",
        "next_actions": [
            "Use this H5AD for four-state internal training/evaluation after the nonfinite/representation audit passes.",
            "Do not treat CGGA Unknown rows as cell-state truth.",
        ],
    }
    write_json_with_explanation(args.output.with_suffix(".json"), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
