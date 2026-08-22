#!/usr/bin/env python3
"""Safely standardize patient and real Neftel-state columns in an H5AD copy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gbm_study.plain_english import write_json_with_explanation

VALID_STATES = {"AC", "MES", "NPC", "OPC"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patient-source", default="Sample")
    parser.add_argument("--state-source", default="derived_state")
    parser.add_argument("--report", type=Path, default=Path("reports/schema_fix.json"))
    args = parser.parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Refusing to overwrite the input H5AD; choose a new --output path")

    import anndata as ad  # type: ignore[import-untyped]

    data = ad.read_h5ad(args.input)
    if args.patient_source not in data.obs:
        raise ValueError(f"Missing real patient column: {args.patient_source}")
    if args.state_source not in data.obs:
        raise ValueError(f"Missing real state column: {args.state_source}")
    states = data.obs[args.state_source].astype(str)
    unknown = sorted(set(states) - VALID_STATES)
    if unknown:
        raise ValueError(f"State column contains values outside AC/MES/NPC/OPC: {unknown}")
    data.obs["patient_id"] = data.obs[args.patient_source].astype(str).to_numpy()
    data.obs["state"] = states.to_numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.write_h5ad(args.output)
    result = {
        "status": "completed",
        "input": str(args.input),
        "output": str(args.output),
        "rows": int(data.n_obs),
        "patients": int(data.obs["patient_id"].nunique()),
        "state_counts": {
            str(key): int(value) for key, value in data.obs["state"].value_counts().items()
        },
        "reason": "Copied existing patient and state labels; no labels were inferred or randomized.",
    }
    write_json_with_explanation(args.report, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
