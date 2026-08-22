#!/usr/bin/env python3
"""Create a deterministic patient-only split for the supplied pilot H5AD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gbm_study.plain_english import companion_path, explain


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    import anndata as ad

    data = ad.read_h5ad(args.adata, backed="r")
    patients = sorted(data.obs["Sample"].astype(str).unique().tolist())
    if len(patients) < 6:
        raise ValueError("Pilot split requires at least six patients")
    rng = np.random.default_rng(args.seed)
    shuffled = np.asarray(patients, dtype=object)
    rng.shuffle(shuffled)
    n_test = max(2, round(len(shuffled) * 0.2))
    n_validation = max(2, round(len(shuffled) * 0.2))
    payload = {
        "train": sorted(shuffled[: len(shuffled) - n_test - n_validation].tolist()),
        "validation": sorted(
            shuffled[len(shuffled) - n_test - n_validation : len(shuffled) - n_test].tolist()
        ),
        "test": sorted(shuffled[len(shuffled) - n_test :].tolist()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    companion_path(args.output).write_text(
        explain(
            {
                "status": "completed",
                "seed": args.seed,
                "patient_count": len(patients),
                "next_actions": ["Use this same split file for every model arm and seed."],
            },
            source=str(args.output),
        ),
        encoding="utf-8",
    )
    print(
        json.dumps({"status": "completed", "patients": len(patients), "output": str(args.output)})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
