#!/usr/bin/env python3
"""Compatibility entry point for the Week 1 donor/batch data audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.donor_batch_audit import run_audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata", type=Path, default=Path("data/raw/neftel/neftel_qc.h5ad"))
    parser.add_argument("--output", type=Path, default=Path("results/week1_data_audit"))
    args = parser.parse_args()
    result = run_audit(args.adata, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
