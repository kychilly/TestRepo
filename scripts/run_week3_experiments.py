#!/usr/bin/env python3
"""Execute the actual Week 3 experiment matrix and persist every artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.week3 import run_matrix
from models.scgpt_internal import run_internal_cohort


def _config(path: Path) -> dict[str, Any]:
    import yaml  # type: ignore[import-untyped]

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Week 3 experiment config must be a YAML object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/week3_adit.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reports/week3_adit/experiments"))
    args = parser.parse_args()
    result = run_matrix(_config(args.config), run_internal_cohort, args.output)
    print(json.dumps({
        "status": result["status"],
        "completed_runs": result["completed_runs"],
        "blocked_runs": result["blocked_runs"],
        "backbone_scope": result["backbone_scope"],
        "manifest": str(args.output / "manifest.json"),
    }, indent=2, sort_keys=True))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
