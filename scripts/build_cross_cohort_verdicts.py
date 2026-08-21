#!/usr/bin/env python3
"""Convert the current Stage 3/4 JSON assignments to the Week 4 CSV input."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage34-report",
        type=Path,
        default=Path("reports/stage34/combined_full_candidate_run.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/pilot/stage34_verdicts_current.csv"),
    )
    args = parser.parse_args()
    report = json.loads(args.stage34_report.read_text(encoding="utf-8"))
    assignments = report.get("assignments")
    if not isinstance(assignments, list) or not assignments:
        raise ValueError("Stage 3/4 report has no assignments")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["candidate_id", "gene", "outcome"])
        writer.writeheader()
        for row in assignments:
            writer.writerow(
                {
                    "candidate_id": str(row["candidate_id"]),
                    "gene": str(row["gene"]).upper(),
                    "outcome": str(row["real_outcome"]),
                }
            )
    print(json.dumps({"status": "completed", "rows": len(assignments), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
