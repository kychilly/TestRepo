#!/usr/bin/env python3
"""Materialize the frozen Stage 3/4 outcomes for the Week 3 consumer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gbm_study.plain_english import write_jsonl_explanation


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
        default=Path("data/pilot/week3_validator_outcomes.jsonl"),
    )
    args = parser.parse_args()
    report: dict[str, Any] = json.loads(args.stage34_report.read_text(encoding="utf-8"))
    assignments = report.get("assignments")
    if not isinstance(assignments, list) or not assignments:
        raise ValueError("Stage 3/4 report has no assignments")
    rows = []
    for item in assignments:
        if not isinstance(item, dict):
            raise ValueError("Stage 3/4 assignment is not an object")
        rows.append(
            {
                "candidate_id": str(item["candidate_id"]),
                "gene": str(item["gene"]).upper(),
                "outcome": str(item["real_outcome"]),
                "source_report": str(args.stage34_report),
                "source_status": str(report.get("status")),
                "scientifically_complete": bool(report.get("scientifically_complete", False)),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    write_jsonl_explanation(
        args.output,
        row_count=len(rows),
        description="Frozen real Stage 3/4 outcomes consumed by the Week 3 validator-on arm.",
        next_actions=[
            "Replace this file only by rerunning Stage 3/4 after independent protein evidence is added.",
            "A zero-confirmed outcome is a valid data result and causes the validator-on arm to remain blocked rather than inventing genes.",
        ],
    )
    print(json.dumps({"status": "completed", "rows": len(rows), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
