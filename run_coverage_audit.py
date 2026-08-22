#!/usr/bin/env python3
"""Report candidate-level validator coverage from the current Stage 3/4 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gbm_study.plain_english import write_json_with_explanation


def run(stage34_report: Path) -> dict[str, Any]:
    source = json.loads(stage34_report.read_text(encoding="utf-8"))
    assignments = source.get("assignments")
    if not isinstance(assignments, list) or not assignments:
        raise ValueError("Stage 3/4 report has no candidate assignments")
    total = len(assignments)
    deficient = sum(
        isinstance(row, dict) and row.get("real_outcome") == "data_deficient" for row in assignments
    )
    abstain = sum(
        isinstance(row, dict) and row.get("real_outcome") == "abstain" for row in assignments
    )
    return {
        "status": "completed",
        "scope": "candidate_level_stage34_assignments",
        "source_report": str(stage34_report),
        "candidate_count": total,
        "data_deficient_count": deficient,
        "data_deficient_fraction": deficient / total,
        "abstain_count": abstain,
        "abstain_fraction": abstain / total,
        "usable_non_abstain_non_deficient_count": total - deficient - abstain,
        "next_actions": [
            "Report this coverage rather than silently dropping deficient candidates.",
            "Add independent protein evidence and rerun Stage 3/4 to improve coverage.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage34-report",
        type=Path,
        default=Path("reports/stage34/combined_full_candidate_run.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("reports/stage34/coverage_audit.json"))
    args = parser.parse_args()
    try:
        result = run(args.stage34_report)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        result = {
            "status": "failed",
            "reason": str(exc),
            "next_actions": ["Regenerate the real Stage 3/4 report and rerun."],
        }
    write_json_with_explanation(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
