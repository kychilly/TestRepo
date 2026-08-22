#!/usr/bin/env python3
"""Production Stage 3/4 entry point using candidate-aligned real inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gbm_study.plain_english import write_json_with_explanation
from scripts.run_stage34_validation import run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--records", type=Path, default=Path("data/pilot/stage34_combined_records.jsonl")
    )
    parser.add_argument(
        "--candidates", type=Path, default=Path("data/pilot/internal_candidate_universe.jsonl")
    )
    parser.add_argument("--pool-candidates", type=Path, action="append", default=[])
    parser.add_argument("--gold-outcomes", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config/stage34.yaml"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/stage34/combined_full_candidate_run.json"),
    )
    args = parser.parse_args()
    try:
        result = run(
            args.records,
            args.config,
            candidates_path=args.candidates,
            pool_candidate_paths=args.pool_candidates,
            gold_outcomes_path=args.gold_outcomes,
            seed=args.seed,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        result = {"status": "failed", "scientifically_complete": False, "reason": str(exc)}
    write_json_with_explanation(args.output, result)
    console_summary = {
        key: result.get(key)
        for key in (
            "status",
            "scientifically_complete",
            "candidate_count",
            "bucket_counts",
            "confirmable_count_primary",
            "feasibility",
            "comparison",
        )
        if key in result
    }
    console_summary["full_report"] = str(args.output)
    print(json.dumps(console_summary, indent=2, sort_keys=True))
    return 0 if result.get("status") == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
