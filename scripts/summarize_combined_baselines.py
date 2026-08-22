#!/usr/bin/env python3
"""Summarize combined-data baseline/eval statuses across seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gbm_study.plain_english import write_json_with_explanation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("reports/baselines_combined"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/baselines_combined/summary.json")
    )
    args = parser.parse_args(argv)
    rows: list[dict[str, Any]] = []
    final_dirs = sorted(args.root.glob("final_seed*"))
    seed_dirs = final_dirs or sorted(args.root.glob("fold0_seed*"))
    seed_dirs = seed_dirs or sorted(args.root.glob("seed*"))
    for seed_dir in seed_dirs:
        seed = (
            seed_dir.name.removeprefix("fold0_seed").removeprefix("final_seed").removeprefix("seed")
        )
        for method in ("pca_logreg", "scvi_probe", "harmony_knn"):
            metadata = seed_dir / method / "run_metadata.json"
            error = seed_dir / method / "run_error.json"
            evaluation = seed_dir / method / "eval" / "metrics.json"
            if not evaluation.is_file():
                evaluation = seed_dir / f"{method}_eval" / "metrics.json"
            if metadata.is_file():
                run = json.loads(metadata.read_text())
                item = {
                    "seed": int(seed),
                    "method": method,
                    "status": run.get("status"),
                    "run": str(metadata),
                }
                if evaluation.is_file():
                    metrics = json.loads(evaluation.read_text())
                    item["evaluation"] = (
                        metrics.get("tasks", {})
                        .get("cell_state", {})
                        .get("metrics", {})
                        .get("point_estimate", {})
                    )
                rows.append(item)
            elif error.is_file():
                run = json.loads(error.read_text())
                rows.append(
                    {
                        "seed": int(seed),
                        "method": method,
                        "status": run.get("status"),
                        "reason": run.get("reason"),
                        "run": str(error),
                    }
                )
    completed = sum(item["status"] == "completed" for item in rows)
    missing_methods = [item["method"] for item in rows if item.get("status") != "completed"]
    result = {
        "status": "completed_with_blockers" if completed < len(rows) else "completed",
        "scope": "three requested baselines on combined Neftel analysis cohort",
        "rows": rows,
        "completed_runs": completed,
        "non_applicable_runs": sum(item["status"] == "not_applicable" for item in rows),
        "scientific_note": (
            "PCA/logistic regression and Harmony/kNN are evaluated on the same "
            "patient-held-out Neftel test split. scVI is not scientifically applicable "
            "without true raw integer counts."
        ),
        "methods_not_completed": missing_methods,
        "next_actions": [
            "Supply the original raw Neftel count matrix, preserve it in layers['counts'], then rerun scVI with the same patient split."
        ],
    }
    write_json_with_explanation(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
