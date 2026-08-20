#!/usr/bin/env python3
"""Create a read-only evidence inventory for Adit's Week 1-4 deliverables."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from gbm_study.plain_english import write_json_with_explanation


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def classify(path: Path, *, stale: bool = False) -> dict[str, Any]:
    payload = read_json(path)
    if payload is None:
        status = "missing_or_invalid"
    else:
        raw = str(payload.get("status", "unknown"))
        status = "stale" if stale else raw
    return {
        "path": str(path),
        "status": status,
        "sha256": sha256(path),
        "declared_status": payload.get("status") if payload else None,
        "reason": payload.get("reason") if payload else None,
        "blockers": payload.get("blockers", []) if payload else [],
    }


def run() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    assets = root / "artifacts/models/scGPT_pancancer"
    artifacts = {
        "week1_environment": classify(root / "results/compute/week1_environment_check.json"),
        "week1_benchmark": classify(root / "results/compute/week1_scgpt_benchmark.json", stale=True),
        "latest_model_benchmark": classify(root / "reports/week3_adit/scgpt_timing.json"),
        "week2_adit": classify(root / "reports/week2_adit/report_current.json", stale=True),
        "week3_manifest": classify(root / "reports/week3_adit/experiments/manifest.json"),
        "stage34": classify(root / "reports/stage34/fixture_feasibility_clean.json"),
        "cross_cohort": classify(root / "reports/cross_cohort_combined_v2/results.json"),
        "readiness": classify(root / "reports/readiness/current.json"),
    }
    model_assets = {
        name: {"path": str(path), "exists": path.is_file(), "sha256": sha256(path)}
        for name, path in {
            "checkpoint": assets / "best_model.pt",
            "vocabulary": assets / "vocab.json",
            "model_args": assets / "args.json",
        }.items()
    }
    return {
        "status": "completed",
        "scope": "Adit Week 1-4 evidence inventory",
        "generated_from": str(root),
        "artifacts": artifacts,
        "model_assets": model_assets,
        "current_scientific_call": "no_result_until_real_scGPT_and_external_endpoint_runs_complete",
        "known_issues": [
            "Historical benchmark and Week 2 reports are stale relative to the downloaded official model assets.",
            "Week 3 has 12 blocked runs and no completed real scGPT arm.",
            "The current cross-cohort result is a clean-CGGA IDH CPU feature-mask analysis, not the scGPT ablation matrix.",
            "CGGA bulk rows do not contain AC/MES/NPC/OPC truth.",
            "The GRN holdout has one unique positive edge and cannot support a paper claim.",
        ],
        "next_actions": [
            "Update configs to the verified checkpoint and model arguments.",
            "Run the A100 preflight and exactly 1,000-cell benchmark.",
            "Run the complete Week 3 matrix with persistent checkpoints.",
            "Add an external single-cell state endpoint and independent variant/abstention truth.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("reports/adit_week_audit/current.json"))
    args = parser.parse_args()
    result = run()
    write_json_with_explanation(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
