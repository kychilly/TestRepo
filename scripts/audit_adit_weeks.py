#!/usr/bin/env python3
"""Create a read-only evidence inventory for Adit's Week 1-4 deliverables."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
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
        "warnings": payload.get("warnings", []) if payload else [],
    }


def branch_status(root: Path, ref: str) -> dict[str, Any]:
    """Record whether an available team ref is already in the current HEAD."""
    present = (
        subprocess.run(
            ["git", "show-ref", "--verify", "--quiet", f"refs/remotes/{ref}"],
            cwd=root,
            check=False,
        ).returncode
        == 0
    )
    integrated = False
    if present:
        integrated = (
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", ref, "HEAD"],
                cwd=root,
                check=False,
            ).returncode
            == 0
        )
    return {"ref": ref, "present": present, "integrated_into_head": integrated}


def run() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    assets = root / "artifacts/models/scGPT_pancancer"
    artifacts = {
        "execution_readiness": classify(root / "reports/readiness/execution_readiness.json"),
        "baseline_bar": classify(root / "reports/pilot_baselines_verified/baseline_bar.json"),
        "grn_sanity": classify(root / "reports/jeffrey_grn_run/grn_sanity_current.json"),
        "stage34": classify(root / "reports/stage34/combined_full_candidate_run.json"),
        "cross_cohort": classify(root / "reports/cross_cohort_current/results.json"),
        "a100_benchmark_existing": classify(
            root / "reports/week3_adit/scgpt_timing.json", stale=True
        ),
        "week3_manifest_existing": classify(
            root / "reports/week3_adit/experiments/manifest.json", stale=True
        ),
    }
    model_assets = {
        name: {"path": str(path), "exists": path.is_file(), "sha256": sha256(path)}
        for name, path in {
            "checkpoint": assets / "best_model.pt",
            "vocabulary": assets / "vocab.json",
            "model_args": assets / "args.json",
        }.items()
    }
    team_refs = {
        "jeffrey_week2": branch_status(root, "origin/Jeffrey-week2"),
        "alexis_week2": branch_status(root, "origin/alexis-week2"),
        "ishaan_week2": branch_status(root, "origin/ishaan-week2"),
        "ishaan_week1": branch_status(root, "origin/ishaan-week1"),
    }
    return {
        "status": "completed",
        "scope": "Adit Week 1-4 evidence inventory",
        "generated_from": str(root),
        "artifacts": artifacts,
        "model_assets": model_assets,
        "team_week2_integration": {
            "refs": team_refs,
            "jeffrey_inputs": {
                "dataset_archive_audit": str(root / "reports/readiness/dataset_archives.json"),
                "mutation_join": str(
                    root
                    / "data/import_20260820/TP53 Dataset(preprocessed)/pilot/patient_gene_mutation_join.csv"
                ),
                "grn_train_prior": str(
                    root
                    / "data/import_20260820/TP53 Dataset(preprocessed)/prior/grn_pilot_train_prior.csv"
                ),
                "grn_holdout": str(
                    root
                    / "data/import_20260820/TP53 Dataset(preprocessed)/prior/grn_pilot_adit_holdout_check.csv"
                ),
            },
            "ishaan_inputs": {
                "validator": str(root / "validator.py"),
                "shuffled_validator": str(root / "shuffled_validator.py"),
                "stage34_report": str(root / "reports/stage34/combined_full_candidate_run.json"),
            },
            "alexis_inputs": {
                "baselines": str(root / "scripts/run_baseline.py"),
                "evaluation": str(root / "eval.py"),
                "baseline_summary": str(
                    root / "reports/pilot_baselines_verified/baseline_bar.json"
                ),
            },
            "week3_consumers": [
                "config/week3_adit.yaml",
                "scripts/run_week3_experiments.py",
                "src/experiments/week3.py",
                "src/models/scgpt_internal.py",
            ],
            "week4_consumers": [
                "scripts/run_cross_cohort.py",
                "reports/cross_cohort_current/results.json",
            ],
        },
        "current_scientific_call": "no_result_until_protein_evidence_and_a100_and_external_truth_are_complete",
        "known_issues": [
            "The A100 benchmark and Week 3 matrix have not been rerun with the current checkpoint-compatible environment and code fingerprint.",
            "Stage 3/4 has zero confirmed genes because independent protein evidence is absent; only three validator-off GPU runs are currently runnable.",
            "scVI is data-blocked because the supplied counts layer contains non-integer log-scale values.",
            "CGGA bulk rows do not contain AC/MES/NPC/OPC truth.",
            "The GRN holdout has one unique positive edge and cannot support a paper claim.",
        ],
        "next_actions": [
            "Run the A100 preflight and exactly 1,000-cell benchmark.",
            "Run the currently runnable validator-off seeds with persistent checkpoints.",
            "Add independent protein evidence, rerun Stage 3/4, then complete all 12 ablations.",
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
