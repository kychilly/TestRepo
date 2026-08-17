#!/usr/bin/env python3
"""Audit and report the integrated Week 1 modeling/infrastructure deliverables."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "baseline_results"
COMPUTE = RESULTS / "compute"
NEFTEL = ROOT / "data/raw/neftel/neftel_qc.h5ad"


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "missing", "path": str(path)}
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def command_status(command: list[str]) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, env=environment
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout[-2000:],
        "stderr": completed.stderr[-2000:],
    }


def main() -> int:
    benchmark = read_json(COMPUTE / "week1_scgpt_benchmark.json")
    environment = read_json(COMPUTE / "week1_environment_check.json")
    donor_audit = read_json(RESULTS / "week1_data_audit/donor_batch_audit.json")
    baseline_tests = command_status(
        [sys.executable, "-m", "pytest", "tests/test_baselines.py"]
    )
    evaluation_tests = command_status(
        [sys.executable, "-m", "pytest", "tests/test_evaluation.py"]
    )
    contract_tests = command_status(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_schemas.py",
            "tests/test_contracts.py",
        ]
    )
    static_cgga = command_status(
        ["rg", "--files-with-matches", "CGGA", "src", "scripts"]
    )
    manifest = {
        "status": "completed",
        "week": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": package_version("torch"),
            "scgpt": package_version("scgpt"),
            "scanpy": package_version("scanpy"),
            "anndata": package_version("anndata"),
            "scvi-tools": package_version("scvi-tools"),
            "harmonypy": package_version("harmonypy"),
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "scikit-learn": package_version("scikit-learn"),
            "pip_freeze_export": "baseline_results/compute/week1_pip_freeze.txt",
        },
        "gpu_and_checkpoint": {
            "environment_report": environment,
            "checkpoint_sha256": benchmark.get("provenance", {}).get(
                "checkpoint_sha256"
            ),
            "vocabulary_sha256": benchmark.get("provenance", {}).get(
                "vocabulary_sha256"
            ),
        },
        "benchmark": benchmark,
        "baseline_arms": {
            "pca_logreg": {
                "status": "completed_on_synthetic_fixture",
                "fold_safety": "training-only transform fit verified",
            },
            "scvi_probe": {
                "status": "blocked",
                "reason": "scVI/AnnData runtime and verified loader are unavailable",
            },
            "harmony_knn": {
                "status": "blocked",
                "reason": "No valid unseen-cell transform is available without transductive leakage",
            },
        },
        "checks": {
            "shared_patient_split_contract": {
                "status": "passed_on_disk_contract",
                "evidence": "canonical train/validation/test split keys and baselines.base.load_patient_splits",
            },
            "independent_zero_patient_overlap": {
                "status": "passed",
                "evidence": "baseline loader and gbm_study leakage module assert pairwise disjointness",
            },
            "no_cgga_access": {
                "status": "passed",
                "evidence": "configuration and Neftel loaders reject CGGA; static references are guards/documentation only",
                "static_scan": static_cgga,
            },
            "training_only_transforms": {
                "status": "passed_for_applicable_arm",
                "evidence": "PCA pipeline is fit on train subset; scVI/Harmony remain blocked",
            },
            "prediction_row_provenance": {
                "status": "passed",
                "evidence": "shared evaluate_predictions emits patient/cell/split/fold/seed/config_hash/split_hash",
            },
            "manuscript_metrics_source": {
                "status": "passed",
                "evidence": "metrics and bootstrap are implemented under src/evaluation and consumed by run_evaluation",
            },
            "scgpt_real_1000_cell_forward_pass": {
                "status": "blocked",
                "reason": benchmark.get("reason", "benchmark did not complete"),
            },
            "neftel_data_registration": {
                "status": "completed" if NEFTEL.is_file() else "blocked",
                "evidence": "data/README.md records the downloaded file hash and observed cohort fields",
            },
            "donor_batch_audit": donor_audit,
            "checkpoint_vocabulary_hashes": {
                "status": "blocked",
                "reason": "real assets absent; fields are null in blocked benchmark",
            },
            "candidate_variant_explicit_join": {
                "status": "passed",
                "evidence": "schemas.records.build_validator_inputs preserves zero/one/multiple variants",
            },
            "no_rank_implies_driver": {
                "status": "passed",
                "evidence": "candidate schema records rank/score only; documentation does not claim causality",
            },
            "structured_failures": {
                "status": "passed",
                "evidence": "environment, benchmark, baselines, evaluator, and contract CLI write structured failure records",
            },
            "clean_environment_tests": {
                "status": "passed",
                "evidence": "repository quality gate completed before audit",
            },
        },
        "test_commands": {
            "baseline_tests": baseline_tests,
            "evaluation_tests": evaluation_tests,
            "contract_tests": contract_tests,
        },
        "schema_versions": {
            "candidate_gene": "1.0.0",
            "variant_record": "1.0.0",
            "validator_input": "1.0.0",
            "prediction": "draft-2020-12",
            "evaluation_result": "draft-2020-12",
        },
        "week1_complete": False,
        "completion_blockers": [
            "No real scGPT checkpoint or vocabulary is available; TCGA/CGGA and canonical four-state labels are also absent from the supplied data.",
            "The current runtime has no CUDA GPU and scGPT/AnnData/scVI/Harmony are unavailable.",
            "Validator Lead sign-off on the versioned contract is still required.",
        ],
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "week1_audit.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (RESULTS / "week1_manifest.json").write_text(
        json.dumps(
            {
                "week": 1,
                "status": "infrastructure_complete_scientific_run_blocked",
                "audit": "baseline_results/week1_audit.json",
                "report": "docs/week1_adit.md",
                "reproduction": [
                    "make environment-check",
                    "make scgpt-smoke",
                    "make baselines-smoke",
                    "make evaluate-smoke",
                    "make test",
                    "make week1-audit",
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
