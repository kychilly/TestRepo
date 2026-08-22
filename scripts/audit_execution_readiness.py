#!/usr/bin/env python3
"""Audit the complete Week 2-to-4 execution contract before using GPU time."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from baselines.base import load_patient_splits
from experiments.week3 import ablation_matrix
from gbm_study.plain_english import write_json_with_explanation
from models.grn import load_edges


ROOT = Path(__file__).resolve().parents[1]
CONFIRMED = {"destabilizing_driver", "functional_driver"}
REQUIRED_STATES = {"AC", "MES", "NPC", "OPC"}
REQUIRED_GENES = {"TP53", "IDH1", "EGFR", "RPRM"}


def _yaml(path: Path) -> dict[str, Any]:
    import yaml  # type: ignore[import-untyped]

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML object")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _resolve(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def run() -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    week3 = _yaml(ROOT / "config/week3_adit.yaml")
    cross = _yaml(ROOT / "config/cross_cohort.yaml")
    required_paths = {
        "internal_h5ad": _resolve(week3.get("cell_data_path")),
        "patient_split": _resolve(week3.get("split_file")),
        "checkpoint": _resolve(week3.get("checkpoint_path")),
        "vocabulary": _resolve(week3.get("vocabulary_path")),
        "model_args": _resolve(week3.get("model_args_path")),
        "candidate_genes": _resolve(week3.get("candidate_genes_path")),
        "validator_outcomes": _resolve(week3.get("validator_outcomes_path")),
        "grn_train": _resolve(week3.get("grn_train_prior_path")),
        "grn_holdout": _resolve(cross.get("grn_holdout_path")),
        "tcga": _resolve(cross.get("tcga_path")),
        "combined_external": _resolve(cross.get("full_cohort_path")),
        "cross_cohort_verdicts": _resolve(cross.get("verdicts_path")),
        "stage34_report": ROOT / "reports/stage34/combined_full_candidate_run.json",
    }
    missing = [name for name, path in required_paths.items() if not path.is_file()]
    if missing:
        blockers.append("Missing required files: " + ", ".join(missing))

    seeds = [int(value) for value in week3.get("seeds", [])]
    passes = int(week3.get("mc_dropout_passes", 0))
    token_length = int(week3.get("token_length", 0))
    if len(set(seeds)) < 3:
        blockers.append("Week 3 requires at least three distinct seeds")
    if not 20 <= passes <= 50:
        blockers.append("MC dropout must use 20-50 passes")
    if token_length < 2:
        blockers.append("A finite scGPT token_length is required")

    split_summary: dict[str, Any] = {}
    cohort_summary: dict[str, Any] = {}
    if not missing:
        split = load_patient_splits(required_paths["patient_split"], int(week3.get("fold", 0)))
        split_summary = {
            "train_patients": len(split.train),
            "validation_patients": len(split.validation),
            "test_patients": len(split.test),
            "overlap": bool(
                set(split.train) & set(split.validation)
                or set(split.train) & set(split.test)
                or set(split.validation) & set(split.test)
            ),
        }
        if split_summary["overlap"]:
            blockers.append("Patient split partitions overlap")

        import anndata as ad  # type: ignore[import-untyped]

        data = ad.read_h5ad(required_paths["internal_h5ad"], backed="r")
        patient_key = str(week3["patient_id_column"])
        state_key = str(week3["state_key"])
        if patient_key not in data.obs or state_key not in data.obs:
            blockers.append("Internal H5AD lacks the configured patient/state columns")
        else:
            states = Counter(data.obs[state_key].astype(str))
            patients = set(data.obs[patient_key].astype(str))
            observed_states = set(states)
            if not REQUIRED_STATES.issubset(observed_states):
                blockers.append("Internal H5AD lacks one or more required cell states")
            split_patients = set(split.train) | set(split.validation) | set(split.test)
            if not split_patients.issubset(patients):
                blockers.append("Patient split contains patients absent from the internal H5AD")
            cohort_summary = {
                "cells": int(data.n_obs),
                "genes": int(data.n_vars),
                "patients": len(patients),
                "state_counts": dict(sorted(states.items())),
                "required_genes_present": sorted(REQUIRED_GENES & set(map(str, data.var_names))),
                "all_required_genes_present": REQUIRED_GENES.issubset(
                    set(map(str, data.var_names))
                ),
            }
            if "counts" in data.layers:
                count_sample = data.layers["counts"][
                    : min(128, data.n_obs), : min(256, data.n_vars)
                ]
                if hasattr(count_sample, "toarray"):
                    count_sample = count_sample.toarray()
                count_values = np.asarray(count_sample, dtype=np.float64)
                sampled_integer_like = bool(
                    np.isfinite(count_values).all()
                    and (count_values >= 0).all()
                    and np.allclose(count_values, np.rint(count_values))
                )
                cohort_summary["counts_layer_sampled_integer_like"] = sampled_integer_like
                if not sampled_integer_like:
                    warnings.append(
                        "H5AD layers['counts'] contains sampled non-integer values; scVI is data-blocked"
                    )
            else:
                cohort_summary["counts_layer_sampled_integer_like"] = False
                warnings.append("H5AD has no counts layer; scVI is data-blocked")
            if not cohort_summary["all_required_genes_present"]:
                blockers.append("Internal H5AD lacks TP53, IDH1, EGFR, or RPRM")

    validator_summary: dict[str, Any] = {}
    if required_paths["validator_outcomes"].is_file():
        outcomes = _jsonl(required_paths["validator_outcomes"])
        outcome_counts = Counter(str(row.get("outcome")) for row in outcomes)
        confirmed = sorted(
            {str(row.get("gene")) for row in outcomes if row.get("outcome") in CONFIRMED}
        )
        validator_summary = {
            "rows": len(outcomes),
            "bucket_counts": dict(sorted(outcome_counts.items())),
            "confirmed_genes": confirmed,
            "confirmed_gene_count": len(confirmed),
        }
        if not confirmed:
            warnings.append(
                "Validator evidence has zero confirmed genes; validator-on arms are data-blocked"
            )

    grn_summary: dict[str, Any] = {}
    if required_paths["grn_train"].is_file() and required_paths["grn_holdout"].is_file():
        train_edges = load_edges(required_paths["grn_train"])
        held_edges = load_edges(required_paths["grn_holdout"])
        train_pairs = {
            (str(edge.data["source_gene"]), str(edge.data["target_gene"])) for edge in train_edges
        }
        held_pairs = {
            (str(edge.data["source_gene"]), str(edge.data["target_gene"])) for edge in held_edges
        }
        overlap = sorted(train_pairs & held_pairs)
        grn_summary = {
            "train_edges": len(train_pairs),
            "held_out_edges": len(held_pairs),
            "train_holdout_overlap": len(overlap),
        }
        if overlap:
            blockers.append("GRN training and held-out edge sets overlap")
        if len(held_pairs) < 20:
            warnings.append("GRN holdout has fewer than 20 unique positive edges")

    verdict_count = 0
    if required_paths["cross_cohort_verdicts"].is_file():
        with required_paths["cross_cohort_verdicts"].open(newline="", encoding="utf-8") as stream:
            verdict_count = sum(1 for _ in csv.DictReader(stream))
        if validator_summary and verdict_count != validator_summary["rows"]:
            blockers.append("Week 3 JSONL and Week 4 CSV verdict counts disagree")

    confirmed_count = int(validator_summary.get("confirmed_gene_count", 0))
    matrix = ablation_matrix()
    runnable_conditions = [
        item.name for item in matrix if not item.validator or confirmed_count > 0
    ]
    blocked_conditions = [item.name for item in matrix if item.name not in runnable_conditions]
    return {
        "status": "completed" if not blockers else "completed_with_blockers",
        "code_contract_ready": not blockers,
        "full_scientific_matrix_ready": not blockers and not blocked_conditions,
        "required_paths": {
            name: {
                "path": str(path),
                "exists": path.is_file(),
                "bytes": path.stat().st_size if path.is_file() else None,
            }
            for name, path in required_paths.items()
        },
        "week3_design": {
            "seeds": seeds,
            "mc_dropout_passes": passes,
            "token_length": token_length,
            "conditions": [item.name for item in matrix],
            "gpu_runnable_conditions": runnable_conditions,
            "data_blocked_conditions": blocked_conditions,
            "expected_runnable_runs": len(runnable_conditions) * len(seeds),
            "expected_total_runs": len(matrix) * len(seeds),
        },
        "internal_cohort": cohort_summary,
        "patient_split": split_summary,
        "validator": validator_summary,
        "grn": grn_summary,
        "week4_verdict_rows": verdict_count,
        "blockers": blockers,
        "warnings": warnings,
        "next_actions": [
            "Run the local CPU confirmation commands in docs/team_week2_to_week4_execution.md.",
            "Run scripts/bootstrap_a100.sh and the A100 preflight before starting scGPT.",
            "Add independent protein evidence before expecting validator-on conditions to complete.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("reports/readiness/execution_readiness.json")
    )
    args = parser.parse_args()
    try:
        result = run()
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        result = {
            "status": "completed_with_blockers",
            "code_contract_ready": False,
            "blockers": [str(exc)],
            "next_actions": ["Fix the reported contract error and rerun this audit."],
        }
    write_json_with_explanation(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("code_contract_ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())
