#!/usr/bin/env python3
"""Run resumable Neftel/TCGA to held-out CGGA evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from evaluation.cross_cohort import (
    feature_groups,
    fit_arm,
    load_cohorts,
    load_completed_jsonl,
    sha256,
    summarize_runs,
)
from gbm_study.plain_english import write_json_with_explanation, write_jsonl_explanation
from models.grn import load_edges, score_held_out_edges


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def run(config_path: Path, output: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tcga_path = Path(config["tcga_path"])
    full_path = Path(config["full_cohort_path"])
    verdicts_path = Path(config["verdicts_path"])
    seeds = [int(seed) for seed in config.get("seeds", [17, 42, 101])]
    if len(set(seeds)) < 3:
        raise ValueError("At least three distinct seeds are required")
    fingerprint_payload = {
        "config_sha256": sha256(config_path),
        "tcga_sha256": sha256(tcga_path),
        "full_cohort_sha256": sha256(full_path),
        "verdicts_sha256": sha256(verdicts_path),
        "grn_train_sha256": sha256(Path(config["grn_train_prior_path"])),
        "grn_holdout_sha256": sha256(Path(config["grn_holdout_path"])),
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    checkpoint_path = output / "checkpoint.json"
    if checkpoint_path.is_file():
        prior = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        prior_fingerprint = prior.get("fingerprint")
        if prior_fingerprint is not None and prior_fingerprint != fingerprint:
            raise ValueError("Checkpoint fingerprint changed; use a new output directory")
    data = load_cohorts(tcga_path, full_path)
    try:
        groups = feature_groups(data["genes"], verdicts_path)
    except ValueError as exc:
        result: dict[str, Any] = {
            "status": "completed_with_blockers",
            "endpoint": "patient_level_IDH_mutation_status",
            "scope": "Week 4 integration preflight; no model arm was fit",
            "data": {
                "tcga_samples": int(len(data["tcga"][1])),
                "neftel_pseudobulk_samples": int(len(data["neftel"][1])),
                "cgga_labeled_samples": int(len(data["cgga"][1])),
                "common_genes": int(len(data["genes"])),
            },
            "proposal_metrics": {
                "macro_f1": "blocked: confirmed feature group is unavailable",
                "variant_auroc": "blocked: no independent variant-effect labels and scores",
                "abstention_accuracy": "blocked: no independent abstention gold labels",
                "grn_edge_auroc": "not_run",
                "internal_to_external_performance_drop": "blocked: no confirmed arm",
            },
            "provenance": fingerprint_payload,
            "blockers": [
                str(exc),
                "Current Stage 3/4 evidence has zero confirmed genes because protein/AlphaFold evidence is absent.",
                "Do not substitute the stale reports/stage34/verdicts.csv file.",
            ],
            "next_actions": [
                "Add independent AlphaFold/protein evidence and rerun Stage 3/4.",
                "Run this same Week 4 command after the current verdict CSV is regenerated.",
            ],
        }
        output.mkdir(parents=True, exist_ok=True)
        write_json_with_explanation(output / "results.json", result)
        return result

    runs_path = output / "runs.jsonl"
    completed = load_completed_jsonl(runs_path)
    done = {(str(row["arm"]), int(row["seed"])) for row in completed}
    prediction_dir = output / "predictions"
    for seed in seeds:
        rng = np.random.default_rng(seed)
        shuffled = rng.choice(
            groups["unconfirmed_genes"], size=len(groups["confirmed_genes"]), replace=False
        )
        run_groups = {**groups, "shuffled_size_matched_control": np.sort(shuffled)}
        for arm, indices in run_groups.items():
            if (arm, seed) in done:
                continue
            result, predictions = fit_arm(data, indices, seed=seed)
            result["arm"] = arm
            result["genes"] = data["genes"][indices].astype(str).tolist()
            prediction_path = prediction_dir / f"{arm}_seed{seed}.jsonl"
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            prediction_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions),
                encoding="utf-8",
            )
            write_jsonl_explanation(
                prediction_path,
                row_count=len(predictions),
                description="Internal TCGA and never-fit CGGA IDH predictions for one arm and seed.",
            )
            result["predictions_path"] = str(prediction_path)
            append_jsonl(runs_path, result)
            completed.append(result)
            write_json_with_explanation(
                checkpoint_path,
                {
                    "status": "completed_with_blockers",
                    "completed_runs": len(completed),
                    "expected_runs": len(seeds) * len(run_groups),
                    "fingerprint": fingerprint,
                    "next_actions": ["Re-run the same command to resume any missing arm/seed."],
                },
            )
    write_jsonl_explanation(
        runs_path,
        row_count=len(completed),
        description="One completed cross-cohort model arm per seed; this is the resumable ledger.",
    )
    write_json_with_explanation(
        checkpoint_path,
        {
            "status": "completed",
            "completed_runs": len(completed),
            "expected_runs": len(seeds) * 4,
            "fingerprint": fingerprint,
            "next_actions": [
                "Use a new output directory if any input, seed, or configuration changes."
            ],
        },
    )

    train_edges = load_edges(Path(config["grn_train_prior_path"]))
    held_edges = load_edges(Path(config["grn_holdout_path"]))
    grn = score_held_out_edges(
        train_edges, held_edges, lambda edge: float(edge.get("confidence", 0.0))
    )
    grn["limitation"] = (
        "This is a prior-confidence sanity check with one unique held-out positive, not a learned GRN result."
    )
    summary = summarize_runs(completed)
    result = {
        "status": "completed_with_blockers",
        "endpoint": "patient_level_IDH_mutation_status",
        "scope": "CPU tabular feature-mask analysis; not the blocked scGPT ablation matrix",
        "training": "TCGA training split plus 27 IDH-wildtype Neftel patient pseudobulks",
        "internal_test": "stratified held-out TCGA patients",
        "external_test": f"{len(data['cgga'][1])} labeled CGGA patients never used for fitting",
        "normalization": "within-sample percentile ranks fixed without labels",
        "seeds": seeds,
        "data": {
            "tcga_samples": len(data["tcga"][1]),
            "neftel_pseudobulk_samples": len(data["neftel"][1]),
            "cgga_labeled_samples": len(data["cgga"][1]),
            "common_genes": len(data["genes"]),
            "confirmed_genes": data["genes"][groups["confirmed_genes"]].tolist(),
            "unconfirmed_gene_count": len(groups["unconfirmed_genes"]),
        },
        "proposal_metrics": {
            "macro_f1": "completed for the IDH endpoint",
            "variant_auroc": "blocked: no independent benign/pathogenic variant-effect labels and scores",
            "abstention_accuracy": "blocked: no independent gold abstain/non-abstain verdicts",
            "grn_edge_auroc": grn,
            "internal_to_external_performance_drop": "completed for the IDH endpoint",
        },
        **summary,
        "provenance": fingerprint_payload,
        "blockers": [
            "CGGA has zero AC/MES/NPC/OPC labels, so external four-state macro-F1 cannot be calculated.",
            "Variant-effect AUROC needs independent positive and negative labels plus variant scores; these are absent.",
            "Abstention accuracy needs independent gold verdicts; reusing validator outputs as truth would be circular.",
            "GRN holdout contains one unique positive edge, so its AUROC is only a software sanity check.",
            "Only TP53 and IDH1 are confirmed; the headline comparison is exploratory and underpowered.",
            "The confirmed pair did not beat the seeded size-matched shuffled control, so the required call is no_result.",
            "The all-on/validator-off/GRN-off/MC-off scGPT arms still require CUDA and an external endpoint compatible with CGGA labels.",
        ],
        "next_actions": [
            "Add authoritative CGGA state labels if the intended endpoint is AC/MES/NPC/OPC.",
            "Add ClinVar/DMS-labeled benign and pathogenic variants with model scores for variant AUROC.",
            "Add independently adjudicated validator outcomes for abstention accuracy.",
            "Replace the one-edge GRN holdout with a frozen larger positive set and matched negatives.",
            "Expand confirmed protein evidence beyond TP53 and IDH1 before treating the headline number as paper-ready.",
            "Do not make the paper claim unless the confirmed arm beats the prespecified shuffled control.",
        ],
    }
    write_json_with_explanation(output / "results.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/cross_cohort.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reports/cross_cohort"))
    args = parser.parse_args()
    result = run(args.config, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
