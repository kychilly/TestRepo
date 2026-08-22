#!/usr/bin/env python3
"""Run repository-only synthetic checks with unbiased status reporting.

Synthetic metrics are labeled synthetic and are never suitable for the paper.
GPU and Ishaan-owned validator checks are reported as blocked/held when their
real prerequisites are absent; this script never substitutes CPU timings or
invented protein evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from baselines.base import CellData, MethodNotApplicable
from baselines.harmony_knn import HarmonyKNN
from baselines.pca_logreg import PCALogReg
from baselines.scvi_probe import ScVIProbe
from evaluation.metrics import binary_metrics, cell_metrics
from gbm_study.gpu_planner import GPUPlanningError, plan_cuda_work
from gbm_study.plain_english import write_json_with_explanation


def synthetic_data() -> CellData:
    rng = np.random.default_rng(17)
    states = np.array(["AC", "MES", "NPC", "OPC"] * 8)
    patients = np.repeat(np.array(["p1", "p2", "p3", "p4"]), 8)
    batches = np.repeat(np.array(["b1", "b2", "b1", "b2"]), 8)
    X = rng.poisson(2, size=(32, 8)).astype(float)
    for row, state in enumerate(states):
        X[row, ("AC", "MES", "NPC", "OPC").index(state)] += 8
    return CellData(
        X,
        patients,
        np.array([f"cell-{i}" for i in range(32)]),
        states,
        tuple(f"G{i}" for i in range(8)),
        batches,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("baseline_results/synthetic_smoke_report.json")
    )
    args = parser.parse_args(argv)
    data = synthetic_data()
    train = data.subset(np.flatnonzero(np.isin(data.patient_id, ["p1", "p2"])))
    test = data.subset(np.flatnonzero(np.isin(data.patient_id, ["p3", "p4"])))
    report: dict[str, Any] = {
        "synthetic": True,
        "scientific_use": "forbidden",
        "baseline_results": {},
    }

    pca = PCALogReg(components=2, seed=17).fit(train, {"split": "train"})
    pca_pred = pca.predict(test, {"split": "test"})
    pca_prob = pca.predict_proba(test, {"split": "test"})
    report["baseline_results"]["pca_logreg"] = {
        "status": "completed",
        "macro_f1": cell_metrics(test.state, pca_pred, pca_prob)["macro_f1"],
    }

    for name, model in (
        ("scvi", ScVIProbe(latent_size=4, epochs=2, seed=17)),
        (
            "harmony_knn",
            HarmonyKNN(components=2, n_neighbors=3, harmony_covariate="batch", seed=17),
        ),
    ):
        try:
            model.fit(train, {"split": "train", "early_stopping": False})
            prediction = model.predict(test, {"split": "test"})
            probability = model.predict_proba(test, {"split": "test"})
            report["baseline_results"][name] = {
                "status": "completed",
                "macro_f1": cell_metrics(test.state, prediction, probability)["macro_f1"],
            }
        except (MethodNotApplicable, ValueError, RuntimeError, ImportError) as exc:
            report["baseline_results"][name] = {
                "status": "blocked",
                "metric": None,
                "reason": str(exc),
            }

    idh_true = np.array([0, 1, 0, 1])
    idh_scores = np.array([0.1, 0.9, 0.2, 0.8])
    report["baseline_results"]["idh_evaluation"] = {
        "status": "completed",
        "metric_scope": "synthetic_patient_level_only",
        "auroc": binary_metrics(idh_true, idh_scores, ("auroc",))["auroc"],
    }

    try:
        report["baseline_results"]["gpu_plan"] = plan_cuda_work(
            cells=1000, token_length=2048, batch_size=32
        ).to_dict()
    except GPUPlanningError as exc:
        report["baseline_results"]["gpu_plan"] = {
            "status": "blocked",
            "metric": None,
            "reason": str(exc),
        }

    report["baseline_results"]["candidate_schema"] = {
        "status": "completed",
        "metric": None,
        "producer": "implemented_and_schema_validated",
        "validator_consumer": "held_for_Ishaan_signoff",
    }
    write_json_with_explanation(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
