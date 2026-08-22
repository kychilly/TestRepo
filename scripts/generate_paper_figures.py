#!/usr/bin/env python3
"""Generate figures exclusively from current, machine-recorded run artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from gbm_study.plain_english import write_json_with_explanation

ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required current report is missing: {path}")
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _cell_metrics(path: Path) -> dict[str, Any]:
    payload = _read(path)
    return cast(dict[str, Any], payload["tasks"]["cell_state"]["metrics"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("reports/paper_figures"))
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    readiness_path = ROOT / "reports/readiness/execution_readiness.json"
    stage_path = ROOT / "reports/stage34/combined_full_candidate_run.json"
    metric_paths = {
        "PCA/logistic": ROOT
        / "reports/pilot_baselines_verified/seed42/pca_logreg_eval/metrics.json",
        "Harmony/kNN": ROOT
        / "reports/pilot_baselines_verified/seed42/harmony_knn_eval/metrics.json",
    }
    readiness = _read(readiness_path)
    stage = _read(stage_path)
    metrics = {name: _cell_metrics(path) for name, path in metric_paths.items()}

    figure_paths: list[str] = []
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    names = list(metrics)
    values = [float(metrics[name]["point_estimate"]["macro_f1"]) for name in names]
    lows = [
        values[index] - float(metrics[name]["bootstrap"]["macro_f1"]["ci_2_5"])
        for index, name in enumerate(names)
    ]
    highs = [
        float(metrics[name]["bootstrap"]["macro_f1"]["ci_97_5"]) - values[index]
        for index, name in enumerate(names)
    ]
    ax.bar(names, values, yerr=[lows, highs], capsize=5, color=["#4c78a8", "#72b7b2"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Held-out macro-F1")
    ax.set_title("Patient-held-out Neftel baseline bar (seed 42)")
    baseline_figure = output / "baseline_macro_f1.png"
    fig.savefig(baseline_figure, dpi=220, bbox_inches="tight")
    plt.close(fig)
    figure_paths.append(str(baseline_figure))

    bucket_order = [
        "destabilizing_driver",
        "functional_driver",
        "abstain",
        "unconfirmed",
        "data_deficient",
    ]
    counts = stage["bucket_counts"]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(
        [name.replace("_", "\n") for name in bucket_order],
        [int(counts.get(name, 0)) for name in bucket_order],
        color="#f58518",
    )
    ax.set_ylabel("Candidate genes")
    ax.set_title("Real Stage 3/4 outcomes (missing protein evidence retained)")
    stage_figure = output / "stage34_bucket_counts.png"
    fig.savefig(stage_figure, dpi=220, bbox_inches="tight")
    plt.close(fig)
    figure_paths.append(str(stage_figure))

    state_counts = readiness["internal_cohort"]["state_counts"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    state_order = ["AC", "MES", "NPC", "OPC"]
    ax.bar(state_order, [int(state_counts[name]) for name in state_order], color="#54a24b")
    ax.set_ylabel("Cells")
    ax.set_title("Internal Neftel cohort state composition")
    state_figure = output / "internal_state_counts.png"
    fig.savefig(state_figure, dpi=220, bbox_inches="tight")
    plt.close(fig)
    figure_paths.append(str(state_figure))

    result = {
        "status": "completed",
        "figures": figure_paths,
        "sources": [str(readiness_path), str(stage_path), *map(str, metric_paths.values())],
        "warnings": readiness.get("warnings", []),
        "reason": "All plotted values were loaded from current run artifacts; no readiness or performance value was hard-coded.",
    }
    write_json_with_explanation(output / "figure_manifest.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
