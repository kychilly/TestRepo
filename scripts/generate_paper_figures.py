#!/usr/bin/env python3
"""Generate evidence-labeled figures from the current repository artifacts.

The figures intentionally distinguish measured, synthetic, and blocked results.
They are suitable for internal manuscript preparation and should be regenerated
after every locked real-data run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def save(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "reports/paper_figures")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out = args.output if args.output.is_absolute() else ROOT / args.output
    plt.rcParams.update({"font.size": 9, "axes.titleweight": "bold"})

    # Figure 1: readiness and evidence status.
    fig, ax = plt.subplots(figsize=(9, 4.8))
    labels = [
        "Software\ncontracts",
        "Local pilot\nmeasurements",
        "Scientific\nreadiness",
        "Publication\nreadiness",
    ]
    values = [8, 3, 3, 4]
    colors = ["#2ca02c", "#d9a441", "#d62728", "#d62728"]
    bars = ax.bar(labels, values, color=colors, width=0.62)
    ax.set_ylim(0, 10)
    ax.set_ylabel("Readiness score / 10")
    ax.set_title("Current project readiness (audit dated 2026-08-15)")
    ax.text(
        0.01,
        -0.18,
        "Scores are audit judgments, not model performance; scientific and publication readiness remain blocked.",
        transform=ax.transAxes,
        fontsize=8,
    )
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, value + 0.25, str(value), ha="center", weight="bold"
        )
    save(fig, out / "01_readiness_status.png")
    plt.close(fig)

    # Figure 2: data availability and missing study inputs.
    fig, ax = plt.subplots(figsize=(9, 5))
    items = [
        "Neftel\nlocal",
        "TCGA\nrequired",
        "CGGA 325\nrequired",
        "CGGA 693\nrequired",
        "scGPT\ncheckpoint",
        "Vocabulary",
        "CUDA GPU",
        "Raw counts",
    ]
    vals = [7930, 528, 325, 693, 1, 1, 1, 1]
    avail = ["present", "missing", "missing", "missing", "missing", "missing", "missing", "missing"]
    colors = ["#2ca02c"] + ["#d62728"] * 7
    bars = ax.bar(items, vals, color=colors)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylabel("Count (log scale for heterogeneous assets)")
    ax.set_title("Registered data and required assets")
    ax.text(
        0.01,
        -0.2,
        "Neftel count is cells; cohort counts are patients; binary assets are shown at 1 only to mark presence/absence.",
        transform=ax.transAxes,
        fontsize=8,
    )
    for bar, state in zip(bars, avail):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(bar.get_height(), 1) * 1.25,
            state,
            ha="center",
            rotation=90,
            fontsize=8,
        )
    save(fig, out / "02_data_and_asset_status.png")
    plt.close(fig)

    # Figure 3: baseline test performance with patient bootstrap intervals.
    metric_files = {
        "PCA/logistic": "reports/pilot_baselines/pca_test_eval/metrics.json",
        "Harmony/kNN": "reports/pilot_baselines/harmony_test_eval/metrics.json",
    }
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(metric_files))
    width = 0.36
    for j, metric in enumerate(["macro_f1", "balanced_accuracy"]):
        vals, lows, highs = [], [], []
        for path in metric_files.values():
            d = read_json(path)
            vals.append(d["point_estimate"][metric])
            lows.append(vals[-1] - d["bootstrap"][metric]["ci_2_5"])
            highs.append(d["bootstrap"][metric]["ci_97_5"] - vals[-1])
        bars = ax.bar(
            x + (j - 0.5) * width,
            vals,
            width,
            yerr=[lows, highs],
            capsize=4,
            label=metric.replace("_", " ").title(),
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.035,
                f"{val:.3f}",
                ha="center",
                fontsize=8,
            )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test score")
    ax.set_title("Measured pilot baseline performance (cell-state; patient bootstrap 95% CI)")
    ax.set_xticks(x, list(metric_files))
    ax.legend()
    save(fig, out / "03_baseline_performance.png")
    plt.close(fig)

    # Figure 4: confusion matrices, fixed state order.
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    states = ["AC", "MES", "NPC", "OPC"]
    for ax, (name, path) in zip(axes, metric_files.items()):
        cm = np.asarray(read_json(path)["point_estimate"]["confusion_matrix"]["values"])
        row_norm = cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(row_norm, vmin=0, vmax=1, cmap="Blues")
        for i in range(4):
            for j in range(4):
                ax.text(
                    j,
                    i,
                    f"{row_norm[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if row_norm[i, j] > 0.5 else "black",
                    fontsize=8,
                )
        ax.set_title(name)
        ax.set_xticks(range(4), states)
        ax.set_yticks(range(4), states)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.suptitle("Row-normalized pilot test confusion matrices")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="Fraction of true state")
    save(fig, out / "04_confusion_matrices.png")
    plt.close(fig)

    # Figure 5: batch-risk and split coverage.
    audit = read_json("results/week1_data_audit/donor_batch_audit.json")
    splits = read_json("TP53 Dataset(preprocessed)/pilot/patient_splits.json")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    sil = audit["silhouette"]
    axes[0].bar(
        ["Cell assignment\nproxy", "Donor"],
        [sil["cell_assignment_proxy"], sil["donor"]],
        color=["#4c78a8", "#f58518"],
    )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Silhouette score")
    axes[0].set_title("Batch-risk audit")
    counts = [len(splits[k]) for k in ["train", "validation", "test"]]
    axes[1].bar(["Train", "Validation", "Test"], counts, color="#72b7b2")
    axes[1].set_ylabel("Patients")
    axes[1].set_title("Pilot patient split (26 total)")
    for i, n in enumerate(counts):
        axes[1].text(i, n + 0.2, str(n), ha="center")
    fig.suptitle("Data structure and leakage-risk evidence")
    save(fig, out / "05_batch_and_split_audit.png")
    plt.close(fig)

    # Figure 6: GRN and validator status, with blocked publication gate visible.
    grn = read_json("reports/jeffrey_grn_run/grn_sanity.json")
    gate = read_json("reports/jeffrey_grn_run/validator_gate.json")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(
        ["Held-out\npositive", "Negatives"],
        [grn["held_out_edges"], grn["negative_edges"]],
        color=["#e45756", "#bab0ab"],
    )
    axes[0].set_title(f"GRN sanity check: AUROC={grn['auroc']:.1f}")
    axes[0].set_ylabel("Edges")
    axes[1].bar(
        ["Classification\ngate", "Publication\ngate"],
        [int(gate["classification_gate_passed"]), int(gate["publication_gate_passed"])],
        color=["#2ca02c", "#d62728"],
    )
    axes[1].set_ylim(0, 1.2)
    axes[1].set_yticks([0, 1], ["blocked/fail", "pass"])
    axes[1].set_title("Validator gate status")
    fig.suptitle("Mechanistic branch: software pass does not equal publication readiness")
    save(fig, out / "06_grn_validator_status.png")
    plt.close(fig)

    # Figure 7: compute blockers and measured-vs-missing values.
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = ["CUDA available", "Checkpoint", "Vocabulary", "Real GPU timing", "MC-dropout timing"]
    values = [0, 0, 0, 0, 0]
    ax.barh(labels, values, color="#d62728")
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 1], ["missing", "available"])
    ax.set_title("Compute readiness: no measured scGPT/MC-dropout run yet")
    ax.text(
        0.01,
        -0.18,
        "Do not estimate GPU seconds from this Mac/CPU environment; obtain a CUDA benchmark after assets are fixed.",
        transform=ax.transAxes,
        fontsize=8,
    )
    save(fig, out / "07_compute_blockers.png")
    plt.close(fig)

    manifest = {
        "status": "completed",
        "evidence_policy": "figures distinguish measured local artifacts from blocked or synthetic branches",
        "figures": sorted(p.name for p in out.glob("*.png")),
        "measured": [
            "pilot baseline test metrics",
            "donor/batch audit",
            "pilot split counts",
            "GRN sanity edge counts",
            "validator classification/publication gate flags",
        ],
        "blocked_or_missing": [
            "TCGA/CGGA integrated evaluation",
            "scGPT checkpoint/vocabulary/CUDA",
            "raw counts for scVI",
            "real candidate Stage 3-4 audit",
            "complete protein-evidence provenance",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
