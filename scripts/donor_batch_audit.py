#!/usr/bin/env python3
"""Audit donor versus available cell-assignment structure in a Neftel H5AD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def _categorical(group: Any) -> np.ndarray[str]:
    categories = np.asarray(group["categories"][...]).astype(str)
    codes = np.asarray(group["codes"][...])
    return np.asarray([categories[int(code)] if int(code) >= 0 else "missing" for code in codes])


def run_audit(path: Path, output: Path, *, seed: int = 17) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as exc:
        return {"status": "blocked", "reason": f"h5py is required to inspect H5AD: {exc}"}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        return {"status": "blocked", "reason": f"matplotlib is required for audit visuals: {exc}"}
    if not path.is_file():
        return {"status": "blocked", "reason": f"H5AD does not exist: {path}"}
    with h5py.File(path, "r") as handle:
        if "Sample" not in handle["obs"] or "CellAssignment" not in handle["obs"]:
            return {
                "status": "blocked",
                "reason": "H5AD requires obs/Sample and obs/CellAssignment",
            }
        donor = _categorical(handle["obs"]["Sample"])
        cell_assignment = _categorical(handle["obs"]["CellAssignment"])
        hvg = np.asarray(handle["var"]["highly_variable"][...], dtype=bool)
        matrix = np.asarray(handle["X"][:, hvg], dtype=np.float32)
    if len(np.unique(donor)) < 2 or len(np.unique(cell_assignment)) < 2:
        return {
            "status": "blocked",
            "reason": "Donor and cell-assignment labels need at least two groups",
        }
    embedding = PCA(n_components=20, random_state=seed).fit_transform(matrix)
    donor_silhouette = float(silhouette_score(embedding, donor))
    state_silhouette = float(silhouette_score(embedding, cell_assignment))
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for axis, labels, title in (
        (axes[0], donor, "Donor / Sample"),
        (axes[1], cell_assignment, "CellAssignment (proxy; not AC/MES/NPC/OPC)"),
    ):
        for label in sorted(set(labels)):
            selected = labels == label
            axis.scatter(
                embedding[selected, 0], embedding[selected, 1], s=3, alpha=0.45, label=label
            )
        axis.set_title(title)
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
        axis.legend(fontsize=6, markerscale=3)
    fig.tight_layout()
    figure_path = output / "donor_vs_cellassignment_pca.png"
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    risk = (
        "donor explains more separation than available cell-assignment labels"
        if donor_silhouette > state_silhouette
        else "available cell-assignment labels explain equal or more separation than donor"
    )
    result = {
        "status": "completed",
        "n_cells": int(len(donor)),
        "n_hvg": int(hvg.sum()),
        "n_donors": int(len(set(donor))),
        "n_cell_assignment_groups": int(len(set(cell_assignment))),
        "silhouette": {"donor": donor_silhouette, "cell_assignment_proxy": state_silhouette},
        "batch_risk_interpretation": risk,
        "state_label_warning": "CellAssignment is not the agreed AC/MES/NPC/OPC state label.",
    }
    (output / "donor_batch_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata", type=Path, default=Path("data/raw/neftel/neftel_qc.h5ad"))
    parser.add_argument("--output", type=Path, default=Path("results/week1_data_audit"))
    args = parser.parse_args()
    result = run_audit(args.adata, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
