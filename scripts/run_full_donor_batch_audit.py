from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def _run_pca_silhouette(matrix: np.ndarray, labels: np.ndarray, seed: int, n_components: int = 20):
    embedding = PCA(n_components=min(n_components, matrix.shape[0] - 1, matrix.shape[1]),
                    random_state=seed).fit_transform(matrix)
    score = float(silhouette_score(embedding, labels))
    return embedding, score


def cohort_level_audit(adata, output_dir: Path, seed: int) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hvg_mask = adata.var["highly_variable"].to_numpy()
    matrix = adata.X[:, hvg_mask]
    matrix = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)

    cohort = adata.obs["Cohort"].astype(str).to_numpy()

    embedding, cohort_silhouette = _run_pca_silhouette(matrix, cohort, seed)

    fig, ax = plt.subplots(figsize=(7, 5))
    for label in sorted(set(cohort)):
        selected = cohort == label
        ax.scatter(embedding[selected, 0], embedding[selected, 1], s=3, alpha=0.45, label=label)
    ax.set_title("Full cohort PCA, colored by data source")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=8, markerscale=3)
    fig.tight_layout()
    figure_path = output_dir / "cohort_level_pca.png"
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)

    return {
        "n_cells": int(matrix.shape[0]),
        "n_hvg": int(hvg_mask.sum()),
        "cohort_counts": {k: int(v) for k, v in adata.obs["Cohort"].value_counts().items()},
        "silhouette_by_cohort": cohort_silhouette,
        "figure": str(figure_path),
        "interpretation": (
            "Cohort-level separation here reflects Neftel (single-cell) vs "
            "TCGA (bulk RNA-seq) being different assay types, not "
            "purely a donor/technical batch effect within one assay. "
            "Report as an assay-type effect, not conflated with the "
            "within-cohort donor/state finding below."
        ),
    }


def neftel_donor_audit(adata, output_dir: Path, seed: int) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    neftel = adata[adata.obs["Cohort"] == "Neftel"].copy()
    if neftel.n_obs == 0:
        return {"status": "blocked", "reason": "No Neftel cells found in this h5ad."}

    state_col = None
    for candidate in ("state", "CellAssignment"):
        if candidate in neftel.obs.columns:
            state_col = candidate
            break
    if state_col is None:
        return {
            "status": "blocked",
            "reason": "Neither 'state' nor 'CellAssignment' present in obs; "
                      "cannot run the within-Neftel donor-vs-state comparison.",
        }

    donor = neftel.obs["Sample"].astype(str).to_numpy()
    cell_label = neftel.obs[state_col].astype(str).to_numpy()

    if len(set(donor)) < 2 or len(set(cell_label)) < 2:
        return {"status": "blocked", "reason": "Need >=2 donors and >=2 label groups."}

    hvg_mask = neftel.var["highly_variable"].to_numpy()
    matrix = neftel.X[:, hvg_mask]
    matrix = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)

    embedding, donor_silhouette = _run_pca_silhouette(matrix, donor, seed)
    _, state_silhouette = _run_pca_silhouette(matrix, cell_label, seed)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for axis, labels, title in (
            (axes[0], donor, "Donor / Sample"),
            (axes[1], cell_label, f"{state_col}" + ("" if state_col == "state" else " (proxy; not AC/MES/NPC/OPC)")),
    ):
        for label in sorted(set(labels)):
            selected = labels == label
            axis.scatter(embedding[selected, 0], embedding[selected, 1], s=3, alpha=0.45, label=label)
        axis.set_title(title)
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
        axis.legend(fontsize=6, markerscale=3)
    fig.tight_layout()
    figure_path = output_dir / "neftel_donor_vs_state_pca.png"
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)

    risk = (
        "donor explains more separation than cell state"
        if donor_silhouette > state_silhouette
        else "cell state explains equal or more separation than donor"
    )

    result = {
        "status": "completed",
        "state_column_used": state_col,
        "n_cells": int(neftel.n_obs),
        "n_donors": int(len(set(donor))),
        "n_state_groups": int(len(set(cell_label))),
        "silhouette": {"donor": donor_silhouette, "state": state_silhouette},
        "batch_risk_interpretation": risk,
        "figure": str(figure_path),
    }
    if state_col != "state":
        result["state_label_warning"] = (
            f"'{state_col}' is a proxy; it is not the agreed AC/MES/NPC/OPC "
            "state label. Join the real Neftel four-state metadata before "
            "treating this as the paper's final figure."
        )
    return result


def run_full_audit(adata_path: Path, output_dir: Path, seed: int = 17) -> dict[str, Any]:
    if not adata_path.is_file():
        return {"status": "blocked", "reason": f"h5ad does not exist: {adata_path}"}

    adata = sc.read_h5ad(adata_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "status": "completed",
        "adata_path": str(adata_path),
        "cohort_level": cohort_level_audit(adata, output_dir, seed),
        "neftel_donor_vs_state": neftel_donor_audit(adata, output_dir, seed),
    }

    (output_dir / "donor_batch_audit_full.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Full-scale donor/batch audit across all internal cohorts.")
    parser.add_argument("--adata", type=Path, default=Path("data/processed/full_cohort_neftel_tcga.h5ad"))
    parser.add_argument("--output", type=Path, default=Path("results/full_data_audit"))
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    result = run_full_audit(args.adata, args.output, args.seed)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
