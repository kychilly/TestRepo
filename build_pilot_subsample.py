import json
import numpy as np
import pandas as pd
import scanpy as sc
from load_cgga_cohort import load_cgga_cohort
from load_tcga_cohort import load_tcga_cohort


def generate_pilot_subsample():
    print("[1/6] Loading Neftel single-cell dataset...")
    adata = sc.read_h5ad("data/processed/neftel_qc.h5ad")

    print("[2/6] Loading split definitions...")
    with open("splits/patient_splits.json", "r") as f:
        splits_data = json.load(f)

    # Collect all valid patient IDs across all folds (train, validation, test)
    valid_split_patients = set()
    for fold in splits_data.get("folds", []):
        for split_type in ["train", "validation", "test"]:
            valid_split_patients.update(fold.get(split_type, []))

    print(f"      Found {len(valid_split_patients)} total valid patient IDs in splits.")

    print("[3/6] Loading bulk cohorts (CGGA & TCGA)...")
    cgga_adata = load_cgga_cohort()
    tcga_adata = load_tcga_cohort()
    print(f"      Loaded CGGA cohort: {cgga_adata.n_obs} samples x {cgga_adata.n_vars} genes.")
    print(f"      Loaded TCGA cohort: {tcga_adata.n_obs} samples x {tcga_adata.n_vars} genes.")

    # Standard clean state labels (no trailing hyphens)
    state_cols = ["MESlike1", "MESlike2", "AClike", "OPClike", "NPClike1", "NPClike2"]
    state_map = {
        "MESlike1": "MES", "MESlike2": "MES",
        "AClike": "AC", "OPClike": "OPC",
        "NPClike1": "NPC", "NPClike2": "NPC"
    }

    print("[4/6] Deriving cellular states and mapping patient metadata...")
    state_scores = adata.obs[state_cols].apply(pd.to_numeric, errors="coerce")

    has_valid_state = state_scores.notna().any(axis=1)
    max_cols = state_scores.fillna(-np.inf).idxmax(axis=1)

    adata.obs["derived_state"] = max_cols.map(state_map)
    adata.obs.loc[~has_valid_state, "derived_state"] = "Unknown"

    # Standardize column mappings required by baselines.py
    adata.obs["patient_id"] = adata.obs["Sample"].astype(str)
    adata.obs["state"] = adata.obs["derived_state"]

    # Strictly retain cells that have a valid state AND exist in patient_splits.json
    adata_valid = adata[
        (adata.obs["derived_state"] != "Unknown") &
        (adata.obs["patient_id"].isin(valid_split_patients))
        ].copy()

    print(f"      Retained {adata_valid.n_obs}/{adata.n_obs} cells with valid states and matching split IDs.")

    print("[5/6] Subsampling Neftel patients across derived states and GBMType...")
    patient_df = adata_valid.obs[["patient_id", "derived_state", "GBMType"]].drop_duplicates()

    sampled_patients = (
        patient_df.groupby(["derived_state", "GBMType"], group_keys=False)
        .apply(lambda x: x["patient_id"].sample(
            n=min(10, len(x["patient_id"])),
            random_state=42
        ))
        .tolist()
    )

    adata_pilot = adata_valid[adata_valid.obs["patient_id"].isin(sampled_patients)].copy()

    print("[6/6] Subsetting highly variable & mandatory target genes...")
    sc.pp.highly_variable_genes(adata_pilot, n_top_genes=2500)
    hvg_list = set(adata_pilot.var_names[adata_pilot.var["highly_variable"]])

    mandatory_genes = {"TP53", "IDH1", "EGFR", "RPRM"}
    final_genes = list(hvg_list.union(mandatory_genes.intersection(adata_pilot.var_names)))

    # Subset features on Neftel
    adata_pilot = adata_pilot[:, final_genes].copy()

    # Align CGGA & TCGA features to match the pilot gene set
    cgga_shared_genes = [g for g in final_genes if g in cgga_adata.var_names]
    cgga_pilot = cgga_adata[:, cgga_shared_genes].copy()

    tcga_shared_genes = [g for g in final_genes if g in tcga_adata.var_names]
    tcga_pilot = tcga_adata[:, tcga_shared_genes].copy()

    # Save all pilot objects
    out_neftel = "data/pilot/pilot_subsample.h5ad"
    out_cgga = "data/pilot/cgga_pilot_subsample.h5ad"
    out_tcga = "data/pilot/tcga_pilot_subsample.h5ad"

    adata_pilot.write_h5ad(out_neftel)
    cgga_pilot.write_h5ad(out_cgga)
    tcga_pilot.write_h5ad(out_tcga)

    print(f"\n[PASS] Neftel pilot saved to: {out_neftel} ({adata_pilot.n_obs} cells x {adata_pilot.n_vars} genes)")
    print(f"[PASS] CGGA pilot saved to:   {out_cgga} ({cgga_pilot.n_obs} samples x {cgga_pilot.n_vars} genes)")
    print(f"[PASS] TCGA pilot saved to:   {out_tcga} ({tcga_pilot.n_obs} samples x {tcga_pilot.n_vars} genes)")


if __name__ == "__main__":
    generate_pilot_subsample()