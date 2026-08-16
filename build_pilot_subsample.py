import numpy as np
import pandas as pd
import scanpy as sc
from load_cgga_cohort import load_cgga_cohort


def generate_pilot_subsample():
    print("[1/5] Loading Neftel single-cell dataset...")
    adata = sc.read_h5ad("data/processed/neftel_qc.h5ad")

    print("[2/5] Loading CGGA bulk dataset...")
    cgga_adata = load_cgga_cohort()
    print(f"      Loaded CGGA cohort: {cgga_adata.n_obs} samples x {cgga_adata.n_vars} genes.")

    state_cols = ["MESlike1", "MESlike2", "AClike", "OPClike", "NPClike1", "NPClike2"]
    state_map = {
        "MESlike1": "MES-like", "MESlike2": "MES-like",
        "AClike": "AC-like", "OPClike": "OPC-like",
        "NPClike1": "NPC-like", "NPClike2": "NPC-like"
    }

    print("[3/5] Deriving cellular states from Neftel module scores...")
    state_scores = adata.obs[state_cols].apply(pd.to_numeric, errors="coerce")

    has_valid_state = state_scores.notna().any(axis=1)
    max_cols = state_scores.fillna(-np.inf).idxmax(axis=1)

    adata.obs["derived_state"] = max_cols.map(state_map)
    adata.obs.loc[~has_valid_state, "derived_state"] = "Unknown"

    # Filter out Unknown states from Neftel for sampling
    adata_valid = adata[adata.obs["derived_state"] != "Unknown"].copy()
    print(f"      Retained {adata_valid.n_obs}/{adata.n_obs} Neftel cells with valid states.")

    print("[4/5] Subsampling Neftel patients across derived states and GBMType...")
    sampled_patients = (
        adata_valid.obs.groupby(["derived_state", "GBMType"], group_keys=False)
        .apply(lambda x: x["Sample"].drop_duplicates().sample(
            n=min(10, len(x["Sample"].unique())),
            random_state=42
        ))
    )

    adata_pilot = adata_valid[adata_valid.obs["Sample"].isin(sampled_patients)].copy()

    print("[5/5] Subsetting highly variable & mandatory target genes...")
    sc.pp.highly_variable_genes(adata_pilot, n_top_genes=2500)
    hvg_list = set(adata_pilot.var_names[adata_pilot.var["highly_variable"]])

    mandatory_genes = {"TP53", "IDH1", "EGFR", "RPRM"}
    final_genes = list(hvg_list.union(mandatory_genes.intersection(adata_pilot.var_names)))

    # Subset features on Neftel
    adata_pilot = adata_pilot[:, final_genes].copy()

    # Align CGGA features to match the pilot gene set
    cgga_shared_genes = [g for g in final_genes if g in cgga_adata.var_names]
    cgga_pilot = cgga_adata[:, cgga_shared_genes].copy()

    # Save both pilot objects
    out_neftel = "data/pilot/pilot_subsample.h5ad"
    out_cgga = "data/pilot/cgga_pilot_subsample.h5ad"

    adata_pilot.write_h5ad(out_neftel)
    cgga_pilot.write_h5ad(out_cgga)

    print(f"\n[PASS] Neftel pilot saved to: {out_neftel} ({adata_pilot.n_obs} cells x {adata_pilot.n_vars} genes)")
    print(f"[PASS] CGGA pilot saved to:   {out_cgga} ({cgga_pilot.n_obs} samples x {cgga_pilot.n_vars} genes)")


if __name__ == "__main__":
    generate_pilot_subsample()