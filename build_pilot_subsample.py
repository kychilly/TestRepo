import os
import scanpy as sc
import pandas as pd
import numpy as np

os.makedirs("data/pilot", exist_ok=True)

# 1. Load Preprocessed Data
adata = sc.read_h5ad("data/processed/neftel_qc.h5ad")

# Filter to malignant cells if present
if "CellAssignment" in adata.obs.columns:
    adata = adata[adata.obs["CellAssignment"] == "Malignant"].copy()

# 2. Derive 4 Neftel States from Continuous Score Columns
state_cols = ["MESlike1", "MESlike2", "AClike", "OPClike", "NPClike1", "NPClike2"]

# Find max scoring state column per cell
max_state_col = adata.obs[state_cols].idxmax(axis=1)

# Collapse sub-states into the 4 primary states
state_map = {
    "MESlike1": "MES-like",
    "MESlike2": "MES-like",
    "AClike": "AC-like",
    "OPClike": "OPC-like",
    "NPClike1": "NPC-like",
    "NPClike2": "NPC-like"
}
adata.obs["derived_state"] = max_state_col.map(state_map)

# 3. Handle IDH Status
# Check if an IDH/Type column exists or derive from Sample name
if "IDH" in adata.obs.columns:
    adata.obs["idh_group"] = adata.obs["IDH"]
elif "IDH_status" in adata.obs.columns:
    adata.obs["idh_group"] = adata.obs["IDH_status"]
else:
    # Fallback: Group by Sample name and GBMType if explicit IDH status isn't annotated
    adata.obs["idh_group"] = adata.obs["GBMType"]

# 4. Stratified Subsampling (5-10 patients per group)
patient_df = adata.obs[["derived_state", "idh_group", "Sample"]].drop_duplicates()

sampled_patients = (
    patient_df.groupby(["derived_state", "idh_group"], group_keys=False)
    .apply(lambda g: g["Sample"].sample(n=min(10, len(g)), random_state=42))
    .unique()
)

# Filter cells to sampled patients
adata_pilot = adata[adata.obs["Sample"].isin(sampled_patients)].copy()

# 5. HVG Selection (~2500)
sc.pp.highly_variable_genes(adata_pilot, n_top_genes=2500)
hvg_list = set(adata_pilot.var_names[adata_pilot.var["highly_variable"]])

# 6. Force-Include Required Driver Genes
mandatory_genes = {"TP53", "IDH1", "EGFR", "RPRM"}
final_genes = list(hvg_list.union(mandatory_genes.intersection(adata_pilot.var_names)))

# Check which driver genes were found
found_mandatory = mandatory_genes.intersection(adata_pilot.var_names)
print(f"Driver genes found in dataset: {found_mandatory}")

# 7. Subset and Save
adata_pilot = adata_pilot[:, final_genes].copy()
adata_pilot.write_h5ad("data/pilot/pilot_subsample.h5ad")

print(f"Done! Saved pilot dataset to data/pilot/pilot_subsample.h5ad with {adata_pilot.n_obs} cells and {adata_pilot.n_vars} genes.")