import numpy as np
import scanpy as sc
import pandas as pd

adata = sc.read_h5ad("data/processed/neftel_qc.h5ad")

# Some debugging
#print("Metadata columns:", adata.obs.columns.tolist())

#print("GBMType values:", adata.obs['GBMType'].value_counts())
#print("CellAssignment values:", adata.obs['CellAssignment'].value_counts())

# Derive states from Neftel score columns(found from the debugging)
state_cols = ["MESlike1", "MESlike2", "AClike", "OPClike", "NPClike1", "NPClike2"]
state_map = {
    "MESlike1": "MES-like", "MESlike2": "MES-like",
    "AClike": "AC-like", "OPClike": "OPC-like",
    "NPClike1": "NPC-like", "NPClike2": "NPC-like"
}

# Handle rows with all NaN/DNE/Unknown
state_scores = adata.obs[state_cols].apply(pd.to_numeric, errors="coerce")

# Find row max values to detect rows with no state data
has_valid_state = state_scores.notna().any(axis=1)

# Compute idxmax safely by filling NaN with a very low number (-inf)
max_cols = state_scores.fillna(-np.inf).idxmax(axis=1)

# Map to 4 primary states, assigning 'Unknown' where state scores were missing
adata.obs["derived_state"] = max_cols.map(state_map)
adata.obs.loc[~has_valid_state, "derived_state"] = "Unknown"

# Filter out 'Unknown' states if present so sampling only picks valid cell states
adata_valid = adata[adata.obs["derived_state"] != "Unknown"].copy()
# 5-10 paitents per subsample
sampled_patients = (
    adata.obs.groupby(["derived_state", "GBMType"], group_keys=False)
    .apply(lambda x: x["Sample"].drop_duplicates().sample(n=min(10, len(x["Sample"].unique())), random_state=42))
)
adata_pilot = adata[adata.obs["Sample"].isin(sampled_patients)].copy()

# Filter out 2k-3k most variable genes + TP53, IDH1, EGFR and RPRM requirements
sc.pp.highly_variable_genes(adata_pilot, n_top_genes=2500)
hvg_list = set(adata_pilot.var_names[adata_pilot.var["highly_variable"]])

mandatory_genes = {"TP53", "IDH1", "EGFR", "RPRM"}
final_genes = list(hvg_list.union(mandatory_genes.intersection(adata_pilot.var_names)))

# Save data
adata_pilot = adata_pilot[:, final_genes]
adata_pilot.write_h5ad("data/pilot/pilot_subsample.h5ad")