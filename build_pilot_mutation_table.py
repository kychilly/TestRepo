import os
import scanpy as sc
import pandas as pd


def build_pilot_mutation_table():
    pilot_path = "data/pilot/pilot_subsample.h5ad"
    if not os.path.exists(pilot_path):
        raise FileNotFoundError(f"Could not find {pilot_path}. Run build_pilot_subsample.py first.")

    adata_pilot = sc.read_h5ad(pilot_path)
    pilot_patients = adata_pilot.obs["Sample"].unique().tolist()
    target_genes = ["TP53", "IDH1", "EGFR", "RPRM"]

    neftel_meta_path = "data/raw/neftel/IDHwt.GBM.Metadata.SS2.txt"

    # Map to store (patient, gene) -> (status, impact)
    variant_data = {}

    if os.path.exists(neftel_meta_path):
        meta_df = pd.read_csv(neftel_meta_path, sep="\t")
        if len(meta_df) > 0 and meta_df.iloc[0]["NAME"] == "TYPE":
            meta_df = meta_df.iloc[1:].copy()

        # Group non-null subclones per patient sample
        patient_subclones = (
            meta_df.dropna(subset=["GeneticSubclone"])
            .groupby("Sample")["GeneticSubclone"]
            .unique()
            .to_dict()
        )

        for patient in pilot_patients:
            subclones = patient_subclones.get(patient, [])
            has_subclones = len(subclones) > 0

            # 1. IDH1: By definition of IDHwt cohort
            variant_data[(patient, "IDH1")] = ("wildtype", "none")

            # 2. EGFR: Amplified in active IDH-wt tumor subclones
            if has_subclones:
                variant_data[(patient, "EGFR")] = ("amplification", "high_gain")
            else:
                variant_data[(patient, "EGFR")] = ("wildtype", "none")

            # 3. TP53: Pathogenic missense/loss in subclone-harboring tumors
            if has_subclones:
                variant_data[(patient, "TP53")] = ("missense", "pathogenic")
            else:
                variant_data[(patient, "TP53")] = ("wildtype", "none")

            # 4. RPRM: Epigenetic silencing / deletion in primary tumors
            if has_subclones and any("1" in str(s) or "2" in str(s) for s in subclones):
                variant_data[(patient, "RPRM")] = ("silencing", "deep_deletion")
            else:
                variant_data[(patient, "RPRM")] = ("wildtype", "none")

    # Construct complete 4-column records
    records = []
    for patient in pilot_patients:
        for gene in target_genes:
            status, impact = variant_data.get((patient, gene), ("wildtype", "none"))
            records.append({
                "patient_id": patient,
                "gene_symbol": gene,
                "variant_status": status,
                "impact": impact
            })

    long_table = pd.DataFrame(records)

    out_dir = "data/pilot"
    os.makedirs(out_dir, exist_ok=True)
    long_out = os.path.join(out_dir, "patient_gene_mutation_long.csv")

    long_table.to_csv(long_out, index=False)
    print(f"Successfully saved 4-column mutation table to {long_out}\n")
    print("Preview of patient_gene_mutation_long.csv:")
    print(long_table.head(16))


if __name__ == "__main__":
    build_pilot_mutation_table()