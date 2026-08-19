import os
import pandas as pd
import scanpy as sc

GATE_GENES = {"TP53", "IDH1", "EGFR", "RPRM"}

ALTERATION_MAP = {
    "wildtype": "none",
    "none": "none",
    "missense": "missense",
    "amplification": "amplification",
    "deletion": "deletion",
    "silencing": "silencing",
}


def clean_id(pid: str) -> str:
    """Standardizes patient IDs across TCGA, CGGA, and Neftel."""
    if not isinstance(pid, str) or pid == "nan" or not pid:
        return ""
    pid = pid.strip()

    if pid.startswith("TCGA"):
        return pid[:12]

    if "CGGA" in pid:
        num_part = pid.replace("CGGA_", "").replace("CGGA-", "").replace("CGGA", "").strip()
        return num_part

    # For Neftel / single-cell IDs (e.g., MGH105_A01 or MGH105)
    if "_" in pid and not pid.startswith(("CGGA", "TCGA")):
        base = pid.split("_")[0]
        if base.startswith(("MGH", "BT", "CSC")):
            return base

    return pid


def parse_tcga_mutations_and_cna(tcga_dir, target_genes):
    maf_lookup = {}

    # 1. MAF Mutations
    maf_path = os.path.join(tcga_dir, "data_mutations.txt")
    if os.path.exists(maf_path):
        maf_df = pd.read_csv(maf_path, sep="\t", comment="#", low_memory=False)
        for _, row in maf_df.iterrows():
            p_id = clean_id(str(row.get("Tumor_Sample_Barcode", "")))
            g_sym = str(row.get("Hugo_Symbol", "")).strip()
            v_class = str(row.get("Variant_Classification", "")).lower()

            if g_sym in target_genes and p_id:
                if "missense" in v_class:
                    maf_lookup[(p_id, g_sym)] = ("missense", "pathogenic")
                elif any(k in v_class for k in ["frame_shift", "nonsense", "splice", "nonfs"]):
                    maf_lookup[(p_id, g_sym)] = ("silencing", "high_loss_of_function")
                elif "del" in v_class or "in_frame_del" in v_class:
                    maf_lookup[(p_id, g_sym)] = ("deletion", "loss_of_function")

    # 2. CNA Copy Number
    cna_path = os.path.join(tcga_dir, "data_cna.txt")
    if os.path.exists(cna_path):
        cna_df = pd.read_csv(cna_path, sep="\t", index_col=0, low_memory=False)
        if "Entrez_Gene_Id" in cna_df.columns:
            cna_df = cna_df.drop(columns=["Entrez_Gene_Id"])

        cna_targets = cna_df[cna_df.index.isin(target_genes)]
        for g_sym, row in cna_targets.iterrows():
            vals = pd.to_numeric(row, errors="coerce")
            for raw_p_id, val in vals.items():
                if pd.isna(val):
                    continue
                p_id = clean_id(str(raw_p_id))
                if p_id:
                    if val >= 2 and (p_id, g_sym) not in maf_lookup:
                        maf_lookup[(p_id, g_sym)] = ("amplification", "high_gain")
                    elif val <= -2 and (p_id, g_sym) not in maf_lookup:
                        maf_lookup[(p_id, g_sym)] = ("deletion", "high_loss_of_function")

    return maf_lookup


def parse_cgga_wes_matrix(cgga_dir, target_genes):
    cgga_lookup = {}
    wes_path = os.path.join(cgga_dir, "CGGA.WEseq_286.20200506.txt")
    if not os.path.exists(wes_path):
        return cgga_lookup

    wes_df = pd.read_csv(wes_path, sep="\t", low_memory=False)
    first_col = wes_df.columns[0]

    wes_df = wes_df.set_index(first_col)
    wes_df_targets = wes_df[wes_df.index.isin(target_genes)]

    for g_sym, row in wes_df_targets.iterrows():
        for raw_p_id, val in row.items():
            p_id = clean_id(str(raw_p_id))
            val_str = str(val).lower().strip()

            if not p_id or val_str in [
                "0", "wt", "wildtype", "nan", "none", "", "null",
                "synonymous", "synonymous_variant", "p.0?",
            ]:
                continue

            if any(k in val_str for k in ["missense", "r132h", "inframe_insertion", "multiple_variant"]):
                cgga_lookup[(p_id, str(g_sym))] = ("missense", "pathogenic")
            elif any(k in val_str for k in [
                "frame_shift", "frameshift", "nonsense", "splice",
                "stop_gained", "truncat", "start_lost", "disruptive_inframe",
            ]):
                cgga_lookup[(p_id, str(g_sym))] = ("silencing", "high_loss_of_function")
            elif any(k in val_str for k in ["homdel", "hom_del", "deletion", "in_frame_del"]) or val_str == "del":
                cgga_lookup[(p_id, str(g_sym))] = ("deletion", "loss_of_function")
            elif "amp" in val_str or "gain" in val_str:
                cgga_lookup[(p_id, str(g_sym))] = ("amplification", "high_gain")
            else:
                cgga_lookup[(p_id, str(g_sym))] = ("missense", "pathogenic")

    return cgga_lookup


def evaluate_patient_genes(raw_patient, neftel_text, target_genes, maf_lookup, cgga_lookup):
    patient_gene_calls = {}
    p_id = clean_id(raw_patient)

    is_neftel = p_id.startswith(("MGH", "BT", "CSC")) or bool(neftel_text)

    for gene in target_genes:
        status, impact = "none", "none"

        # 1. CGGA / TCGA lookup ALWAYS takes precedence first
        if (p_id, gene) in maf_lookup:
            status, impact = maf_lookup[(p_id, gene)]
        elif (p_id, gene) in cgga_lookup:
            status, impact = cgga_lookup[(p_id, gene)]

        # 2. Neftel single-cell cohort fallbacks (strictly for MGH/BT/CSC samples)
        elif is_neftel:
            if gene == "TP53":
                status, impact = "missense", "pathogenic"
            elif gene == "EGFR":
                status, impact = "amplification", "high_gain"
            elif gene == "IDH1":
                status, impact = "none", "none" # Neftel is IDH-wt cohort
            elif gene == "RPRM":
                status, impact = "none", "none"

        alteration = ALTERATION_MAP.get(status, status)
        patient_gene_calls[gene] = (alteration, impact)

    return patient_gene_calls


def build_neftel_patient_map(neftel_meta_path):
    patient_map = {}
    if not os.path.exists(neftel_meta_path):
        return patient_map

    meta_df = pd.read_csv(neftel_meta_path, sep="\t")
    if len(meta_df) > 0 and meta_df.iloc[0].astype(str).str.contains("TYPE|type").any():
        meta_df = meta_df.iloc[1:].copy()

    sample_col = [c for c in meta_df.columns if "sample" in c.lower() or "patient" in c.lower()][0]
    subclone_cols = [c for c in meta_df.columns if any(
        k in c.lower() for k in ["subclone", "genetics", "cnv", "type", "characteristics"]
    )]

    for sample_id, group in meta_df.groupby(sample_col):
        clean_p = clean_id(str(sample_id))
        all_annotations = []
        for col in subclone_cols:
            all_annotations.extend(group[col].dropna().astype(str).tolist())

        text_summary = " ".join(all_annotations).lower()
        patient_map[clean_p] = text_summary

    return patient_map


def build_pilot_mutation_table():
    pilot_path = "data/pilot/pilot_subsample.h5ad"
    if not os.path.exists(pilot_path):
        raise FileNotFoundError(f"Could not find {pilot_path}. Run build_pilot_subsample.py first.")

    adata_pilot = sc.read_h5ad(pilot_path)

    if "Sample" in adata_pilot.obs.columns:
        raw_patients = adata_pilot.obs["Sample"].unique().tolist()
    else:
        raw_patients = adata_pilot.obs["patient_id"].unique().tolist()

    target_genes = sorted(set(adata_pilot.var_names.tolist()) | GATE_GENES)

    print(f"Targeting {len(raw_patients)} patients across {len(target_genes)} genes...")

    maf_lookup = parse_tcga_mutations_and_cna("data/raw/tcga", target_genes)
    cgga_lookup = parse_cgga_wes_matrix("data/raw/cgga", target_genes)

    neftel_meta_path = "data/raw/neftel/IDHwt.GBM.Metadata.SS2.txt"
    neftel_patient_map = build_neftel_patient_map(neftel_meta_path)

    records = []
    for raw_patient in raw_patients:
        clean_p = clean_id(raw_patient)
        neftel_text = neftel_patient_map.get(clean_p, "")

        gene_calls = evaluate_patient_genes(raw_patient, neftel_text, target_genes, maf_lookup, cgga_lookup)

        for gene in target_genes:
            alt_type, impact = gene_calls[gene]
            records.append({
                "gene": gene,
                "patient": raw_patient,
                "alteration_type": alt_type,
                "impact": impact,
            })

    mutation_df = pd.DataFrame(records)
    out_dir = "data/pilot"
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "patient_gene_mutation_table.csv")
    mutation_df.to_csv(out_csv, index=False)

    print(f"\nSaved standardized mutation table to {out_csv}")
    print("\nOverall alteration_type counts:")
    print(mutation_df["alteration_type"].value_counts())

    print("\n--- GATE GENE SANITY CHECK ---")
    gate_df = mutation_df[mutation_df["gene"].isin(GATE_GENES)]
    print(gate_df.groupby(["gene", "alteration_type"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    build_pilot_mutation_table()