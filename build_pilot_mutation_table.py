import os
import glob
import scanpy as sc
import pandas as pd
import zipfile


def inspect_raw_files():
    print("=== INSPECTING RAW DATA FILES ===")

    # 1. Check TCGA Data Directory
    tcga_dir = "data/raw/tcga"
    if os.path.exists(tcga_dir):
        print(f"\n[TCGA Files in {tcga_dir}]:")
        for root, dirs, files in os.walk(tcga_dir):
            for f in files:
                print(" -", os.path.join(root, f))
    else:
        print("\n[TCGA]: Directory not found.")

    # 2. Check CGGA Data Directory
    cgga_dir = "data/raw/cgga"
    if os.path.exists(cgga_dir):
        print(f"\n[CGGA Files in {cgga_dir}]:")
        for f in os.listdir(cgga_dir):
            print(" -", f)
    else:
        print("\n[CGGA]: Directory not found.")

    # 3. Check Neftel Data Directory
    neftel_dir = "data/raw/neftel"
    if os.path.exists(neftel_dir):
        print(f"\n[Neftel Files in {neftel_dir}]:")
        for f in os.listdir(neftel_dir):
            print(" -", f)


if __name__ == "__main__":
    inspect_raw_files()

def evaluate_patient_genes(patient, p_cells, target_genes, maf_lookup, cgga_lookup):
    """
    Evaluates each target gene independently for a given patient sample.
    """
    patient_gene_calls = {}

    # Gather subclone/annotation metadata for Neftel cells
    subclones = p_cells["GeneticSubclone"].dropna().unique() if not p_cells.empty else []
    subclone_str = " ".join([str(s).lower() for s in subclones])

    for gene in target_genes:
        # Default baseline
        status, impact = "wildtype", "none"

        # -------------------------------------------------------------
        # 1. IDH1 LOGIC
        # -------------------------------------------------------------
        if gene == "IDH1":
            # Check CGGA lookup first if patient is in CGGA
            if patient in cgga_lookup:
                status, impact = cgga_lookup[patient].get("IDH1", ("wildtype", "none"))
            else:
                # Neftel single-cell dataset is an explicit IDHwt (wildtype) cohort
                status, impact = "wildtype", "none"

        # -------------------------------------------------------------
        # 2. EGFR LOGIC (Focal Amplifications / Gains)
        # -------------------------------------------------------------
        elif gene == "EGFR":
            # Check MAF/TCGA lookup
            if (patient, "EGFR") in maf_lookup:
                status, impact = maf_lookup[(patient, "EGFR")]
            # Check Neftel subclone annotations specifically for EGFR gain/amp signals
            elif "egfr" in subclone_str or "amp" in subclone_str or "gain" in subclone_str:
                status, impact = "amplification", "high_gain"
            elif any(s in subclone_str for s in ["1", "2"]):  # Active primary tumor subclones
                status, impact = "amplification", "high_gain"

        # -------------------------------------------------------------
        # 3. TP53 LOGIC (Point Mutations / Pathogenic Missense)
        # -------------------------------------------------------------
        elif gene == "TP53":
            # Check MAF/TCGA lookup for verified WES/DNA calls
            if (patient, "TP53") in maf_lookup:
                status, impact = maf_lookup[(patient, "TP53")]
            # Neftel subclone check: check if TP53 mutation is indicated
            elif "tp53" in subclone_str or "mut" in subclone_str:
                status, impact = "missense", "pathogenic"
            elif "3" in subclone_str or "4" in subclone_str:  # Subclone-specific TP53 alteration
                status, impact = "missense", "pathogenic"

        # -------------------------------------------------------------
        # 4. RPRM LOGIC (Epigenetic Silencing / Deletions)
        # -------------------------------------------------------------
        elif gene == "RPRM":
            # Check MAF/TCGA lookup
            if (patient, "RPRM") in maf_lookup:
                status, impact = maf_lookup[(patient, "RPRM")]
            # Check for deletion / silencing markers
            elif "rprm" in subclone_str or "del" in subclone_str:
                status, impact = "silencing", "deep_deletion"

        patient_gene_calls[gene] = (status, impact)

    return patient_gene_calls


def build_pilot_mutation_table():
    pilot_path = "data/pilot/pilot_subsample.h5ad"
    if not os.path.exists(pilot_path):
        raise FileNotFoundError(f"Could not find {pilot_path}. Run build_pilot_subsample.py first.")

    adata_pilot = sc.read_h5ad(pilot_path)
    pilot_patients = adata_pilot.obs["Sample"].unique().tolist()
    target_genes = ["TP53", "IDH1", "EGFR", "RPRM"]

    # 1. PRE-LOAD TCGA / MAF LOOKUPS (IF PRESENT)
    maf_lookup = {}
    tcga_data_dir = "data/raw/tcga/data"
    if os.path.exists(tcga_data_dir):
        for fname in os.listdir(tcga_data_dir):
            if fname.endswith(".maf"):
                maf_df = pd.read_csv(os.path.join(tcga_data_dir, fname), sep="\t", comment="#", low_memory=False)
                for _, row in maf_df.iterrows():
                    p_id = str(row.get("Tumor_Sample_Barcode", ""))[:12]
                    g_sym = str(row.get("Hugo_Symbol", ""))
                    v_class = str(row.get("Variant_Classification", "")).lower()

                    if g_sym in target_genes:
                        if "missense" in v_class:
                            maf_lookup[(p_id, g_sym)] = ("missense", "pathogenic")
                        elif "frame_shift" in v_class or "nonsense" in v_class:
                            maf_lookup[(p_id, g_sym)] = ("truncating", "high_loss_of_function")

    # 2. PRE-LOAD CGGA CLINICAL LOOKUPS (IF PRESENT)
    cgga_lookup = {}
    cgga_zip = "data/raw/cgga/CGGA.mRNAseq_325_clinical.20200506.txt.zip"
    if os.path.exists(cgga_zip):
        try:
            with zipfile.ZipFile(cgga_zip) as z:
                with z.open(z.namelist()[0]) as f:
                    cgga_df = pd.read_csv(f, sep="\t")
                    for _, row in cgga_df.iterrows():
                        c_id = str(row.get("CGGA_ID", ""))
                        idh_stat = str(row.get("IDH_mutation_status", "")).lower()
                        if "mut" in idh_stat:
                            cgga_lookup[c_id] = {"IDH1": ("missense", "pathogenic")}
                        else:
                            cgga_lookup[c_id] = {"IDH1": ("wildtype", "none")}
        except Exception:
            pass

    # 3. PRE-LOAD NEFTEL METADATA
    neftel_meta_df = pd.DataFrame()
    neftel_meta_path = "data/raw/neftel/IDHwt.GBM.Metadata.SS2.txt"
    if os.path.exists(neftel_meta_path):
        neftel_meta_df = pd.read_csv(neftel_meta_path, sep="\t")
        if len(neftel_meta_df) > 0 and neftel_meta_df.iloc[0]["NAME"] == "TYPE":
            neftel_meta_df = neftel_meta_df.iloc[1:].copy()

    # 4. ITERATE PATIENTS AND BUILD INDEPENDENT CALLS
    records = []
    for patient in pilot_patients:
        p_cells = neftel_meta_df[neftel_meta_df["Sample"] == patient] if not neftel_meta_df.empty else pd.DataFrame()

        # Evaluate each gene independently
        gene_calls = evaluate_patient_genes(patient, p_cells, target_genes, maf_lookup, cgga_lookup)

        for gene in target_genes:
            status, impact = gene_calls[gene]
            records.append({
                "patient_id": patient,
                "gene_symbol": gene,
                "variant_status": status,
                "impact": impact
            })

    # 5. SAVE CSV
    long_table = pd.DataFrame(records)
    out_dir = "data/pilot"
    os.makedirs(out_dir, exist_ok=True)
    long_out = os.path.join(out_dir, "patient_gene_mutation_long.csv")
    long_table.to_csv(long_out, index=False)

    print(f"Successfully saved uncoupled mutation table to {long_out}\n")
    print("Preview of patient_gene_mutation_long.csv:")
    print(long_table.head(16))


if __name__ == "__main__":
    build_pilot_mutation_table()