import glob
import os
import zipfile

import anndata as ad
import pandas as pd
import scanpy as sc
import yaml

from build_pilot_mutation_table import clean_id  # single source of truth for ID cleaning

GATE_GENES = {"TP53", "IDH1", "EGFR", "RPRM"}


# ---------------------------------------------------------------------------
# QC config
# ---------------------------------------------------------------------------

def load_qc_config(config_path="config/qc.yaml"):
    """Load frozen QC parameters from YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"QC configuration file not found at {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"[QC] Loaded frozen parameters from {config_path}")
    return config


# ---------------------------------------------------------------------------
# Per-matrix QC + normalization
# ---------------------------------------------------------------------------

def process_single_matrix(adata, qc_cfg, cohort_name):
    """Applies frozen cell QC, gene QC, CP10K + log1p normalization on an AnnData object."""
    print(f"\n--- Processing Cohort: {cohort_name} ---")
    print(f"[Initial shape] Cells: {adata.n_obs}, Genes: {adata.n_vars}")

    adata.var["mito"] = adata.var_names.str.startswith(("MT-", "mt-"))
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mito"], percent_top=None, log1p=False, inplace=True)

    min_genes = qc_cfg["cell_qc"]["min_genes_per_cell"]
    max_mito = qc_cfg["cell_qc"]["max_pct_mito"]

    sc.pp.filter_cells(adata, min_genes=min_genes)
    if "pct_counts_mito" in adata.obs:
        adata = adata[adata.obs["pct_counts_mito"] <= max_mito, :].copy()

    min_cells = qc_cfg["gene_qc"]["min_cells_expressing"]
    sc.pp.filter_genes(adata, min_cells=min_cells)

    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    target_sum = qc_cfg["normalization"]["target_sum"]
    sc.pp.normalize_total(adata, target_sum=target_sum)

    if qc_cfg["normalization"]["log_transform"]:
        sc.pp.log1p(adata)

    adata.obs["Cohort"] = cohort_name
    print(f"[Filtered & Normalized shape] Cells: {adata.n_obs}, Genes: {adata.n_vars}")
    return adata


# ---------------------------------------------------------------------------
# Neftel (single-cell) loader
# ---------------------------------------------------------------------------

def load_neftel(data_dir="data/raw/neftel", processed_dir="data/processed"):
    """Loads Neftel dataset from preprocessed h5ad or processes raw logTPM.txt.gz."""
    preprocessed_path = os.path.join(processed_dir, "neftel_qc.h5ad")
    if os.path.exists(preprocessed_path):
        print(f"[Loading] Found existing Neftel AnnData at {preprocessed_path}")
        adata = sc.read_h5ad(preprocessed_path)
    else:
        expr_path = os.path.join(data_dir, "IDHwtGBM.processed.SS2.logTPM.txt.gz")
        meta_path = os.path.join(data_dir, "IDHwt.GBM.Metadata.SS2.txt")

        if not os.path.exists(expr_path):
            raise FileNotFoundError(f"Neftel expression file not found at {expr_path}")

        print(f"[Loading] Parsing raw Neftel matrix: {expr_path}...")
        df_expr = pd.read_csv(expr_path, sep="\t", compression="gzip", index_col=0)

        if df_expr.shape[0] > df_expr.shape[1]:
            df_expr = df_expr.T

        adata = ad.AnnData(
            X=df_expr.values,
            obs=pd.DataFrame(index=df_expr.index),
            var=pd.DataFrame(index=df_expr.columns),
        )

        if os.path.exists(meta_path):
            df_meta = pd.read_csv(meta_path, sep="\t", index_col=0)

            # FIX: drop the leading "TYPE" annotation row (values like
            # "group"/"numeric") before joining -- otherwise every numeric
            # score column (MESlike1/2, AClike, OPClike, NPClike1/2, G1S,
            # G2M) gets silently coerced to object/string dtype cohort-wide.
            if "TYPE" in df_meta.index:
                df_meta = df_meta.drop(index="TYPE")

            numeric_cols = [
                "GenesExpressed", "MESlike2", "MESlike1", "AClike", "OPClike",
                "NPClike1", "NPClike2", "G1S", "G2M",
            ]
            for col in numeric_cols:
                if col in df_meta.columns:
                    df_meta[col] = pd.to_numeric(df_meta[col], errors="coerce")

            adata.obs = adata.obs.join(df_meta, how="left")

    if "Sample" not in adata.obs.columns:
        if "patient_id" in adata.obs.columns:
            adata.obs["Sample"] = adata.obs["patient_id"]
        else:
            adata.obs["Sample"] = adata.obs.index.map(lambda x: str(x).split("_")[0].split("-")[0])

    return adata


# ---------------------------------------------------------------------------
# CGGA (bulk RSEM) loader
# ---------------------------------------------------------------------------

def _find_cgga_zip(cgga_dir, tokens):
    """Matches CGGA zip filenames regardless of dot vs. underscore naming
    convention -- your uploads use underscores throughout
    (CGGA_mRNAseq_325_RSEM-genes_20200506_txt.zip), the CGGA site's own
    convention uses dots (CGGA.mRNAseq_325.RSEM-genes.20200506.txt.zip)."""
    for c in glob.glob(os.path.join(cgga_dir, "*.zip")):
        base = os.path.basename(c).lower()
        if all(tok.lower() in base for tok in tokens):
            return c
    return None


def _read_first_real_member(zpath):
    """Reads the first zip member that isn't __MACOSX junk (namelist()[0]
    order isn't guaranteed to put the real file first on every platform)."""
    with zipfile.ZipFile(zpath, "r") as z:
        real_members = [
            n for n in z.namelist()
            if not n.startswith("__MACOSX")
            and not os.path.basename(n).startswith("._")
            and n.lower().endswith(".txt")
        ]
        if not real_members:
            raise FileNotFoundError(f"No usable .txt member found in {zpath}")
        with z.open(real_members[0]) as f:
            return pd.read_csv(f, sep="\t", index_col=0, low_memory=False)


def load_cgga(cgga_dir="data/raw/cgga"):
    """Parses zipped CGGA RSEM mRNA-seq matrices (325 and 693 cohorts)."""
    cgga_adatas = []
    cohort_specs = [("325", ["325", "rsem"]), ("693", ["693", "rsem"])]

    for cohort_id, tokens in cohort_specs:
        zpath = _find_cgga_zip(cgga_dir, tokens)
        if zpath is None:
            print(f"[Warning] Could not find a CGGA {cohort_id} RSEM zip in {cgga_dir} "
                  f"(looked for files containing {tokens}).")
            continue

        print(f"[Loading] Unzipping and reading {os.path.basename(zpath)}...")
        df = _read_first_real_member(zpath)

        if df.shape[0] > df.shape[1]:
            df = df.T

        adata_part = ad.AnnData(
            X=df.values,
            obs=pd.DataFrame(index=df.index),
            var=pd.DataFrame(index=df.columns),
        )
        adata_part.obs["Sample"] = adata_part.obs.index
        adata_part.obs["CGGA_cohort_size"] = cohort_id
        cgga_adatas.append(adata_part)

    if not cgga_adatas:
        return None

    print("[Merging] Combining CGGA 325 and 693 cohorts...")
    return ad.concat(cgga_adatas, merge="unique", join="inner")


# ---------------------------------------------------------------------------
# Clinical metadata join (new -- did not exist in the original pipeline)
# ---------------------------------------------------------------------------

def _load_cgga_clinical(cgga_dir):
    frames = []
    for tokens in (["325", "clinical"], ["693", "clinical"]):
        zpath = _find_cgga_zip(cgga_dir, tokens)
        if zpath is None:
            continue
        df = _read_first_real_member(zpath).reset_index()
        id_col = df.columns[0]  # CGGA_ID
        df["clean_id"] = df[id_col].astype(str).apply(clean_id)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    clinical = pd.concat(frames, ignore_index=True).drop_duplicates("clean_id")
    clinical = clinical.set_index("clean_id")
    clinical = clinical.rename(columns={"IDH_mutation_status": "IDH_status"})
    return clinical[["IDH_status"]] if "IDH_status" in clinical.columns else pd.DataFrame()


def _load_tcga_clinical(tcga_dir):
    """data_clinical_sample.txt has its real header on line 5 (four leading
    cBioPortal '#' metadata lines); the IDH column is IDH_STATUS (all caps),
    not IDH_status/IDH1_status."""
    path = os.path.join(tcga_dir, "data_clinical_sample.txt")
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
    df["clean_id"] = df["SAMPLE_ID"].astype(str).apply(clean_id)
    df = df.drop_duplicates("clean_id").set_index("clean_id")
    df = df.rename(columns={"IDH_STATUS": "IDH_status", "TRANSCRIPTOME_SUBTYPE": "Bulk_subtype"})
    keep = [c for c in ["IDH_status", "Bulk_subtype"] if c in df.columns]
    return df[keep]


def merge_clinical_metadata(adata_full, raw_dir="data/raw"):
    """Joins CGGA + TCGA clinical metadata into adata_full.obs on cleaned
    patient ID, producing one standardized 'IDH_status' column
    ('WT'/'Mutant'/NaN). Call after ad.concat(), before HVG selection."""
    cgga_clin = _load_cgga_clinical(os.path.join(raw_dir, "cgga"))
    tcga_clin = _load_tcga_clinical(os.path.join(raw_dir, "tcga"))

    clinical = pd.concat([cgga_clin, tcga_clin])
    clinical = clinical[~clinical.index.duplicated(keep="first")]

    adata_full.obs["clean_id"] = adata_full.obs["Sample"].astype(str).apply(clean_id)
    adata_full.obs = adata_full.obs.join(clinical, on="clean_id")

    if "IDH_status" in adata_full.obs.columns:
        adata_full.obs["IDH_status"] = adata_full.obs["IDH_status"].replace(
            {"Wildtype": "WT", "wildtype": "WT"}
        )
        n_missing = adata_full.obs["IDH_status"].isna().sum()
        print(f"[Clinical Join] IDH_status resolved for "
              f"{adata_full.n_obs - n_missing}/{adata_full.n_obs} samples "
              f"({n_missing} unmatched).")
    else:
        print("[Warning] No IDH_status could be joined -- check that raw_dir/cgga "
              "and raw_dir/tcga clinical files are present.")

    return adata_full


# ---------------------------------------------------------------------------
# Full cohort assembly
# ---------------------------------------------------------------------------

def build_full_cohort(
        raw_dir="data/raw", output_dir="data/processed", config_path="config/qc.yaml"
):
    qc_cfg = load_qc_config(config_path)
    adatas = []

    try:
        adata_neftel = load_neftel(os.path.join(raw_dir, "neftel"), output_dir)
        adata_neftel = process_single_matrix(adata_neftel, qc_cfg, "Neftel")
        adatas.append(adata_neftel)
    except Exception as e:
        print(f"[Warning] Could not load Neftel: {e}")

    try:
        adata_cgga = load_cgga(os.path.join(raw_dir, "cgga"))
        if adata_cgga is not None:
            adata_cgga = process_single_matrix(adata_cgga, qc_cfg, "CGGA")
            adatas.append(adata_cgga)
    except Exception as e:
        print(f"[Warning] Could not load CGGA: {e}")

    if not adatas:
        raise FileNotFoundError(f"No valid cohort matrices could be built from {raw_dir}.")

    print("\n[Merging] Combining all datasets into full cohort...")
    adata_full = ad.concat(adatas, merge="unique", join="inner")
    adata_full.obs_names_make_unique()

    # NEW: join clinical metadata (IDH status, TCGA bulk subtype) BEFORE HVG
    # selection so downstream pilot stratification has something to work with.
    adata_full = merge_clinical_metadata(adata_full, raw_dir)

    n_top_genes = qc_cfg["hvg_selection"]["n_top_genes"]
    flavor = qc_cfg["hvg_selection"]["flavor"]
    print(f"\n[HVG Selection] Identifying top {n_top_genes} HVGs (flavor='{flavor}')...")
    sc.pp.highly_variable_genes(adata_full, n_top_genes=n_top_genes, flavor=flavor)

    for gene in GATE_GENES:
        if gene in adata_full.var_names:
            adata_full.var.loc[gene, "highly_variable"] = True

    os.makedirs(output_dir, exist_ok=True)
    full_out_path = os.path.join(output_dir, "full_cohort.h5ad")
    adata_full.write(full_out_path)

    print(f"\n[Saved] Successfully compiled full dataset to: {full_out_path}")
    print(f"Total cells/samples: {adata_full.n_obs}, Total genes: {adata_full.n_vars}")
    print(f"Cohort Breakdown:\n{adata_full.obs['Cohort'].value_counts()}")


if __name__ == "__main__":
    build_full_cohort()