import os

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
# TCGA (bulk RSEM) loader
# ---------------------------------------------------------------------------

def load_tcga(tcga_dir="data/raw/tcga"):
    """Loads TCGA PanCancer Atlas RSEM expression matrix (data_mrna_seq_v2_rsem.txt).

    cBioPortal's RSEM export has genes as rows (Hugo_Symbol, Entrez_Gene_Id,
    then one column per sample barcode) -- opposite orientation from Neftel's
    raw matrix, so this transposes explicitly rather than relying on the
    shape-guess used elsewhere. A small number of rows have no Hugo_Symbol
    (unnamed/uncharacterized loci) or a duplicate symbol across different
    Entrez IDs; both are dropped since var_names must be unique and
    symbol-matchable against Neftel.
    """
    expr_path = os.path.join(tcga_dir, "data_mrna_seq_v2_rsem.txt")
    if not os.path.exists(expr_path):
        raise FileNotFoundError(f"TCGA expression file not found at {expr_path}")

    print(f"[Loading] Parsing raw TCGA RSEM matrix: {expr_path}...")
    df = pd.read_csv(expr_path, sep="\t")

    n_before = len(df)
    df = df[df["Hugo_Symbol"].notna()].copy()
    df = df.drop_duplicates(subset="Hugo_Symbol", keep="first")
    print(f"[TCGA] Dropped {n_before - len(df)} rows with missing/duplicate Hugo_Symbol "
          f"({len(df)} genes retained).")

    df = df.set_index("Hugo_Symbol").drop(columns=["Entrez_Gene_Id"])
    df = df.T  # genes are rows in the raw file -- transpose to samples x genes

    adata = ad.AnnData(
        X=df.values,
        obs=pd.DataFrame(index=df.index),
        var=pd.DataFrame(index=df.columns),
    )
    adata.obs["Sample"] = adata.obs.index.astype(str)

    return adata


# ---------------------------------------------------------------------------
# Clinical metadata join (TCGA only -- CGGA excluded from the internal cohort)
# ---------------------------------------------------------------------------

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
    """Joins TCGA clinical metadata into adata_full.obs on cleaned patient ID,
    producing one standardized 'IDH_status' column ('WT'/'Mutant'/NaN).
    Call after ad.concat(), before HVG selection.
    (CGGA clinical join intentionally omitted -- CGGA is not part of the
    internal cohort this function operates on.)
    """
    tcga_clin = _load_tcga_clinical(os.path.join(raw_dir, "tcga"))

    adata_full.obs["clean_id"] = adata_full.obs["Sample"].astype(str).apply(clean_id)
    adata_full.obs = adata_full.obs.join(tcga_clin, on="clean_id")

    if "IDH_status" in adata_full.obs.columns:
        adata_full.obs["IDH_status"] = adata_full.obs["IDH_status"].replace(
            {"Wildtype": "WT", "wildtype": "WT"}
        )
        n_missing = adata_full.obs["IDH_status"].isna().sum()
        print(f"[Clinical Join] IDH_status resolved for "
              f"{adata_full.n_obs - n_missing}/{adata_full.n_obs} samples "
              f"({n_missing} unmatched -- expected for all Neftel cells, "
              f"which have their own IDH annotation elsewhere).")
    else:
        print("[Warning] No IDH_status could be joined -- check that raw_dir/tcga "
              "clinical file is present.")

    return adata_full


# ---------------------------------------------------------------------------
# Full internal cohort assembly (Neftel + TCGA only -- CGGA excluded)
# ---------------------------------------------------------------------------

def build_full_cohort(
        raw_dir="data/raw", output_dir="data/processed", config_path="config/qc.yaml"
):
    """Builds the full INTERNAL cohort: Neftel + TCGA only.

    CGGA is deliberately excluded here. Per Week 1 ("CGGA held out entirely")
    and Week 4 ("Preprocess CGGA with the frozen Week 1 thresholds... the
    whole external comparison dies if we touch this"), CGGA's expression
    data must remain completely unprocessed until its own dedicated Week 4
    step, where it serves as the untouched external validation cohort.
    """
    qc_cfg = load_qc_config(config_path)
    adatas = []
    neftel_states = None

    try:
        adata_neftel = load_neftel(os.path.join(raw_dir, "neftel"), output_dir)
        # Capture state BEFORE process_single_matrix/concat: TCGA has no
        # equivalent per-cell state column, and ad.concat(join="inner")
        # silently drops any obs column not common to every input cohort.
        if "state" in adata_neftel.obs.columns:
            neftel_states = adata_neftel.obs[["state"]].copy()
            neftel_states.index = adata_neftel.obs.index
        adata_neftel = process_single_matrix(adata_neftel, qc_cfg, "Neftel")
        adatas.append(adata_neftel)
    except Exception as e:
        print(f"[Warning] Could not load Neftel: {e}")

    try:
        adata_tcga = load_tcga(os.path.join(raw_dir, "tcga"))
        adata_tcga = process_single_matrix(adata_tcga, qc_cfg, "TCGA")
        adatas.append(adata_tcga)
    except Exception as e:
        print(f"[Warning] Could not load TCGA: {e}")

    if not adatas:
        raise FileNotFoundError(f"No valid cohort matrices could be built from {raw_dir}.")

    print("\n[Merging] Combining Neftel + TCGA into full internal cohort...")
    adata_full = ad.concat(adatas, merge="unique", join="inner")
    adata_full.obs_names_make_unique()

    # Rejoin Neftel's per-cell state (dropped by the inner-join concat above,
    # since TCGA has no equivalent column). Uses the pre-concat Neftel index,
    # matched by position within the Neftel block of adata_full since
    # obs_names_make_unique() may have altered raw index strings.
    if neftel_states is not None:
        neftel_mask = adata_full.obs["Cohort"] == "Neftel"
        n_neftel_in_full = int(neftel_mask.sum())
        if n_neftel_in_full == len(neftel_states):
            adata_full.obs.loc[neftel_mask, "state"] = neftel_states["state"].to_numpy()
        else:
            print(f"[Warning] Neftel cell count changed during QC filtering "
                  f"({len(neftel_states)} pre-QC vs {n_neftel_in_full} post-QC); "
                  f"re-deriving state alignment by index instead of position.")
            state_lookup = neftel_states["state"]
            adata_full.obs["state"] = adata_full.obs_names.map(state_lookup)
        n_with_state = adata_full.obs["state"].notna().sum()
        print(f"[State Rejoin] state resolved for {n_with_state}/{adata_full.n_obs} cells "
              f"(non-Neftel rows correctly have no state).")

    adata_full = merge_clinical_metadata(adata_full, raw_dir)

    n_top_genes = qc_cfg["hvg_selection"]["n_top_genes"]
    flavor = qc_cfg["hvg_selection"]["flavor"]
    print(f"\n[HVG Selection] Identifying top {n_top_genes} HVGs (flavor='{flavor}')...")
    sc.pp.highly_variable_genes(adata_full, n_top_genes=n_top_genes, flavor=flavor)

    for gene in GATE_GENES:
        if gene in adata_full.var_names:
            adata_full.var.loc[gene, "highly_variable"] = True

    os.makedirs(output_dir, exist_ok=True)
    full_out_path = os.path.join(output_dir, "full_cohort_neftel_tcga.h5ad")
    adata_full.write(full_out_path)

    print(f"\n[Saved] Successfully compiled full INTERNAL cohort (Neftel + TCGA) to: {full_out_path}")
    print(f"Total cells/samples: {adata_full.n_obs}, Total genes: {adata_full.n_vars}")
    print(f"Cohort Breakdown:\n{adata_full.obs['Cohort'].value_counts()}")
    print("\nCGGA was NOT included -- it is preprocessed separately in Week 4 "
          "as the untouched external validation cohort.")


if __name__ == "__main__":
    raise SystemExit(main())
