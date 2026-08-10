import os
import shutil
from pathlib import Path
import yaml
import pandas as pd
import scanpy as sc
import anndata as ad


def load_qc_config(config_path="config/qc.yaml"):
    """Load frozen QC parameters from YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"QC configuration file not found at {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"[QC] Loaded frozen parameters from {config_path}")
    return config


def validate_preprocessed_h5ad(input_path, config_path="config/qc.yaml"):
    """Validate Jeffrey's preprocessed H5AD without silently reprocessing it.

    The shared artifact is already CP10K/log1p/HVG processed. This function
    checks the frozen QC contract and reports missing study labels explicitly;
    it never invents an AC/MES/NPC/OPC mapping from ``CellAssignment``.
    """
    try:
        import anndata as ad
    except ImportError as exc:
        raise RuntimeError("H5AD validation requires anndata") from exc
    qc_cfg = load_qc_config(config_path)
    adata = ad.read_h5ad(input_path, backed="r")
    required_obs = {"Sample", "CellAssignment", "n_genes_by_counts", "pct_counts_mito"}
    missing_obs = sorted(required_obs - set(adata.obs.columns))
    if missing_obs:
        raise ValueError(f"Preprocessed H5AD is missing required obs fields: {missing_obs}")
    if "highly_variable" not in adata.var:
        raise ValueError("Preprocessed H5AD is missing var/highly_variable")
    expected_hvg = int(qc_cfg["hvg_selection"]["n_top_genes"])
    actual_hvg = int(adata.var["highly_variable"].sum())
    if actual_hvg != expected_hvg:
        raise ValueError(f"HVG count {actual_hvg} does not match frozen config {expected_hvg}")
    state_warning = "state column absent; CellAssignment is not AC/MES/NPC/OPC"
    return {
        "status": "validated",
        "cells": int(adata.n_obs),
        "genes": int(adata.n_vars),
        "hvg_count": actual_hvg,
        "sample_count": int(adata.obs["Sample"].nunique()),
        "state_warning": state_warning,
        "counts_layer_present": "counts" in adata.layers,
    }


def preprocess_neftel(
    data_dir="data/raw/neftel", output_dir="data/processed", config_path="config/qc.yaml"
):
    """
    Executes cell QC, gene QC, CP10K + log1p normalization, and HVG selection
    on Neftel et al. single-cell expression data based on frozen config/qc.yaml parameters.
    """
    qc_cfg = load_qc_config(config_path)

    h5ad_files = sorted(Path(data_dir).glob("*.h5ad"))
    if h5ad_files:
        input_path = h5ad_files[0]
        report = validate_preprocessed_h5ad(input_path, config_path)
        os.makedirs(output_dir, exist_ok=True)
        output_path = Path(output_dir) / "neftel_qc.h5ad"
        if input_path.resolve() != output_path.resolve():
            shutil.copyfile(input_path, output_path)
        report["output_path"] = str(output_path)
        print(f"[Validated] Existing preprocessed H5AD copied to {output_path}")
        return report

    # 1. Locate expression and metadata files
    expr_file = None
    meta_file = None

    for f in os.listdir(data_dir):
        if "logTPM" in f or "counts" in f or "expression" in f:
            expr_file = os.path.join(data_dir, f)
        elif "Metadata" in f or "metadata" in f or "Hierarchy" in f:
            meta_file = os.path.join(data_dir, f)

    if not expr_file:
        raise FileNotFoundError(f"Could not find expression matrix file in {data_dir}")

    print(f"[Loading] Reading expression data from: {expr_file}")

    # Load expression matrix (genes x cells or cells x genes)
    if expr_file.endswith(".gz"):
        df_expr = pd.read_csv(expr_file, sep="\t", compression="gzip", index_col=0)
    else:
        df_expr = pd.read_csv(expr_file, sep="\t", index_col=0)

    # Transpose if genes are rows (Scanpy expects cells as rows, genes as columns)
    if df_expr.shape[0] > df_expr.shape[1]:
        print("[Format] Transposing expression matrix (cells -> rows, genes -> columns)...")
        df_expr = df_expr.T

    adata = ad.AnnData(
        X=df_expr.values,
        obs=pd.DataFrame(index=df_expr.index),
        var=pd.DataFrame(index=df_expr.columns),
    )
    print(f"[Initial shape] Cells: {adata.n_obs}, Genes: {adata.n_vars}")

    # Attach metadata if present
    if meta_file and os.path.exists(meta_file):
        print(f"[Loading] Attaching metadata from: {meta_file}")
        df_meta = pd.read_csv(meta_file, sep="\t", index_col=0)
        adata.obs = adata.obs.join(df_meta, how="left")

    # 2. Compute Mitochondrial Percentage & QC Metrics
    # Identify mitochondrial genes starting with MT- or mt-
    adata.var["mito"] = adata.var_names.str.startswith(("MT-", "mt-"))

    sc.pp.calculate_qc_metrics(adata, qc_vars=["mito"], percent_top=None, log1p=False, inplace=True)

    # 3. Cell QC Filtering
    min_genes = qc_cfg["cell_qc"]["min_genes_per_cell"]
    max_mito = qc_cfg["cell_qc"]["max_pct_mito"]

    print(
        f"[Filtering Cells] Keeping cells with >= {min_genes} genes & <= {max_mito}% mito content..."
    )
    initial_cells = adata.n_obs

    sc.pp.filter_cells(adata, min_genes=min_genes)
    if "pct_counts_mito" in adata.obs:
        adata = adata[adata.obs["pct_counts_mito"] <= max_mito, :].copy()

    print(f"[Cells Filtered] {initial_cells - adata.n_obs} cells removed ({adata.n_obs} retained).")

    # 4. Gene QC Filtering
    min_cells = qc_cfg["gene_qc"]["min_cells_expressing"]
    print(f"[Filtering Genes] Keeping genes expressed in >= {min_cells} cells...")
    initial_genes = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=min_cells)
    print(
        f"[Genes Filtered] {initial_genes - adata.n_vars} genes removed ({adata.n_vars} retained)."
    )

    # 5. Normalization: CP10K + log1p
    target_sum = qc_cfg["normalization"]["target_sum"]
    print(f"[Normalization] Performing CP10K normalization (target_sum={target_sum})...")
    sc.pp.normalize_total(adata, target_sum=target_sum)

    if qc_cfg["normalization"]["log_transform"]:
        print("[Normalization] Applying log1p transformation...")
        sc.pp.log1p(adata)

    # Store raw normalized counts before subsetting HVGs
    adata.raw = adata

    # 6. HVG Selection
    n_top_genes = qc_cfg["hvg_selection"]["n_top_genes"]
    flavor = qc_cfg["hvg_selection"]["flavor"]
    print(
        f"[HVG Selection] Identifying top {n_top_genes} highly variable genes (flavor='{flavor}')..."
    )
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor=flavor)

    # 7. Save Cleaned AnnData Object
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "neftel_qc.h5ad")
    adata.write(save_path)
    print(f"[Saved] Preprocessed AnnData object successfully written to: {save_path}")


if __name__ == "__main__":
    preprocess_neftel()
