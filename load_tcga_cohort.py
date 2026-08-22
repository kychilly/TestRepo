import os
import zipfile
import pandas as pd
from anndata import AnnData


def load_tcga_cohort(data_dir: str = "data/raw/tcga") -> AnnData:

    expr_file = os.path.join(data_dir, "tcga_gbm_expression.tsv")
    meta_file = os.path.join(data_dir, "tcga_gbm_clinical.csv")
    zip_archive = os.path.join(data_dir, "tcga_data.zip")

    # 1. PARSE FROM UNCOMPRESSED TSV / CSV
    if os.path.exists(expr_file):
        print(f"Loading TCGA bulk dataset from {expr_file}...")
        expr_df = pd.read_csv(expr_file, sep="\t", index_col=0)

        obs_df = pd.DataFrame(index=expr_df.columns)
        if os.path.exists(meta_file):
            meta_df = pd.read_csv(meta_file, index_col=0)
            obs_df = obs_df.join(meta_df, how="left")
        else:
            obs_df["GBMType"] = "Primary"
            obs_df["Cohort"] = "TCGA-GBM"

        var_df = pd.DataFrame(index=expr_df.index)

        # Transpose expression matrix so obs = samples, vars = genes
        adata = AnnData(X=expr_df.values.T, obs=obs_df, var=var_df)
        adata.var_names_make_unique()
        return adata

    # 2. PARSE FROM ZIP ARCHIVE (IF DOWNLOADED AS ZIP)
    elif os.path.exists(zip_archive):
        print(f"Extracting TCGA dataset from zip archive: {zip_archive}...")
        with zipfile.ZipFile(zip_archive, "r") as z:
            txt_files = [f for f in z.namelist() if f.endswith(".txt") or f.endswith(".tsv")]
            if txt_files:
                with z.open(txt_files[0]) as f:
                    expr_df = pd.read_csv(f, sep="\t", index_col=0)
                obs_df = pd.DataFrame(index=expr_df.columns)
                obs_df["GBMType"] = "Primary"
                obs_df["Cohort"] = "TCGA-GBM"
                var_df = pd.DataFrame(index=expr_df.index)

                adata = AnnData(X=expr_df.values.T, obs=obs_df, var=var_df)
                adata.var_names_make_unique()
                return adata

    raise FileNotFoundError(
        f"No real TCGA expression file or archive was found under {data_dir}; "
        "synthetic fallback is prohibited"
    )


if __name__ == "__main__":
    tcga_adata = load_tcga_cohort()
    print(f"Loaded TCGA cohort: {tcga_adata.n_obs} samples x {tcga_adata.n_vars} genes.")
