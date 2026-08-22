import numpy as np
import os
import pandas as pd
import zipfile
import anndata as ad


def _load_cgga_clinical_zip(zip_path: str) -> pd.DataFrame:
    """
    Parse a CGGA clinical ZIP and return all well-formed tabular rows.

    CRC failures are treated as data-integrity failures. Silently bypassing the
    archive checksum can turn damaged expression or labels into plausible-looking
    scientific results.
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"CGGA clinical file not found: {zip_path}")

    with zipfile.ZipFile(zip_path) as z:
        real_names = [
            n
            for n in z.namelist()
            if not n.startswith("__MACOSX") and not os.path.basename(n).startswith(".")
        ]
        if not real_names:
            raise ValueError(f"No usable file found inside {zip_path}")
        try:
            raw = z.read(real_names[0])
        except zipfile.BadZipFile as exc:
            raise zipfile.BadZipFile(
                f"CRC/integrity failure in {zip_path}; re-download or replace the archive"
            ) from exc

    lines = [ln for ln in raw.decode("utf-8", errors="replace").split("\r") if ln.strip()]
    header = lines[0].split("\t")
    n_cols = len(header)

    records = []
    skipped = []
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) != n_cols:
            skipped.append(line)
            continue
        records.append(fields)

    if skipped:
        print(
            f"CGGA clinical ({os.path.basename(zip_path)}): skipped {len(skipped)} "
            f"malformed row(s): {[r.split(chr(9))[0] for r in skipped]}"
        )

    return pd.DataFrame(records, columns=header)


def _load_cgga_expression(expr_path: str) -> pd.DataFrame:
    """
    Gene_Name x CGGA_ID matrix -> returns patients-as-rows DataFrame
    (genes as columns), matching AnnData's expected obs/var orientation.

    The real CGGA RSEM download ships this as a .txt.zip, not a plain .txt
    (confirmed against the actual filenames on disk: CGGA.mRNAseq_325.RSEM-genes.
    20200506.txt.zip). Handles both: if expr_path ends in .zip, extracts the
    inner .txt in memory first; otherwise reads it directly as a plain TSV,
    so this still works if a future release ships unzipped.
    """
    if not os.path.exists(expr_path):
        raise FileNotFoundError(f"CGGA expression file not found: {expr_path}")

    if expr_path.endswith(".zip"):
        with zipfile.ZipFile(expr_path) as z:
            real_names = [
                n
                for n in z.namelist()
                if not n.startswith("__MACOSX") and not os.path.basename(n).startswith(".")
            ]
            if not real_names:
                raise ValueError(f"No usable file found inside {expr_path}")
            try:
                with z.open(real_names[0]) as zf:
                    expr = pd.read_csv(zf, sep="\t", index_col=0)
            except zipfile.BadZipFile as exc:
                raise zipfile.BadZipFile(
                    f"CRC/integrity failure in {expr_path}; re-download or replace the archive"
                ) from exc
    else:
        expr = pd.read_csv(expr_path, sep="\t", index_col=0)

    # Gene_Name can have duplicate rows in some CGGA releases; keep first occurrence
    expr = expr[~expr.index.duplicated(keep="first")]
    return expr.T  # patients x genes


def load_cgga_cohort(data_dir: str = "data/raw/cgga") -> ad.AnnData:
    """
    Loads and merges both CGGA batches (mRNAseq_325, mRNAseq_693) into a
    single AnnData with the same obs schema build_pilot_subsample.py expects
    from Neftel: 'Sample', 'GBMType', 'derived_state'.

    IDH_mutation_status from the clinical file is NOT written into obs here.
    It stays exclusively in the clinical DataFrame available via
    adata.uns['cgga_clinical'] — sampling on IDH status directly would leak
    the mutation label into the variable build_pilot_mutation_table.py is
    supposed to independently determine. The two scripts should not both
    look at IDH_mutation_status through different paths for the same patient.
    """
    batches = [
        (
            "mRNAseq_325",
            os.path.join(data_dir, "CGGA.mRNAseq_325.RSEM-genes.20200506.txt.zip"),
            os.path.join(data_dir, "CGGA.mRNAseq_325_clinical.20200506.txt.zip"),
        ),
        (
            "mRNAseq_693",
            os.path.join(data_dir, "CGGA.mRNAseq_693.RSEM-genes.20200506.txt.zip"),
            os.path.join(data_dir, "CGGA.mRNAseq_693_clinical.20200506.txt.zip"),
        ),
    ]

    adatas = []
    for batch_name, expr_path, clinical_zip in batches:
        if not os.path.exists(expr_path) or not os.path.exists(clinical_zip):
            print(f"CGGA {batch_name}: missing expression or clinical file, skipping.")
            continue

        expr_df = _load_cgga_expression(expr_path)
        clinical_df = _load_cgga_clinical_zip(clinical_zip)
        clinical_df = clinical_df.set_index("CGGA_ID")

        # Only keep patients present in BOTH expression and clinical files.
        # Silently dropping the rest would be the same class of bug as the
        # original mutation-mapping fallback: don't guess, and don't include
        # a patient we can't cross-reference.
        shared_ids = expr_df.index.intersection(clinical_df.index)
        dropped = expr_df.index.difference(clinical_df.index)
        if len(dropped) > 0:
            print(
                f"CGGA {batch_name}: {len(dropped)} patient(s) in expression matrix "
                f"have no clinical record and are excluded: {list(dropped)[:5]}"
                f"{'...' if len(dropped) > 5 else ''}"
            )

        expr_df = expr_df.loc[shared_ids]
        clinical_df = clinical_df.loc[shared_ids]

        adata = ad.AnnData(X=expr_df.values.astype(np.float32))
        adata.obs_names = shared_ids
        adata.var_names = expr_df.columns
        adata.obs["Sample"] = shared_ids
        adata.obs["GBMType"] = clinical_df["Histology"].values  # see module docstring, point 1
        adata.obs["derived_state"] = "Unknown"  # see module docstring, point 2
        adata.obs["cgga_batch"] = batch_name
        adata.uns[f"cgga_clinical_{batch_name}"] = clinical_df

        adatas.append(adata)

    if not adatas:
        raise FileNotFoundError(f"No CGGA batches found under {data_dir}")

    merged = ad.concat(adatas, join="outer", label="cgga_batch_concat", index_unique=None)
    return merged


if __name__ == "__main__":
    cgga = load_cgga_cohort()
    print(f"Loaded {cgga.n_obs} CGGA patients, {cgga.n_vars} genes")
    print(cgga.obs["GBMType"].value_counts())
    print(cgga.obs["derived_state"].value_counts())
