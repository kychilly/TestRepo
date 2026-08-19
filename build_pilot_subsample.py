import os
import scanpy as sc
import pandas as pd


def clean_id(pid: str) -> str:
    """Standardizes patient IDs to match across WES and single-cell metadata."""
    if not isinstance(pid, str) or pid == "nan" or not pid:
        return ""
    pid = pid.strip()

    if pid.startswith("TCGA"):
        return pid[:12]

    if "CGGA" in pid:
        num_part = pid.replace("CGGA_", "").replace("CGGA-", "").replace("CGGA", "").strip()
        return num_part

    if "_" in pid and not pid.startswith(("CGGA", "TCGA")):
        base = pid.split("_")[0]
        if base.startswith(("MGH", "BT", "CSC")):
            return base

    return pid


def get_idh1_mutant_patient_ids(cgga_dir):
    """Parses CGGA WES to extract patient IDs carrying IDH1 R132H / missense mutations."""
    wes_path = os.path.join(cgga_dir, "CGGA.WEseq_286.20200506.txt")
    if not os.path.exists(wes_path):
        print(f"[Subsample Warning] CGGA WES not found at {wes_path}")
        return set()

    wes_df = pd.read_csv(wes_path, sep="\t", index_col=0, low_memory=False)

    if "IDH1" not in wes_df.index:
        return set()

    idh1_row = wes_df.loc["IDH1"].astype(str).str.lower()

    # Identify columns containing R132H, missense, or point mutations
    mut_mask = idh1_row.str.contains("r132h|missense|point|multiple_variant", na=False)
    raw_mut_cols = idh1_row[mut_mask].index.tolist()

    idh1_cleaned_ids = {clean_id(str(col)) for col in raw_mut_cols if clean_id(str(col))}
    return idh1_cleaned_ids


def build_pilot_subsample(target_patient_count=20, target_idh1_count=12):
    # 1. Locate dataset
    data_dir = "data/processed"
    raw_dir = "data/raw"
    out_dir = "data/pilot"

    # Check potential input locations
    possible_paths = [
        os.path.join(data_dir, "full_dataset.h5ad"),
        os.path.join(data_dir, "merged_cohort.h5ad"),
        os.path.join(raw_dir, "neftel/neftel_subsample.h5ad"),
    ]

    h5ad_path = None
    for path in possible_paths:
        if os.path.exists(path):
            h5ad_path = path
            break

    if not h5ad_path:
        # Fallback to scanning data directory for any valid h5ad
        for root, _, files in os.walk("data"):
            for f in files:
                if f.endswith(".h5ad") and "pilot_subsample" not in f:
                    h5ad_path = os.path.join(root, f)
                    break
            if h5ad_path:
                break

    if not h5ad_path:
        raise FileNotFoundError(
            "Could not locate an existing .h5ad dataset in 'data/'. "
            "Please ensure your full or processed single-cell/bulk AnnData file exists."
        )

    print(f"Loading base dataset from: {h5ad_path} ...")
    adata = sc.read_h5ad(h5ad_path)

    # Determine patient column
    sample_col = None
    for candidate in ["Sample", "patient_id", "patient", "sample_id", "Donor"]:
        if candidate in adata.obs.columns:
            sample_col = candidate
            break

    if not sample_col:
        raise KeyError(
            f"Could not find patient ID column in adata.obs. Available columns: {list(adata.obs.columns)}"
        )

    print(f"Using '{sample_col}' to stratify patient selection.")

    # 2. Find IDH1-mutant patient candidates from CGGA WES
    cgga_dir = os.path.join(raw_dir, "cgga")
    idh1_mut_ids = get_idh1_mutant_patient_ids(cgga_dir)
    print(f"Extracted {len(idh1_mut_ids)} IDH1-mutant candidate patient IDs from CGGA WES.")

    # Get unique patient IDs in dataset
    all_raw_patients = adata.obs[sample_col].dropna().unique().tolist()

    # Partition patients into IDH1-mutant candidates vs others
    idh1_selected = []
    other_patients = []

    for raw_p in all_raw_patients:
        c_p = clean_id(str(raw_p))
        if c_p in idh1_mut_ids:
            idh1_selected.append(raw_p)
        else:
            other_patients.append(raw_p)

    print(f"Matched {len(idh1_selected)} IDH1-mutant patients within current dataset.")

    # 3. Stratified Subsampling Selection
    # Prioritize IDH1-mutant patients to make IDH1 missense dominant in Gate Gene check
    selected_patients = []

    # Pick target number of IDH1 mutants (or as many as available)
    n_idh1_to_pick = min(target_idh1_count, len(idh1_selected))
    selected_patients.extend(idh1_selected[:n_idh1_to_pick])

    # Fill the remaining patient slots with Neftel / TCGA / non-IDH1 patients
    remaining_slots = target_patient_count - len(selected_patients)
    selected_patients.extend(other_patients[:remaining_slots])

    print(f"Final pilot cohort patient composition ({len(selected_patients)} total):")
    print(f"  - IDH1-mutant enriched: {len(selected_patients[:n_idh1_to_pick])}")
    print(f"  - Non-IDH1 / Primary GBM / Controls: {len(selected_patients[n_idh1_to_pick:])}")

    # 4. Filter AnnData object to selected patients
    adata_pilot = adata[adata.obs[sample_col].isin(selected_patients)].copy()

    # 5. Export pilot subsample
    os.makedirs(out_dir, exist_ok=True)
    out_h5ad = os.path.join(out_dir, "pilot_subsample.h5ad")
    adata_pilot.write_h5ad(out_h5ad)

    print(f"\n[Success] Pilot subsample saved to: {out_h5ad}")
    print(f"Matrix shape: {adata_pilot.shape} ({adata_pilot.n_obs} cells x {adata_pilot.n_vars} genes)")


if __name__ == "__main__":
    build_pilot_subsample()