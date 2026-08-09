import os
import glob
import json
import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split


def find_files(directory, extensions=(".txt", ".tsv", ".csv", ".zip")):
    """Finds matching files ignoring system files like __MACOSX."""
    matched = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith(extensions) and not f.startswith(".") and "__MACOSX" not in root:
                matched.append(os.path.join(root, f))
    return matched


def read_cgga_clinical(filepath):
    """Reads CGGA clinical data whether plain text or zip containing __MACOSX."""
    if filepath.endswith(".zip"):
        with zipfile.ZipFile(filepath, 'r') as z:
            # Find the actual data file inside the zip, ignoring __MACOSX
            target_file = [f for f in z.namelist() if not f.startswith("__MACOSX") and f.endswith(".txt")][0]
            with z.open(target_file) as f:
                return pd.read_csv(f, sep="\t")
    return pd.read_csv(filepath, sep="\t")


def build_patient_splits():
    print("[1] Extracting Patient IDs from metadata...")

    # 1. Neftel Patients
    neftel_files = find_files("data/raw/neftel")
    neftel_meta_file = [f for f in neftel_files if "logTPM" not in f][0]
    print(f"    Reading Neftel file: {neftel_meta_file}")

    neftel_df = pd.read_csv(neftel_meta_file, sep="\t", index_col=0)
    neftel_patients = list(set([str(idx).split('-')[0] for idx in neftel_df.index]))
    print(f"    Found {len(neftel_patients)} Neftel patient IDs.")

    # 2. TCGA Patients (skip cBioPortal '#' comments)
    tcga_files = find_files("data/raw/tcga", extensions=(".tsv", ".txt"))
    tcga_file = tcga_files[0]
    print(f"    Reading TCGA file: {tcga_file}")

    tcga_df = pd.read_csv(tcga_file, sep="\t", comment='#')

    # Identify Patient ID column dynamically
    tcga_col = None
    for col in tcga_df.columns:
        if "patient" in col.lower():
            tcga_col = col
            break
    if not tcga_col:
        tcga_col = tcga_df.columns[0]

    tcga_patients = list(set(tcga_df[tcga_col].dropna().astype(str)))
    print(f"    Found {len(tcga_patients)} TCGA patient IDs.")

    # 3. CGGA Patients
    cgga_files = find_files("data/raw/cgga")
    cgga_patients = set()

    for f in cgga_files:
        if "RSEM" not in f and "genes" not in f:
            print(f"    Reading CGGA file: {f}")
            cdf = read_cgga_clinical(f)
            cgga_patients.update(cdf.iloc[:, 0].dropna().astype(str))

    cgga_patients = list(cgga_patients)
    print(f"    Found {len(cgga_patients)} CGGA patient IDs.")

    # Combine Neftel and TCGA into Train/Val pool
    train_val_pool = sorted(list(set(neftel_patients + tcga_patients)))

    print("\n[2] Splitting Neftel + TCGA into 80% Train / 20% Validation...")
    train_patients, val_patients = train_test_split(train_val_pool, test_size=0.20, random_state=42)
    test_patients = sorted(cgga_patients)

    print("\n[3] Running strict overlap assertions...")
    train_set = set(train_patients)
    val_set = set(val_patients)
    test_set = set(test_patients)

    # ASSERT ZERO PATIENT OVERLAP
    intersection_train_val = train_set.intersection(val_set)
    intersection_train_test = train_set.intersection(test_set)
    intersection_val_test = val_set.intersection(test_set)

    assert len(intersection_train_val) == 0, f"CRITICAL ERROR: Leak between Train & Val: {intersection_train_val}"
    assert len(intersection_train_test) == 0, f"CRITICAL ERROR: Leak between Train & CGGA: {intersection_train_test}"
    assert len(intersection_val_test) == 0, f"CRITICAL ERROR: Leak between Val & CGGA: {intersection_val_test}"

    print("    [PASS] Zero patient overlap across all splits confirmed!")

    # Save to JSON
    splits = {
        "train": train_patients,
        "val": val_patients,
        "test_cgga": test_patients
    }

    os.makedirs("splits", exist_ok=True)
    out_path = "splits/patient_splits.json"

    with open(out_path, "w") as f:
        json.dump(splits, f, indent=4)

    print(f"\n[4] Saved patient splits to: {out_path}")


if __name__ == "__main__":
    build_patient_splits()