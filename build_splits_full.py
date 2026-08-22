import glob
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from build_pilot_subsample import clean_id


def find_files(directory, extensions=(".txt", ".tsv", ".csv", ".zip")):
    """Finds matching files ignoring system files like __MACOSX."""
    matched = []
    for root, _, files in os.walk(directory):
        for f in files:
            if (
                f.endswith(extensions)
                and not f.startswith(".")
                and "__MACOSX" not in root
            ):
                matched.append(os.path.join(root, f))
    return matched


def build_patient_splits():
    print("[1] Extracting Patient IDs from metadata...")

    # 1. Neftel Patients
    neftel_files = find_files("data/raw/neftel")
    neftel_meta_file = [f for f in neftel_files if "logTPM" not in f][0]
    print(f"    Reading Neftel file: {neftel_meta_file}")

    neftel_df = pd.read_csv(neftel_meta_file, sep="\t", index_col=0)
    raw_neftel_ids = [str(idx).split("-")[0] for idx in neftel_df.index]
    neftel_patients = sorted({clean_id(pid) for pid in raw_neftel_ids if clean_id(pid)})
    print(f"    Found {len(neftel_patients)} Neftel patient IDs.")

    # 2. TCGA Patients
    tcga_files = find_files("data/raw/tcga", extensions=(".tsv", ".txt"))
    tcga_file = tcga_files[0]
    print(f"    Reading TCGA file: {tcga_file}")

    tcga_df = pd.read_csv(tcga_file, sep="\t", comment="#")

    # Identify Patient ID column dynamically
    tcga_col = None
    for col in tcga_df.columns:
        if "patient" in col.lower():
            tcga_col = col
            break
    if not tcga_col:
        tcga_col = tcga_df.columns[0]

    raw_tcga_ids = tcga_df[tcga_col].dropna().astype(str).tolist()
    tcga_patients = sorted({clean_id(pid) for pid in raw_tcga_ids if clean_id(pid)})
    print(f"    Found {len(tcga_patients)} TCGA patient IDs.")

    # Combine Neftel and TCGA into total pool
    patient_pool = sorted(list(set(neftel_patients + tcga_patients)))

    print("\n[2] Splitting Neftel + TCGA (70% Train, 15% Validation, 15% Test)...")
    # Step 1: Hold out 70% for training, 30% for temp (val + test)
    train_patients, temp_patients = train_test_split(
        patient_pool, test_size=0.30, random_state=42
    )
    # Step 2: Divide the 30% temp evenly into Validation (15%) and Test (15%)
    val_patients, test_patients = train_test_split(
        temp_patients, test_size=0.50, random_state=42
    )

    print("\n[3] Running strict overlap assertions...")
    train_set = set(train_patients)
    val_set = set(val_patients)
    test_set = set(test_patients)

    # ASSERT ZERO PATIENT OVERLAP ACROSS ALL SPLITS
    assert len(train_set.intersection(val_set)) == 0, (
        "CRITICAL ERROR: Leak between Train & Validation!"
    )
    assert len(train_set.intersection(test_set)) == 0, (
        "CRITICAL ERROR: Leak between Train & Test!"
    )
    assert len(val_set.intersection(test_set)) == 0, (
        "CRITICAL ERROR: Leak between Validation & Test!"
    )

    print("    [PASS] Zero patient overlap across Train, Validation, and Test confirmed!")

    # Exactly matches your indexing code: assignments["train"], assignments["validation"], assignments["test"]
    splits = {
        "train": sorted(train_patients),
        "validation": sorted(val_patients),
        "test": sorted(test_patients),
    }

    os.makedirs("splits", exist_ok=True)
    out_path = "splits/patient_splits_full.json"

    with open(out_path, "w") as f:
        json.dump(splits, f, indent=4)

    print(f"\n[4] Saved patient splits to: {out_path}")
    print(f"    Train patients:      {len(train_patients)}")
    print(f"    Validation patients: {len(val_patients)}")
    print(f"    Test patients:       {len(test_patients)}")


if __name__ == "__main__":
    build_patient_splits()