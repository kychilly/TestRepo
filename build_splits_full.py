#!/usr/bin/env python3
"""Build the canonical Neftel patient split used by every scGPT arm.

The retired script discovered an arbitrary first TCGA text file and mixed
bulk TCGA patients into a split consumed by a Neftel-only single-cell model.
This wrapper defaults to the exact H5AD and split configured for Adit's run.
"""

from __future__ import annotations

import sys

from scripts.build_pilot_splits import main


DEFAULT_ARGUMENTS = [
    "--adata",
    "data/import_20260820/TP53 Dataset(preprocessed)/processed/neftel_analysis_cohort.h5ad",
    "--output",
    "splits/combined_full_cohort_neftel_patient_splits.json",
    "--seed",
    "17",
]


if __name__ == "__main__":
    raise SystemExit(main(DEFAULT_ARGUMENTS if len(sys.argv) == 1 else None))
