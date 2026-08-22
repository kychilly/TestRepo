#!/usr/bin/env python3
"""Compatibility entry point for the maintained cross-cohort assembler.

The former ``preprocess_full.py`` concatenated single-cell logTPM with bulk
RSEM, normalized already-normalized matrices again, and copied those values
into a layer named ``counts``. It also appended CGGA twice in one code path.
Those behaviours make the file unsafe for scVI and misleading for scGPT.

This shim keeps Jeffrey's documented command working while delegating to the
provenance-preserving builder. The resulting combined H5AD is only for the
patient-level cross-cohort IDH analysis. Adit's scGPT run continues to use the
Neftel-only H5AD configured in ``config/week3_adit.yaml``.
"""

from __future__ import annotations

from scripts.build_analysis_ready_combined_h5ad import main


if __name__ == "__main__":
    raise SystemExit(main())
