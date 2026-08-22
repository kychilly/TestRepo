#!/usr/bin/env python3
"""Compatibility entry point for provenance-safe cohort assembly.

The former implementation silently continued after cohort-loading failures and
could copy already-normalized expression into a layer named ``counts``. Both
behaviours can invalidate downstream scVI and cross-cohort analyses. This entry
point delegates to the maintained builder, which requires explicit H5AD inputs
and records input/output hashes plus a plain-English companion report.
"""

from __future__ import annotations

from scripts.build_analysis_ready_combined_h5ad import main


if __name__ == "__main__":
    raise SystemExit(main())
