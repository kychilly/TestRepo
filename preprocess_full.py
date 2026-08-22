#!/usr/bin/env python3
"""Compatibility entry point for the maintained Week 4 cohort assembler.

Jeffrey's updated branch correctly removes CGGA from the internal training
cohort. This command delegates to the maintained Week 4 builder, which reads
the already-frozen Neftel input and separately rebuilt finite CGGA input. The
result is used only for held-out patient-level IDH evaluation. Adit's scGPT
training remains Neftel-only through ``config/week3_adit.yaml``.

This entry point deliberately does not copy logTPM/RSEM into a layer named
``counts`` or normalize those already-normalized representations again.
"""

from __future__ import annotations

from scripts.build_analysis_ready_combined_h5ad import main


if __name__ == "__main__":
    raise SystemExit(main())
