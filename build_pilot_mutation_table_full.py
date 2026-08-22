#!/usr/bin/env python3
"""Compatibility entry point for the provenance-preserving mutation join.

The retired implementation treated all Neftel patients as TP53-missense and
EGFR-amplified without patient-level calls and converted unknown CGGA entries
to missense. Those are cohort assumptions, not mutation evidence, so they
must not enter Stage 3/4 or Week 4 metrics.
"""

from __future__ import annotations

from scripts.build_tcga_pilot_mutation_join import main


if __name__ == "__main__":
    raise SystemExit(main())
