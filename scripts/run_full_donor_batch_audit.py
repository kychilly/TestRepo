#!/usr/bin/env python3
"""Compatibility entry point for the maintained Neftel donor audit.

Jeffrey's updated Neftel+TCGA figures remain historical audit artifacts. The
paper-relevant donor-versus-state calculation is run on the exact Neftel-only
H5AD consumed by scGPT, without conflating single-cell and bulk assay effects.
"""

from __future__ import annotations

from donor_batch_audit import main


if __name__ == "__main__":
    raise SystemExit(main())
