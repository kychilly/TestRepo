#!/usr/bin/env python3
"""Compatibility entry point for the maintained Neftel donor audit.

The paper-relevant batch question is donor versus biological state within the
single-cell Neftel cohort. Bulk CGGA/TCGA rows are deliberately excluded.
"""

from __future__ import annotations

from donor_batch_audit import main


if __name__ == "__main__":
    raise SystemExit(main())
