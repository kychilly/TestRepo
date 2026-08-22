#!/usr/bin/env python3
"""Compatibility entry point for the provenance-preserving mutation join."""

from __future__ import annotations

from scripts.build_tcga_pilot_mutation_join import main


if __name__ == "__main__":
    raise SystemExit(main())
