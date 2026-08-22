#!/usr/bin/env python3
"""Compatibility entry point for deterministic patient-only pilot splits."""

from __future__ import annotations

from scripts.build_pilot_splits import main


if __name__ == "__main__":
    raise SystemExit(main())
