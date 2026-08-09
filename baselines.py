#!/usr/bin/env python3
"""Public entry point for the three leakage-safe conventional baselines.

The implementation lives in ``scripts.run_baseline`` so the same validated
runner is usable from the documented ``PYTHONPATH=src`` command.  All methods
accept the same ``--splits`` argument and therefore use the same patient-level
partition contract; select a method with ``--method``.
"""

from __future__ import annotations

# Keep the requested executable name while exposing the existing package under
# ``src/baselines``.  Without this, Python resolves ``baselines.py`` before the
# package and imports such as ``baselines.base`` fail during test collection.
import importlib.util
import sys
import argparse
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent / "src" / "baselines"
__path__ = [str(_PACKAGE_DIR)]


def _load_package_for_script_execution() -> None:
    """Make ``baselines.base`` resolve to src/baselines when run as a file."""
    if "baselines" in sys.modules:
        return
    package_init = _PACKAGE_DIR / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "baselines", package_init, submodule_search_locations=[str(_PACKAGE_DIR)]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load baseline package from {_PACKAGE_DIR}")
    package = importlib.util.module_from_spec(spec)
    sys.modules["baselines"] = package
    spec.loader.exec_module(package)


def main(argv: list[str] | None = None) -> int:
    """Run one baseline or all three into separate output directories."""
    _load_package_for_script_execution()
    from scripts.run_baseline import main as run_one

    if argv is not None and "--help" in argv:
        return run_one(argv)
    if argv is None and "--help" in sys.argv[1:]:
        return run_one(sys.argv[1:])

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--method", required=True)
    parser.add_argument("--adata", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parsed, _ = parser.parse_known_args(argv)
    methods = (
        ("pca_logreg", "scvi_probe", "harmony_knn")
        if parsed.method == "all"
        else (parsed.method,)
    )
    if parsed.method != "all":
        return run_one(argv)
    exit_code = 0
    for method in methods:
        method_args = [
            "--method",
            method,
            "--adata",
            parsed.adata,
            "--splits",
            parsed.splits,
            "--fold",
            parsed.fold,
            "--seed",
            parsed.seed,
            "--config",
            parsed.config,
            "--output",
            str(Path(parsed.output) / method),
        ]
        exit_code = max(exit_code, run_one(method_args))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
