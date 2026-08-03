#!/usr/bin/env python3
"""Run or explicitly block the fixed 1,000-cell scGPT benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from baselines.base import BaselineError, load_patient_splits
from models.scgpt_adapter import AdapterError, deterministic_indices


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml  # type: ignore[import-untyped]

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AdapterError("Benchmark configuration must be a YAML object")
    return payload


def _blocked(config: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "status": "blocked",
        "reason": reason,
        "benchmark_cells": int(config.get("benchmark_cells", 1000)),
        "selection": {
            "seed": config.get("seed"),
            "patients_represented": None,
            "training_patients_only": False,
        },
        "mapping": {
            "genes_before": None,
            "genes_after": None,
            "retained": None,
            "dropped": None,
            "duplicated": None,
            "unmapped": None,
        },
        "timing": {
            "wall_clock_seconds": None,
            "gpu_synchronization": None,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
            "cells_per_second": None,
            "gpu_seconds_per_1000_cells": None,
            "projected_gpu_seconds_per_10000_cells": None,
        },
        "model": {
            "precision": config.get("precision"),
            "batch_size": config.get("batch_size"),
            "token_length": config.get("token_length"),
        },
        "provenance": {
            "checkpoint_sha256": None,
            "vocabulary_sha256": None,
        },
    }


def run_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    """Validate the real-data prerequisites before performing inference."""
    required = (
        "cell_data_path",
        "patient_id_column",
        "gene_id_column",
        "gene_id_type",
        "split_file",
        "checkpoint_path",
        "vocabulary_path",
    )
    missing = [key for key in required if not config.get(key)]
    if missing:
        return _blocked(config, "Missing real benchmark inputs: " + ", ".join(missing))
    data_path = Path(str(config["cell_data_path"]))
    split_path = Path(str(config["split_file"]))
    checkpoint_path = Path(str(config["checkpoint_path"]))
    vocabulary_path = Path(str(config["vocabulary_path"]))
    missing_paths = [
        str(path)
        for path in (data_path, split_path, checkpoint_path, vocabulary_path)
        if not path.is_file()
    ]
    if missing_paths:
        return _blocked(
            config, "Configured benchmark asset does not exist: " + ", ".join(missing_paths)
        )
    try:
        import anndata as ad  # type: ignore[import-not-found]
    except ImportError as exc:
        return _blocked(config, f"anndata is required for the configured AnnData input: {exc}")
    try:
        split = load_patient_splits(split_path, int(config.get("fold", 0)))
    except BaselineError as exc:
        raise AdapterError(f"Invalid patient split contract: {exc}") from exc
    data = ad.read_h5ad(data_path, backed="r")
    patient_column = str(config["patient_id_column"])
    gene_column = str(config["gene_id_column"])
    if patient_column not in data.obs or gene_column not in data.var:
        raise AdapterError("Configured patient or gene identifier column is absent from AnnData")
    train_patients = set(split.train)
    patient_values = np.asarray(data.obs[patient_column].astype(str))
    train_indices = np.flatnonzero(np.isin(patient_values, list(train_patients)))
    selected_relative = deterministic_indices(
        len(train_indices), int(config.get("benchmark_cells", 1000)), int(config["seed"])
    )
    selected = train_indices[selected_relative]
    raise AdapterError(
        "scGPT model construction is checkpoint-specific; configure a verified loader before "
        f"benchmarking the selected {len(selected)} training cells"
    )


def _unavailable_model() -> Any:
    raise AdapterError("No model loader is available without a verified scGPT checkpoint contract")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/model.yaml"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config = _load_yaml(args.config)
    try:
        result = run_benchmark(config)
        exit_code = 0 if result["status"] == "completed" else 2
    except (AdapterError, OSError, ValueError) as exc:
        result = _blocked(config, str(exc))
        exit_code = 2
    output = args.output or Path(
        str(config.get("output_path", "results/compute/week1_scgpt_benchmark.json"))
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(result, indent=2, sort_keys=True), file=sys.stderr if exit_code else sys.stdout
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
