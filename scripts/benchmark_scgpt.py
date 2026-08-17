#!/usr/bin/env python3
"""Run or explicitly block the fixed 1,000-cell scGPT benchmark."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from baselines.base import BaselineError, load_patient_splits
from gbm_study.storage_policy import StoragePolicyError, policy_from_config
from models.scgpt_adapter import AdapterError, ScGPTAdapter, deterministic_indices
from models.scgpt_loader import load_official_scgpt, load_vocabulary


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
    data_source = str(config.get("data_source", "local_h5ad"))
    required = [
        "patient_id_column",
        "gene_id_column",
        "gene_id_type",
        "split_file",
        "checkpoint_path",
        "vocabulary_path",
    ]
    if data_source == "local_h5ad":
        required.append("cell_data_path")
    elif data_source == "huggingface_stream":
        required.extend(["hf_dataset_id", "hf_split", "hf_expression_column"])
    else:
        return _blocked(config, f"Unsupported data_source: {data_source}")
    missing = [key for key in required if not config.get(key)]
    if missing:
        return _blocked(config, "Missing real benchmark inputs: " + ", ".join(missing))
    split_path = Path(str(config["split_file"]))
    checkpoint_path = Path(str(config["checkpoint_path"]))
    vocabulary_path = Path(str(config["vocabulary_path"]))
    data_path = Path(str(config["cell_data_path"])) if data_source == "local_h5ad" else None
    missing_paths = [
        str(path)
        for path in (data_path, split_path, checkpoint_path, vocabulary_path)
        if path is not None
        if not path.is_file()
    ]
    if missing_paths:
        return _blocked(
            config,
            "Configured benchmark asset does not exist: " + ", ".join(missing_paths),
        )
    local_inputs = [split_path, checkpoint_path, vocabulary_path]
    if data_path is not None:
        local_inputs.append(data_path)
    args_value = config.get("model_args_path")
    args_path = Path(str(args_value)) if args_value else checkpoint_path.parent / "args.json"
    if args_path.is_file():
        local_inputs.append(args_path)
    try:
        storage = policy_from_config(config).validate(local_inputs)
    except StoragePolicyError as exc:
        return _blocked(config, str(exc))
    try:
        split = load_patient_splits(split_path, int(config.get("fold", 0)))
    except BaselineError as exc:
        raise AdapterError(f"Invalid patient split contract: {exc}") from exc
    patient_column = str(config["patient_id_column"])
    gene_column = str(config["gene_id_column"])
    train_patients = set(split.train)
    requested = int(config.get("benchmark_cells", 1000))
    if data_source == "local_h5ad":
        try:
            import anndata as ad  # type: ignore[import-untyped]
        except ImportError as exc:
            return _blocked(config, f"anndata is required for AnnData input: {exc}")
        assert data_path is not None
        data = ad.read_h5ad(data_path, backed="r")
        gene_column_present = gene_column == "var_names" or gene_column in data.var
        if patient_column not in data.obs or not gene_column_present:
            raise AdapterError(
                "Configured patient or gene identifier column is absent from AnnData"
            )
        patient_values = np.asarray(data.obs[patient_column].astype(str))
        train_indices = np.flatnonzero(np.isin(patient_values, list(train_patients)))
        selected_relative = deterministic_indices(
            len(train_indices), requested, int(config["seed"])
        )
        selected = train_indices[selected_relative]
        subset = data[selected].to_memory()
        matrix = subset.X.toarray() if hasattr(subset.X, "toarray") else np.asarray(subset.X)
        gene_ids = (
            [str(value) for value in subset.var_names]
            if gene_column == "var_names"
            else subset.var[gene_column].astype(str).tolist()
        )
        selected_patients = subset.obs[patient_column].astype(str).tolist()
    else:
        matrix, gene_ids, selected_patients = _stream_huggingface_training_sample(
            config, train_patients, requested
        )
    vocabulary = load_vocabulary(vocabulary_path)
    device = str(config.get("requested_device", "cuda"))
    model = load_official_scgpt(checkpoint_path, vocabulary_path, device, config)
    adapter = ScGPTAdapter(model, vocabulary, checkpoint_path, vocabulary_path, device=device)
    prepared = adapter.prepare_inputs(
        {"X": matrix, "var_names": gene_ids}, str(config["gene_id_type"])
    )
    import torch

    for _ in range(int(config.get("warmup_iterations", 2))):
        adapter.infer(
            prepared,
            batch_size=int(config.get("batch_size", 32)),
            precision=str(config.get("precision", "float32")),
        )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    output = adapter.infer(
        prepared,
        batch_size=int(config.get("batch_size", 32)),
        precision=str(config.get("precision", "float32")),
    )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    duration = time.perf_counter() - started
    if output.shape[0] != requested:
        raise AdapterError("scGPT benchmark output row count is incorrect")
    report = prepared.report
    return {
        "status": "completed",
        "benchmark_cells": requested,
        "selection": {
            "seed": int(config["seed"]),
            "patients_represented": sorted(set(selected_patients)),
            "training_patients_only": set(selected_patients).issubset(train_patients),
            "data_source": data_source,
        },
        "mapping": {
            "genes_before": len(gene_ids),
            "genes_after": report.retained_count,
            "retained": report.retained_count,
            "dropped": len(report.dropped),
            "duplicated": len(report.duplicated),
            "unmapped": len(report.unmapped),
        },
        "timing": {
            "wall_clock_seconds": duration,
            "gpu_synchronization": bool(device.startswith("cuda")),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated())
            if device.startswith("cuda")
            else None,
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())
            if device.startswith("cuda")
            else None,
            "cells_per_second": requested / duration,
            "gpu_seconds_per_1000_cells": duration * 1000 / requested,
            "projected_gpu_seconds_per_10000_cells": duration * 10000 / requested,
        },
        "model": {
            "precision": config.get("precision"),
            "batch_size": config.get("batch_size"),
            "token_length": config.get("token_length"),
            "device": device,
        },
        "provenance": {**adapter.provenance(), **storage},
    }


def _stream_huggingface_training_sample(
    config: dict[str, Any], train_patients: set[str], requested: int
) -> tuple[np.ndarray[Any, Any], list[str], list[str]]:
    """Reservoir-sample training cells from a HF iterable dataset without downloading it."""
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as exc:
        raise AdapterError(f"Hugging Face streaming requires datasets: {exc}") from exc
    stream = load_dataset(
        str(config["hf_dataset_id"]),
        name=str(config["hf_dataset_config"]) if config.get("hf_dataset_config") else None,
        split=str(config["hf_split"]),
        revision=str(config["hf_revision"]) if config.get("hf_revision") else None,
        streaming=True,
    )
    patient_key = str(config["patient_id_column"])
    expression_key = str(config["hf_expression_column"])
    gene_key = str(config.get("hf_gene_ids_column", "gene_ids"))
    rng = np.random.default_rng(int(config["seed"]))
    reservoir: list[tuple[np.ndarray[Any, Any], str]] = []
    gene_ids: list[str] | None = None
    seen = 0
    for row in stream:
        patient = str(row[patient_key])
        if patient not in train_patients:
            continue
        expression = np.asarray(row[expression_key], dtype=np.float32)
        if expression.ndim != 1:
            raise AdapterError("HF expression rows must be one-dimensional")
        if gene_ids is None:
            gene_ids = [str(value) for value in row[gene_key]]
        if len(expression) != len(gene_ids):
            raise AdapterError("HF expression and gene ID lengths do not match")
        seen += 1
        item = (expression, patient)
        if len(reservoir) < requested:
            reservoir.append(item)
        else:
            replacement = int(rng.integers(0, seen))
            if replacement < requested:
                reservoir[replacement] = item
    if len(reservoir) < requested or gene_ids is None:
        raise AdapterError(
            f"HF stream contains only {len(reservoir)} eligible training cells; need {requested}"
        )
    return (
        np.stack([item[0] for item in reservoir]),
        gene_ids,
        [item[1] for item in reservoir],
    )


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
        str(config.get("output_path", "baseline_results/compute/week1_scgpt_benchmark.json"))
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(result, indent=2, sort_keys=True),
        file=sys.stderr if exit_code else sys.stdout,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
