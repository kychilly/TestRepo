#!/usr/bin/env python3
"""Fail-closed A100, streaming-data, storage, and model-asset preflight."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gbm_study.plain_english import write_json_with_explanation


DATA_PATTERNS = ("*.h5ad", "*.h5", "*.loom", "*.zip", "*.tar", "*.tar.gz", "*.tgz")
REQUIRED_STREAM_FIELDS = (
    "hf_dataset_id",
    "hf_revision",
    "hf_split",
    "hf_expression_column",
    "hf_gene_ids_column",
    "patient_id_column",
)


def _yaml(path: Path) -> dict[str, Any]:
    import yaml  # type: ignore[import-untyped]

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("A100 model config must be a YAML object")
    return payload


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def inspect(
    config_path: Path,
    repo: Path,
    scratch: Path,
    deadline_utc: str,
) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    config = _yaml(config_path)
    data_source = str(config.get("data_source"))
    if data_source not in {"huggingface_stream", "local_h5ad"}:
        blockers.append("data_source must be huggingface_stream or local_h5ad")
    if data_source == "huggingface_stream":
        missing_stream = [field for field in REQUIRED_STREAM_FIELDS if not config.get(field)]
        if missing_stream:
            blockers.append("Missing immutable Hugging Face fields: " + ", ".join(missing_stream))
    elif data_source == "local_h5ad":
        value = config.get("cell_data_path")
        data_path = Path(str(value)) if value else None
        if data_path is not None and not data_path.is_absolute():
            data_path = repo / data_path
        if data_path is None or not data_path.is_file():
            blockers.append(f"Missing local H5AD cell_data_path: {value!r}")

    asset_paths: dict[str, str | None] = {}
    for field in ("split_file", "checkpoint_path", "vocabulary_path"):
        value = config.get(field)
        path = Path(str(value)) if value else None
        if path is not None and not path.is_absolute():
            path = repo / path
        asset_paths[field] = str(path) if path else None
        if path is None or not path.is_file():
            blockers.append(f"Missing required local asset {field}: {value!r}")
    checkpoint_value = config.get("checkpoint_path")
    checkpoint = Path(str(checkpoint_value)) if checkpoint_value else None
    if checkpoint is not None and not checkpoint.is_absolute():
        checkpoint = repo / checkpoint
    args_value = config.get("model_args_path")
    args_path = (
        Path(str(args_value))
        if args_value
        else (checkpoint.parent / "args.json" if checkpoint else None)
    )
    if args_path is not None and not args_path.is_absolute():
        args_path = repo / args_path
    asset_paths["model_args_path"] = str(args_path) if args_path else None
    if args_path is None or not args_path.is_file():
        blockers.append(f"Missing checkpoint-matched args.json: {args_path}")

    scratch.mkdir(parents=True, exist_ok=True)
    if _inside(scratch, repo):
        blockers.append("GBM_A100_SCRATCH must be outside the Git repository")
    disk = shutil.disk_usage(scratch)
    if disk.free < 10 * 1024**3:
        warnings.append(f"Scratch has less than 10 GiB free: {disk.free} bytes")

    cache_vars = ("HF_HOME", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE")
    caches: dict[str, str | None] = {}
    for name in cache_vars:
        value = os.environ.get(name)
        caches[name] = value
        if not value:
            blockers.append(f"{name} is not set")
        elif not _inside(Path(value), scratch):
            blockers.append(f"{name} must be inside GBM_A100_SCRATCH")

    local_data = sorted(
        str(path.relative_to(repo))
        for pattern in DATA_PATTERNS
        for path in repo.glob(f"**/{pattern}")
        if ".git" not in path.parts and path.is_file()
    )
    if local_data and data_source == "huggingface_stream":
        blockers.append(
            "Dataset/archive files are present inside the Git workspace; move them to scratch: "
            + ", ".join(local_data[:10])
        )

    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        if not cuda_available:
            blockers.append("CUDA-enabled PyTorch is unavailable")
        gpu = {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": cuda_available,
            "device_count": torch.cuda.device_count(),
            "devices": [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ],
            "bf16_supported": bool(cuda_available and torch.cuda.is_bf16_supported()),
        }
    except ImportError as exc:
        blockers.append(f"PyTorch is unavailable: {exc}")
        gpu = {"cuda_available": False}

    try:
        nvidia_smi = (
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
            .splitlines()
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        blockers.append(f"nvidia-smi failed: {exc}")
        nvidia_smi = []

    deadline = datetime.fromisoformat(deadline_utc.replace("Z", "+00:00"))
    remaining = max(0.0, (deadline - datetime.now(timezone.utc)).total_seconds())
    if remaining <= 0:
        blockers.append("The configured A100 shutdown deadline has passed")

    packages = {
        name: _version(name)
        for name in ("torch", "scgpt", "datasets", "numpy", "anndata", "scanpy", "pyarrow")
    }
    for required in ("scgpt", "datasets", "PyYAML"):
        if _version(required) is None:
            blockers.append(f"Required package is not installed: {required}")
    if _version("scgpt") is not None:
        try:
            from scgpt.model import TransformerModel  # type: ignore[import-not-found]
            from scgpt.tokenizer.gene_tokenizer import (  # type: ignore[import-not-found]
                GeneVocab,
            )

            del TransformerModel, GeneVocab
        except (ImportError, OSError, RuntimeError) as exc:
            blockers.append(f"Official scGPT model imports failed: {exc}")

    return {
        "status": "passed" if not blockers else "blocked",
        "blockers": blockers,
        "warnings": warnings,
        "config": str(config_path),
        "data_source": data_source,
        "assets": asset_paths,
        "local_dataset_files": local_data,
        "scratch": {
            "path": str(scratch),
            "free_bytes": disk.free,
            "total_bytes": disk.total,
            "caches": caches,
        },
        "gpu": gpu,
        "nvidia_smi": nvidia_smi,
        "packages": packages,
        "deadline_utc": deadline.isoformat(),
        "remaining_seconds": remaining,
        "python": sys.version,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/model_shared_gpu.yaml"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--scratch", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-utc", default="2026-08-18T06:41:00Z")
    args = parser.parse_args(argv)
    scratch = args.scratch or Path(
        os.environ.get("GBM_A100_SCRATCH", f"/tmp/gbm-a100-{os.environ.get('USER', 'researcher')}")
    )
    try:
        result = inspect(
            args.config.resolve(), args.repo.resolve(), scratch.resolve(), args.deadline_utc
        )
    except (OSError, ValueError, TypeError) as exc:
        result = {"status": "blocked", "blockers": [str(exc)]}
    write_json_with_explanation(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
