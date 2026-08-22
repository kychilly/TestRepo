#!/usr/bin/env python3
"""Validate scGPT assets and runtime capabilities for a reproducible run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from gbm_study.plain_english import write_json_with_explanation


class EnvironmentError(RuntimeError):
    """Raised for a required environment or model-contract violation."""


def sha256_file(path: Path) -> str:
    """Hash a file in bounded-memory chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML configuration using PyYAML, with a clear dependency error."""
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise EnvironmentError("PyYAML is required to load model.yaml") from exc
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise EnvironmentError(f"Cannot read configuration {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EnvironmentError("Model configuration must be a YAML object")
    return payload


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _torch_report() -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        return {"installed": False, "error": str(exc)}
    cuda_available = bool(torch.cuda.is_available())
    devices: list[dict[str, Any]] = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
            }
        )
    return {
        "installed": True,
        "version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),  # type: ignore[no-untyped-call]
        "cuda_available": cuda_available,
        "gpu_count": torch.cuda.device_count(),
        "devices": devices,
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "mixed_precision_capability": {
            "float16": cuda_available,
            "bfloat16": cuda_available and torch.cuda.is_bf16_supported(),
        },
    }


def _load_vocabulary(path: Path) -> dict[str, int]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EnvironmentError(f"Vocabulary must be a JSON gene-to-token mapping: {exc}") from exc
    if not isinstance(payload, dict) or not all(
        isinstance(k, str) and isinstance(v, int) for k, v in payload.items()
    ):
        raise EnvironmentError("Vocabulary must map string identifiers to integer token IDs")
    return payload


def _checkpoint_shapes(path: Path) -> dict[str, list[int]]:
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise EnvironmentError(f"Cannot load checkpoint {path}: {exc}") from exc
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise EnvironmentError("Checkpoint does not contain a state-dict mapping")
    return {
        str(name): [int(size) for size in value.shape]
        for name, value in state.items()
        if hasattr(value, "shape")
    }


def inspect_environment(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    """Return a JSON-compatible report and raise on non-negotiable failures."""
    requested_device = str(config.get("requested_device", "cpu"))
    checkpoint_value = config.get("checkpoint_path")
    vocabulary_value = config.get("vocabulary_path")
    checkpoint = Path(str(checkpoint_value)) if checkpoint_value else None
    vocabulary = Path(str(vocabulary_value)) if vocabulary_value else None
    report: dict[str, Any] = {
        "status": "passed",
        "config_path": str(config_path),
        "operating_system": platform.platform(),
        "python_version": platform.python_version(),
        "package_versions": {
            name: _version(name) for name in ("torch", "scgpt", "scanpy", "anndata")
        },
        "checkpoint": None,
        "vocabulary": None,
        "torch": _torch_report(),
    }
    failures: list[str] = []
    if requested_device == "cuda" and not report["torch"].get("cuda_available", False):
        failures.append("CUDA was requested but is unavailable")
    if checkpoint is None or not checkpoint.is_file():
        failures.append(f"checkpoint is missing: {checkpoint_value!r}")
    else:
        report["checkpoint"] = {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "tensor_shapes": _checkpoint_shapes(checkpoint),
        }
    if vocabulary is None or not vocabulary.is_file():
        failures.append(f"vocabulary is missing: {vocabulary_value!r}")
    else:
        vocabulary_mapping = _load_vocabulary(vocabulary)
        required_tokens = [str(token) for token in config.get("required_special_tokens", [])]
        missing_tokens = [token for token in required_tokens if token not in vocabulary_mapping]
        report["vocabulary"] = {
            "path": str(vocabulary),
            "sha256": sha256_file(vocabulary),
            "size": len(vocabulary_mapping),
            "missing_special_tokens": missing_tokens,
        }
        if missing_tokens:
            failures.append(f"required vocabulary tokens are absent: {missing_tokens}")
    expected_shapes = config.get("expected_tensor_shapes", {})
    if report["checkpoint"] and isinstance(expected_shapes, dict):
        actual_shapes = report["checkpoint"]["tensor_shapes"]
        incompatible = {
            name: [expected_shapes[name], actual_shapes.get(name)]
            for name in expected_shapes
            if actual_shapes.get(name) != expected_shapes[name]
        }
        report["checkpoint"]["incompatible_shapes"] = incompatible
        if incompatible:
            failures.append(f"incompatible checkpoint tensor shapes: {sorted(incompatible)}")
    report["scgpt"] = {
        "version": _version("scgpt"),
        "git_commit": config.get("scgpt_git_commit"),
    }
    report["determinism"] = {
        "seed": config.get("seed"),
        "requested_device": requested_device,
    }
    if failures:
        report["status"] = "failed"
        report["failures"] = failures
        raise EnvironmentError(json.dumps(report, sort_keys=True))
    return report


def _export_environment(path: Path) -> None:
    """Persist the complete pip freeze and host metadata after successful setup."""
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    payload = {
        "status": "completed",
        "python": sys.version,
        "platform": platform.platform(),
        "pip_freeze": sorted(freeze),
        "environment": dict(os.environ).get("CONDA_PREFIX") or "venv/system",
    }
    write_json_with_explanation(path, payload)


def main(argv: list[str] | None = None) -> int:
    """Validate the configured environment and optionally export it."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/model.yaml"))
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--environment-export", type=Path)
    args = parser.parse_args(argv)
    try:
        report = inspect_environment(load_config(args.config), args.config)
    except EnvironmentError as exc:
        try:
            report = json.loads(str(exc))
        except json.JSONDecodeError:
            report = {"status": "failed", "error": str(exc)}
        if args.json_out:
            write_json_with_explanation(args.json_out, report)
        print(json.dumps(report, indent=2, sort_keys=True), file=sys.stderr)
        return 2
    if args.environment_export:
        _export_environment(args.environment_export)
        report["environment_export"] = str(args.environment_export)
    if args.json_out:
        write_json_with_explanation(args.json_out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
