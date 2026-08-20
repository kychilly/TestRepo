#!/usr/bin/env python3
"""Run Adit's Week 3 compute and uncertainty stages with fail-closed outputs."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from models.mc_dropout import blocked_result
from gbm_study.plain_english import write_json_with_explanation


def _yaml(path: Path) -> dict[str, Any]:
    import yaml  # type: ignore[import-untyped]

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Week 3 config must be a YAML object")
    return value


def _mc(config: dict[str, Any], root: Path) -> dict[str, Any]:
    passes = int(config.get("mc_dropout_passes", 20))
    batch = int(config.get("mc_dropout_batch_size", 32))
    device = str(config.get("mc_dropout_device", "cuda"))
    checkpoint = config.get("checkpoint_path")
    vocabulary = config.get("vocabulary_path")
    if not checkpoint or not (root / str(checkpoint)).is_file():
        return blocked_result("scGPT checkpoint is missing", n_passes=passes, batch_size=batch, device=device)
    if not vocabulary or not (root / str(vocabulary)).is_file():
        return blocked_result("scGPT vocabulary is missing", n_passes=passes, batch_size=batch, device=device)
    try:
        import torch
    except ImportError:
        return blocked_result("PyTorch is unavailable", n_passes=passes, batch_size=batch, device=device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        return blocked_result("CUDA is unavailable", n_passes=passes, batch_size=batch, device=device)
    spec = config.get("mc_dropout_runner")
    if not spec:
        return blocked_result("mc_dropout_runner is not configured", n_passes=passes, batch_size=batch, device=device)
    try:
        module_name, function_name = str(spec).split(":", 1)
        result = getattr(importlib.import_module(module_name), function_name)(config)
    except (ValueError, ImportError, AttributeError, RuntimeError) as exc:
        return blocked_result(f"MC-dropout runner failed: {exc}", n_passes=passes, batch_size=batch, device=device)
    if not isinstance(result, dict) or result.get("status") != "completed":
        return blocked_result("MC-dropout runner returned no measured completed artifact", n_passes=passes, batch_size=batch, device=device)
    return result


def run(model_config_path: Path, adit_config_path: Path, output: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    model_config = _yaml(model_config_path)
    adit_config = _yaml(adit_config_path)
    try:
        from benchmark_scgpt import run_benchmark

        benchmark = run_benchmark(model_config)
    except (ImportError, OSError, RuntimeError, ValueError, KeyError) as exc:
        benchmark = {"status": "blocked", "reason": f"Benchmark failed: {exc}"}
    mc = _mc({**model_config, **adit_config}, root)
    blockers = []
    for name, component in (("benchmark", benchmark), ("mc_dropout", mc)):
        if component.get("status") != "completed":
            blockers.append(f"{name}: {component.get('reason', 'blocked')}")
    result = {
        "status": "completed" if not blockers else "completed_with_blockers",
        "owner": "Adit",
        "week": 3,
        "components": {"scgpt_1000_cell_benchmark": benchmark, "mc_dropout": mc},
        "blockers": blockers,
        "required_evidence": [
            "1,000 real training-patient cells",
            "synchronized CUDA timing and projected_gpu_seconds_per_10000_cells",
            "20-50 active-dropout passes with per-gene mean and variance",
            "checkpoint, vocabulary, split, and environment hashes",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_with_explanation(output, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=Path, default=Path("config/model.yaml"))
    parser.add_argument("--adit-config", type=Path, default=Path("config/week2_adit.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.model_config, args.adit_config, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
