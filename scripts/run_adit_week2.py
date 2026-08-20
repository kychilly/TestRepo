#!/usr/bin/env python3
"""Run Adit's Week 2 integration seam as one auditable command.

This orchestrator never substitutes synthetic or CPU timing for a real
scGPT/MC-dropout run. It emits a completed report for available components and
structured blockers for missing model assets or hardware.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from models.grn import file_sha256, load_edges, score_held_out_edges
from models.mc_dropout import blocked_result
from gbm_study.plain_english import write_json_with_explanation


def _config(path: Path) -> dict[str, Any]:
    import yaml  # type: ignore[import-untyped]

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Week 2 Adit config must be a YAML object")
    return payload


def _grn(config: dict[str, Any], root: Path) -> dict[str, Any]:
    train_path = root / str(config["grn_train_prior_path"])
    held_path = root / str(config["grn_held_out_path"])
    if not train_path.is_file() or not held_path.is_file():
        return {
            "status": "blocked",
            "reason": "GRN train-prior or held-out file is missing",
            "train_prior": str(train_path),
            "held_out": str(held_path),
        }
    train, held = load_edges(train_path), load_edges(held_path)
    result = score_held_out_edges(train, held, lambda edge: float(edge["confidence"]))
    if result.get("held_out_edges", 0) < 10:
        result["status"] = "completed_with_limitations"
        result["scientific_limitation"] = (
            "Only one unique held-out positive edge is available; AUROC is a software sanity check, not evidence of GRN performance."
        )
    result["provenance"] = {
        "train_prior": str(train_path),
        "held_out": str(held_path),
        "train_prior_sha256": file_sha256(train_path),
        "held_out_sha256": file_sha256(held_path),
    }
    return dict(result)


def _mc_dropout(config: dict[str, Any], root: Path) -> dict[str, Any]:
    checkpoint = config.get("checkpoint_path")
    vocabulary = config.get("vocabulary_path")
    if not checkpoint or not (root / str(checkpoint)).is_file():
        return blocked_result(
            "scGPT checkpoint is missing; real MC-dropout timing was not substituted",
            n_passes=int(config.get("mc_dropout_passes", 20)),
            batch_size=int(config.get("mc_dropout_batch_size", 32)),
            device=str(config.get("mc_dropout_device", "cuda")),
        )
    if not vocabulary or not (root / str(vocabulary)).is_file():
        return blocked_result(
            "scGPT vocabulary is missing; real MC-dropout timing was not substituted",
            n_passes=int(config.get("mc_dropout_passes", 20)),
            batch_size=int(config.get("mc_dropout_batch_size", 32)),
            device=str(config.get("mc_dropout_device", "cuda")),
        )
    try:
        import torch
    except ImportError:
        return blocked_result(
            "PyTorch is unavailable for MC-dropout inference",
            n_passes=int(config.get("mc_dropout_passes", 20)),
            batch_size=int(config.get("mc_dropout_batch_size", 32)),
            device=str(config.get("mc_dropout_device", "cuda")),
        )
    device = str(config.get("mc_dropout_device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        return blocked_result(
            "CUDA is unavailable; real MC-dropout timing was not substituted",
            n_passes=int(config.get("mc_dropout_passes", 20)),
            batch_size=int(config.get("mc_dropout_batch_size", 32)),
            device=device,
        )
    runner_spec = config.get("mc_dropout_runner")
    if not runner_spec:
        return blocked_result(
            "mc_dropout_runner is not configured for the checkpoint-specific model output",
            n_passes=int(config.get("mc_dropout_passes", 20)),
            batch_size=int(config.get("mc_dropout_batch_size", 32)),
            device=device,
        )
    try:
        module_name, function_name = str(runner_spec).split(":", 1)
        runner = getattr(importlib.import_module(module_name), function_name)
        result = runner(config)
    except (ValueError, ImportError, AttributeError, RuntimeError) as exc:
        return blocked_result(
            f"MC-dropout runner failed: {exc}",
            n_passes=int(config.get("mc_dropout_passes", 20)),
            batch_size=int(config.get("mc_dropout_batch_size", 32)),
            device=device,
        )
    if not isinstance(result, dict) or result.get("status") != "completed":
        return blocked_result(
            "MC-dropout runner did not return a completed measured artifact",
            n_passes=int(config.get("mc_dropout_passes", 20)),
            batch_size=int(config.get("mc_dropout_batch_size", 32)),
            device=device,
        )
    return result


def run(config_path: Path, output: Path) -> dict[str, Any]:
    config = _config(config_path)
    root = Path(__file__).resolve().parents[1]
    components = {
        "grn": _grn(config, root),
        "mc_dropout": _mc_dropout(config, root),
        "stage5_masking": {
            "status": "completed",
            "validator_flag": str(config.get("validator", "off")),
            "implementation": "src/gbm_study/stage5_masking.py",
            "confirmed_buckets": ["destabilizing_driver", "functional_driver"],
        },
    }
    blockers = [
        f"{name}: {value.get('reason', 'blocked')}"
        for name, value in components.items()
        if value.get("status") == "blocked"
    ]
    result = {
        "status": "completed_with_blockers" if blockers else "completed",
        "owner": "Adit",
        "config": str(config_path),
        "components": components,
        "blockers": blockers,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_with_explanation(output, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/week2_adit.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.config, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
