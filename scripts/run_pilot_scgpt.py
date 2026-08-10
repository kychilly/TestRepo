#!/usr/bin/env python3
"""Run the pilot candidate producer or emit an explicit prerequisite blocker."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from models.candidate_generation import build_candidate_records
from models.scgpt_adapter import AdapterError

INTERPRETATION_LABEL = "candidate/suspect gene ranking — not a confirmed driver"


def _yaml(path: Path) -> dict[str, Any]:
    import yaml  # type: ignore[import-untyped]

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AdapterError("Pilot configuration must be a YAML object")
    return value


def _blocked(config: Mapping[str, Any], reason: str) -> dict[str, Any]:
    return {
        "status": "blocked",
        "reason": reason,
        "interpretation_label": INTERPRETATION_LABEL,
        "provenance": {"checkpoint_sha256": None, "vocabulary_sha256": None, "config_sha256": None},
    }


def build_from_scores(
    config: Mapping[str, Any],
    scores: list[Mapping[str, Any]],
    metadata: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    records = build_candidate_records(
        scores,
        run_id=str(config["run_id"]),
        backbone="scGPT",
        checkpoint_hash=str(config["checkpoint_hash"]),
        vocabulary_hash=str(config["vocabulary_hash"]),
        cohort=str(config["cohort"]),
        fold=int(config.get("fold", 0)),
        seed=int(config["seed"]),
        split_hash=str(config["split_hash"]),
        patient_scope="train_only",
        patient_ids=[str(value) for value in config["training_patient_ids"]],
        state=str(config["state"]),
        score_method=str(config.get("score_method", "mask_delta_logit")),
        config_hash=str(config["config_hash"]),
        gene_metadata=metadata,
        n_cells=int(config["n_cells"]),
        created_at=str(config.get("created_at", "2026-08-09T00:00:00Z")),
    )
    return {
        "status": "completed",
        "interpretation_label": INTERPRETATION_LABEL,
        "candidates": [record.to_dict() for record in records],
    }


def run_pilot(config: dict[str, Any]) -> dict[str, Any]:
    required = (
        "cell_data_path",
        "split_file",
        "checkpoint_path",
        "vocabulary_path",
        "gene_id_type",
    )
    missing = [key for key in required if not config.get(key)]
    if missing:
        return _blocked(config, "Missing real pilot inputs: " + ", ".join(missing))
    paths = [
        Path(str(config[key]))
        for key in ("cell_data_path", "split_file", "checkpoint_path", "vocabulary_path")
    ]
    absent = [str(path) for path in paths if not path.is_file()]
    if absent:
        return _blocked(config, "Configured pilot asset does not exist: " + ", ".join(absent))
    if str(config.get("requested_device", "cuda")) != "cuda":
        return _blocked(config, "CUDA device is required for the real pilot run")
    try:
        import torch
    except ImportError:
        return _blocked(config, "PyTorch is required for the real pilot run")
    if not torch.cuda.is_available():
        return _blocked(config, "CUDA GPU is unavailable for the real pilot run")
    return _blocked(config, "No verified scGPT checkpoint/model loader contract is configured")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/pilot.yaml"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config = _yaml(args.config)
    try:
        result = run_pilot(config)
    except (AdapterError, OSError, ValueError, KeyError) as exc:
        result = _blocked(config, str(exc))
    output = args.output or Path(str(config.get("output_path", "results/compute/pilot_scgpt.json")))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(
        json.dumps(result, indent=2, sort_keys=True),
        file=sys.stderr if result["status"] == "blocked" else sys.stdout,
    )
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
