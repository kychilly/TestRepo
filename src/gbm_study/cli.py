"""Command-line entry points for Week 1 study validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import ConfigurationError, StudyConfig
from .leakage import LeakageError, normalize_split_keys
from .provenance import atomic_write_json, run_metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gbm-study")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("validate-config", "validate-split"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", type=Path, required=True)
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Cannot read JSON input {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigurationError(f"JSON input {path} must contain an object")
    return payload


def _validate_split(config: StudyConfig) -> dict[str, Any]:
    config.require_run_inputs()
    assert config.split_file is not None
    payload = _load_json(config.split_file)
    splits = normalize_split_keys(payload)
    return {
        "split_schema": "patient_ids_v1",
        "patient_counts": {name: len(ids) for name, ids in splits.items()},
    }


def _run(command: str, config_path: Path) -> int:
    config = StudyConfig.from_json(config_path)
    result: dict[str, Any] = {"command": command, "status": "passed"}
    if command == "validate-split":
        result.update(_validate_split(config))
    else:
        config.require_run_inputs()
        result["status"] = "configuration_valid"

    assert config.data_manifest is not None
    assert config.split_file is not None
    assert config.vocabulary is not None
    root = Path(__file__).resolve().parents[2]
    result["provenance"] = run_metadata(
        root=root,
        config_path=config_path,
        manifest=config.data_manifest,
        split=config.split_file,
        vocabulary=config.vocabulary,
        seed=config.seed,
    )
    output_path = config.output_dir / f"{command}.json"
    atomic_write_json(output_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run a validation command and emit structured errors on stderr."""
    args = _parser().parse_args(argv)
    try:
        return _run(args.command, args.config)
    except (ConfigurationError, LeakageError, OSError) as exc:
        print(
            json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
