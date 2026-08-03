#!/usr/bin/env python3
"""Validate versioned records and create explicit validator inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from schemas.records import ContractError, build_validator_inputs, read_jsonl, write_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--variants", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validator-config-version", required=True)
    parser.add_argument("--aliases", type=Path)
    parser.add_argument("--forbidden-patients", type=Path)
    args = parser.parse_args(argv)
    try:
        aliases = json.loads(args.aliases.read_text(encoding="utf-8")) if args.aliases else None
        forbidden = (
            json.loads(args.forbidden_patients.read_text(encoding="utf-8"))
            if args.forbidden_patients
            else []
        )
        inputs = build_validator_inputs(
            read_jsonl(args.candidates),
            read_jsonl(args.variants),
            args.validator_config_version,
            aliases=aliases,
            forbidden_patient_ids=forbidden,
        )
        write_jsonl(args.output, [item.to_dict() for item in inputs])
    except (ContractError, OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {"status": "completed", "inputs_written": len(inputs), "output": str(args.output)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
