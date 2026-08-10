#!/usr/bin/env python3
"""Run the validated evaluation pathway; never import a model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evaluation.metrics import EvaluationError
from evaluation.reporting import run_evaluation
from gbm_study.stage5_masking import (
    PassthroughStubOutcomeSource,
    load_outcome_source,
    mask_candidates,
    source_provenance,
)
from schemas.records import read_jsonl, write_jsonl


def main(argv: list[str] | None = None) -> int:
    """Validate inputs, write evaluation artifacts, and return a loud failure code."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/evaluation.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validator", choices=("on", "off"), default="off")
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--validator-payloads", type=Path)
    parser.add_argument("--outcome-source-module")
    args = parser.parse_args(argv)
    try:
        if args.candidates is not None:
            if args.validator == "on" and args.validator_payloads is None:
                raise EvaluationError("--validator-payloads is required when --validator on")
            candidates = read_jsonl(args.candidates)
            payloads = (
                {str(item["input_id"]): item for item in read_jsonl(args.validator_payloads)}
                if args.validator_payloads
                else {}
            )
            if args.validator == "on" and not args.outcome_source_module:
                raise EvaluationError(
                    "--outcome-source-module is required when --validator on; "
                    "the test stub cannot be used for a scientific run"
                )
            source = (
                load_outcome_source(args.outcome_source_module)
                if args.outcome_source_module
                else PassthroughStubOutcomeSource()
            )
            masked, provenance = mask_candidates(
                candidates,
                payloads,
                validator=args.validator,
                outcome_source=source,
            )
            provenance["outcome_source_provenance"] = source_provenance(source)
            args.output.mkdir(parents=True, exist_ok=True)
            write_jsonl(args.output / "masked_candidates.jsonl", masked)
            (args.output / "validator_provenance.json").write_text(
                json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        manifest = run_evaluation(args.predictions, args.splits, args.config, args.output)
    except (EvaluationError, OSError, ValueError, ImportError) as exc:
        error = {"status": "failed", "error": str(exc)}
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "evaluation_error.json").write_text(
            json.dumps(error, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(error, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps({"status": "completed", "manifest": manifest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
