#!/usr/bin/env python3
"""Run the validated evaluation pathway; never import a model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evaluation.metrics import EvaluationError
from evaluation.reporting import run_evaluation


def main(argv: list[str] | None = None) -> int:
    """Validate inputs, write evaluation artifacts, and return a loud failure code."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/evaluation.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
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
