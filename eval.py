#!/usr/bin/env python3
"""Combined manuscript evaluator for cell states and patient-level IDH status."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from evaluation.reporting import run_evaluation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate cell-state macro-F1 and optional patient-level IDH AUROC."
    )
    parser.add_argument("--cell-predictions", type=Path)
    parser.add_argument("--idh-predictions", type=Path)
    # Backward-compatible single-task form.
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/evaluation.yaml"))
    parser.add_argument("--idh-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    cell_predictions = args.cell_predictions or args.predictions
    if cell_predictions is None and args.idh_predictions is None:
        parser.error("provide --cell-predictions/--predictions or --idh-predictions")
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        payload: dict[str, Any] = {"status": "completed", "tasks": {}}
        if cell_predictions is not None:
            cell_dir = args.output / "cell_state"
            cell_manifest = run_evaluation(cell_predictions, args.splits, args.config, cell_dir)
            cell_metrics = json.loads((cell_dir / "metrics.json").read_text(encoding="utf-8"))
            payload["tasks"]["cell_state"] = {
                "metrics": cell_metrics,
                "manifest": cell_manifest,
            }
        if args.idh_predictions is not None:
            idh_dir = args.output / "idh"
            idh_config = args.idh_config or Path("config/evaluation_idh.yaml")
            idh_manifest = run_evaluation(args.idh_predictions, args.splits, idh_config, idh_dir)
            idh_metrics = json.loads((idh_dir / "metrics.json").read_text(encoding="utf-8"))
            payload["tasks"]["idh"] = {
                "metrics": idh_metrics,
                "manifest": idh_manifest,
            }
        (args.output / "metrics.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (args.output / "evaluation_manifest.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "tasks": sorted(payload["tasks"]),
                    "cell_prediction_file": str(cell_predictions) if cell_predictions else None,
                    "idh_prediction_file": str(args.idh_predictions)
                    if args.idh_predictions
                    else None,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"status": "completed", "output": str(args.output)}, sort_keys=True))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        error = {"status": "failed", "error": str(exc)}
        (args.output / "evaluation_error.json").write_text(
            json.dumps(error, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(error, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
