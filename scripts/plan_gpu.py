#!/usr/bin/env python3
"""Create a workload plan from real visible CUDA devices only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gbm_study.gpu_planner import GPUPlanningError, plan_cuda_work


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, default=10000)
    parser.add_argument("--token-length", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--precision",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--device-ids", type=int, nargs="*")
    parser.add_argument(
        "--output", type=Path, default=Path("baseline_results/compute/gpu_plan.json")
    )
    args = parser.parse_args(argv)
    try:
        plan = plan_cuda_work(
            cells=args.cells,
            token_length=args.token_length,
            batch_size=args.batch_size,
            precision=args.precision,
            device_ids=None if args.device_ids is None else tuple(args.device_ids),
        )
        result = plan.to_dict()
        code = 0
    except GPUPlanningError as exc:
        result = {
            "status": "blocked",
            "reason": str(exc),
            "measured_gpu_seconds_per_10000_cells": None,
        }
        code = 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
