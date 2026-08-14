#!/usr/bin/env python3
"""Run held-out GRN recovery, or preserve a structured missing-edge-list blocker."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from models.grn import run_sanity_check
from schemas.records import ContractError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/model.yaml"))
    parser.add_argument("--edge-list", type=Path, help="Override config grn_edge_list_path")
    parser.add_argument("--held-out-fraction", type=float, default=0.2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    import yaml  # type: ignore[import-untyped]

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    configured_path = args.edge_list or (
        Path(str(config["grn_edge_list_path"]))
        if isinstance(config, dict) and config.get("grn_edge_list_path")
        else None
    )
    if configured_path is None:
        result: dict[str, Any] = {
            "status": "blocked",
            "reason": "No real GRN edge list is configured (Data Lead delivery pending)",
        }
    else:
        path = configured_path
        if not path.is_file():
            result = {
                "status": "blocked",
                "reason": f"Configured GRN edge list does not exist: {path}",
            }
        else:
            try:
                result = run_sanity_check(
                    path,
                    seed=int(config.get("seed", 17)) if isinstance(config, dict) else 17,
                    score_fn=lambda edge: float(edge.get("confidence", 0.0)),
                    held_out_fraction=args.held_out_fraction,
                )
            except ContractError as exc:
                result = {"status": "failed", "reason": str(exc)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(result, indent=2, sort_keys=True),
        file=sys.stderr if result["status"] == "blocked" else sys.stdout,
    )
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
