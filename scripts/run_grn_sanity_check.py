#!/usr/bin/env python3
"""Run held-out GRN recovery, or preserve a structured missing-edge-list blocker."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from models.grn import load_edges, run_sanity_check, score_held_out_edges
from gbm_study.plain_english import write_json_with_explanation
from schemas.records import ContractError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/model.yaml"))
    parser.add_argument("--edge-list", type=Path, help="Override config grn_edge_list_path")
    parser.add_argument("--train-prior", type=Path, help="Explicit train-prior CSV/JSONL")
    parser.add_argument("--held-out", type=Path, help="Explicit held-out CSV/JSONL")
    parser.add_argument("--held-out-fraction", type=float, default=0.2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    import yaml  # type: ignore[import-untyped]

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if bool(args.train_prior) != bool(args.held_out):
        parser.error("--train-prior and --held-out must be supplied together")
    configured_path = args.edge_list or (
        Path(str(config["grn_edge_list_path"]))
        if isinstance(config, dict) and config.get("grn_edge_list_path")
        else None
    )
    result: dict[str, Any]
    if args.train_prior and args.held_out:
        try:
            train_path = args.train_prior
            held_path = args.held_out
            if not train_path.is_file() or not held_path.is_file():
                result = {
                    "status": "blocked",
                    "reason": "Explicit GRN train/holdout file is missing",
                }
            else:
                train, held = load_edges(train_path), load_edges(held_path)
                result = score_held_out_edges(
                    train, held, lambda edge: float(edge.get("confidence", 0.0))
                )
                result["provenance"] = {
                    "train_prior": str(train_path),
                    "held_out": str(held_path),
                    "train_prior_sha256": __import__(
                        "models.grn", fromlist=["file_sha256"]
                    ).file_sha256(train_path),
                    "held_out_sha256": __import__(
                        "models.grn", fromlist=["file_sha256"]
                    ).file_sha256(held_path),
                }
        except ContractError as exc:
            result = {"status": "failed", "reason": str(exc)}
    elif configured_path is None:
        result = {
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
    write_json_with_explanation(args.output, result)
    print(
        json.dumps(result, indent=2, sort_keys=True),
        file=sys.stderr if result["status"] == "blocked" else sys.stdout,
    )
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
