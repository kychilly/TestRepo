#!/usr/bin/env python3
"""Convert scGPT mask-logit score rows into validated candidate-gene JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from models.candidate_generation import build_candidate_records
from models.candidate_scoring import aggregate_mask_delta_scores
from schemas.records import read_jsonl, write_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--gene-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state", choices=("AC", "MES", "NPC", "OPC"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--backbone", default="scGPT")
    parser.add_argument("--checkpoint-hash", required=True)
    parser.add_argument("--vocabulary-hash", required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split-hash", required=True)
    parser.add_argument("--patient-scope", default="train_only")
    parser.add_argument("--config-hash", required=True)
    parser.add_argument("--created-at")
    args = parser.parse_args(argv)
    rows = read_jsonl(args.scores)
    scored = aggregate_mask_delta_scores(rows, state=args.state)
    patients = sorted({str(row["patient_id"]) for row in rows})
    n_cells = len({str(row["cell_id"]) for row in rows})
    metadata = json.loads(args.gene_metadata.read_text(encoding="utf-8"))
    records = build_candidate_records(
        scored,
        run_id=args.run_id,
        backbone=args.backbone,
        checkpoint_hash=args.checkpoint_hash,
        vocabulary_hash=args.vocabulary_hash,
        cohort=args.cohort,
        fold=args.fold,
        seed=args.seed,
        split_hash=args.split_hash,
        patient_scope=args.patient_scope,
        patient_ids=patients,
        state=args.state,
        score_method="mask_delta_logit",
        config_hash=args.config_hash,
        gene_metadata=metadata,
        n_cells=n_cells,
        created_at=args.created_at,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, [record.to_dict() for record in records])
    print(
        json.dumps(
            {
                "status": "completed",
                "candidates": len(records),
                "output": str(args.output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
