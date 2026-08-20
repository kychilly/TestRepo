#!/usr/bin/env python3
"""Create candidate-aligned Stage 3 records without inventing protein scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from gbm_study.plain_english import write_jsonl_explanation


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=Path("data/pilot/internal_candidate_universe.jsonl"))
    parser.add_argument("--mutations", type=Path, default=Path("data/import_20260820/TP53 Dataset(preprocessed)/pilot/patient_gene_mutation_join.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/pilot/stage34_combined_records.jsonl"))
    args = parser.parse_args(argv)
    candidate_rows = [json.loads(line) for line in args.candidates.read_text().splitlines() if line.strip()]
    by_gene: dict[str, set[str]] = {}
    with args.mutations.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            by_gene.setdefault(str(row["gene_symbol"]).upper(), set()).add(str(row["alteration_type"]))
    output_rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidate_rows, 1):
        gene = str(candidate["gene"]).upper()
        types = by_gene.get(gene, set())
        alteration = "missense" if "missense" in types else "amplification" if "amplification" in types else "deletion" if "deletion" in types else "other"
        output_rows.append({
            "gene": gene,
            "candidate_id": f"stage34-{index:05d}-{gene}",
            "mutation": "raw_TCGA_join_present" if types else "no_TCGA_pilot_call",
            "alteration_type": alteration,
            "plddt": None,
            "esm1b": None,
            "ddg": None,
            "evidence_source": str(args.mutations),
            "evidence_sha256": sha256(args.mutations),
            "protein_evidence_status": "not_present_in_supplied_ZIP",
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows), encoding="utf-8")
    write_jsonl_explanation(args.output, row_count=len(output_rows), description="Candidate-aligned Stage 3 records built from the raw TCGA-derived join. Protein scores are deliberately null when not supplied.", next_actions=["Add AlphaFold pLDDT and independent protein-effect evidence, then rerun Stage 3/4."])
    print(json.dumps({"status": "completed", "candidate_count": len(output_rows), "output": str(args.output), "candidates_sha256": sha256(args.candidates), "mutation_join_sha256": sha256(args.mutations)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
