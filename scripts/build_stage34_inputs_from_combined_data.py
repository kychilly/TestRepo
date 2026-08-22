#!/usr/bin/env python3
"""Create candidate-aligned Stage 3 records without inventing protein scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from gbm_study.plain_english import write_jsonl_explanation


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _protein_evidence(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load one preselected variant-level protein record per candidate gene."""
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"Protein evidence file does not exist: {path}")
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    required = {"gene", "mutation", "plddt", "esm1b", "ddg"}
    by_gene: dict[str, dict[str, Any]] = {}
    for number, row in enumerate(rows, 1):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"Protein evidence row {number} lacks: {', '.join(missing)}")
        gene = str(row["gene"]).strip().upper()
        if not gene:
            raise ValueError(f"Protein evidence row {number} has an empty gene")
        if gene in by_gene:
            raise ValueError(
                f"Protein evidence has multiple rows for {gene}; preselect one variant "
                "with a documented rule before running the gene-level validator"
            )
        parsed = dict(row)
        parsed["gene"] = gene
        for field in ("plddt", "esm1b", "ddg"):
            value = row.get(field)
            parsed[field] = None if value in (None, "", "NA", "NaN", "nan") else float(value)
            if parsed[field] is not None and not math.isfinite(parsed[field]):
                raise ValueError(f"Protein evidence row {number} has non-finite {field}")
        if parsed["plddt"] is not None and not 0.0 <= parsed["plddt"] <= 100.0:
            raise ValueError(f"Protein evidence row {number} has pLDDT outside [0, 100]")
        alteration_type = str(parsed.get("alteration_type", "missense")).strip().lower()
        if alteration_type != "missense":
            raise ValueError(
                f"Protein evidence row {number} is {alteration_type!r}; protein scores "
                "may only be attached to a missense variant"
            )
        parsed["alteration_type"] = alteration_type
        by_gene[gene] = parsed
    return by_gene


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates", type=Path, default=Path("data/pilot/internal_candidate_universe.jsonl")
    )
    parser.add_argument(
        "--pool-candidates",
        type=Path,
        action="append",
        default=[],
        help="Additional-cancer candidate JSONL; may be repeated",
    )
    parser.add_argument(
        "--mutations",
        type=Path,
        default=Path(
            "data/import_20260820/TP53 Dataset(preprocessed)/pilot/patient_gene_mutation_join.csv"
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/pilot/stage34_combined_records.jsonl")
    )
    parser.add_argument(
        "--protein-evidence",
        type=Path,
        help=(
            "CSV/JSONL with exactly one preselected variant per gene and columns "
            "gene,mutation,plddt,esm1b,ddg; optional alteration_type/evidence_source"
        ),
    )
    args = parser.parse_args(argv)
    primary_rows = [
        json.loads(line) for line in args.candidates.read_text().splitlines() if line.strip()
    ]
    pooled_rows = [
        json.loads(line)
        for path in args.pool_candidates
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    primary_genes = {str(row["gene"]).upper() for row in primary_rows}
    pooled_genes = [str(row["gene"]).upper() for row in pooled_rows]
    pooled_gene_counts = Counter(pooled_genes)
    duplicate_pool_genes = sorted(
        primary_genes.intersection(pooled_genes)
        | {gene for gene, count in pooled_gene_counts.items() if count > 1}
    )
    if duplicate_pool_genes:
        raise ValueError(
            "Pooled candidate genes must be unique and absent from the primary universe: "
            + ", ".join(duplicate_pool_genes[:20])
        )
    candidate_rows = primary_rows + pooled_rows
    by_gene: dict[str, set[str]] = {}
    with args.mutations.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            by_gene.setdefault(str(row["gene_symbol"]).upper(), set()).add(
                str(row["alteration_type"])
            )
    protein = _protein_evidence(args.protein_evidence)
    unused_protein_genes = sorted(
        set(protein) - {str(row["gene"]).upper() for row in candidate_rows}
    )
    if unused_protein_genes:
        raise ValueError(
            "Protein evidence contains genes outside the frozen candidate universe: "
            + ", ".join(unused_protein_genes[:20])
        )
    output_rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidate_rows, 1):
        gene = str(candidate["gene"]).upper()
        types = by_gene.get(gene, set())
        evidence = protein.get(gene)
        alteration = (
            str(evidence.get("alteration_type", "missense"))
            if evidence is not None
            else "missense"
            if "missense" in types
            else "amplification"
            if "amplification" in types
            else "deletion"
            if "deletion" in types
            else "other"
        )
        output_rows.append(
            {
                "gene": gene,
                "candidate_id": f"stage34-{index:05d}-{gene}",
                "mutation": (
                    str(evidence["mutation"])
                    if evidence is not None
                    else "raw_TCGA_join_present"
                    if types
                    else "no_TCGA_pilot_call"
                ),
                "alteration_type": alteration,
                "plddt": evidence["plddt"] if evidence is not None else None,
                "esm1b": evidence["esm1b"] if evidence is not None else None,
                "ddg": evidence["ddg"] if evidence is not None else None,
                "evidence_source": (
                    str(evidence.get("evidence_source", args.protein_evidence))
                    if evidence is not None
                    else str(args.mutations)
                ),
                "evidence_sha256": (
                    sha256(args.protein_evidence)
                    if evidence is not None and args.protein_evidence is not None
                    else sha256(args.mutations)
                ),
                "protein_evidence_status": "supplied" if evidence is not None else "missing",
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows), encoding="utf-8"
    )
    write_jsonl_explanation(
        args.output,
        row_count=len(output_rows),
        description="Candidate-aligned Stage 3 records built from the raw TCGA-derived join. Protein scores are deliberately null when not supplied.",
        next_actions=[
            "Add AlphaFold pLDDT and independent protein-effect evidence, then rerun Stage 3/4."
        ],
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "candidate_count": len(output_rows),
                "primary_candidate_count": len(primary_rows),
                "pooled_candidate_count": len(pooled_rows),
                "output": str(args.output),
                "candidates_sha256": sha256(args.candidates),
                "mutation_join_sha256": sha256(args.mutations),
                "protein_evidence_rows": len(protein),
                "protein_evidence_sha256": (
                    sha256(args.protein_evidence) if args.protein_evidence is not None else None
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
