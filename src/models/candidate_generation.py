"""Convert scGPT gene scores into validated candidate-gene records."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from schemas.records import CandidateGene, ContractError, normalize_gene_symbol


def build_candidate_records(
    scored_genes: Iterable[Mapping[str, Any]],
    *,
    run_id: str,
    backbone: str,
    checkpoint_hash: str,
    vocabulary_hash: str,
    cohort: str,
    fold: int,
    seed: int,
    split_hash: str,
    patient_scope: str,
    patient_ids: Iterable[str],
    state: str,
    score_method: str,
    config_hash: str,
    gene_metadata: Mapping[str, Mapping[str, str]],
    n_cells: int,
    created_at: str | None = None,
    aliases: Mapping[str, str] | None = None,
) -> list[CandidateGene]:
    """Build and validate one-based per-state candidate rankings.

    ``scored_genes`` contains only expression-derived scores. Protein evidence
    is deliberately absent and is joined later by Ishaan's validator input
    builder.
    """
    if n_cells < 1:
        raise ContractError("n_cells must be positive")
    patients = sorted(set(patient_ids))
    if not patients:
        raise ContractError("patient_ids must be non-empty")
    timestamp = created_at or datetime.now(timezone.utc).isoformat()
    prepared: list[dict[str, Any]] = []
    for row in scored_genes:
        if "gene" not in row or "score" not in row:
            raise ContractError("Each scored gene requires gene and score")
        gene = normalize_gene_symbol(str(row["gene"]), aliases)
        score = float(row["score"])
        score_sd = float(row.get("score_sd", 0.0))
        if not math.isfinite(score) or not math.isfinite(score_sd) or score_sd < 0:
            raise ContractError(f"Invalid finite score/score_sd for {gene}")
        if gene not in gene_metadata:
            raise ContractError(f"Missing canonical metadata for {gene}")
        metadata = gene_metadata[gene]
        prepared.append(
            {"gene": gene, "score": score, "score_sd": score_sd, "metadata": metadata}
        )
    prepared.sort(key=lambda row: (-row["score"], row["gene"]))
    records: list[CandidateGene] = []
    for rank, row in enumerate(prepared, start=1):
        metadata = row["metadata"]
        payload = {
            "schema_version": "1.0.0",
            "candidate_id": f"{run_id}-{state}-{row['gene']}-rank{rank:03d}",
            "run_id": run_id,
            "created_at": timestamp,
            "backbone": backbone,
            "checkpoint_hash": checkpoint_hash,
            "vocabulary_hash": vocabulary_hash,
            "cohort": cohort,
            "fold": fold,
            "seed": seed,
            "split_hash": split_hash,
            "patient_scope": patient_scope,
            "contributing_patient_ids": patients,
            "state": state,
            "gene": row["gene"],
            "gene_namespace": metadata["gene_namespace"],
            "ensembl_gene_id": metadata["ensembl_gene_id"],
            "gene_id_version": metadata["gene_id_version"],
            "score_method": score_method,
            "score": row["score"],
            "score_sd": row["score_sd"],
            "rank": rank,
            "rank_scope": "run_state",
            "n_cells": int(row.get("n_cells", n_cells)),
            "n_patients": int(row.get("n_patients", len(patients))),
            "training_only": patient_scope == "train_only",
            "config_hash": config_hash,
        }
        records.append(CandidateGene.from_dict(payload, aliases=aliases))
    return records
