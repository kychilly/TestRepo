"""Patient-aware conversion of scGPT mask perturbations into gene scores.

The model-specific adapter must provide state logits for an unmasked input and
the same input with one gene masked. This module defines the reproducible
aggregation step; it never invents scores from cell embeddings.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Mapping

from schemas.records import ContractError, normalize_gene_symbol


def aggregate_mask_delta_scores(
    rows: Iterable[Mapping[str, Any]],
    *,
    state: str,
    aliases: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Aggregate ``baseline_logit - masked_logit`` by patient, then gene.

    Cells are first averaged within each patient so donors with more cells
    cannot dominate the candidate ranking. ``score_sd`` is the sample standard
    deviation across patient-level scores; one patient has SD zero.
    """
    patient_gene: dict[tuple[str, str], list[float]] = defaultdict(list)
    seen_cells: set[tuple[str, str, str]] = set()
    for row in rows:
        gene = normalize_gene_symbol(str(row.get("gene", "")), aliases)
        patient = str(row.get("patient_id", ""))
        cell = str(row.get("cell_id", ""))
        row_state = str(row.get("state", state))
        if row_state != state:
            raise ContractError(f"Mixed state rows: expected {state}, found {row_state}")
        if not patient or not cell:
            raise ContractError("Each mask score requires patient_id and cell_id")
        key = (patient, cell, gene)
        if key in seen_cells:
            raise ContractError(f"Duplicate mask score for {key}")
        seen_cells.add(key)
        try:
            baseline = float(row["baseline_logit"])
            masked = float(row["masked_logit"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(
                "Each mask score requires numeric baseline_logit and masked_logit"
            ) from exc
        delta = baseline - masked
        if not math.isfinite(delta):
            raise ContractError(f"Non-finite mask delta for {gene}")
        patient_gene[(patient, gene)].append(delta)

    by_gene: dict[str, list[float]] = defaultdict(list)
    for (patient, gene), deltas in sorted(patient_gene.items()):
        by_gene[gene].append(sum(deltas) / len(deltas))

    scored: list[dict[str, Any]] = []
    for gene, patient_scores in sorted(by_gene.items()):
        mean = sum(patient_scores) / len(patient_scores)
        variance = (
            sum((value - mean) ** 2 for value in patient_scores) / (len(patient_scores) - 1)
            if len(patient_scores) > 1
            else 0.0
        )
        scored.append(
            {
                "gene": gene,
                "score": mean,
                "score_sd": math.sqrt(variance),
                "n_cells": sum(
                    len(patient_gene[(patient, gene)])
                    for patient in {p for p, g in patient_gene if g == gene}
                ),
                "n_patients": len(patient_scores),
            }
        )
    return sorted(scored, key=lambda row: (-row["score"], row["gene"]))
