"""Synthetic AlphaFold evidence-shape test; no real evidence is downloaded.

This deliberately tests only the future data boundary. It does not implement
Ishaan's validator, choose thresholds, register cohorts, or compute pLDDT.
"""

from __future__ import annotations


def synthetic_alphafold_record(gene: str) -> dict[str, object]:
    return {
        "gene": gene,
        "protein_accession": f"AF-{gene}-F1-model_v4",
        "source": "AlphaFold Protein Structure Database",
        "source_version": "synthetic-fixture",
        "residue_numbers": [1, 2, 3],
        "plddt": [91.2, 87.4, 72.0],
    }


def test_alphafold_fixture_preserves_per_residue_confidence_and_provenance() -> None:
    records = [
        synthetic_alphafold_record(gene) for gene in ("TP53", "IDH1", "EGFR", "RPRM")
    ]
    for record in records:
        assert record["source"] == "AlphaFold Protein Structure Database"
        assert record["source_version"]
        assert len(record["residue_numbers"]) == len(record["plddt"])
        assert all(0.0 <= value <= 100.0 for value in record["plddt"])
