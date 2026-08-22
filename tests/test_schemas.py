"""Contract tests for records, schema versions, and explicit candidate/variant joins."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from schemas.records import (
    CONTRACT_VERSION,
    ContractError,
    build_validator_inputs,
    normalize_gene_symbol,
    validate_candidate_gene,
)

EXAMPLE_DIR = Path(__file__).parents[1] / "examples"


def candidate(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": CONTRACT_VERSION,
        "candidate_id": "cand-EGFR-p1",
        "run_id": "run-1",
        "created_at": "2026-08-02T12:00:00Z",
        "backbone": "scGPT",
        "checkpoint_hash": "checkpoint",
        "vocabulary_hash": "vocab",
        "cohort": "Neftel",
        "fold": 0,
        "seed": 17,
        "split_hash": "split",
        "patient_scope": "train_only",
        "contributing_patient_ids": ["P001"],
        "state": "MES",
        "gene": "EGFR",
        "gene_namespace": "HGNC",
        "ensembl_gene_id": "ENSG00000146648",
        "gene_id_version": "Ensembl-110",
        "score_method": "mask_delta_logit",
        "score": 1.0,
        "score_sd": 0.1,
        "rank": 1,
        "rank_scope": "run_state",
        "n_cells": 10,
        "n_patients": 1,
        "training_only": True,
        "config_hash": "config",
    }
    value.update(overrides)
    return value


def variant(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": CONTRACT_VERSION,
        "variant_id": "var-1",
        "patient_id": "P001",
        "cohort": "Neftel",
        "gene_symbol": "EGFR",
        "ensembl_gene_id": "ENSG00000146648",
        "genome_build": "GRCh38",
        "chromosome": "chr7",
        "position": 55249071,
        "reference_allele": "T",
        "alternate_allele": "G",
        "alteration_type": "missense",
        "transcript_id": "ENST00000275493",
        "transcript_version": ".6",
        "protein_accession": "NP_005219.2",
        "protein_isoform": "1",
        "protein_change": "p.L858R",
        "variant_source": "synthetic",
        "variant_source_version": "1",
        "mapping_status": "resolved",
        "mapping_notes": "Synthetic MANE mapping.",
    }
    value.update(overrides)
    return value


def test_valid_missense_mapping_is_eligible() -> None:
    inputs = build_validator_inputs([candidate()], [variant()], "1.0.0")
    result = inputs[0].to_dict()
    assert result["join_cardinality"] == "one"
    assert result["validator_eligibility"] == "eligible_missense"
    assert result["protein_mapping"][0]["protein_isoform"] == "1"


def test_validator_payload_is_simplified_for_consumer() -> None:
    result = build_validator_inputs([candidate()], [variant()], "1.0.0")[0].to_validator_payload()
    assert set(result["candidate"]) == {"gene", "state", "score", "rank", "seed"}
    assert result["candidate"] == {
        "gene": "EGFR",
        "state": "MES",
        "score": 1.0,
        "rank": 1,
        "seed": 17,
    }
    assert "checkpoint_hash" not in result["candidate"]
    assert result["variants"][0]["variant_id"] == "var-1"


def test_protein_evidence_is_joined_by_variant_id_and_returned() -> None:
    evidence = {
        "schema_version": CONTRACT_VERSION,
        "evidence_id": "ev-1",
        "variant_id": "var-1",
        "gene": "EGFR",
        "protein_accession": "NP_005219.2",
        "evidence_status": "complete",
        "alphafold_source": "AlphaFold DB",
        "alphafold_version": "2024-01",
        "plddt_score": 92.0,
        "esm1b_source": "ESM1b",
        "esm1b_version": "1.0",
        "esm1b_score": -2.1,
        "stability_source": "precomputed",
        "stability_version": "1.0",
        "delta_delta_g": 1.7,
    }
    joined = build_validator_inputs(
        [candidate()], [variant()], "1.0.0", protein_evidence=[evidence]
    )[0]
    assert joined.to_dict()["protein_evidence"][0]["evidence_id"] == "ev-1"
    assert joined.to_validator_payload()["protein_evidence"][0]["variant_id"] == "var-1"


def test_non_missense_events_remain_and_abstain() -> None:
    amplification = variant(
        variant_id="var-amp",
        alteration_type="amplification",
        position=1,
        reference_allele="N",
        alternate_allele="A",
        transcript_id=None,
        transcript_version=None,
        protein_accession=None,
        protein_isoform=None,
        protein_change=None,
        mapping_status="non_protein_resolved",
        mapping_notes="Copy-number event; no protein substitution.",
    )
    inputs = build_validator_inputs([candidate()], [amplification], "1.0.0")
    assert inputs[0].to_dict()["validator_eligibility"] == "abstain_non_missense"
    assert inputs[0].to_dict()["variant_provenance"][0]["alteration_type"] == "amplification"


def test_silencing_and_missing_protein_are_preserved() -> None:
    silencing = variant(
        variant_id="var-silence",
        alteration_type="silencing",
        position=1,
        reference_allele="N",
        alternate_allele="A",
        transcript_id=None,
        transcript_version=None,
        protein_accession=None,
        protein_isoform=None,
        protein_change=None,
        mapping_status="non_protein_resolved",
        mapping_notes="Silencing has no protein substitution.",
    )
    result = build_validator_inputs([candidate()], [silencing], "1.0.0")[0].to_dict()
    assert result["variant_provenance"][0]["protein_accession"] is None
    assert result["validator_eligibility"] == "abstain_non_missense"


def test_multiple_transcripts_are_not_collapsed() -> None:
    second = variant(
        variant_id="var-2",
        transcript_id="ENST00000342988",
        transcript_version=".4",
        protein_isoform="2",
        protein_change="p.L858R",
    )
    result = build_validator_inputs([candidate()], [variant(), second], "1.0.0")[0].to_dict()
    assert result["join_cardinality"] == "multiple"
    assert len(result["variant_provenance"]) == 2
    assert result["validator_eligibility"] == "abstain_ambiguous"


def test_unknown_alias_fails_closed() -> None:
    with pytest.raises(ContractError, match="Unknown or undocumented"):
        normalize_gene_symbol("EGFR_ALIAS")
    with pytest.raises(ContractError):
        validate_candidate_gene(candidate(gene="egfr_alias"))


def test_validation_patient_candidate_is_rejected() -> None:
    with pytest.raises(ContractError, match="forbidden"):
        build_validator_inputs([candidate()], [variant()], "1.0.0", forbidden_patient_ids=["P001"])


def test_checkpoint_omission_is_rejected() -> None:
    record = candidate()
    del record["checkpoint_hash"]
    with pytest.raises(ContractError, match="checkpoint_hash"):
        validate_candidate_gene(record)


def test_schema_version_mismatch_is_rejected() -> None:
    record = deepcopy(candidate())
    record["schema_version"] = "2.0.0"
    with pytest.raises(ContractError, match="schema_version"):
        validate_candidate_gene(record)


def test_zero_join_preserves_missing_evidence() -> None:
    result = build_validator_inputs([candidate()], [], "1.0.0")[0].to_dict()
    assert result["join_cardinality"] == "zero"
    assert result["variant_provenance"] == []
    assert result["validator_eligibility"] == "abstain_missing_evidence"


def test_ambiguous_mapping_requires_notes_and_abstains() -> None:
    record = variant(
        mapping_status="ambiguous",
        protein_accession=None,
        protein_isoform=None,
        protein_change=None,
        mapping_notes="Two transcript mappings remain possible.",
    )
    result = build_validator_inputs([candidate()], [record], "1.0.0")[0].to_dict()
    assert result["ambiguity_status"] == "transcript_ambiguous"
    assert result["validator_eligibility"] == "abstain_ambiguous"
