from __future__ import annotations

import pytest

from models.candidate_generation import build_candidate_records
from schemas.records import ContractError


def metadata() -> dict[str, dict[str, str]]:
    return {
        "TP53": {
            "gene_namespace": "HGNC",
            "ensembl_gene_id": "ENSG00000141510",
            "gene_id_version": "Ensembl-110",
        },
        "EGFR": {
            "gene_namespace": "HGNC",
            "ensembl_gene_id": "ENSG00000146648",
            "gene_id_version": "Ensembl-110",
        },
    }


def test_candidate_generation_ranks_and_validates_full_contract() -> None:
    records = build_candidate_records(
        [
            {"gene": "EGFR", "score": 1.0},
            {"gene": "TP53", "score": 2.0, "score_sd": 0.2},
        ],
        run_id="run-1",
        backbone="scGPT",
        checkpoint_hash="checkpoint",
        vocabulary_hash="vocab",
        cohort="Neftel",
        fold=0,
        seed=17,
        split_hash="split",
        patient_scope="train_only",
        patient_ids=["P001", "P002"],
        state="MES",
        score_method="mask_delta_logit",
        config_hash="config",
        gene_metadata=metadata(),
        n_cells=1000,
        created_at="2026-08-09T18:00:00Z",
    )
    assert [record.data["gene"] for record in records] == ["TP53", "EGFR"]
    assert records[0].data["rank"] == 1
    assert records[0].data["training_only"] is True


def test_candidate_generation_rejects_missing_metadata() -> None:
    with pytest.raises(ContractError, match="Missing canonical metadata"):
        build_candidate_records(
            [{"gene": "RPRM", "score": 1.0}],
            run_id="run-1",
            backbone="scGPT",
            checkpoint_hash="c",
            vocabulary_hash="v",
            cohort="Neftel",
            fold=0,
            seed=17,
            split_hash="s",
            patient_scope="train_only",
            patient_ids=["P001"],
            state="MES",
            score_method="mask_delta_logit",
            config_hash="config",
            gene_metadata={},
            n_cells=10,
        )
