from __future__ import annotations

import pytest

from models.candidate_scoring import aggregate_mask_delta_scores
from schemas.records import ContractError


def rows() -> list[dict[str, object]]:
    return [
        {
            "state": "MES",
            "gene": "TP53",
            "patient_id": "P1",
            "cell_id": "c1",
            "baseline_logit": 3,
            "masked_logit": 1,
        },
        {
            "state": "MES",
            "gene": "TP53",
            "patient_id": "P1",
            "cell_id": "c2",
            "baseline_logit": 2,
            "masked_logit": 1,
        },
        {
            "state": "MES",
            "gene": "TP53",
            "patient_id": "P2",
            "cell_id": "c3",
            "baseline_logit": 4,
            "masked_logit": 1,
        },
        {
            "state": "MES",
            "gene": "EGFR",
            "patient_id": "P1",
            "cell_id": "c1",
            "baseline_logit": 2,
            "masked_logit": 1,
        },
        {
            "state": "MES",
            "gene": "EGFR",
            "patient_id": "P2",
            "cell_id": "c3",
            "baseline_logit": 2,
            "masked_logit": 1,
        },
    ]


def test_scores_are_patient_averaged_before_ranking() -> None:
    scored = aggregate_mask_delta_scores(rows(), state="MES")
    assert [row["gene"] for row in scored] == ["TP53", "EGFR"]
    assert scored[0]["score"] == pytest.approx(2.25)
    assert scored[0]["n_patients"] == 2
    assert scored[0]["n_cells"] == 3


def test_duplicate_cell_gene_and_mixed_state_fail_closed() -> None:
    with pytest.raises(ContractError, match="Duplicate"):
        aggregate_mask_delta_scores(rows() + [rows()[0]], state="MES")
    with pytest.raises(ContractError, match="Mixed state"):
        aggregate_mask_delta_scores(
            rows() + [{**rows()[0], "state": "AC", "cell_id": "c4"}], state="MES"
        )
