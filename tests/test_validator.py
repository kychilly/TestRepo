from __future__ import annotations

from pathlib import Path

from gate import run_gate
from validator import GeneRecord, Outcome, Thresholds, classify


THRESHOLDS = Thresholds(plddt_floor=70.0, esm1b_cutoff=-10.0, ddg_cutoff=1.0)


def test_each_validator_outcome_has_a_distinct_synthetic_input() -> None:
    cases = [
        (GeneRecord("TP53", "R175H", "missense", 96.0, -5.0, 2.0), Outcome.DESTABILIZING_DRIVER),
        (GeneRecord("IDH1", "R132H", "missense", 96.0, -11.0, 0.4), Outcome.FUNCTIONAL_DRIVER),
        (GeneRecord("EGFR", "amplification", "amplification"), Outcome.ABSTAIN),
        (GeneRecord("RPRM", "E21K", "missense", 90.0, -5.0, 0.4), Outcome.UNCONFIRMED),
        (GeneRecord("RPRM", "R1Q", "missense", 50.0, -5.0, None), Outcome.DATA_DEFICIENT),
    ]
    for record, expected in cases:
        assert classify(record, THRESHOLDS).outcome is expected


def test_four_gene_gate_matches_requested_outcomes_without_threshold_tuning() -> None:
    result = run_gate(
        Path("examples/validator_gate_input.jsonl"),
        Path("config/validator.yaml"),
    )
    assert result["status"] == "blocked"
    assert result["gate_passed"] is False
    assert result["classification_gate_passed"] is True
    assert result["publication_gate_passed"] is False
    assert {item["gene"]: item["outcome"] for item in result["verdicts"]} == {
        "TP53": "destabilizing_driver",
        "IDH1": "functional_driver",
        "EGFR": "abstain",
        "RPRM": "abstain",
    }
    assert result["thresholds_moved_after_gate_observation"] is False
    assert result["provenance_warnings"] == [
        "incomplete_and_estimated_ddg",
        "incomplete_source_version_metadata",
    ]
