from __future__ import annotations

from pathlib import Path

from gate import run_gate
from validator import GeneRecord, Outcome, Thresholds, classify
from shuffled_validator import classify_many


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


def test_shuffled_control_preserves_exact_bucket_counts_and_seed() -> None:
    records = [
        GeneRecord(f"G{i}", f"V{i}", "missense", 90.0, -11.0, 0.0)
        for i in range(10)
    ]
    outcomes = list(Outcome) * 2
    real = [
        __import__("validator").Verdict(record.gene, record.mutation, outcome, "real")
        for record, outcome in zip(records, outcomes, strict=True)
    ]
    first = classify_many(records, THRESHOLDS, reference_verdicts=real, seed=17)
    second = classify_many(records, THRESHOLDS, reference_verdicts=real, seed=17)
    assert [item.outcome for item in first] == [item.outcome for item in second]
    assert sorted(item.outcome.value for item in first) == sorted(
        item.outcome.value for item in real
    )


def test_real_and_shuffled_validators_share_batch_drop_in_api() -> None:
    import inspect
    import shuffled_validator
    import validator

    assert inspect.signature(validator.classify_many) == inspect.signature(
        shuffled_validator.classify_many
    )


def test_stage34_runner_triggers_pooling_and_marks_fixture_blocker() -> None:
    from scripts.run_stage34_validation import run

    result = run(
        Path("examples/validator_gate_input.jsonl"),
        Path("config/stage34.yaml"),
        candidates_path=Path("results/contracts/tp53/scgpt_candidate_output.jsonl"),
        pool_candidate_paths=[],
        gold_outcomes_path=Path("examples/validator_gold.synthetic.jsonl"),
        seed=17,
    )
    assert result["bucket_counts"] == {
        "destabilizing_driver": 1,
        "functional_driver": 0,
        "abstain": 2,
        "unconfirmed": 0,
        "data_deficient": 0,
    }
    assert result["candidate_count"] == 3
    assert result["confirmable_count_primary"] == 1
    assert result["candidate_alignment"]["primary"]["unused_evidence_genes"] == ["IDH1"]
    assert result["feasibility"]["branch"] == "pool_additional_cancer_types"
    assert result["feasibility"]["status"] == "blocked_missing_pool_candidates"
    assert result["shuffled_control"]["proportions_preserved_exactly"] is True
    assert result["comparison"]["status"] == "completed"
    assert result["comparison"]["result_call"] in {"validator_result", "no_result"}
    assert result["scope"] == "synthetic_fixture"
    assert result["scientifically_complete"] is False


def test_stage34_counts_only_candidates_and_marks_missing_evidence_data_deficient(
    tmp_path: Path,
) -> None:
    import json
    from scripts.run_stage34_validation import run

    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        json.dumps({"candidate_id": "c1", "gene": "MISSING", "run_id": "real"}) + "\n"
    )
    result = run(
        Path("examples/validator_gate_input.jsonl"),
        Path("config/stage34.yaml"),
        candidates_path=candidates,
        pool_candidate_paths=[],
        gold_outcomes_path=None,
        seed=17,
    )
    assert result["candidate_count"] == 1
    assert result["bucket_counts"]["data_deficient"] == 1
    assert result["candidate_alignment"]["primary"]["missing_evidence_candidate_ids"] == [
        "c1"
    ]
    assert result["comparison"]["result_call"] == "no_result"
