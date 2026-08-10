from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gbm_study.stage5_masking import PassthroughStubOutcomeSource, mask_candidates
from models.grn import GRNEdge, held_out_edges, score_held_out_edges
from models.mc_dropout import blocked_result, infer_mc_dropout, multiplier_from_timings
from models.scgpt_adapter import PreparedInputs, ScGPTAdapter
from schemas.records import ContractError


def test_grn_edge_schema_fixture_and_auroc() -> None:
    fixture = Path(__file__).parents[1] / "examples/grn_edge.synthetic.jsonl"
    edges = [
        GRNEdge.from_dict(__import__("json").loads(line))
        for line in fixture.read_text().splitlines()
    ]
    result = score_held_out_edges(
        edges[:2], edges[2:], lambda edge: float(edge.get("confidence", 0.0))
    )
    assert result["status"] == "completed"
    assert 0.0 <= result["auroc"] <= 1.0


def test_grn_rejects_transitive_training_prior_leakage() -> None:
    def edge(source: str, target: str) -> GRNEdge:
        return GRNEdge.from_dict(
            {
                "schema_version": "1.0.0",
                "source_gene": source,
                "target_gene": target,
                "source_database": "synthetic",
                "confidence": 0.5,
                "edge_list_sha256": "sha",
                "access_date": "2026-08-09",
            }
        )

    with pytest.raises(ContractError, match="training prior"):
        held_out_edges(
            [edge("A", "B"), edge("B", "C"), edge("A", "C")], seed=1, held_out_fraction=0.34
        )


class StochasticModel:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, **kwargs: object) -> dict[str, np.ndarray]:
        self.calls += 1
        return {
            "cell_emb": np.full((len(kwargs["values"]), 2), float(self.calls), dtype=np.float32)
        }


def test_mc_dropout_mean_variance_and_multiplier(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    vocabulary = tmp_path / "vocab"
    checkpoint.write_text("checkpoint")
    vocabulary.write_text("vocab")
    model = StochasticModel()
    adapter = ScGPTAdapter(model, {"TP53": 1}, checkpoint, vocabulary)
    prepared = PreparedInputs(
        np.ones((2, 1), dtype=np.float32),
        np.array([1], dtype=np.int64),
        __import__("models.scgpt_adapter", fromlist=["GeneMappingReport"]).GeneMappingReport(
            ("TP53",), (), (), ()
        ),
    )
    result = infer_mc_dropout(adapter, prepared, n_passes=3, batch_size=2)
    assert np.allclose(result.mean, 2.0)
    assert np.allclose(result.variance, 2 / 3)
    assert multiplier_from_timings(2.0, 3, 6.0) == 3.0
    assert (
        blocked_result("CUDA GPU is unavailable", n_passes=20, batch_size=32, device="cuda")[
            "status"
        ]
        == "blocked"
    )


def test_stage5_validator_on_masks_and_off_is_noop() -> None:
    candidates = [{"candidate_id": "a", "gene": "TP53"}, {"candidate_id": "b", "gene": "EGFR"}]
    payloads = {
        "a": {"input_id": "a", "validator_eligibility": "eligible_missense"},
        "b": {"input_id": "b", "validator_eligibility": "abstain_ambiguous"},
    }
    source = PassthroughStubOutcomeSource()
    kept, on_provenance = mask_candidates(
        candidates, payloads, validator="on", outcome_source=source
    )
    unchanged, off_provenance = mask_candidates(
        candidates, {}, validator="off", outcome_source=source
    )
    assert [row["gene"] for row in kept] == ["TP53"]
    assert unchanged == candidates
    assert on_provenance["outcome_source_version"] == "stub-1.0.0"
    assert off_provenance["masked"] == 0


def test_pilot_blocker_names_missing_prerequisites() -> None:
    from importlib.machinery import SourceFileLoader

    pilot = SourceFileLoader(
        "run_pilot_scgpt", str(Path(__file__).parents[1] / "scripts/run_pilot_scgpt.py")
    ).load_module()
    result = pilot.run_pilot(
        {
            "cell_data_path": None,
            "split_file": None,
            "checkpoint_path": None,
            "vocabulary_path": None,
            "gene_id_type": None,
        }
    )
    assert result["status"] == "blocked"
    assert "cell_data_path" in result["reason"]


def test_pilot_synthetic_producer_has_label_and_valid_records() -> None:
    from importlib.machinery import SourceFileLoader

    pilot = SourceFileLoader(
        "run_pilot_scgpt_scores", str(Path(__file__).parents[1] / "scripts/run_pilot_scgpt.py")
    ).load_module()
    result = pilot.build_from_scores(
        {
            "run_id": "r",
            "checkpoint_hash": "c",
            "vocabulary_hash": "v",
            "cohort": "Neftel",
            "seed": 17,
            "split_hash": "s",
            "config_hash": "cfg",
            "training_patient_ids": ["P001"],
            "state": "MES",
            "n_cells": 2,
        },
        [{"gene": "TP53", "score": 1.0, "score_sd": 0.25, "n_cells": 2, "n_patients": 1}],
        {
            "TP53": {
                "gene_namespace": "HGNC",
                "ensembl_gene_id": "ENSG00000141510",
                "gene_id_version": "Ensembl-110",
            }
        },
    )
    assert result["interpretation_label"].startswith("candidate/suspect")
    assert result["candidates"][0]["gene"] == "TP53"


def test_grn_cli_blocker_is_structured(tmp_path: Path) -> None:
    from importlib.machinery import SourceFileLoader

    script = SourceFileLoader(
        "run_grn_sanity_check", str(Path(__file__).parents[1] / "scripts/run_grn_sanity_check.py")
    ).load_module()
    output = tmp_path / "grn.json"
    assert (
        script.main(
            [
                "--config",
                str(Path(__file__).parents[1] / "config/model.yaml"),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert __import__("json").loads(output.read_text())["status"] == "blocked"
