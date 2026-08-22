from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.week3 import ablation_matrix, run_matrix, select_backbones
from gbm_study.plain_english import write_json_with_explanation
from models.scgpt_internal import _checkpoint_status, _fingerprint


def test_ablation_matrix_changes_one_variable_at_a_time() -> None:
    reference, *others = ablation_matrix()
    assert reference.name == "all_on"
    for value in others:
        changes = sum(
            left != right
            for left, right in (
                (reference.validator, value.validator),
                (reference.grn, value.grn),
                (reference.mc_dropout, value.mc_dropout),
            )
        )
        assert changes == 1


def test_backbone_scope_requires_measured_timing() -> None:
    blocked = select_backbones({"status": "blocked"}, seeds=3, ablations=4, budget_gpu_seconds=10)
    assert blocked["backbones"] == ["scGPT"]
    allowed = select_backbones(
        {"status": "completed", "timing": {"projected_gpu_seconds_per_10000_cells": 1.0}},
        seeds=3,
        ablations=4,
        budget_gpu_seconds=12,
    )
    assert allowed["backbones"] == ["scGPT", "CellFM", "Geneformer"]


def test_matrix_persists_embeddings_rankings_predictions_and_seeds(tmp_path: Path) -> None:
    timing = tmp_path / "timing.json"
    timing.write_text(json.dumps({"status": "blocked"}), encoding="utf-8")

    def runner(config: object, ablation: object, seed: int, backbone: str) -> dict[str, object]:
        return {
            "embeddings": np.full((2, 3), seed, dtype=np.float32),
            "embedding_variance": np.zeros((2, 3), dtype=np.float32),
            "embedding_cell_ids": ["c1", "c2"],
            "embedding_patient_ids": ["p1", "p2"],
            "rankings": [{"gene": "TP53", "score": 1.0, "rank": 1, "seed": seed}],
            "predictions": [{"cell_id": "c1", "predicted_state": "MES", "seed": seed}],
            "metrics": {"macro_f1": 1.0},
        }

    output = tmp_path / "runs"
    result = run_matrix({"seeds": [1, 2, 3], "week2_timing_path": str(timing)}, runner, output)
    assert result["status"] == "completed"
    assert result["completed_runs"] == 12
    assert {run["seed"] for run in result["runs"]} == {1, 2, 3}
    first = Path(result["runs"][0]["artifacts"]["embeddings"])
    assert set(np.load(first).files) == {
        "embeddings",
        "embedding_variance",
        "cell_ids",
        "patient_ids",
    }
    assert (first.parent / "rankings.jsonl").is_file()
    assert (first.parent / "predictions.jsonl").is_file()
    assert (first.parent / "rankings.txt").is_file()
    assert (first.parent / "predictions.txt").is_file()
    assert (output / "manifest.txt").is_file()
    assert (first.parent / "run.txt").is_file()


def test_plain_english_companion_has_required_sections(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    write_json_with_explanation(
        path,
        {"status": "blocked", "reason": "CUDA is unavailable", "next_actions": ["Use a GPU."]},
    )
    text = path.with_suffix(".txt").read_text(encoding="utf-8")
    for heading in (
        "WHAT HAPPENED",
        "WHY",
        "WHAT IS IMPORTANT",
        "WHAT IS CONCERNING",
        "NEXT ACTIONS",
    ):
        assert heading in text


def test_resume_fingerprint_protects_seed_and_checkpoint(tmp_path: Path) -> None:
    config = {"fold": 0, "target_states": ["AC", "MES"], "mc_dropout_passes": 20}
    ablation = ablation_matrix()[0]
    provenance = {"checkpoint_sha256": "checkpoint", "vocabulary_sha256": "vocab"}
    first = _fingerprint(config, ablation, 17, "scGPT", provenance, ["TP53"])
    second = _fingerprint(config, ablation, 42, "scGPT", provenance, ["TP53"])
    assert first != second
    status = tmp_path / "checkpoint_status.json"
    _checkpoint_status(
        status,
        fingerprint=first,
        stage="ranking_genes",
        completed_genes=10,
        total_genes=100,
    )
    assert json.loads(status.read_text())["resume_fingerprint"] == first
    assert status.with_suffix(".txt").is_file()
