from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from gbm_study.storage_policy import SharedGPUStoragePolicy, StoragePolicyError
from scripts.benchmark_scgpt import _stream_huggingface_training_sample


def test_shared_gpu_policy_accepts_compact_inputs(tmp_path: Path) -> None:
    first = tmp_path / "pilot.bin"
    second = tmp_path / "checkpoint.bin"
    first.write_bytes(b"a" * 10)
    second.write_bytes(b"b" * 20)
    result = SharedGPUStoragePolicy(max_single_file_bytes=25, max_total_input_bytes=40).validate(
        [first, second]
    )
    assert result == {"local_input_bytes": 30, "local_input_files": 2}


def test_shared_gpu_policy_rejects_oversized_input(tmp_path: Path) -> None:
    path = tmp_path / "full-dataset.bin"
    path.write_bytes(b"x" * 11)
    with pytest.raises(StoragePolicyError, match="per-file storage limit"):
        SharedGPUStoragePolicy(max_single_file_bytes=10, max_total_input_bytes=20).validate([path])


def test_shared_gpu_policy_requires_explicit_large_approval(tmp_path: Path) -> None:
    path = tmp_path / "approved.bin"
    path.write_bytes(b"x" * 11)
    result = SharedGPUStoragePolicy(
        max_single_file_bytes=10,
        max_total_input_bytes=10,
        large_downloads_approved=True,
    ).validate([path])
    assert result["local_input_bytes"] == 11


def test_huggingface_loader_streams_and_samples_training_patients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_load_dataset(dataset_id: str, **kwargs: object) -> list[dict[str, object]]:
        calls.append({"dataset_id": dataset_id, **kwargs})
        return [
            {"patient_id": "train-a", "expression": [1.0, 0.0], "gene_ids": ["A", "B"]},
            {"patient_id": "test-z", "expression": [9.0, 9.0], "gene_ids": ["A", "B"]},
            {"patient_id": "train-b", "expression": [0.0, 2.0], "gene_ids": ["A", "B"]},
        ]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))
    matrix, genes, patients = _stream_huggingface_training_sample(
        {
            "hf_dataset_id": "owner/compact-neftel",
            "hf_split": "analysis",
            "hf_expression_column": "expression",
            "patient_id_column": "patient_id",
            "seed": 17,
        },
        {"train-a", "train-b"},
        2,
    )
    assert calls[0]["streaming"] is True
    assert matrix.shape == (2, 2)
    assert genes == ["A", "B"]
    assert set(patients) == {"train-a", "train-b"}
