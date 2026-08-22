"""Environment, mapping, sampling, and result-schema tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from models.scgpt_adapter import AdapterError, ScGPTAdapter, deterministic_indices
from models.scgpt_loader import _truncate_and_pad


def adapter(tmp_path: Path, vocabulary: dict[str, int]) -> ScGPTAdapter:
    checkpoint = tmp_path / "checkpoint.pt"
    vocab = tmp_path / "vocab.json"
    checkpoint.write_bytes(b"checkpoint")
    vocab.write_text(json.dumps(vocabulary), encoding="utf-8")
    return ScGPTAdapter(
        lambda **_: {"cell_emb": np.ones((1, 2), dtype=np.float32)},
        vocabulary,
        checkpoint,
        vocab,
    )


def test_missing_checkpoint(tmp_path: Path) -> None:
    vocabulary = tmp_path / "vocab.json"
    vocabulary.write_text(json.dumps({"G1": 1}), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        ScGPTAdapter(lambda **_: None, {"G1": 1}, tmp_path / "missing.pt", vocabulary).provenance()


def test_missing_vocabulary(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    with pytest.raises(FileNotFoundError):
        ScGPTAdapter(
            lambda **_: None, {"G1": 1}, checkpoint, tmp_path / "missing.json"
        ).provenance()


def test_zero_overlapping_genes(tmp_path: Path) -> None:
    with pytest.raises(AdapterError, match="No genes map"):
        adapter(tmp_path, {"G1": 1}).map_genes(["G2"])


def test_duplicated_gene_symbols_are_reported(tmp_path: Path) -> None:
    _, report = adapter(tmp_path, {"G1": 1}).map_genes(["G1", "G1"])
    assert report.duplicated == ("G1",)


def test_duplicate_gene_expression_keeps_only_first_matching_column(tmp_path: Path) -> None:
    prepared = adapter(tmp_path, {"G1": 1}).prepare_inputs(
        {"X": [[3.0, 99.0]], "var_names": ["G1", "G1"]}, "symbol"
    )
    assert prepared.values.shape == (1, 1)
    assert prepared.values[0, 0] == 3.0
    assert prepared.token_ids.tolist() == [1]


def test_scgpt_truncation_is_per_cell_expression_ranked_and_padded() -> None:
    gene_ids = np.asarray([[10, 11, 12, 13], [10, 11, 12, 13]], dtype=np.int64)
    raw = np.asarray([[0.0, 5.0, 2.0, 9.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    transformed = raw + 100.0
    ids, values = _truncate_and_pad(
        gene_ids,
        raw,
        transformed,
        token_length=3,
        cls_id=1,
        pad_id=0,
        pad_value=-2.0,
    )
    assert ids.tolist() == [[1, 13, 11], [1, 10, 0]]
    assert values.tolist() == [[0.0, 109.0, 105.0], [0.0, 101.0, -2.0]]


def test_incompatible_output_shape(tmp_path: Path) -> None:
    def model(**_: object) -> dict[str, np.ndarray[Any, Any]]:
        return {"cell_emb": np.ones((2, 2), dtype=np.float32)}

    instance = ScGPTAdapter(model, {"G1": 1}, tmp_path / "checkpoint", tmp_path / "vocab")
    with pytest.raises(AdapterError, match="first dimension"):
        instance.infer(instance.prepare_inputs({"X": [[1.0]], "var_names": ["G1"]}, "symbol"), 1)


def test_deterministic_sampling_of_1000_cells() -> None:
    first = deterministic_indices(1200, 1000, 17)
    second = deterministic_indices(1200, 1000, 17)
    assert np.array_equal(first, second)
    assert len(first) == 1000
    assert len(set(first.tolist())) == 1000


def test_stable_output_shape_and_finite_embeddings(tmp_path: Path) -> None:
    instance = adapter(tmp_path, {"G1": 1})
    prepared = instance.prepare_inputs({"X": [[1.0], [2.0]], "var_names": ["G1"]}, "symbol")
    output = instance.infer(prepared, 1)
    assert output.shape == (2, 2)
    assert np.isfinite(output).all()


def test_benchmark_json_schema() -> None:
    path = Path("baseline_results/compute/week1_scgpt_benchmark.json")
    if not path.is_file():
        path = Path("results/compute/week1_scgpt_benchmark.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] in {"blocked", "completed"}
    assert payload["benchmark_cells"] == 1000
    assert {"selection", "mapping", "timing", "model", "provenance"} <= payload.keys()
