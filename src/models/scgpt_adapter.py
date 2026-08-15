"""Leakage-conscious scGPT input mapping and inference wrapper.

The adapter deliberately does not guess a gene identifier namespace. Callers
must provide the namespace used by the checkpoint vocabulary and a vocabulary
mapping with one integer token ID per gene identifier.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray


class AdapterError(ValueError):
    """Raised when model assets or input contracts are invalid."""


@dataclass(frozen=True)
class GeneMappingReport:
    """Audit trail for mapping source genes to model vocabulary tokens."""

    retained: tuple[str, ...]
    dropped: tuple[str, ...]
    duplicated: tuple[str, ...]
    unmapped: tuple[str, ...]

    @property
    def retained_count(self) -> int:
        return len(self.retained)


@dataclass(frozen=True)
class PreparedInputs:
    """Tokenized expression matrix and mapping audit for model inference."""

    values: NDArray[np.float32]
    token_ids: NDArray[np.int64]
    report: GeneMappingReport


class ModelProtocol(Protocol):
    """Minimum callable surface needed by the adapter."""

    def __call__(self, **kwargs: Any) -> Any: ...


class ScGPTAdapter:
    """Wrap a loaded scGPT-compatible model and checkpoint vocabulary."""

    def __init__(
        self,
        model: ModelProtocol,
        vocabulary: dict[str, int],
        checkpoint_path: Path,
        vocabulary_path: Path,
        device: str = "cpu",
    ) -> None:
        if not vocabulary:
            raise AdapterError("The model vocabulary is empty")
        if any(
            not isinstance(key, str) or not isinstance(value, int)
            for key, value in vocabulary.items()
        ):
            raise AdapterError("Vocabulary must map string gene identifiers to integer token IDs")
        self.model = model
        self.vocabulary = vocabulary
        self.checkpoint_path = checkpoint_path
        self.vocabulary_path = vocabulary_path
        self.device = device

    @staticmethod
    def file_sha256(path: Path) -> str:
        """Hash a model asset for inclusion in every inference result."""
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def map_genes(self, gene_ids: list[str]) -> tuple[NDArray[np.int64], GeneMappingReport]:
        """Map exact gene IDs, retaining first occurrences and reporting duplicates."""
        if not gene_ids:
            raise AdapterError("No input genes were supplied")
        seen: set[str] = set()
        retained: list[str] = []
        duplicated: list[str] = []
        unmapped: list[str] = []
        token_ids: list[int] = []
        for gene_id in gene_ids:
            if gene_id in seen:
                duplicated.append(gene_id)
                continue
            seen.add(gene_id)
            token_id = self.vocabulary.get(gene_id)
            if token_id is None:
                unmapped.append(gene_id)
                continue
            retained.append(gene_id)
            token_ids.append(token_id)
        report = GeneMappingReport(
            retained=tuple(retained),
            dropped=tuple(unmapped),
            duplicated=tuple(sorted(set(duplicated))),
            unmapped=tuple(unmapped),
        )
        if not token_ids:
            raise AdapterError("No genes map to the model vocabulary")
        return np.asarray(token_ids, dtype=np.int64), report

    def prepare_inputs(self, data: Any, gene_id_type: str | None) -> PreparedInputs:
        """Prepare an AnnData-like object or explicit representation.

        Required representation: ``{"X": array-like, "var_names": list[str]}``.
        ``gene_id_type`` is mandatory to prevent silent namespace selection.
        """
        if not gene_id_type:
            raise AdapterError("gene_id_type is required; identifier namespaces are never inferred")
        if isinstance(data, dict):
            if "X" not in data or "var_names" not in data:
                raise AdapterError("Intermediate representation requires X and var_names")
            matrix = np.asarray(data["X"])
            gene_ids = [str(value) for value in data["var_names"]]
        else:
            try:
                matrix = np.asarray(data.X)
                gene_ids = [str(value) for value in data.var_names]
            except AttributeError as exc:
                raise AdapterError("Input must be AnnData-like or contain X and var_names") from exc
        if matrix.ndim != 2 or matrix.shape[1] != len(gene_ids):
            raise AdapterError(
                "Expression matrix columns must match the number of gene identifiers"
            )
        token_ids, report = self.map_genes(gene_ids)
        retained_indices = [
            index for index, gene_id in enumerate(gene_ids) if gene_id in report.retained
        ]
        values = matrix[:, retained_indices].astype(np.float32, copy=False)
        if not np.isfinite(values).all():
            raise AdapterError("Input expression matrix contains NaN or infinite values")
        return PreparedInputs(values=values, token_ids=token_ids, report=report)

    def infer(
        self, prepared: PreparedInputs, batch_size: int, precision: str = "float32"
    ) -> NDArray[np.float32]:
        """Run batched inference using the checkpoint's token IDs and expression values."""
        if batch_size < 1:
            raise AdapterError("batch_size must be positive")
        if precision not in {"float32", "float16", "bfloat16"}:
            raise AdapterError(f"Unsupported precision mode: {precision}")
        outputs: list[NDArray[np.float32]] = []
        for start in range(0, prepared.values.shape[0], batch_size):
            values = prepared.values[start : start + batch_size]
            token_ids = np.broadcast_to(prepared.token_ids, values.shape)
            model_inputs = {
                "gene_ids": token_ids,
                "values": values,
                "precision": precision,
            }
            result = self.model(**model_inputs)
            embedding = result.get("cell_emb") if isinstance(result, dict) else result
            array = np.asarray(cast(Any, embedding), dtype=np.float32)
            if array.shape[0] != values.shape[0]:
                raise AdapterError("Model output first dimension does not match batch size")
            outputs.append(array)
        if not outputs:
            raise AdapterError("Cannot infer on zero cells")
        result = np.concatenate(outputs, axis=0)
        if not np.isfinite(result).all():
            raise AdapterError("Model embeddings contain NaN or infinite values")
        return cast(NDArray[np.float32], result)

    def provenance(self) -> dict[str, str]:
        """Return immutable asset hashes for result metadata."""
        return {
            "checkpoint_sha256": self.file_sha256(self.checkpoint_path),
            "vocabulary_sha256": self.file_sha256(self.vocabulary_path),
        }


def deterministic_indices(n_cells: int, requested: int, seed: int) -> NDArray[np.int64]:
    """Select exactly ``requested`` row indices reproducibly without replacement."""
    if n_cells < requested:
        raise AdapterError(f"Need {requested} cells, but only {n_cells} are available")
    if requested < 1:
        raise AdapterError("requested cell count must be positive")
    return np.random.default_rng(seed).choice(n_cells, size=requested, replace=False)


def elapsed_seconds(start: float, end: float) -> float:
    """Validate and return a monotonic elapsed duration."""
    duration = end - start
    if not math.isfinite(duration) or duration < 0:
        raise AdapterError("Invalid wall-clock measurement")
    return duration
