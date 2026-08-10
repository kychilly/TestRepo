"""Leakage-checked held-out edge recovery for a gene regulatory prior."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from evaluation.metrics import binary_metrics
from schemas.records import ContractError, _validate

Edge = tuple[str, str]
ScoreFn = Callable[[Mapping[str, Any]], float]


@dataclass(frozen=True)
class GRNEdge:
    data: dict[str, Any]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GRNEdge":
        payload = dict(value)
        _validate(payload, "grn_edge.schema.json")
        if payload["source_gene"] == payload["target_gene"]:
            raise ContractError("GRN self-edges are not supported")
        return cls(payload)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_edges(path: Path) -> list[GRNEdge]:
    records: list[GRNEdge] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(GRNEdge.from_dict(json.loads(line)))
    if not records:
        raise ContractError("GRN edge list is empty")
    return records


def _reachable(start: str, target: str, graph: Mapping[str, set[str]]) -> bool:
    pending = [start]
    visited: set[str] = set()
    while pending:
        node = pending.pop()
        if node == target:
            return True
        if node in visited:
            continue
        visited.add(node)
        pending.extend(graph.get(node, set()) - visited)
    return False


def held_out_edges(
    edges: Iterable[GRNEdge], *, seed: int, held_out_fraction: float = 0.2
) -> tuple[list[GRNEdge], list[GRNEdge]]:
    """Deterministically split edges; reject transitive-prior leakage."""
    import random

    values = list(edges)
    if not values or not 0 < held_out_fraction < 1:
        raise ContractError("Need edges and a held_out_fraction strictly between 0 and 1")
    ordered = sorted(values, key=lambda edge: (edge.data["source_gene"], edge.data["target_gene"]))
    random.Random(seed).shuffle(ordered)
    n_test = max(1, min(len(ordered) - 1, round(len(ordered) * held_out_fraction)))
    held = ordered[:n_test]
    train = ordered[n_test:]
    graph: dict[str, set[str]] = {}
    for edge in train:
        graph.setdefault(edge.data["source_gene"], set()).add(edge.data["target_gene"])
    leaked = [
        edge
        for edge in held
        if _reachable(edge.data["source_gene"], edge.data["target_gene"], graph)
    ]
    if leaked:
        names = [f"{e.data['source_gene']}->{e.data['target_gene']}" for e in leaked]
        raise ContractError("Held-out edges reachable through training prior: " + ", ".join(names))
    return train, held


def score_held_out_edges(
    train: Iterable[GRNEdge], held_out: Iterable[GRNEdge], score_fn: ScoreFn
) -> dict[str, Any]:
    """Compute AUROC for held-out positives against deterministic unknown negatives."""
    train_values, held_values = list(train), list(held_out)
    known = {(e.data["source_gene"], e.data["target_gene"]) for e in train_values + held_values}
    genes = sorted({g for e in known for g in e})
    negatives = [
        {
            "source_gene": source,
            "target_gene": target,
            "source_database": "negative",
            "confidence": 0.0,
        }
        for source in genes
        for target in genes
        if source != target and (source, target) not in known
    ]
    positives = [e.data for e in held_values]
    if not negatives:
        raise ContractError("GRN edge list has no unknown negative pairs for AUROC")
    y_true = [1] * len(positives) + [0] * len(negatives)
    scores = [float(score_fn(edge)) for edge in positives + negatives]
    result = binary_metrics(
        __import__("numpy").asarray(y_true), __import__("numpy").asarray(scores), ("auroc",)
    )
    return {
        "status": "completed",
        "auroc": result["auroc"],
        "held_out_edges": len(positives),
        "negative_edges": len(negatives),
    }


def run_sanity_check(
    path: Path, *, seed: int, score_fn: ScoreFn, held_out_fraction: float = 0.2
) -> dict[str, Any]:
    edges = load_edges(path)
    train, held = held_out_edges(edges, seed=seed, held_out_fraction=held_out_fraction)
    result = score_held_out_edges(train, held, score_fn)
    result["provenance"] = {
        "edge_list_sha256": file_sha256(path),
        "access_date": date.today().isoformat(),
    }
    return result
