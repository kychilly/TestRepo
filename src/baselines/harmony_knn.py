"""Harmony representation plus KNN, with an explicit unseen-data guard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from numpy.typing import NDArray

from .base import Baseline, CellData, MethodNotApplicable


class HarmonyKNN(Baseline):
    """Document Harmony's invalid unseen-point transform rather than leaking data."""

    method = "harmony_knn"

    def __init__(
        self,
        components: int = 16,
        n_neighbors: int = 15,
        harmony_covariate: str | None = None,
        seed: int = 17,
    ) -> None:
        self.components = components
        self.n_neighbors = n_neighbors
        self.harmony_covariate = harmony_covariate
        self.seed = seed
        self.applicability = "not evaluated"

    def fit(self, train_data: CellData, train_metadata: dict[str, Any]) -> "HarmonyKNN":
        if not self.harmony_covariate:
            self.applicability = "Harmony covariate is not prespecified"
        else:
            self.applicability = "selected Harmony implementation has no valid transform for unseen validation/test cells"
        raise MethodNotApplicable(self.applicability)

    def predict(self, data: CellData, metadata: dict[str, Any]) -> NDArray[Any]:
        raise MethodNotApplicable(self.applicability)

    def predict_proba(self, data: CellData, metadata: dict[str, Any]) -> NDArray[Any]:
        raise MethodNotApplicable(self.applicability)

    def save(self, path: Path) -> None:
        raise MethodNotApplicable(self.applicability)

    @classmethod
    def load(cls, path: Path) -> "HarmonyKNN":
        raise MethodNotApplicable(
            "Harmony artifact unavailable because unseen-data mapping is invalid"
        )

    def get_run_metadata(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "components": self.components,
            "n_neighbors": self.n_neighbors,
            "harmony_covariate": self.harmony_covariate,
            "seed": self.seed,
            "applicability": self.applicability,
            "analysis_type": "not_run",
        }
