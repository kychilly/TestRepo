"""scVI latent representation plus the shared multinomial probe."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .base import Baseline, BaselineError, CellData, MethodNotApplicable
from .pca_logreg import PCALogReg


class ScVIProbe(Baseline):
    """Train scVI on training cells, then fit the same logistic probe."""

    method = "scvi_probe"

    def __init__(
        self,
        latent_size: int = 10,
        epochs: int = 100,
        seed: int = 17,
        count_layer: str = "X",
        batch_key: str | None = None,
    ) -> None:
        self.latent_size = latent_size
        self.epochs = epochs
        self.seed = seed
        self.count_layer = count_layer
        self.batch_key = batch_key
        self.model: Any = None
        self.probe: PCALogReg | None = None
        self.gene_ids: tuple[str, ...] | None = None
        self.applicability: str | None = None

    def fit(self, train_data: CellData, train_metadata: dict[str, Any]) -> "ScVIProbe":
        try:
            import scvi  # type: ignore[import-not-found]
        except ImportError as exc:
            self.applicability = "scvi-tools is not installed"
            raise MethodNotApplicable(self.applicability) from exc
        self.applicability = f"scvi-tools {scvi.__version__} requires a verified AnnData loader"
        if self.count_layer != "X" and self.count_layer not in train_metadata:
            raise BaselineError(f"Configured count layer {self.count_layer!r} was not supplied")
        if not np.allclose(train_data.X, np.floor(train_data.X)) or np.any(train_data.X < 0):
            raise BaselineError("scVI input must be non-negative integer-like raw counts")
        raise MethodNotApplicable(
            "scVI training requires an AnnData loader and checkpointed architecture configuration"
        )

    def predict(self, data: CellData, metadata: dict[str, Any]) -> NDArray[Any]:
        return self._probe().predict(data, metadata)

    def predict_proba(self, data: CellData, metadata: dict[str, Any]) -> NDArray[Any]:
        return self._probe().predict_proba(data, metadata)

    def save(self, path: Path) -> None:
        raise MethodNotApplicable("No scVI model was fitted")

    @classmethod
    def load(cls, path: Path) -> "ScVIProbe":
        raise MethodNotApplicable("No scVI model artifact is available")

    def get_run_metadata(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "latent_size": self.latent_size,
            "epochs": self.epochs,
            "seed": self.seed,
            "count_layer": self.count_layer,
            "batch_key": self.batch_key,
            "applicability": self.applicability,
        }

    def _probe(self) -> PCALogReg:
        if self.probe is None:
            raise BaselineError("scVI probe is unfitted")
        return self.probe
