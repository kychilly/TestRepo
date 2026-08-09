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
        self._scvi: Any = None
        self.probe: PCALogReg | None = None
        self.gene_ids: tuple[str, ...] | None = None
        self.latent_gene_ids: tuple[str, ...] | None = None
        self.applicability: str | None = None

    def fit(self, train_data: CellData, train_metadata: dict[str, Any]) -> "ScVIProbe":
        try:
            import scvi  # type: ignore[import-not-found]
        except ImportError as exc:
            self.applicability = "scvi-tools is not installed"
            raise MethodNotApplicable(self.applicability) from exc
        self.applicability = f"scvi-tools {scvi.__version__}"
        if self.count_layer != "X" and self.count_layer not in train_metadata:
            raise BaselineError(
                f"Configured count layer {self.count_layer!r} was not supplied"
            )
        if not np.allclose(train_data.X, np.floor(train_data.X)) or np.any(
            train_data.X < 0
        ):
            raise BaselineError(
                "scVI input must be non-negative integer-like raw counts"
            )
        try:
            import anndata as ad  # type: ignore[import-not-found]

            scvi.settings.seed = self.seed
            adata = self._to_anndata(ad, train_data)
            scvi.model.SCVI.setup_anndata(
                adata,
                layer=None if self.count_layer == "X" else self.count_layer,
                batch_key=self.batch_key,
            )
            self.model = scvi.model.SCVI(adata, n_latent=self.latent_size)
            self.model.train(
                max_epochs=self.epochs,
                early_stopping=bool(train_metadata.get("early_stopping", True)),
            )
            self._scvi = scvi
            latent = np.asarray(
                self.model.get_latent_representation(), dtype=np.float64
            )
        except (ImportError, AttributeError, RuntimeError, ValueError) as exc:
            raise MethodNotApplicable(f"scVI training failed: {exc}") from exc
        if latent.ndim != 2 or not np.isfinite(latent).all():
            raise BaselineError(
                "scVI latent representation is not finite and two-dimensional"
            )
        self.gene_ids = train_data.gene_ids
        self.latent_gene_ids = tuple(
            f"scvi_latent_{index}" for index in range(latent.shape[1])
        )
        latent_data = self._latent_data(train_data, latent)
        components = min(8, latent_data.X.shape[0], latent_data.X.shape[1])
        if components < 1:
            raise BaselineError("scVI latent representation is empty")
        self.probe = PCALogReg(components=components, seed=self.seed).fit(
            latent_data, {"split": "train"}
        )
        return self

    def predict(self, data: CellData, metadata: dict[str, Any]) -> NDArray[Any]:
        return self._probe().predict(self._latent_for(data), metadata)

    def predict_proba(self, data: CellData, metadata: dict[str, Any]) -> NDArray[Any]:
        return self._probe().predict_proba(self._latent_for(data), metadata)

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
            "analysis_type": "scvi_latent_train_only_probe",
            "latent_gene_ids": self.latent_gene_ids,
        }

    def _probe(self) -> PCALogReg:
        if self.probe is None:
            raise BaselineError("scVI probe is unfitted")
        return self.probe

    def _to_anndata(self, ad: Any, data: CellData) -> Any:
        obs: dict[str, Any] = {"state": data.state.astype(str)}
        if self.batch_key:
            if data.batch is None:
                raise BaselineError("Configured scVI batch_key requires CellData.batch")
            obs[self.batch_key] = data.batch.astype(str)
        return ad.AnnData(
            X=np.asarray(data.X, dtype=np.float32),
            obs=obs,
            var={"gene_id": list(data.gene_ids)},
        )

    def _latent_data(self, data: CellData, latent: NDArray[Any]) -> CellData:
        if self.latent_gene_ids is None:
            raise BaselineError("scVI latent gene identifiers are unavailable")
        return CellData(
            latent,
            data.patient_id,
            data.cell_id,
            data.state,
            self.latent_gene_ids,
            data.batch,
        )

    def _latent_for(self, data: CellData) -> CellData:
        if self.model is None or self._scvi is None:
            raise BaselineError("scVI model is unfitted")
        try:
            import anndata as ad  # type: ignore[import-not-found]

            query = self._to_anndata(ad, data)
            self._scvi.model.SCVI.prepare_query_anndata(query, self.model)
            query_model = self._scvi.model.SCVI.load_query_data(query, self.model)
            latent = np.asarray(
                query_model.get_latent_representation(), dtype=np.float64
            )
        except (ImportError, AttributeError, RuntimeError, ValueError) as exc:
            raise MethodNotApplicable(f"scVI query transform failed: {exc}") from exc
        return self._latent_data(data, latent)
