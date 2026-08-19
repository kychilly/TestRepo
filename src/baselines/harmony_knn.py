"""Train-only Harmony embedding with an explicit out-of-sample projection."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.decomposition import PCA  # type: ignore[import-untyped]
from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
from sklearn.neighbors import KNeighborsClassifier  # type: ignore[import-untyped]
from sklearn.preprocessing import OneHotEncoder  # type: ignore[import-untyped]

from .base import Baseline, BaselineError, CellData, MethodNotApplicable, STATE_LABELS


class HarmonyKNN(Baseline):
    """Harmony on training cells, then a frozen train-learned query map and kNN.

    Harmony itself is fit only on training cells. Since harmonypy does not
    expose a stable transform API for unseen cells, the class learns a Ridge
    map from raw training PCA/batch features to the training Harmony embedding.
    Validation/test rows are transformed with that frozen map; they never enter
    Harmony fitting. This approximation is recorded in run metadata.
    """

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
        self.pca: PCA | None = None
        self.batch_encoder: OneHotEncoder | None = None
        self.query_map: Ridge | None = None
        self.knn: KNeighborsClassifier | None = None
        self.gene_ids: tuple[str, ...] | None = None
        self.applicability = "not evaluated"
        self.harmony_version: str | None = None

    def fit(self, train_data: CellData, train_metadata: dict[str, Any]) -> "HarmonyKNN":
        if not self.harmony_covariate:
            raise MethodNotApplicable("Harmony covariate must be prespecified")
        if train_data.batch is None:
            raise MethodNotApplicable(
                "Harmony requires CellData.batch for the configured covariate"
            )
        try:
            import harmonypy  # type: ignore[import-not-found]
        except ImportError as exc:
            raise MethodNotApplicable("harmonypy is not installed") from exc
        batches = train_data.batch.astype(str).reshape(-1, 1)
        if len(np.unique(batches)) < 2:
            raise MethodNotApplicable("Harmony requires at least two training batches")
        n_components = min(self.components, train_data.X.shape[0], train_data.X.shape[1])
        if n_components < 1:
            raise BaselineError("Harmony PCA has no valid components")
        self.gene_ids = train_data.gene_ids
        self.pca = PCA(n_components=n_components, random_state=self.seed).fit(train_data.X)
        raw = self.pca.transform(train_data.X)
        metadata = pd.DataFrame({self.harmony_covariate: batches[:, 0]})
        try:
            harmony = harmonypy.run_harmony(
                raw, metadata, [self.harmony_covariate], random_state=self.seed
            )
            corrected = np.asarray(harmony.Z_corr, dtype=np.float64)
            if corrected.shape == (raw.shape[1], raw.shape[0]):
                corrected = corrected.T
            if corrected.shape != raw.shape:
                raise BaselineError("Harmony returned an unexpected corrected embedding shape")
            self.harmony_version = str(getattr(harmonypy, "__version__", "unknown"))
        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
            raise MethodNotApplicable(f"Harmony training failed: {exc}") from exc
        self.batch_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoded = self.batch_encoder.fit_transform(batches)
        self.query_map = Ridge(alpha=1.0).fit(np.column_stack([raw, encoded]), corrected)
        transformed = self._transform(train_data)
        self.knn = KNeighborsClassifier(
            n_neighbors=min(self.n_neighbors, len(train_data.state)), weights="distance"
        ).fit(transformed, train_data.state)
        self.applicability = "completed_train_only_with_frozen_query_projection"
        return self

    def _transform(self, data: CellData) -> NDArray[Any]:
        if self.pca is None or self.batch_encoder is None or self.query_map is None:
            raise BaselineError("Harmony model is unfitted")
        if data.batch is None:
            raise BaselineError("Harmony query transform requires batch values")
        raw = self.pca.transform(data.X)
        encoded = self.batch_encoder.transform(data.batch.astype(str).reshape(-1, 1))
        transformed = self.query_map.predict(np.column_stack([raw, encoded]))
        if not np.isfinite(transformed).all():
            raise BaselineError("Harmony transformed embedding is not finite")
        return transformed

    def predict(self, data: CellData, metadata: dict[str, Any]) -> NDArray[Any]:
        if self.knn is None:
            raise BaselineError("Harmony kNN model is unfitted")
        return cast(NDArray[Any], self.knn.predict(self._transform(data)))

    def predict_proba(self, data: CellData, metadata: dict[str, Any]) -> NDArray[Any]:
        if self.knn is None:
            raise BaselineError("Harmony kNN model is unfitted")
        probabilities = self.knn.predict_proba(self._transform(data))
        ordered = np.zeros((len(data.X), len(STATE_LABELS)), dtype=np.float64)
        for source, label in enumerate(self.knn.classes_.tolist()):
            if label not in STATE_LABELS:
                raise BaselineError(f"Unexpected classifier state {label!r}")
            ordered[:, STATE_LABELS.index(label)] = probabilities[:, source]
        return ordered

    def save(self, path: Path) -> None:
        if self.knn is None or self.gene_ids is None:
            raise BaselineError("Cannot save an unfitted Harmony baseline")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            pickle.dump(self, stream)

    @classmethod
    def load(cls, path: Path) -> "HarmonyKNN":
        with path.open("rb") as stream:
            result = pickle.load(stream)
        if not isinstance(result, cls):
            raise BaselineError("Serialized model is not a HarmonyKNN")
        return result

    def get_run_metadata(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "components": self.components,
            "n_neighbors": self.n_neighbors,
            "harmony_covariate": self.harmony_covariate,
            "seed": self.seed,
            "applicability": self.applicability,
            "harmony_version": self.harmony_version,
            "analysis_type": "harmony_train_only_frozen_query_projection_knn",
        }
