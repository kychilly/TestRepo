"""Training-only PCA followed by multinomial logistic regression."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from .base import Baseline, BaselineError, CellData, STATE_LABELS, model_hash


class PCALogReg(Baseline):
    """Frozen-gene, training-only scaling/PCA/multinomial logistic regression."""

    method = "pca_logreg"

    def __init__(
        self,
        components: int = 16,
        class_weight: str | dict[str, float] | None = "balanced",
        max_iter: int = 1000,
        C: float = 1.0,
        seed: int = 17,
    ) -> None:
        self.components = components
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.C = C
        self.seed = seed
        self.pipeline: Pipeline | None = None
        self.gene_ids: tuple[str, ...] | None = None
        self.convergence_status: bool | None = None
        self.coefficient_norm: float | None = None

    def fit(self, train_data: CellData, train_metadata: dict[str, Any]) -> "PCALogReg":
        if len(set(train_data.state.tolist())) < 2:
            raise BaselineError("Training data must contain at least two state classes")
        if self.components > min(train_data.X.shape):
            raise BaselineError("PCA components exceed training matrix rank bound")
        self.gene_ids = train_data.gene_ids
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=self.components, random_state=self.seed)),
                (
                    "classifier",
                    LogisticRegression(
                        # multi_class="multinomial", Removed since scvi_probe and harmony dont touch this method, but might need to undo in the future
                        class_weight=self.class_weight,
                        max_iter=self.max_iter,
                        C=self.C,
                        random_state=self.seed,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        self.pipeline.fit(train_data.X, train_data.state)
        classifier = self.pipeline.named_steps["classifier"]
        self.convergence_status = bool(classifier.n_iter_.max() < self.max_iter)
        self.coefficient_norm = float(np.linalg.norm(classifier.coef_))
        return self

    def _check(self, data: CellData) -> Pipeline:
        if self.pipeline is None or self.gene_ids != data.gene_ids:
            raise BaselineError(
                "Baseline is unfitted or input genes differ from training genes"
            )
        return self.pipeline

    def predict(self, data: CellData, metadata: dict[str, Any]) -> NDArray[Any]:
        return cast(NDArray[Any], self._check(data).predict(data.X))

    def predict_proba(self, data: CellData, metadata: dict[str, Any]) -> NDArray[Any]:
        pipeline = self._check(data)
        classifier = pipeline.named_steps["classifier"]
        probabilities = pipeline.predict_proba(data.X)
        return _state_order(probabilities, classifier.classes_)

    def save(self, path: Path) -> None:
        self._check_saved()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            pickle.dump(self, stream)

    @classmethod
    def load(cls, path: Path) -> "PCALogReg":
        with path.open("rb") as stream:
            result = pickle.load(stream)
        if not isinstance(result, cls):
            raise BaselineError("Serialized model is not a PCALogReg")
        return result

    def get_run_metadata(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "components": self.components,
            "class_weight": self.class_weight,
            "max_iter": self.max_iter,
            "C": self.C,
            "seed": self.seed,
            "convergence_status": self.convergence_status,
            "coefficient_norm": self.coefficient_norm,
            "model_hash": model_hash(
                {
                    "method": self.method,
                    "components": self.components,
                    "C": self.C,
                    "seed": self.seed,
                }
            ),
        }

    def _check_saved(self) -> None:
        if self.pipeline is None or self.gene_ids is None:
            raise BaselineError("Cannot save an unfitted baseline")


def _state_order(probabilities: NDArray[Any], classes: NDArray[Any]) -> NDArray[Any]:
    """Expand classifier columns into the fixed AC/MES/NPC/OPC order."""
    ordered = np.zeros((probabilities.shape[0], len(STATE_LABELS)), dtype=np.float64)
    for source, label in enumerate(classes.tolist()):
        if label not in STATE_LABELS:
            raise BaselineError(f"Unexpected classifier state {label!r}")
        ordered[:, STATE_LABELS.index(label)] = probabilities[:, source]
    return ordered
