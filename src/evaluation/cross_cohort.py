"""Leakage-resistant TCGA/Neftel to CGGA patient-level evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from numpy.typing import NDArray


CONFIRMED_OUTCOMES = {"destabilizing_driver", "functional_driver"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rank_rows(values: NDArray[Any]) -> NDArray[np.float32]:
    """Map each sample to within-sample percentile ranks.

    The source files use incompatible units (TCGA rank-like values and
    normalized Neftel/CGGA expression).  A within-sample transform is fixed in
    advance and does not inspect labels from any cohort.
    """
    from scipy.stats import rankdata

    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expression must be a finite sample-by-gene matrix")
    denominator = max(1, values.shape[1] - 1)
    return np.vstack(
        [(rankdata(row, method="average") - 1.0) / denominator for row in values]
    ).astype(np.float32)


def _dense(value: Any) -> NDArray[Any]:
    return np.asarray(value.toarray() if hasattr(value, "toarray") else value)  # type: ignore[no-any-return]


def _labels(values: Iterable[Any]) -> NDArray[np.int8]:
    mapping = {"WT": 0, "Wildtype": 0, "Mutant": 1}
    return np.asarray([mapping[str(value)] for value in values], dtype=np.int8)


def load_cohorts(tcga_path: Path, full_path: Path) -> dict[str, Any]:
    """Load aligned TCGA, Neftel pseudobulk, and labeled held-out CGGA."""
    import anndata as ad

    tcga = ad.read_h5ad(tcga_path)
    full = ad.read_h5ad(full_path)
    common = sorted(set(map(str, tcga.var_names)) & set(map(str, full.var_names)))
    if len(common) < 100:
        raise ValueError("Fewer than 100 genes overlap TCGA and full-cohort files")

    tcga_obs = tcga.obs
    tcga_valid = tcga_obs["IDH_status"].astype(str).isin(["WT", "Wildtype", "Mutant"]).to_numpy()
    x_tcga = rank_rows(_dense(tcga[tcga_valid, common].X))
    y_tcga = _labels(tcga_obs.loc[tcga_valid, "IDH_status"])
    id_tcga = tcga_obs.index[tcga_valid].astype(str).to_numpy()

    cohort = full.obs["Cohort"].astype(str)
    neftel_mask = (cohort == "Neftel").to_numpy()
    neftel = full[neftel_mask, common]
    neftel_cell = rank_rows(_dense(neftel.X))
    neftel_samples = neftel.obs["Sample"].astype(str).to_numpy()
    patient_ids = sorted(set(neftel_samples))
    x_neftel = np.vstack(
        [neftel_cell[neftel_samples == patient].mean(axis=0) for patient in patient_ids]
    ).astype(np.float32)
    # The source dataset is explicitly IDH-wildtype GBM. This is a cohort-level
    # label, not an inference from expression or CGGA test labels.
    y_neftel = np.zeros(len(patient_ids), dtype=np.int8)

    cgga_valid = (
        (cohort == "CGGA") & full.obs["IDH_status"].astype(str).isin(["WT", "Wildtype", "Mutant"])
    ).to_numpy()
    x_cgga = rank_rows(_dense(full[cgga_valid, common].X))
    y_cgga = _labels(full.obs.loc[cgga_valid, "IDH_status"])
    id_cgga = full.obs.index[cgga_valid].astype(str).to_numpy()

    return {
        "genes": np.asarray(common, dtype=str),
        "tcga": (x_tcga, y_tcga, id_tcga),
        "neftel": (x_neftel, y_neftel, np.asarray(patient_ids, dtype=str)),
        "cgga": (x_cgga, y_cgga, id_cgga),
    }


def feature_groups(genes: NDArray[Any], verdicts_path: Path) -> dict[str, NDArray[np.int64]]:
    import pandas as pd

    verdicts = pd.read_csv(verdicts_path)
    required = {"gene", "outcome"}
    if not required.issubset(verdicts.columns):
        raise ValueError(f"Verdict table must contain {sorted(required)}")
    confirmed = set(verdicts.loc[verdicts["outcome"].isin(CONFIRMED_OUTCOMES), "gene"].astype(str))
    candidates = set(verdicts["gene"].astype(str))
    gene_list = list(map(str, genes))
    confirmed_idx = np.asarray([i for i, gene in enumerate(gene_list) if gene in confirmed])
    unconfirmed_idx = np.asarray(
        [i for i, gene in enumerate(gene_list) if gene in candidates and gene not in confirmed]
    )
    if len(confirmed_idx) == 0 or len(unconfirmed_idx) == 0:
        raise ValueError("Both confirmed and unconfirmed feature groups must be non-empty")
    return {
        "all_genes": np.arange(len(genes)),
        "confirmed_genes": confirmed_idx,
        "unconfirmed_genes": unconfirmed_idx,
    }


def binary_scores(y_true: NDArray[Any], probability: NDArray[Any]) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    prediction = (probability >= 0.5).astype(np.int8)
    return {
        "macro_f1": float(f1_score(y_true, prediction, average="macro")),
        "idh_mutation_auroc": float(roc_auc_score(y_true, probability)),
        "accuracy": float(accuracy_score(y_true, prediction)),
    }


def fit_arm(
    data: dict[str, Any], feature_idx: NDArray[np.int64], *, seed: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x_tcga, y_tcga, id_tcga = data["tcga"]
    x_neftel, y_neftel, _ = data["neftel"]
    x_external, y_external, external_ids = data["cgga"]
    train_idx, internal_idx = train_test_split(
        np.arange(len(y_tcga)), test_size=0.25, random_state=seed, stratify=y_tcga
    )
    x_train = np.vstack([x_tcga[train_idx][:, feature_idx], x_neftel[:, feature_idx]])
    y_train = np.concatenate([y_tcga[train_idx], y_neftel])
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced", max_iter=5000, random_state=seed, solver="liblinear"
        ),
    )
    model.fit(x_train, y_train)
    p_internal = model.predict_proba(x_tcga[internal_idx][:, feature_idx])[:, 1]
    p_external = model.predict_proba(x_external[:, feature_idx])[:, 1]
    internal = binary_scores(y_tcga[internal_idx], p_internal)
    external = binary_scores(y_external, p_external)
    result = {
        "status": "completed",
        "seed": seed,
        "feature_count": int(len(feature_idx)),
        "fit_samples": int(len(y_train)),
        "internal_test_samples": int(len(internal_idx)),
        "external_test_samples": int(len(y_external)),
        "internal": internal,
        "external": external,
        "internal_to_external_macro_f1_drop": float(internal["macro_f1"] - external["macro_f1"]),
    }
    predictions: list[dict[str, Any]] = []
    for scope, ids, labels, probabilities in (
        ("internal_tcga", id_tcga[internal_idx], y_tcga[internal_idx], p_internal),
        ("external_cgga", external_ids, y_external, p_external),
    ):
        predictions.extend(
            {
                "scope": scope,
                "sample_id": str(sample_id),
                "true_idh_mutant": int(label),
                "probability_idh_mutant": float(probability),
                "predicted_idh_mutant": int(probability >= 0.5),
                "seed": seed,
            }
            for sample_id, label, probability in zip(ids, labels, probabilities)
        )
    return result, predictions


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        by_arm.setdefault(str(run["arm"]), []).append(run)
    summaries: dict[str, Any] = {}
    for arm, values in by_arm.items():
        summaries[arm] = {}
        metrics: tuple[tuple[str, Callable[[dict[str, Any]], float]], ...] = (
            ("internal_macro_f1", lambda row: row["internal"]["macro_f1"]),
            ("external_macro_f1", lambda row: row["external"]["macro_f1"]),
            ("external_idh_mutation_auroc", lambda row: row["external"]["idh_mutation_auroc"]),
            (
                "internal_to_external_macro_f1_drop",
                lambda row: row["internal_to_external_macro_f1_drop"],
            ),
        )
        for key, getter in metrics:
            measured_values = [float(getter(row)) for row in values]
            measured = np.asarray(measured_values, dtype=np.float64)
            summaries[arm][key] = {
                "mean": float(measured.mean()),
                "standard_deviation": float(measured.std(ddof=1)) if len(measured) > 1 else 0.0,
                "values": measured.tolist(),
            }
    confirmed = summaries["confirmed_genes"]["internal_to_external_macro_f1_drop"]["mean"]
    unconfirmed = summaries["unconfirmed_genes"]["internal_to_external_macro_f1_drop"]["mean"]
    shuffled = summaries["shuffled_size_matched_control"]["internal_to_external_macro_f1_drop"][
        "mean"
    ]
    confirmed_external = summaries["confirmed_genes"]["external_macro_f1"]["mean"]
    shuffled_external = summaries["shuffled_size_matched_control"]["external_macro_f1"]["mean"]
    confirmed_beats_shuffled = confirmed < shuffled and confirmed_external > shuffled_external
    return {
        "arms": summaries,
        "headline": {
            "confirmed_gene_drop": confirmed,
            "unconfirmed_gene_drop": unconfirmed,
            "shuffled_control_drop": shuffled,
            "confirmed_drop_advantage": float(unconfirmed - confirmed),
            "confirmed_beats_shuffled": confirmed_beats_shuffled,
            "result_call": "candidate_result" if confirmed_beats_shuffled else "no_result",
            "interpretation": "Positive means the confirmed-gene model lost less performance externally.",
        },
    }


def load_completed_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
