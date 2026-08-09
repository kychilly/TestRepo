#!/usr/bin/env python3
"""Run one patient-aware conventional baseline and write auditable outputs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from baselines.base import (
    BaselineError,
    CellData,
    MethodNotApplicable,
    assign_cells,
    config_hash,
    evaluate_predictions,
    load_patient_splits,
    model_hash,
    runtime_metadata,
)
from baselines.harmony_knn import HarmonyKNN
from baselines.pca_logreg import PCALogReg
from baselines.scvi_probe import ScVIProbe


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-untyped]

        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise BaselineError(f"Cannot read baseline config {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BaselineError("Baseline config must be a YAML object")
    return payload


def load_data(path: Path, config: dict[str, Any], method: str | None = None) -> CellData:
    """Load Jeffrey's processed H5AD or the existing intermediate NPZ format."""
    if "cgga" in str(path).lower():
        raise BaselineError("CGGA data are prohibited for the Neftel baseline track")
    try:
        if path.suffix.lower() == ".h5ad":
            try:
                import anndata as ad  # type: ignore[import-untyped]
            except ImportError as exc:
                raise BaselineError("H5AD input requires anndata") from exc
            adata = ad.read_h5ad(path)
            patient_key = str(config.get("patient_id_key", "patient_id"))
            cell_key = str(config.get("cell_id_key", "cell_id"))
            state_key = str(config.get("state_key", "state"))
            gene_key = str(config.get("gene_id_key", "gene_ids"))
            for key in (patient_key, state_key):
                if key not in adata.obs:
                    raise BaselineError(f"H5AD obs is missing configured column {key!r}")
            patient_id = adata.obs[patient_key].astype(str).to_numpy()
            cell_id = (
                adata.obs[cell_key].astype(str).to_numpy()
                if cell_key in adata.obs
                else np.asarray(adata.obs_names).astype(str)
            )
            gene_ids = (
                adata.var[gene_key].astype(str).tolist()
                if gene_key in adata.var
                else [str(value) for value in adata.var_names]
            )
            batch_key = config.get("batch_key")
            batch = (
                adata.obs[str(batch_key)].astype(str).to_numpy()
                if batch_key and str(batch_key) in adata.obs
                else None
            )
            if method == "scvi_probe":
                if "counts" not in adata.layers:
                    raise BaselineError(
                        "scVI requires raw integer counts in H5AD layer 'counts'; "
                        "Jeffrey's QC output contains log-normalized X and cannot be used for scVI"
                    )
                matrix_source = adata.layers["counts"]
            else:
                matrix_source = adata.X
            matrix = (
                matrix_source.toarray()
                if hasattr(matrix_source, "toarray")
                else np.asarray(matrix_source)
            )
            return CellData(
                np.asarray(matrix),
                patient_id,
                cell_id,
                adata.obs[state_key].astype(str).to_numpy(),
                tuple(gene_ids),
                batch,
            )
        with np.load(path, allow_pickle=False) as archive:
            required = {"X", "patient_id", "cell_id", "state", "gene_ids"}
            if set(archive.files) < required:
                raise BaselineError(f"NPZ data must contain {sorted(required)}")
            batch = archive["batch"] if "batch" in archive.files else None
            data = CellData(
                archive["X"],
                archive["patient_id"].astype(str),
                archive["cell_id"].astype(str),
                archive["state"].astype(str),
                tuple(archive["gene_ids"].astype(str).tolist()),
                batch,
            )
            if any("cgga" in value.lower() for value in data.patient_id.tolist()):
                raise BaselineError(
                    "CGGA patients are prohibited for the Neftel baseline track"
                )
            return data
    except OSError as exc:
        raise BaselineError(f"Cannot read data {path}: {exc}") from exc


def build_baseline(method: str, settings: dict[str, Any], seed: int) -> Any:
    if method == "pca_logreg":
        return PCALogReg(
            components=int(settings.get("components", 16)),
            class_weight=settings.get("class_weight"),
            max_iter=int(settings.get("max_iter", 1000)),
            C=float(settings.get("C", 1.0)),
            seed=seed,
        )
    if method == "scvi_probe":
        return ScVIProbe(
            latent_size=int(settings.get("latent_size", 10)),
            epochs=int(settings.get("epochs", 100)),
            seed=seed,
            count_layer=str(settings.get("count_layer", "X")),
            batch_key=settings.get("batch_key"),
        )
    if method == "harmony_knn":
        return HarmonyKNN(
            components=int(settings.get("components", 16)),
            n_neighbors=int(settings.get("n_neighbors", 15)),
            harmony_covariate=settings.get("covariate"),
            seed=seed,
        )
    raise BaselineError(f"Unknown method {method!r}")


def failure_record(
    method: str, fold: int, seed: int, reason: str, config: dict[str, Any]
) -> dict[str, Any]:
    return {
        "status": "not_applicable",
        "method": method,
        "fold": fold,
        "seed": seed,
        "reason": reason,
        "config_hash": config_hash(config),
        "runtime": runtime_metadata(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", required=True, choices=("pca_logreg", "scvi_probe", "harmony_knn")
    )
    parser.add_argument("--adata", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_yaml(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        data = load_data(args.adata, config, args.method)
        splits = load_patient_splits(args.splits, args.fold)
        if any(
            "cgga" in patient.lower()
            for patients in splits.as_dict().values()
            for patient in patients
        ):
            raise BaselineError("CGGA patients are prohibited in the patient split")
        assignments = assign_cells(data, splits)
        method_config = dict(config.get(args.method, {}))
        frozen_genes = config.get("frozen_genes")
        if frozen_genes is not None and tuple(frozen_genes) != data.gene_ids:
            raise BaselineError(
                "Input genes do not exactly match the frozen preprocessing gene list"
            )
        train = data.subset(assignments["train"])
        train_metadata = {"split": "train", "batch": train.batch}
        validation = data.subset(assignments["validation"])
        validation_selection: dict[str, Any] = {"rule": "not_applicable"}
        candidates = method_config.get("C_candidates", [method_config.get("C", 1.0)])
        if args.method == "pca_logreg" and isinstance(candidates, list) and candidates:
            scored: list[tuple[float, float]] = []
            for candidate in candidates:
                candidate_config = {**method_config, "C": float(candidate)}
                candidate_model = build_baseline(
                    args.method, candidate_config, args.seed
                )
                candidate_model.fit(train, train_metadata)
                score = float(
                    np.mean(
                        candidate_model.predict(validation, {"split": "validation"})
                        == validation.state
                    )
                )
                scored.append((score, float(candidate)))
            best_score, best_c = max(scored, key=lambda item: (item[0], -item[1]))
            method_config["C"] = best_c
            validation_selection = {
                "rule": "validation_cell_accuracy_max",
                "candidates": [value for _, value in scored],
                "selected_C": best_c,
                "selected_validation_accuracy": best_score,
            }
        baseline = build_baseline(args.method, method_config, args.seed)
        started = time.perf_counter()
        baseline.fit(train, train_metadata)
        fit_seconds = time.perf_counter() - started
        run = {
            "run_id": f"{args.method}_fold{args.fold}_seed{args.seed}",
            "method": args.method,
            "fold": args.fold,
            "seed": args.seed,
            "split_hash": splits.split_hash,
            "config_hash": config_hash(config),
            "model_hash": model_hash(baseline.get_run_metadata()),
        }
        predictions, patients = evaluate_predictions(baseline, data, assignments, run)
        for row in predictions:
            row["fit_seconds"] = fit_seconds
        (args.output / "predictions.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions),
            encoding="utf-8",
        )
        (args.output / "patient_summary.json").write_text(
            json.dumps(patients, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (args.output / "run_metadata.json").write_text(
            json.dumps(
                {
                    **run,
                    "status": "completed",
                    "baseline": baseline.get_run_metadata(),
                    "validation_selection": validation_selection,
                    "runtime": runtime_metadata(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return 0
    except (BaselineError, MethodNotApplicable, OSError, ValueError) as exc:
        record = failure_record(args.method, args.fold, args.seed, str(exc), config)
        (args.output / "run_error.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(record, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
