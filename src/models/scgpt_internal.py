"""Actual internal-cohort scGPT embedding, probe, and mask-ranking run."""

from __future__ import annotations

import importlib
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score

from baselines.base import load_patient_splits
from experiments.week3 import Ablation, SCIENTIFIC_GUIDELINES
from gbm_study.plain_english import write_json_with_explanation, write_jsonl_explanation
from models.candidate_scoring import aggregate_mask_delta_scores
from models.grn import load_edges
from models.mc_dropout import infer_mc_dropout
from models.scgpt_adapter import PreparedInputs, ScGPTAdapter
from models.scgpt_loader import load_official_scgpt, load_vocabulary


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _candidate_genes(config: Mapping[str, Any], available: list[str], validator: bool) -> list[str]:
    value = config.get("candidate_genes_path")
    candidates = (
        available
        if not value
        else [str(row.get("gene", "")).upper() for row in _rows(Path(str(value)))]
    )
    candidates = list(dict.fromkeys(gene for gene in candidates if gene))
    if not validator:
        limit = config.get("candidate_limit")
        if limit is not None:
            limit = int(limit)
            if limit < 1:
                raise ValueError("candidate_limit must be positive")
            candidates = candidates[:limit]
        return candidates
    outcomes_value = config.get("validator_outcomes_path")
    if not outcomes_value:
        raise ValueError("validator_outcomes_path is required for validator-on ablations")
    confirmed = {
        str(row.get("gene", "")).upper()
        for row in _rows(Path(str(outcomes_value)))
        if row.get("outcome") in {"destabilizing_driver", "functional_driver"}
    }
    selected = [gene for gene in candidates if gene in confirmed]
    if not selected:
        raise ValueError("validator-on ablation retained no confirmed candidate genes")
    limit = config.get("candidate_limit")
    if limit is not None:
        limit = int(limit)
        if limit < 1:
            raise ValueError("candidate_limit must be positive")
        selected = selected[:limit]
    return selected


def _with_grn_neighbors(genes: list[str], config: Mapping[str, Any], enabled: bool) -> list[str]:
    if not enabled:
        return genes
    value = config.get("grn_train_prior_path")
    if not value:
        raise ValueError("grn_train_prior_path is required for GRN-on ablations")
    edges = load_edges(Path(str(value)))
    original = set(genes)
    selected = set(original)
    for edge in edges:
        source, target = edge.data["source_gene"], edge.data["target_gene"]
        if source in original or target in original:
            selected.update((source, target))
    return sorted(selected)


def _decision_column(
    probe: LogisticRegression, embeddings: np.ndarray[Any, Any], state: str
) -> np.ndarray[Any, Any]:
    if state not in probe.classes_:
        raise ValueError(f"target state {state!r} is absent from training labels")
    decision = np.asarray(probe.decision_function(embeddings))
    if decision.ndim == 1:
        positive = decision if probe.classes_[1] == state else -decision
        return positive
    return decision[:, list(probe.classes_).index(state)]


def _infer(
    adapter: ScGPTAdapter,
    prepared: PreparedInputs,
    config: Mapping[str, Any],
    mc: bool,
) -> tuple[
    np.ndarray[Any, Any],
    np.ndarray[Any, Any] | None,
    float | None,
    np.ndarray[Any, Any] | None,
]:
    batch = int(config.get("batch_size", 32))
    precision = str(config.get("precision", "float32"))
    if not mc:
        return adapter.infer(prepared, batch, precision), None, None, None
    passes = int(config.get("mc_dropout_passes", 20))
    if not 20 <= passes <= 50:
        raise ValueError("mc_dropout_passes must be between 20 and 50")
    result = infer_mc_dropout(
        adapter, prepared, n_passes=passes, batch_size=batch, precision=precision
    )
    return result.mean, result.variance, result.compute_multiplier, result.samples


def _fingerprint(
    config: Mapping[str, Any],
    ablation: Ablation,
    seed: int,
    backbone: str,
    provenance: Mapping[str, str],
    genes: list[str],
) -> str:
    protected = {
        "seed": seed,
        "fold": int(config.get("fold", 0)),
        "backbone": backbone,
        "ablation": {
            "validator": ablation.validator,
            "grn": ablation.grn,
            "mc_dropout": ablation.mc_dropout,
        },
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "vocabulary_sha256": provenance["vocabulary_sha256"],
        "genes": genes,
        "target_states": config.get("target_states"),
        "mc_dropout_passes": config.get("mc_dropout_passes"),
    }
    return hashlib.sha256(json.dumps(protected, sort_keys=True).encode()).hexdigest()


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _checkpoint_status(
    path: Path,
    *,
    fingerprint: str,
    stage: str,
    completed_genes: int,
    total_genes: int,
) -> None:
    write_json_with_explanation(
        path,
        {
            "status": "completed" if stage == "completed" else "running",
            "stage": stage,
            "resume_fingerprint": fingerprint,
            "completed_genes": completed_genes,
            "total_genes": total_genes,
            "scientific_guidelines": SCIENTIFIC_GUIDELINES,
            "next_actions": [
                "If the job stopped, run the same command again with the same config and output folder."
            ],
        },
    )


def run_internal_cohort(
    config: Mapping[str, Any], ablation: Ablation, seed: int, backbone: str
) -> Mapping[str, Any]:
    """Execute one real ablation. CellFM/Geneformer require explicit adapters."""
    if backbone != "scGPT":
        spec = config.get(f"{backbone.lower()}_runner")
        if not spec:
            raise ValueError(f"{backbone} was compute-approved but no real runner is configured")
        module, function = str(spec).split(":", 1)
        external_result = getattr(importlib.import_module(module), function)(
            config, ablation, seed, backbone
        )
        return cast(Mapping[str, Any], external_result)
    required = (
        "cell_data_path",
        "split_file",
        "checkpoint_path",
        "vocabulary_path",
        "patient_id_column",
        "state_key",
    )
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise ValueError("Missing scGPT experiment inputs: " + ", ".join(missing))
    device = str(config.get("requested_device", "cuda"))
    if not device.startswith("cuda"):
        raise ValueError("The full Week 3 scGPT run requires CUDA")
    import torch

    if not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import anndata as ad  # type: ignore[import-not-found]

    data_path = Path(str(config["cell_data_path"]))
    checkpoint = Path(str(config["checkpoint_path"]))
    vocabulary_path = Path(str(config["vocabulary_path"]))
    for path in (data_path, checkpoint, vocabulary_path, Path(str(config["split_file"]))):
        if not path.is_file():
            raise ValueError(f"Required experiment asset is missing: {path}")
    data = ad.read_h5ad(data_path)
    patient_key, state_key = str(config["patient_id_column"]), str(config["state_key"])
    if patient_key not in data.obs or state_key not in data.obs:
        raise ValueError("Configured patient/state columns are absent from the H5AD")
    available = [str(gene).upper() for gene in data.var_names]
    final_genes = _candidate_genes(config, available, ablation.validator)
    input_genes = _with_grn_neighbors(final_genes, config, ablation.grn)
    positions = [index for index, gene in enumerate(available) if gene in set(input_genes)]
    if not positions:
        raise ValueError("No selected candidate/GRN genes occur in the H5AD")
    matrix = data.X[:, positions]
    matrix = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    gene_ids = [available[index] for index in positions]
    patients = data.obs[patient_key].astype(str).to_numpy()
    states = data.obs[state_key].astype(str).to_numpy()
    split = load_patient_splits(Path(str(config["split_file"])), int(config.get("fold", 0)))
    train_mask = np.isin(patients, list(split.train))
    test_mask = np.isin(patients, list(split.test))
    if not train_mask.any() or not test_mask.any():
        raise ValueError("Patient split has no train or test cells in the cohort")
    vocabulary = load_vocabulary(vocabulary_path)
    model = load_official_scgpt(checkpoint, vocabulary_path, device, dict(config))
    adapter = ScGPTAdapter(model, vocabulary, checkpoint, vocabulary_path, device=device)
    prepared = adapter.prepare_inputs(
        {"X": matrix, "var_names": gene_ids}, str(config.get("gene_id_type", "HGNC"))
    )
    retained = list(prepared.report.retained)
    final_genes = [gene for gene in final_genes if gene in retained]
    if not final_genes:
        raise ValueError("No candidate genes map to the scGPT vocabulary")
    run_dir = Path(str(config.get("_run_dir", "reports/week3_adit/checkpoint")))
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "checkpoint_status.json"
    embedding_path = run_dir / "embedding_checkpoint.npz"
    ranking_path = run_dir / "ranking_checkpoint.jsonl"
    provenance = adapter.provenance()
    resume_fingerprint = _fingerprint(config, ablation, seed, backbone, provenance, final_genes)
    existing_status = (
        json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else None
    )
    if existing_status and existing_status.get("resume_fingerprint") != resume_fingerprint:
        raise ValueError(
            "Checkpoint settings changed. Use a new output folder or restore the original seed, fold, model, genes, and ablation."
        )
    started = time.perf_counter()
    if embedding_path.is_file() and existing_status:
        with np.load(embedding_path, allow_pickle=False) as saved:
            embeddings = np.asarray(saved["embeddings"], dtype=np.float32)
            embedding_samples = (
                np.asarray(saved["embedding_samples"], dtype=np.float32)
                if "embedding_samples" in saved.files
                else None
            )
            variance = (
                np.asarray(saved["embedding_variance"], dtype=np.float32)
                if "embedding_variance" in saved.files
                else None
            )
            multiplier_value = (
                float(saved["mc_compute_multiplier"])
                if "mc_compute_multiplier" in saved.files
                else float("nan")
            )
            multiplier = multiplier_value if np.isfinite(multiplier_value) else None
    else:
        embeddings, variance, multiplier, embedding_samples = _infer(
            adapter, prepared, config, ablation.mc_dropout
        )
        arrays: dict[str, Any] = {
            "embeddings": embeddings,
            "mc_compute_multiplier": np.asarray(multiplier if multiplier is not None else np.nan),
        }
        if variance is not None:
            arrays["embedding_variance"] = variance
        if embedding_samples is not None:
            arrays["embedding_samples"] = embedding_samples
        _atomic_npz(embedding_path, **arrays)
        _checkpoint_status(
            status_path,
            fingerprint=resume_fingerprint,
            stage="embeddings_saved",
            completed_genes=0,
            total_genes=len(final_genes),
        )
    probe = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    probe.fit(embeddings[train_mask], states[train_mask])
    probabilities = probe.predict_proba(embeddings[test_mask])
    predicted = probe.classes_[np.argmax(probabilities, axis=1)]
    target_states = [
        str(value) for value in config.get("target_states", [config.get("target_state", "MES")])
    ]
    if not target_states:
        raise ValueError("target_states must be non-empty")
    baseline_logits = {state: _decision_column(probe, embeddings, state) for state in target_states}
    saved_rankings = _rows(ranking_path) if ranking_path.is_file() else []
    ranking_map = {(str(row["state"]), str(row["gene"])): dict(row) for row in saved_rankings}
    completed_genes = {
        gene for gene in final_genes if all((state, gene) in ranking_map for state in target_states)
    }
    retained_index = {gene: index for index, gene in enumerate(retained)}
    for gene in final_genes:
        if gene in completed_genes:
            continue
        masked_values = prepared.values.copy()
        masked_values[:, retained_index[gene]] = 0.0
        masked_prepared = PreparedInputs(masked_values, prepared.token_ids, prepared.report)
        masked_embeddings, _, _, masked_samples = _infer(
            adapter, masked_prepared, config, ablation.mc_dropout
        )
        gene_results: list[dict[str, Any]] = []
        for state in target_states:
            masked_logits = _decision_column(probe, masked_embeddings, state)
            cell_rows: list[dict[str, Any]] = []
            for index in np.flatnonzero(train_mask & (states == state)):
                cell_rows.append(
                    {
                        "gene": gene,
                        "patient_id": str(patients[index]),
                        "cell_id": str(data.obs_names[index]),
                        "state": state,
                        "baseline_logit": float(baseline_logits[state][index]),
                        "masked_logit": float(masked_logits[index]),
                    }
                )
            aggregated = aggregate_mask_delta_scores(cell_rows, state=state)
            if aggregated:
                row = dict(aggregated[0])
                if (
                    ablation.mc_dropout
                    and embedding_samples is not None
                    and masked_samples is not None
                ):
                    sample_deltas = []
                    for sample_index in range(embedding_samples.shape[0]):
                        base_sample = _decision_column(
                            probe, embedding_samples[sample_index], state
                        )
                        masked_sample = _decision_column(probe, masked_samples[sample_index], state)
                        selected = np.flatnonzero(train_mask & (states == state))
                        sample_deltas.append(
                            float(np.mean(base_sample[selected] - masked_sample[selected]))
                        )
                    row["mc_score_mean"] = float(np.mean(sample_deltas))
                    row["mc_score_variance"] = float(np.var(sample_deltas))
                    row["mc_passes"] = len(sample_deltas)
                else:
                    row["mc_score_mean"] = None
                    row["mc_score_variance"] = None
                    row["mc_passes"] = 0
                row.update({"seed": seed, "backbone": backbone, "state": state})
                gene_results.append(row)
                ranking_map[(state, gene)] = row
        if gene_results:
            with ranking_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    "".join(json.dumps(row, sort_keys=True) + "\n" for row in gene_results)
                )
                stream.flush()
                os.fsync(stream.fileno())
            write_jsonl_explanation(
                ranking_path,
                row_count=len(ranking_map),
                description="This is live gene-ranking progress. The run may still be working.",
                status="running",
                next_actions=[
                    "If the job stopped, rerun the same command. Finished genes will be skipped."
                ],
            )
        completed_genes.add(gene)
        interval = int(config.get("checkpoint_every_genes", 10))
        if len(completed_genes) % max(interval, 1) == 0:
            _checkpoint_status(
                status_path,
                fingerprint=resume_fingerprint,
                stage="ranking_genes",
                completed_genes=len(completed_genes),
                total_genes=len(final_genes),
            )
    rankings: list[dict[str, Any]] = []
    for state in target_states:
        state_rankings = [
            dict(ranking_map[(state, gene)]) for gene in final_genes if (state, gene) in ranking_map
        ]
        state_rankings.sort(key=lambda row: (-float(row["score"]), str(row["gene"])))
        for rank, row in enumerate(state_rankings, 1):
            row.update({"rank": rank, "seed": seed, "backbone": backbone, "state": state})
        rankings.extend(state_rankings)
    predictions = []
    test_indices = np.flatnonzero(test_mask)
    for local, index in enumerate(test_indices):
        predictions.append(
            {
                "cell_id": str(data.obs_names[index]),
                "patient_id": str(patients[index]),
                "true_state": str(states[index]),
                "predicted_state": str(predicted[local]),
                "probabilities": {
                    str(label): float(probabilities[local, pos])
                    for pos, label in enumerate(probe.classes_)
                },
                "seed": seed,
            }
        )
    _checkpoint_status(
        status_path,
        fingerprint=resume_fingerprint,
        stage="completed",
        completed_genes=len(completed_genes),
        total_genes=len(final_genes),
    )
    return {
        "embeddings": embeddings,
        "embedding_variance": variance,
        "embedding_cell_ids": data.obs_names.astype(str).tolist(),
        "embedding_patient_ids": patients.tolist(),
        "rankings": rankings,
        "predictions": predictions,
        "metrics": {
            "macro_f1": float(f1_score(states[test_mask], predicted, average="macro")),
            "balanced_accuracy": float(balanced_accuracy_score(states[test_mask], predicted)),
        },
        "timing": {
            "wall_seconds": time.perf_counter() - started,
            "mc_compute_multiplier": multiplier,
        },
        "provenance": provenance,
        "resume_fingerprint": resume_fingerprint,
        "n_input_genes": len(prepared.report.retained),
        "n_ranked_genes": len(rankings),
        "n_final_mask_genes": len(final_genes),
        "ranked_states": target_states,
    }
