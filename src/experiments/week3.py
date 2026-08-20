"""Week 3 multi-seed, one-factor-at-a-time ablation engine."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from gbm_study.plain_english import write_json_with_explanation, write_jsonl_explanation


@dataclass(frozen=True)
class Ablation:
    name: str
    validator: bool
    grn: bool
    mc_dropout: bool


def ablation_matrix() -> tuple[Ablation, ...]:
    """Reference plus three runs, each changing exactly one variable."""
    return (
        Ablation("all_on", True, True, True),
        Ablation("validator_off", False, True, True),
        Ablation("grn_off", True, False, True),
        Ablation("mc_dropout_off", True, True, False),
    )


def select_backbones(
    timing: Mapping[str, Any],
    *,
    seeds: int,
    ablations: int,
    budget_gpu_seconds: float,
    mc_dropout_passes: int = 1,
    ranking_genes: int = 0,
) -> dict[str, Any]:
    """Allow extra backbones only from a measured Week 2 scGPT timing."""
    value = timing.get("timing", {}).get("projected_gpu_seconds_per_10000_cells")
    if timing.get("status") != "completed" or value is None:
        return {
            "backbones": ["scGPT"],
            "branch": "phase1_scgpt_only",
            "reason": "Week 2 measured scGPT timing is unavailable",
            "estimated_scgpt_gpu_seconds": None,
        }
    # Reference, validator-off, and GRN-off retain MC dropout; MC-off is one pass.
    mc_runs = max(0, ablations - 1)
    pass_factor = mc_runs * mc_dropout_passes + 1
    # One unmasked inference plus one inference per ranked gene.
    estimate = float(value) * seeds * pass_factor * (1 + ranking_genes)
    if estimate > budget_gpu_seconds:
        return {
            "backbones": ["scGPT"],
            "branch": "phase1_scgpt_only",
            "reason": f"Measured scGPT matrix estimate {estimate:.3f}s exceeds budget {budget_gpu_seconds:.3f}s",
            "estimated_scgpt_gpu_seconds": estimate,
        }
    return {
        "backbones": ["scGPT", "CellFM", "Geneformer"],
        "branch": "three_backbones",
        "reason": "Measured Week 2 scGPT timing fits the configured compute budget",
        "estimated_scgpt_gpu_seconds": estimate,
    }


def _hash_config(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("JSON artifacts must be objects so they can be explained")
    write_json_with_explanation(path, value)


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    write_jsonl_explanation(
        path,
        row_count=len(rows),
        description="The job saved one machine-readable record on each line.",
    )


Runner = Callable[[Mapping[str, Any], Ablation, int, str], Mapping[str, Any]]

SCIENTIFIC_GUIDELINES = [
    "Use the same patient fold and the same seeds for every ablation.",
    "Change exactly one of validator, GRN, or MC dropout at a time.",
    "Fit probes on training patients only; never use test patients for fitting or ranking.",
    "MC dropout must stay active for 20-50 passes and report mean, variance, and timing.",
    "Do not report synthetic fixtures, blocked runs, or CPU smoke timing as scientific results.",
    "Use CellFM and Geneformer only when measured scGPT timing fits the compute budget.",
    "Keep checkpoint, vocabulary, split, config, seed, and evidence provenance with outputs.",
]


def run_matrix(config: Mapping[str, Any], runner: Runner, output: Path) -> dict[str, Any]:
    seeds = [int(value) for value in config.get("seeds", [17, 42, 101])]
    if len(set(seeds)) < 3:
        raise ValueError("At least three distinct seeds are required")
    timing_path = Path(str(config["week2_timing_path"]))
    timing = json.loads(timing_path.read_text(encoding="utf-8")) if timing_path.is_file() else {"status": "blocked"}
    matrix = ablation_matrix()
    scope = select_backbones(
        timing,
        seeds=len(seeds),
        ablations=len(matrix),
        budget_gpu_seconds=float(config.get("backbone_budget_gpu_seconds", 86400.0)),
        mc_dropout_passes=int(config.get("mc_dropout_passes", 20)),
        ranking_genes=int(config.get("estimated_rank_genes", 2503)),
    )
    config_hash = _hash_config(config)
    runs: list[dict[str, Any]] = []
    for backbone in scope["backbones"]:
        for seed in seeds:
            for ablation in matrix:
                run_id = f"week3-{backbone.lower()}-seed{seed}-{ablation.name}"
                run_dir = output / run_id
                try:
                    run_config = dict(config)
                    run_config["_run_dir"] = str(run_dir)
                    raw = dict(runner(run_config, ablation, seed, backbone))
                    embeddings = np.asarray(raw.pop("embeddings"), dtype=np.float32)
                    embedding_variance = raw.pop("embedding_variance", None)
                    embedding_cell_ids = raw.pop("embedding_cell_ids", None)
                    embedding_patient_ids = raw.pop("embedding_patient_ids", None)
                    rankings = [dict(row) for row in raw.pop("rankings")]
                    predictions = [dict(row) for row in raw.pop("predictions")]
                    if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
                        raise ValueError("runner embeddings must be a finite two-dimensional array")
                    if not rankings or not predictions:
                        raise ValueError("runner must return non-empty rankings and predictions")
                    arrays: dict[str, Any] = {"embeddings": embeddings}
                    if embedding_cell_ids is not None:
                        ids = np.asarray(embedding_cell_ids, dtype=str)
                        if len(ids) != len(embeddings):
                            raise ValueError("embedding cell IDs must match embedding rows")
                        arrays["cell_ids"] = ids
                    if embedding_patient_ids is not None:
                        patients = np.asarray(embedding_patient_ids, dtype=str)
                        if len(patients) != len(embeddings):
                            raise ValueError("embedding patient IDs must match embedding rows")
                        arrays["patient_ids"] = patients
                    if embedding_variance is not None:
                        variance = np.asarray(embedding_variance, dtype=np.float32)
                        if variance.shape != embeddings.shape or not np.isfinite(variance).all():
                            raise ValueError("embedding variance must match embeddings")
                        arrays["embedding_variance"] = variance
                    run_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(run_dir / "embeddings.npz", **arrays)
                    _write_jsonl(run_dir / "rankings.jsonl", rankings)
                    _write_jsonl(run_dir / "predictions.jsonl", predictions)
                    record = {
                        "status": "completed",
                        "run_id": run_id,
                        "backbone": backbone,
                        "seed": seed,
                        "ablation": asdict(ablation),
                        "artifacts": {
                            "embeddings": str(run_dir / "embeddings.npz"),
                            "rankings": str(run_dir / "rankings.jsonl"),
                            "predictions": str(run_dir / "predictions.jsonl"),
                        },
                        **raw,
                    }
                except (ImportError, OSError, RuntimeError, ValueError, KeyError) as exc:
                    record = {
                        "status": "blocked",
                        "run_id": run_id,
                        "backbone": backbone,
                        "seed": seed,
                        "ablation": asdict(ablation),
                        "reason": str(exc),
                    }
                _write_json(run_dir / "run.json", record)
                runs.append(record)
    result = {
        "status": "completed" if runs and all(run["status"] == "completed" for run in runs) else "completed_with_blockers",
        "config_hash": config_hash,
        "seeds": seeds,
        "ablation_design": "one_variable_at_a_time",
        "backbone_scope": scope,
        "runs": runs,
        "completed_runs": sum(run["status"] == "completed" for run in runs),
        "blocked_runs": sum(run["status"] != "completed" for run in runs),
        "scientific_guidelines": SCIENTIFIC_GUIDELINES,
        "checkpointing": {
            "enabled": True,
            "resume_policy": "same run_id, config fingerprint, seed, fold, backbone, and ablation",
            "gene_progress_file": "ranking_checkpoint.jsonl",
            "status_file": "checkpoint_status.json",
        },
    }
    _write_json(output / "manifest.json", result)
    return result
