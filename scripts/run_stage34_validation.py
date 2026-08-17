#!/usr/bin/env python3
"""Run candidate-aligned Stage 4, feasibility pooling, and shuffled control."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import shuffled_validator
import validator
from validator import GeneRecord, Outcome, Thresholds, Verdict

OUTCOME_ORDER = tuple(outcome.value for outcome in Outcome)
CONFIRMED = {Outcome.DESTABILIZING_DRIVER, Outcome.FUNCTIONAL_DRIVER}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number} must be a JSON object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def _record(row: dict[str, Any]) -> GeneRecord:
    return GeneRecord(
        gene=str(row["gene"]).upper(),
        mutation=str(row["mutation"]),
        alteration_type=str(row["alteration_type"]),
        plddt=None if row.get("plddt") is None else float(row["plddt"]),
        esm1b=None if row.get("esm1b") is None else float(row["esm1b"]),
        ddg=None if row.get("ddg") is None else float(row["ddg"]),
    )


def _counts(verdicts: list[Verdict]) -> dict[str, int]:
    found = Counter(verdict.outcome.value for verdict in verdicts)
    return {outcome: int(found.get(outcome, 0)) for outcome in OUTCOME_ORDER}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_fixture(rows: list[dict[str, Any]]) -> bool:
    return any(
        "synthetic" in json.dumps(row, sort_keys=True).lower()
        or "example" in json.dumps(row, sort_keys=True).lower()
        for row in rows
    )


def _align(
    candidates: list[dict[str, Any]], evidence: list[GeneRecord], thresholds: Thresholds
) -> tuple[list[str], list[GeneRecord], list[Verdict], dict[str, Any]]:
    """Emit exactly one record/verdict per Stage 3 candidate, in candidate order."""
    by_gene: dict[str, list[GeneRecord]] = defaultdict(list)
    for record in evidence:
        by_gene[record.gene.upper()].append(record)
    candidate_ids: list[str] = []
    control_records: list[GeneRecord] = []
    verdicts: list[Verdict] = []
    missing: list[str] = []
    ambiguous: list[str] = []
    used_genes: set[str] = set()
    seen_ids: set[str] = set()
    for index, candidate in enumerate(candidates, 1):
        candidate_id = str(candidate.get("candidate_id", f"candidate-{index}"))
        if candidate_id in seen_ids:
            raise ValueError(f"Duplicate candidate_id: {candidate_id}")
        seen_ids.add(candidate_id)
        gene = str(candidate.get("gene", "")).upper()
        if not gene:
            raise ValueError(f"Candidate {candidate_id} has no gene")
        matches = by_gene.get(gene, [])
        candidate_ids.append(candidate_id)
        used_genes.add(gene)
        if len(matches) == 1:
            record = matches[0]
            verdict = validator.classify(record, thresholds)
        elif not matches:
            missing.append(candidate_id)
            record = GeneRecord(gene, "unknown", "missense", None, None, None)
            verdict = Verdict(
                gene,
                "unknown",
                Outcome.DATA_DEFICIENT,
                "Stage 3 candidate has no matching variant/protein evidence record",
            )
        else:
            ambiguous.append(candidate_id)
            record = GeneRecord(gene, "multiple", "other", None, None, None)
            verdict = Verdict(
                gene,
                "multiple",
                Outcome.ABSTAIN,
                "multiple validator records match this candidate; variants were not collapsed",
            )
        control_records.append(record)
        verdicts.append(verdict)
    return candidate_ids, control_records, verdicts, {
        "missing_evidence_candidate_ids": missing,
        "ambiguous_evidence_candidate_ids": ambiguous,
        "unused_evidence_genes": sorted(set(by_gene) - used_genes),
    }


def _confirmable(records: list[GeneRecord], thresholds: Thresholds) -> int:
    return sum(
        record.alteration_type == "missense"
        and record.plddt is not None
        and record.plddt >= thresholds.plddt_floor
        for record in records
    )


def _compare(
    candidate_ids: list[str],
    real: list[Verdict],
    shuffled: list[Verdict],
    gold_path: Path | None,
) -> dict[str, Any]:
    if gold_path is None:
        return {
            "status": "blocked_missing_independent_truth_or_downstream_metric",
            "result_call": "no_result",
            "reason": (
                "Real and shuffled bucket counts are identical by design; provide frozen "
                "independent outcomes or a locked downstream metric."
            ),
        }
    rows = _read_jsonl(gold_path)
    gold = {str(row["candidate_id"]): str(row["outcome"]) for row in rows}
    missing = sorted(set(candidate_ids) - set(gold))
    extra = sorted(set(gold) - set(candidate_ids))
    if missing:
        return {
            "status": "blocked_incomplete_gold_labels",
            "result_call": "no_result",
            "missing_candidate_ids": missing,
            "extra_gold_candidate_ids": extra,
        }
    truth = [gold[candidate_id] for candidate_id in candidate_ids]
    real_labels = [verdict.outcome.value for verdict in real]
    shuffled_labels = [verdict.outcome.value for verdict in shuffled]
    real_accuracy = sum(a == b for a, b in zip(real_labels, truth, strict=True)) / len(truth)
    shuffled_accuracy = sum(a == b for a, b in zip(shuffled_labels, truth, strict=True)) / len(truth)
    confirmed_names = {item.value for item in CONFIRMED}
    real_confirmed = [label in confirmed_names for label in real_labels]
    shuffled_confirmed = [label in confirmed_names for label in shuffled_labels]
    truth_confirmed = [label in confirmed_names for label in truth]
    real_binary = sum(a == b for a, b in zip(real_confirmed, truth_confirmed, strict=True)) / len(truth)
    shuffled_binary = sum(
        a == b for a, b in zip(shuffled_confirmed, truth_confirmed, strict=True)
    ) / len(truth)
    beat = real_accuracy > shuffled_accuracy
    return {
        "status": "completed",
        "gold_path": str(gold_path),
        "gold_sha256": _sha256(gold_path),
        "real_outcome_accuracy": real_accuracy,
        "shuffled_outcome_accuracy": shuffled_accuracy,
        "real_confirmed_binary_accuracy": real_binary,
        "shuffled_confirmed_binary_accuracy": shuffled_binary,
        "real_beats_shuffled": beat,
        "result_call": "validator_result" if beat else "no_result",
        "extra_gold_candidate_ids": extra,
    }


def run(
    records_path: Path,
    config_path: Path,
    *,
    candidates_path: Path,
    pool_candidate_paths: list[Path],
    gold_outcomes_path: Path | None,
    seed: int,
) -> dict[str, Any]:
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Stage 3/4 config must be a YAML object")
    thresholds = Thresholds.from_yaml(Path(str(config.get("validator_config", "config/validator.yaml"))))
    evidence = [_record(row) for row in _read_jsonl(records_path)]
    primary_candidates = _read_jsonl(candidates_path)
    primary_ids, primary_records, primary_real, primary_audit = _align(
        primary_candidates, evidence, thresholds
    )
    decision_count = _confirmable(primary_records, thresholds)
    minimum = int(config.get("confirmable_minimum", 15))
    preferred = int(config.get("confirmable_preferred", 20))
    pooling_triggered = decision_count < minimum

    pooled_candidates: list[dict[str, Any]] = []
    if pooling_triggered:
        for path in pool_candidate_paths:
            pooled_candidates.extend(_read_jsonl(path))
    pooled_ids: list[str] = []
    pooled_records: list[GeneRecord] = []
    pooled_real: list[Verdict] = []
    pooled_audit: dict[str, Any] = {
        "missing_evidence_candidate_ids": [],
        "ambiguous_evidence_candidate_ids": [],
        "unused_evidence_genes": [],
    }
    if pooled_candidates:
        pooled_ids, pooled_records, pooled_real, pooled_audit = _align(
            pooled_candidates, evidence, thresholds
        )
        overlap = sorted(set(primary_ids) & set(pooled_ids))
        if overlap:
            raise ValueError(f"Primary and pooled candidate IDs overlap: {overlap}")

    candidate_ids = primary_ids + pooled_ids
    control_records = primary_records + pooled_records
    real = primary_real + pooled_real
    shuffled = shuffled_validator.classify_many(
        control_records, thresholds, reference_verdicts=real, seed=seed
    )
    real_counts, shuffled_counts = _counts(real), _counts(shuffled)
    if real_counts != shuffled_counts:
        raise RuntimeError("Shuffled control did not preserve bucket proportions")
    comparison = _compare(candidate_ids, real, shuffled, gold_outcomes_path)
    fixture = _is_fixture(primary_candidates)
    pooling_status = (
        "completed" if pooling_triggered and pooled_candidates
        else "blocked_missing_pool_candidates" if pooling_triggered
        else "not_needed"
    )
    scientifically_complete = (
        not fixture
        and pooling_status in {"completed", "not_needed"}
        and comparison["status"] == "completed"
    )
    assignments = []
    for candidate_id, real_verdict, shuffled_verdict in zip(
        candidate_ids, real, shuffled, strict=True
    ):
        assignments.append(
            {
                "candidate_id": candidate_id,
                "gene": real_verdict.gene,
                "mutation": real_verdict.mutation,
                "real_outcome": real_verdict.outcome.value,
                "shuffled_outcome": shuffled_verdict.outcome.value,
                "real_stage5_kept": real_verdict.outcome in CONFIRMED,
                "shuffled_stage5_kept": shuffled_verdict.outcome in CONFIRMED,
                "real_reason": real_verdict.reason,
            }
        )
    return {
        "status": "completed" if scientifically_complete else "completed_with_blockers",
        "scientifically_complete": scientifically_complete,
        "scope": "synthetic_fixture" if fixture else "real_candidates",
        "seed": seed,
        "candidate_count": len(candidate_ids),
        "primary_candidate_count": len(primary_ids),
        "pooled_candidate_count": len(pooled_ids),
        "bucket_counts": real_counts,
        "confirmable_definition": "missense and pLDDT >= configured floor",
        "confirmable_count_primary": decision_count,
        "feasibility": {
            "branch": "pool_additional_cancer_types" if pooling_triggered else "gbm_only",
            "decision_count": decision_count,
            "minimum_required": minimum,
            "preferred_count": preferred,
            "triggered": pooling_triggered,
            "status": pooling_status,
            "requested_cancer_types": list(config.get("pooling_cancer_types", [])),
            "pool_candidate_files": [str(path) for path in pool_candidate_paths],
        },
        "candidate_alignment": {
            "primary": primary_audit,
            "pooled": pooled_audit,
            "records_are_candidate_aligned": True,
        },
        "shuffled_control": {
            "status": "completed",
            "module": "shuffled_validator.py",
            "api": "classify_many(records, thresholds, reference_verdicts=..., seed=...)",
            "seed": seed,
            "bucket_counts": shuffled_counts,
            "proportions_preserved_exactly": True,
        },
        "comparison": comparison,
        "assignments": assignments,
        "provenance": {
            "records_path": str(records_path),
            "records_sha256": _sha256(records_path),
            "candidates_path": str(candidates_path),
            "candidates_sha256": _sha256(candidates_path),
            "config_path": str(config_path),
            "config_sha256": _sha256(config_path),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--pool-candidates", type=Path, action="append", default=[])
    parser.add_argument("--gold-outcomes", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config/stage34.yaml"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run(
            args.records,
            args.config,
            candidates_path=args.candidates,
            pool_candidate_paths=args.pool_candidates,
            gold_outcomes_path=args.gold_outcomes,
            seed=args.seed,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        result = {"status": "failed", "scientifically_complete": False, "reason": str(exc)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") in {"completed", "completed_with_blockers"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
