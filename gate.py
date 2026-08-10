"""Run the frozen four-gene validator gate and write an auditable result."""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from jsonschema import Draft202012Validator, FormatChecker  # type: ignore[import-untyped]

from validator import GeneRecord, Outcome, Thresholds, classify

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "examples/validator_gate_input.jsonl"
DEFAULT_CONFIG = ROOT / "config/validator.yaml"
DEFAULT_OUTPUT = ROOT / "results/validator_gate.json"
SCHEMA = ROOT / "schemas/validator_gate_record.schema.json"
EXPECTED = {
    "TP53": Outcome.DESTABILIZING_DRIVER,
    "IDH1": Outcome.FUNCTIONAL_DRIVER,
    "EGFR": Outcome.ABSTAIN,
    "RPRM": Outcome.ABSTAIN,
}


def _read_records(path: Path) -> list[dict[str, Any]]:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is invalid JSON: {exc}") from exc
        errors = sorted(validator.iter_errors(record), key=lambda error: list(error.path))
        if errors:
            location = ".".join(str(item) for item in errors[0].path) or "$"
            raise ValueError(f"{path}:{line_number} invalid at {location}: {errors[0].message}")
        records.append(record)
    if len(records) != 4:
        raise ValueError(f"Gate requires exactly four records, found {len(records)}")
    genes = [str(record["gene"]) for record in records]
    if set(genes) != set(EXPECTED) or len(set(genes)) != 4:
        raise ValueError(f"Gate requires exactly TP53, IDH1, EGFR, and RPRM; found {genes}")
    return records


def _load_thresholds(path: Path) -> tuple[Thresholds, dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Validator configuration must be a YAML object")
    thresholds = Thresholds.from_yaml(path)
    return thresholds, raw


def run_gate(
    input_path: Path = DEFAULT_INPUT, config_path: Path = DEFAULT_CONFIG
) -> dict[str, Any]:
    thresholds, config = _load_thresholds(config_path)
    records = _read_records(input_path)
    verdicts: list[dict[str, Any]] = []
    for record in records:
        gene_record = GeneRecord(
            gene=str(record["gene"]),
            mutation=str(record["mutation"]),
            alteration_type=str(record["alteration_type"]),
            plddt=record.get("plddt"),
            esm1b=record.get("esm1b"),
            ddg=record.get("ddg"),
        )
        verdict = classify(gene_record, thresholds)
        verdicts.append(
            {
                "gene": verdict.gene,
                "mutation": verdict.mutation,
                "outcome": verdict.outcome.value,
                "expected_outcome": EXPECTED[verdict.gene].value,
                "matches_expected": verdict.outcome == EXPECTED[verdict.gene],
                "counts_toward_prediction": verdict.counts_toward_prediction(),
                "reason": verdict.reason,
                "evidence_provenance": record["evidence_provenance"],
            }
        )
    matches = all(bool(item["matches_expected"]) for item in verdicts)
    provenance_warnings = sorted(
        {
            str(record["evidence_provenance"].get("provenance_status"))
            for record in records
            if record["alteration_type"] == "missense"
            and record["evidence_provenance"].get("provenance_status") != "complete"
        }
    )
    classification_passed = matches
    publication_passed = classification_passed and not (
        bool(config.get("require_complete_evidence_provenance", True)) and provenance_warnings
    )
    return {
        "status": "completed" if publication_passed else "blocked",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate_passed": publication_passed,
        "classification_gate_passed": classification_passed,
        "publication_gate_passed": publication_passed,
        "thresholds": {
            "plddt_floor": thresholds.plddt_floor,
            "esm1b_cutoff": thresholds.esm1b_cutoff,
            "ddg_cutoff": thresholds.ddg_cutoff,
        },
        "threshold_change_log": config.get("threshold_change_log", []),
        "thresholds_moved_after_gate_observation": False,
        "provenance_warnings": provenance_warnings,
        "verdicts": verdicts,
        "interpretation": "Software classifications matched the requested outcomes, but publication gate is blocked until complete precomputed evidence provenance is supplied."
        if provenance_warnings
        else "Gate passed with complete evidence provenance.",
    }


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        result = run_gate(args.input, args.config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        result = {"status": "blocked", "gate_passed": False, "reason": str(exc)}
    _atomic_write(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
