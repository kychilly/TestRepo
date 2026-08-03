"""Typed records and explicit joins for the transcriptomic/protein contract."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, cast

from jsonschema import Draft202012Validator, FormatChecker, RefResolver  # type: ignore[import-untyped]

CONTRACT_VERSION = "1.0.0"
SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas"
GENE_SYMBOL = re.compile(r"^[A-Z][A-Z0-9-]*$")


class ContractError(ValueError):
    """Raised when a contract record or explicit join is invalid."""


def _load_schema(name: str) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8")))


def _validate(document: Mapping[str, Any], schema_name: str) -> None:
    schemas = {
        "candidate_gene.schema.json": _load_schema("candidate_gene.schema.json"),
        "variant_record.schema.json": _load_schema("variant_record.schema.json"),
        "validator_input.schema.json": _load_schema("validator_input.schema.json"),
    }
    schema = schemas[schema_name]
    resolver = RefResolver.from_schema(
        schema,
        store={name: value for name, value in schemas.items()},
    )
    validator = Draft202012Validator(schema, resolver=resolver, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(dict(document)), key=lambda error: list(error.path))
    if errors:
        location = ".".join(str(part) for part in errors[0].path) or "$"
        raise ContractError(f"{schema_name} invalid at {location}: {errors[0].message}")


def normalize_gene_symbol(symbol: str, aliases: Mapping[str, str] | None = None) -> str:
    """Normalize a documented alias; reject unknown aliases instead of guessing."""
    if aliases and symbol in aliases:
        symbol = aliases[symbol]
    if not GENE_SYMBOL.fullmatch(symbol):
        raise ContractError(f"Unknown or undocumented gene alias: {symbol!r}")
    return symbol


@dataclass(frozen=True)
class CandidateGene:
    data: dict[str, Any]

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], aliases: Mapping[str, str] | None = None
    ) -> "CandidateGene":
        payload = dict(data)
        validate_candidate_gene(payload, aliases=aliases)
        return cls(payload)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.data)


@dataclass(frozen=True)
class VariantRecord:
    data: dict[str, Any]

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], aliases: Mapping[str, str] | None = None
    ) -> "VariantRecord":
        payload = dict(data)
        validate_variant_record(payload, aliases=aliases)
        return cls(payload)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.data)


@dataclass(frozen=True)
class ValidatorInput:
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.data)


def validate_candidate_gene(
    record: Mapping[str, Any],
    aliases: Mapping[str, str] | None = None,
    forbidden_patient_ids: Iterable[str] = (),
) -> None:
    _validate(record, "candidate_gene.schema.json")
    normalized = normalize_gene_symbol(str(record["gene_symbol"]), aliases)
    if normalized != record["gene_symbol"]:
        raise ContractError("gene_symbol must be stored in normalized canonical form")
    patients = set(record["contributing_patient_ids"])
    forbidden = patients & set(forbidden_patient_ids)
    if forbidden:
        raise ContractError(
            f"Candidate includes forbidden validation/external patients: {sorted(forbidden)}"
        )
    if int(record["n_patients"]) != len(patients):
        raise ContractError("n_patients must equal the number of contributing_patient_ids")


def validate_variant_record(
    record: Mapping[str, Any], aliases: Mapping[str, str] | None = None
) -> None:
    _validate(record, "variant_record.schema.json")
    normalized = normalize_gene_symbol(str(record["gene_symbol"]), aliases)
    if normalized != record["gene_symbol"]:
        raise ContractError("gene_symbol must be stored in normalized canonical form")
    if record["mapping_status"] == "resolved":
        required = (
            "transcript_id",
            "transcript_version",
            "protein_accession",
            "protein_isoform",
            "protein_change",
        )
        if any(not record[field] for field in required):
            raise ContractError(
                "Resolved protein mappings require every transcript and protein field"
            )
    elif not str(record["mapping_notes"]).strip():
        raise ContractError("Unresolved protein mappings require mapping_notes")


def _eligibility(variants: list[dict[str, Any]]) -> tuple[str, str]:
    if not variants:
        return "none", "abstain_missing_evidence"
    if len(variants) > 1:
        return "multiple_variants", "abstain_ambiguous"
    variant = variants[0]
    if variant["alteration_type"] != "missense":
        return "none", "abstain_non_missense"
    status = variant["mapping_status"]
    if status != "resolved":
        reason = "abstain_ambiguous" if status == "ambiguous" else "abstain_mapping_failed"
        return ("transcript_ambiguous" if status == "ambiguous" else "mapping_failed"), reason
    return "none", "eligible_missense"


def build_validator_inputs(
    candidates: Iterable[Mapping[str, Any]],
    variants: Iterable[Mapping[str, Any]],
    validator_config_version: str,
    aliases: Mapping[str, str] | None = None,
    forbidden_patient_ids: Iterable[str] = (),
) -> list[ValidatorInput]:
    """Create explicit candidate/variant joins; never collapse variant evidence."""
    candidate_records = [CandidateGene.from_dict(item, aliases) for item in candidates]
    variant_records = [VariantRecord.from_dict(item, aliases) for item in variants]
    by_candidate: list[ValidatorInput] = []
    for candidate in candidate_records:
        validate_candidate_gene(candidate.data, aliases, forbidden_patient_ids)
        candidate_gene = candidate.data["gene_symbol"]
        candidate_ensembl = candidate.data["ensembl_gene_id"]
        patient_ids = set(candidate.data["contributing_patient_ids"])
        matches = [
            variant.data
            for variant in variant_records
            if variant.data["patient_id"] in patient_ids
            and variant.data["cohort"] == candidate.data["cohort"]
            and normalize_gene_symbol(variant.data["gene_symbol"], aliases) == candidate_gene
            and variant.data["ensembl_gene_id"] == candidate_ensembl
        ]
        matches.sort(key=lambda item: item["variant_id"])
        cardinality = "zero" if not matches else "one" if len(matches) == 1 else "multiple"
        ambiguity, eligibility = _eligibility(matches)
        protein_mapping = [
            {
                "variant_id": item["variant_id"],
                "mapping_status": item["mapping_status"],
                "protein_accession": item["protein_accession"],
                "protein_isoform": item["protein_isoform"],
                "protein_change": item["protein_change"],
            }
            for item in matches
        ]
        payload = {
            "schema_version": CONTRACT_VERSION,
            "input_id": candidate.data["candidate_id"],
            "candidate_provenance": candidate.to_dict(),
            "variant_provenance": matches,
            "join_key": f"{candidate.data['candidate_id']}|{candidate_gene}|{candidate_ensembl}",
            "join_cardinality": cardinality,
            "ambiguity_status": ambiguity,
            "protein_mapping": protein_mapping,
            "validator_config_version": validator_config_version,
            "validator_eligibility": eligibility,
        }
        _validate(payload, "validator_input.schema.json")
        by_candidate.append(ValidatorInput(payload))
    return by_candidate


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ContractError(f"{path}:{line_number} must contain a JSON object")
        records.append(value)
    return records


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(record), sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
