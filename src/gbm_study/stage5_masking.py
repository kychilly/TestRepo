"""Stage 5 candidate masking seam; validator thresholds intentionally absent."""

from __future__ import annotations

import hashlib
import importlib
import inspect
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, cast

SUPPORTED_OUTCOMES = frozenset({"destabilizing_driver", "functional_driver", "eligible_missense"})
OUTCOMES = frozenset(
    {
        "destabilizing_driver",
        "functional_driver",
        "abstain",
        "unconfirmed",
        "data_deficient",
        "eligible_missense",
        "abstain_missing_evidence",
        "abstain_ambiguous",
        "abstain_non_missense",
        "abstain_mapping_failed",
    }
)


class ValidatorOutcomeSource(Protocol):
    """Return one of the signed-off validator outcome strings for a payload."""

    version: str

    def __call__(self, validator_payload: Mapping[str, Any]) -> str: ...


class PassthroughStubOutcomeSource:
    """Non-scientific test stub: passes through validator_eligibility only."""

    version = "stub-1.0.0"

    def __call__(self, validator_payload: Mapping[str, Any]) -> str:
        outcome = str(
            validator_payload.get("outcome", validator_payload.get("validator_eligibility", ""))
        )
        if outcome not in OUTCOMES:
            raise ValueError(f"Unsupported validator outcome: {outcome!r}")
        return outcome


def load_outcome_source(module_name: str) -> ValidatorOutcomeSource:
    module = importlib.import_module(module_name)
    source = getattr(module, "OUTCOME_SOURCE", None)
    if not callable(source):
        raise ValueError(f"{module_name} must export callable OUTCOME_SOURCE")
    return cast(ValidatorOutcomeSource, source)


def mask_candidates(
    candidates: Sequence[Mapping[str, Any]],
    payloads: Mapping[str, Mapping[str, Any]],
    *,
    validator: str,
    outcome_source: ValidatorOutcomeSource,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if validator not in {"on", "off"}:
        raise ValueError("validator must be on or off")
    if validator == "off":
        return [dict(item) for item in candidates], {
            "validator": "off",
            "masked": 0,
            "outcome_source": "not_invoked",
        }
    kept: list[dict[str, Any]] = []
    outcomes: dict[str, str] = {}
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id", candidate.get("input_id", "")))
        payload = payloads.get(candidate_id)
        if payload is None:
            raise ValueError(f"Missing validator payload for {candidate_id}")
        outcome = outcome_source(payload)
        if outcome not in OUTCOMES:
            raise ValueError(f"Outcome source returned unsupported value: {outcome!r}")
        outcomes[candidate_id] = outcome
        if outcome in SUPPORTED_OUTCOMES:
            kept.append(dict(candidate))
    return kept, {
        "validator": "on",
        "masked": len(candidates) - len(kept),
        "outcomes": outcomes,
        "outcome_source": type(outcome_source).__name__,
        "outcome_source_version": getattr(outcome_source, "version", "unknown"),
    }


def source_provenance(
    source: ValidatorOutcomeSource, module_path: Path | None = None
) -> dict[str, str | None]:
    if module_path is None:
        discovered = inspect.getsourcefile(type(source))
        module_path = Path(discovered) if discovered else None
    return {
        "implementation": type(source).__name__,
        "version": getattr(source, "version", "unknown"),
        "sha256": hashlib.sha256(module_path.read_bytes()).hexdigest() if module_path else None,
    }
