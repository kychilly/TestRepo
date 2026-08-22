"""Patient-level split and observation contract checks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


class LeakageError(ValueError):
    """Raised when a patient-aware study contract is violated."""


def normalize_split_keys(payload: Any) -> dict[str, list[str]]:
    """Normalize Jeffrey's ``val``/``test_cgga`` keys to the study contract.

    The normalized names are always ``train``, ``validation``, and ``test``.
    Both spellings are accepted only when the canonical spelling is absent;
    conflicting duplicate keys fail closed.
    """
    if not isinstance(payload, dict):
        raise LeakageError("Split file must contain a JSON object")
    value = dict(payload)
    aliases = {"validation": "val", "test": "test_cgga"}
    for canonical, alias in aliases.items():
        if canonical in value and alias in value:
            raise LeakageError(f"Split contains both {canonical!r} and alias {alias!r}")
        if canonical not in value and alias in value:
            value[canonical] = value.pop(alias)
    if set(value) != {"train", "validation", "test"}:
        raise LeakageError(
            "Split must contain train/validation/test (or Jeffrey's val/test_cgga aliases)"
        )
    result: dict[str, list[str]] = {}
    for name in ("train", "validation", "test"):
        patients = value[name]
        if (
            not isinstance(patients, list)
            or not patients
            or not all(isinstance(patient, str) and patient for patient in patients)
        ):
            raise LeakageError(f"Split {name!r} must be a non-empty list of patient IDs")
        if len(set(patients)) != len(patients):
            raise LeakageError(f"Split {name!r} contains duplicate patient IDs")
        result[name] = patients
    assert_zero_patient_overlap(result)
    return result


def assert_zero_patient_overlap(splits: Mapping[str, Iterable[str]]) -> None:
    """Ensure train, validation, and test patient IDs are pairwise disjoint."""
    normalized = {name: set(values) for name, values in splits.items()}
    names = list(normalized)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            overlap = normalized[left_name] & normalized[right_name]
            if overlap:
                raise LeakageError(
                    f"Patient overlap between {left_name} and {right_name}: {sorted(overlap)}"
                )


def assert_patient_split(
    patient_ids: Iterable[str], split: str, splits: Mapping[str, Iterable[str]]
) -> None:
    """Verify observations are assigned only to patients in their declared split."""
    if split not in splits:
        raise LeakageError(f"Unknown split {split!r}; expected one of {sorted(splits)}")
    allowed = set(splits[split])
    unexpected = set(patient_ids) - allowed
    if unexpected:
        raise LeakageError(f"Observations contain patients outside {split}: {sorted(unexpected)}")
