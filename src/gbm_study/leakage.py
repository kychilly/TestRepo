"""Patient-level split and observation contract checks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping


class LeakageError(ValueError):
    """Raised when a patient-aware study contract is violated."""


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
