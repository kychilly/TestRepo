"""Fixed-seed shuffled control for the Stage 4 validator.

The public record/threshold/verdict types are re-exported from ``validator`` so
this module can replace it at the pipeline boundary. Exact bucket proportions
are a property of a collection, so scientific control runs must call
``classify_many`` with the real verdicts whose labels are to be permuted.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence

from validator import (  # re-exported drop-in types
    GeneRecord,
    Outcome,
    Thresholds,
    Verdict,
)

DEFAULT_SEED = 17


def classify_many(
    records: Sequence[GeneRecord],
    thresholds: Thresholds,
    *,
    reference_verdicts: Sequence[Verdict] | None = None,
    seed: int = DEFAULT_SEED,
) -> list[Verdict]:
    """Randomly reassign real labels while preserving every bucket count exactly."""
    if reference_verdicts is None:
        from validator import classify_many as real_classify_many

        reference_verdicts = real_classify_many(records, thresholds)
    if len(records) != len(reference_verdicts):
        raise ValueError("records and real_verdicts must have equal length")
    labels = [verdict.outcome for verdict in reference_verdicts]
    random.Random(seed).shuffle(labels)
    return [
        Verdict(
            gene=record.gene,
            mutation=record.mutation,
            outcome=outcome,
            reason=(
                f"fixed-seed shuffled control assignment (seed={seed}); "
                "bucket proportions preserved exactly across this run"
            ),
        )
        for record, outcome in zip(records, labels, strict=True)
    ]


def classify(record: GeneRecord, thresholds: Thresholds, *, seed: int = DEFAULT_SEED) -> Verdict:
    """Single-record API compatibility for consumers that import ``classify``.

    Exact proportion preservation is impossible for one record in isolation;
    this deterministic hash assignment exists only for API compatibility.
    Use ``classify_many`` for the required shuffled-control experiment.
    """
    del thresholds
    digest = hashlib.sha256(
        f"{seed}|{record.gene}|{record.mutation}|{record.alteration_type}".encode()
    ).digest()
    outcomes = tuple(Outcome)
    outcome = outcomes[int.from_bytes(digest[:8], "big") % len(outcomes)]
    return Verdict(
        gene=record.gene,
        mutation=record.mutation,
        outcome=outcome,
        reason=f"deterministic single-record shuffled control assignment (seed={seed})",
    )
