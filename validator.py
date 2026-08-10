"""
validator.py
============

Stage 4 of the pipeline: "the validator (the new part)".

This is deliberately a plain decision tree, not a model. Every number it
uses lives in config/validator.yaml, and every branch below has a one-line
comment explaining the biology it is checking, so a person who has never
seen this codebase can read it top to bottom and follow the logic.

Each candidate gene (a gene the expression model, e.g. scGPT, flagged as
important) is sorted into exactly one of five outcomes:

    DESTABILIZING_DRIVER  - protein shape is disrupted (the TP53 pattern)
    FUNCTIONAL_DRIVER     - protein shape is intact, but the mutation is
                             still clearly damaging (the IDH1 pattern)
    ABSTAIN                - the alteration isn't a point mutation we can
                             run a protein-level test on (amplification,
                             silencing, etc.)
    UNCONFIRMED            - we had enough data to run the test, but
                             neither pattern was confirmed
    DATA_DEFICIENT         - we did NOT have enough data to run the test at
                             all (missing scores)

Only DESTABILIZING_DRIVER and FUNCTIONAL_DRIVER genes are allowed to count
toward the model's final prediction (Stage 5). Everything else is withheld.

NOTE: this file contains ONLY the decision tree itself (Outcome, Thresholds,
GeneRecord, Verdict, classify()). The four-gene gate check (TP53/IDH1/EGFR/
RPRM) lives in its own file, gate.py, which imports classify() from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml

CONFIG_PATH = Path(__file__).parent / "config" / "validator.yaml"


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------

class Outcome(str, Enum):
    DESTABILIZING_DRIVER = "destabilizing_driver"
    FUNCTIONAL_DRIVER = "functional_driver"
    ABSTAIN = "abstain"
    UNCONFIRMED = "unconfirmed"
    DATA_DEFICIENT = "data_deficient"


# Only these two outcomes are allowed to feed Stage 5 (final prediction).
COUNTS_TOWARD_PREDICTION = {Outcome.DESTABILIZING_DRIVER, Outcome.FUNCTIONAL_DRIVER}


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Thresholds:
    plddt_floor: float
    esm1b_cutoff: float
    ddg_cutoff: float

    @classmethod
    def from_yaml(cls, path: Path = CONFIG_PATH) -> "Thresholds":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls(
            plddt_floor=float(raw["plddt_floor"]),
            esm1b_cutoff=float(raw["esm1b_cutoff"]),
            ddg_cutoff=float(raw["ddg_cutoff"]),
        )


@dataclass(frozen=True)
class GeneRecord:
    """One candidate gene's precomputed protein evidence.

    alteration_type distinguishes HOW the gene was altered. Only "missense"
    alterations get a protein-level structural/functional test; amplification
    and silencing events are handled by expression, not by a folded protein
    with a point mutation in it, so they can never be "confirmed" this way.
    """

    gene: str
    mutation: str
    alteration_type: str  # "missense" | "amplification" | "silencing" | "other"
    plddt: Optional[float] = None      # AlphaFold per-residue confidence, 0-100
    esm1b: Optional[float] = None      # ESM1b variant-effect score
    ddg: Optional[float] = None        # ddG stability score, kcal/mol


@dataclass(frozen=True)
class Verdict:
    gene: str
    mutation: str
    outcome: Outcome
    reason: str

    def counts_toward_prediction(self) -> bool:
        return self.outcome in COUNTS_TOWARD_PREDICTION


# ---------------------------------------------------------------------------
# The decision tree itself
# ---------------------------------------------------------------------------

def classify(record: GeneRecord, thresholds: Thresholds) -> Verdict:
    """Sort one candidate gene into one of the five outcomes.

    Read this function top to bottom -- each `if` is one biological
    question, checked in priority order, and the function returns the
    moment a question is answered.
    """

    # --------------------------------------------------------------
    # Step 1: is this even a point mutation we CAN test at the protein
    # level? Amplification (extra copies of a normal gene) and silencing
    # (a gene turned off) don't hand us a mutant protein to fold-check.
    # --------------------------------------------------------------
    if record.alteration_type != "missense":
        return Verdict(
            gene=record.gene,
            mutation=record.mutation,
            outcome=Outcome.ABSTAIN,
            reason=(
                f"alteration_type='{record.alteration_type}', not a missense "
                "mutation; there is no single mutant protein structure or "
                "sequence-effect to test, so we abstain rather than guess"
            ),
        )

    # --------------------------------------------------------------
    # Step 2: for a missense mutation, the ESM1b score is the minimum
    # evidence we need to say anything at all (it's sequence-based, so it
    # doesn't depend on AlphaFold's confidence in the fold). No ESM1b
    # score means we cannot run the test, full stop.
    # --------------------------------------------------------------
    if record.esm1b is None:
        return Verdict(
            gene=record.gene,
            mutation=record.mutation,
            outcome=Outcome.DATA_DEFICIENT,
            reason="missense variant but ESM1b score is missing; cannot run any test",
        )

    # --------------------------------------------------------------
    # Step 3: destabilizing-driver test (the TP53 pattern).
    # This requires BOTH:
    #   (a) a trustworthy structure prediction (pLDDT >= floor) -- without
    #       this, a "the protein falls apart" claim isn't testable, and
    #   (b) a ddG score that clears the destabilizing cutoff.
    # If either piece is missing/untrustworthy, we do NOT fail this gene --
    # we simply move on to the functional-driver test in Step 4, since
    # ESM1b doesn't need a trustworthy structure to be meaningful.
    # --------------------------------------------------------------
    if (
        record.ddg is not None
        and record.plddt is not None
        and record.plddt >= thresholds.plddt_floor
        and record.ddg >= thresholds.ddg_cutoff
    ):
        return Verdict(
            gene=record.gene,
            mutation=record.mutation,
            outcome=Outcome.DESTABILIZING_DRIVER,
            reason=(
                f"pLDDT={record.plddt:.2f} >= floor "
                f"({thresholds.plddt_floor}) so the structure is trustworthy, "
                f"and ddG={record.ddg:.2f} >= cutoff ({thresholds.ddg_cutoff}) "
                "kcal/mol, i.e. the mutation destabilizes the fold"
            ),
        )

    # --------------------------------------------------------------
    # Step 4: functional-driver test (the IDH1 pattern).
    # The protein may stay folded, but the mutation is still clearly
    # damaging by ESM1b's sequence-level measure.
    # --------------------------------------------------------------
    if record.esm1b <= thresholds.esm1b_cutoff:
        return Verdict(
            gene=record.gene,
            mutation=record.mutation,
            outcome=Outcome.FUNCTIONAL_DRIVER,
            reason=(
                f"ESM1b={record.esm1b:.2f} <= cutoff "
                f"({thresholds.esm1b_cutoff}), a clearly damaging sequence-level "
                "signal, without a confirmed destabilizing ddG signal"
            ),
        )

    # --------------------------------------------------------------
    # Step 5: we tried both tests and neither fired. Distinguish "we
    # genuinely lack the data to say anything about stability" (the ddG
    # score is either missing, or present but attached to a structure we
    # can't trust, so it can't be used either way) from "we had a
    # trustworthy ddG value and it just didn't confirm anything".
    # --------------------------------------------------------------
    stability_testable = (
        record.ddg is not None
        and record.plddt is not None
        and record.plddt >= thresholds.plddt_floor
    )
    if not stability_testable:
        return Verdict(
            gene=record.gene,
            mutation=record.mutation,
            outcome=Outcome.DATA_DEFICIENT,
            reason=(
                "no usable stability evidence: "
                f"ddG={record.ddg}, pLDDT={record.plddt} "
                f"(floor={thresholds.plddt_floor}) -- either ddG is missing "
                "or the structure prediction isn't trustworthy enough to act "
                "on it, and ESM1b did not clear the functional-driver cutoff "
                "either"
            ),
        )

    return Verdict(
        gene=record.gene,
        mutation=record.mutation,
        outcome=Outcome.UNCONFIRMED,
        reason=(
            f"had usable data (pLDDT={record.plddt}, ESM1b={record.esm1b:.2f}, "
            f"ddG={record.ddg}) but neither the destabilizing-driver nor the "
            "functional-driver threshold was cleared"
        ),
    )
