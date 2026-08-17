"""
tests/test_validator.py

Task 2: one synthetic GeneRecord per outcome, constructed so it can ONLY
land in that branch of the decision tree. This file tests validator.py
only -- it does not touch the gate. See tests/test_gate.py for the
four-gene gate test (task 3).

Run with:  python -m pytest tests/test_validator.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from validator import (  # noqa: E402
    GeneRecord,
    Outcome,
    Thresholds,
    classify,
)

# Fixed thresholds for the synthetic tests, independent of the checked-in
# config, so these tests keep meaning even if config/validator.yaml changes.
T = Thresholds(plddt_floor=70.0, esm1b_cutoff=-7.5, ddg_cutoff=1.5)


def test_destabilizing_driver_branch():
    """High-confidence structure + ddG over cutoff -> DESTABILIZING_DRIVER.
    ESM1b is deliberately mild (above cutoff) so this can ONLY be reached
    through the ddG/pLDDT path, not accidentally via the ESM1b path."""
    rec = GeneRecord(
        gene="SYN_DESTAB", mutation="X1Y", alteration_type="missense",
        plddt=95.0, esm1b=-2.0, ddg=3.0,
    )
    v = classify(rec, T)
    assert v.outcome == Outcome.DESTABILIZING_DRIVER
    assert v.counts_toward_prediction()


def test_functional_driver_branch():
    """Trustworthy structure, ddG well under cutoff (protein stays folded),
    but ESM1b is strongly damaging -> FUNCTIONAL_DRIVER."""
    rec = GeneRecord(
        gene="SYN_FUNC", mutation="X2Y", alteration_type="missense",
        plddt=90.0, esm1b=-12.0, ddg=0.2,
    )
    v = classify(rec, T)
    assert v.outcome == Outcome.FUNCTIONAL_DRIVER
    assert v.counts_toward_prediction()


def test_abstain_branch_amplification():
    """Non-missense alteration -> ABSTAIN, regardless of how damaging the
    attached scores look (scores are deliberately extreme here to prove
    alteration_type is checked first and short-circuits everything else)."""
    rec = GeneRecord(
        gene="SYN_ABST", mutation="amplification", alteration_type="amplification",
        plddt=99.0, esm1b=-20.0, ddg=10.0,
    )
    v = classify(rec, T)
    assert v.outcome == Outcome.ABSTAIN
    assert not v.counts_toward_prediction()


def test_abstain_branch_silencing():
    """Silencing is also non-missense -> ABSTAIN."""
    rec = GeneRecord(
        gene="SYN_ABST2", mutation="silencing", alteration_type="silencing",
        plddt=None, esm1b=None, ddg=None,
    )
    v = classify(rec, T)
    assert v.outcome == Outcome.ABSTAIN


def test_unconfirmed_branch():
    """Missense, and we HAVE both ddG and pLDDT (so it's testable), but
    neither signal clears its cutoff -> UNCONFIRMED, not DATA_DEFICIENT."""
    rec = GeneRecord(
        gene="SYN_UNCONF", mutation="X3Y", alteration_type="missense",
        plddt=88.0, esm1b=-3.0, ddg=0.3,
    )
    v = classify(rec, T)
    assert v.outcome == Outcome.UNCONFIRMED
    assert not v.counts_toward_prediction()


def test_data_deficient_branch_missing_esm1b():
    """Missense but ESM1b itself is missing -> DATA_DEFICIENT immediately,
    before any other check runs."""
    rec = GeneRecord(
        gene="SYN_DEFICIENT1", mutation="X4Y", alteration_type="missense",
        plddt=95.0, esm1b=None, ddg=3.0,
    )
    v = classify(rec, T)
    assert v.outcome == Outcome.DATA_DEFICIENT


def test_data_deficient_branch_no_usable_stability_evidence():
    """Missense, ESM1b present but mild (fails functional test), AND ddG
    missing AND pLDDT below floor (fails destabilizing test on structure
    grounds, not on ddG value) -> DATA_DEFICIENT, not UNCONFIRMED, because
    we never actually got to test the stability hypothesis at all."""
    rec = GeneRecord(
        gene="SYN_DEFICIENT2", mutation="X5Y", alteration_type="missense",
        plddt=40.0, esm1b=-1.0, ddg=None,
    )
    v = classify(rec, T)
    assert v.outcome == Outcome.DATA_DEFICIENT


def test_low_plddt_does_not_block_functional_driver():
    """Sanity check that the functional-driver path (ESM1b) does NOT
    require a trustworthy structure -- pLDDT is low here but ESM1b alone
    is enough to confirm FUNCTIONAL_DRIVER."""
    rec = GeneRecord(
        gene="SYN_LOWPLDDT_FUNC", mutation="X6Y", alteration_type="missense",
        plddt=40.0, esm1b=-9.0, ddg=None,
    )
    v = classify(rec, T)
    assert v.outcome == Outcome.FUNCTIONAL_DRIVER


def test_low_plddt_blocks_destabilizing_even_with_high_ddg():
    """ddG alone is not enough for DESTABILIZING_DRIVER if the structure
    prediction isn't trustworthy; here it should fall through to check
    ESM1b, and since ESM1b is mild too, land on DATA_DEFICIENT."""
    rec = GeneRecord(
        gene="SYN_LOWPLDDT_DESTAB", mutation="X7Y", alteration_type="missense",
        plddt=40.0, esm1b=-1.0, ddg=5.0,
    )
    v = classify(rec, T)
    assert v.outcome != Outcome.DESTABILIZING_DRIVER
    assert v.outcome == Outcome.DATA_DEFICIENT


if __name__ == "__main__":
    import subprocess

    subprocess.run(
        [sys.executable, "-m", "pytest", str(Path(__file__)), "-v"], check=True
    )
