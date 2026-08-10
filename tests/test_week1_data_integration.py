from __future__ import annotations

from pathlib import Path

from baselines.base import load_patient_splits
from scripts.donor_batch_audit import run_audit


def test_registered_data_manifest_records_download_and_known_gaps() -> None:
    text = (Path(__file__).parents[1] / "data/README.md").read_text(encoding="utf-8")
    assert "neftel_qc.h5ad" in text
    assert "986a2b2bce093334f79232975c064cc2377994fb91316950b4b92e3197288b32" in text
    assert "TCGA-GBM" in text and "CGGA mRNAseq_325" in text


def test_registered_split_has_canonical_keys_and_zero_overlap() -> None:
    split = load_patient_splits(Path(__file__).parents[1] / "splits/patient_splits.json", 0)
    values = split.as_dict()
    assert set(values) == {"train", "validation", "test"}
    assert not values["train"] & values["validation"]
    assert not values["train"] & values["test"]
    assert not values["validation"] & values["test"]


def test_donor_audit_missing_input_is_structured(tmp_path: Path) -> None:
    result = run_audit(tmp_path / "missing.h5ad", tmp_path / "audit")
    assert result["status"] == "blocked"
