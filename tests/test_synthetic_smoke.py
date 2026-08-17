from __future__ import annotations

import json
from pathlib import Path

from scripts.synthetic_smoke import main


def test_synthetic_smoke_reports_statuses_without_fake_gpu_metrics(
    tmp_path: Path,
) -> None:
    output = tmp_path / "synthetic.json"
    assert main(["--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["synthetic"] is True
    assert report["scientific_use"] == "forbidden"
    assert report["baseline_results"]["pca_logreg"]["status"] == "completed"
    assert (
        report["baseline_results"]["idh_evaluation"]["metric_scope"]
        == "synthetic_patient_level_only"
    )
    assert report["baseline_results"]["gpu_plan"]["status"] == "blocked"
    assert report["baseline_results"]["candidate_schema"]["status"] == "completed"
    assert (
        report["baseline_results"]["candidate_schema"]["validator_consumer"]
        == "held_for_Ishaan_signoff"
    )
