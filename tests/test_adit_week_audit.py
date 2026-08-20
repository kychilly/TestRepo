from __future__ import annotations

from pathlib import Path

from scripts.audit_adit_weeks import classify


def test_audit_marks_missing_file(tmp_path: Path) -> None:
    result = classify(tmp_path / "missing.json")
    assert result["status"] == "missing_or_invalid"


def test_audit_marks_stale_json(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text('{"status": "completed"}\n', encoding="utf-8")
    result = classify(path, stale=True)
    assert result["status"] == "stale"
