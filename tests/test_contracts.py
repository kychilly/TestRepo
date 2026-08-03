"""Tests for the study's non-negotiable Week 1 contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gbm_study.config import ConfigurationError, StudyConfig
from gbm_study.leakage import LeakageError, assert_patient_split, assert_zero_patient_overlap


def test_patient_overlap_is_rejected() -> None:
    with pytest.raises(LeakageError, match="Patient overlap"):
        assert_zero_patient_overlap({"train": ["p1"], "validation": ["p1"], "test": ["p2"]})


def test_observation_outside_declared_patient_split_is_rejected() -> None:
    with pytest.raises(LeakageError, match="outside test"):
        assert_patient_split(["p1", "p2"], "test", {"test": ["p1"]})


def test_week_one_cgga_access_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "modality": "neftel",
                "data_manifest": None,
                "split_file": None,
                "checkpoint": None,
                "vocabulary": None,
                "output_dir": str(tmp_path / "results"),
                "seed": 7,
                "week": 1,
                "uses_cgga": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="CGGA"):
        StudyConfig.from_json(config_path)


def test_unconfigured_run_inputs_fail_loudly(tmp_path: Path) -> None:
    config = StudyConfig(
        modality="neftel",
        data_manifest=None,
        split_file=None,
        checkpoint=None,
        vocabulary=None,
        output_dir=tmp_path / "results",
        seed=0,
        week=1,
        uses_cgga=False,
    )
    with pytest.raises(ConfigurationError, match="Run inputs are unspecified"):
        config.require_run_inputs()
