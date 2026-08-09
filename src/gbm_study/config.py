"""Typed configuration and study-design validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigurationError(ValueError):
    """Raised when a study configuration violates the data contract."""


@dataclass(frozen=True)
class StudyConfig:
    """Configuration values required for a Week 1 run."""

    modality: str
    data_manifest: Path | None
    split_file: Path | None
    checkpoint: str | None
    vocabulary: Path | None
    output_dir: Path
    seed: int
    week: int
    uses_cgga: bool

    @classmethod
    def from_json(cls, path: Path) -> "StudyConfig":
        """Load and validate a JSON configuration without resolving missing paths."""
        try:
            payload: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(
                f"Cannot read configuration {path}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise ConfigurationError("Configuration root must be a JSON object")

        def optional_path(key: str) -> Path | None:
            value = payload.get(key)
            return None if value is None else Path(str(value))

        required = {"modality", "output_dir", "seed", "week", "uses_cgga"}
        missing = sorted(required - payload.keys())
        if missing:
            raise ConfigurationError(
                f"Missing configuration keys: {', '.join(missing)}"
            )

        config = cls(
            modality=str(payload["modality"]),
            data_manifest=optional_path("data_manifest"),
            split_file=optional_path("split_file"),
            checkpoint=None
            if payload.get("checkpoint") is None
            else str(payload["checkpoint"]),
            vocabulary=optional_path("vocabulary"),
            output_dir=Path(str(payload["output_dir"])),
            seed=int(payload["seed"]),
            week=int(payload["week"]),
            uses_cgga=bool(payload["uses_cgga"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Reject study-design violations before data or model code can run."""
        if self.modality not in {"neftel", "tcga"}:
            raise ConfigurationError("modality must be 'neftel' or 'tcga'")
        if self.week < 1:
            raise ConfigurationError("week must be positive")
        if self.seed < 0:
            raise ConfigurationError("seed must be non-negative")
        if self.week == 1 and self.uses_cgga:
            raise ConfigurationError("CGGA access is prohibited during Week 1")
        if self.modality == "neftel" and self.uses_cgga:
            raise ConfigurationError("Neftel runs cannot access CGGA")
        if self.modality == "tcga" and self.week == 1:
            raise ConfigurationError(
                "TCGA modeling is not enabled by the Week 1 design"
            )

    def require_run_inputs(self) -> None:
        """Require Data Lead inputs for an executable run, without inventing defaults."""
        missing = [
            name
            for name, value in {
                "data_manifest": self.data_manifest,
                "split_file": self.split_file,
                "checkpoint": self.checkpoint,
                "vocabulary": self.vocabulary,
            }.items()
            if value is None
        ]
        if missing:
            raise ConfigurationError(
                "Run inputs are unspecified; provide real Data Lead values for: "
                + ", ".join(missing)
            )
        assert self.data_manifest is not None
        assert self.split_file is not None
        assert self.vocabulary is not None
        missing_files = [
            str(path)
            for path in (self.data_manifest, self.split_file, self.vocabulary)
            if not path.is_file()
        ]
        if missing_files:
            raise ConfigurationError(
                "Configured input does not exist: " + ", ".join(missing_files)
            )
