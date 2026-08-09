"""Reproducibility metadata and atomic JSON output."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(root: Path) -> str | None:
    """Return the current commit, or None for an uncommitted initial repository."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def runtime_info() -> dict[str, Any]:
    """Collect package-runtime and hardware facts available without optional dependencies."""
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def run_metadata(
    root: Path,
    config_path: Path,
    manifest: Path,
    split: Path,
    vocabulary: Path,
    seed: int,
) -> dict[str, Any]:
    """Build the common provenance envelope required for every command result."""
    return {
        "git_commit": git_commit(root),
        "config_sha256": sha256_file(config_path),
        "data_manifest_sha256": sha256_file(manifest),
        "split_sha256": sha256_file(split),
        "vocabulary_sha256": sha256_file(vocabulary),
        "random_seeds": {"python": seed},
        "runtime": runtime_info(),
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON by replacing a same-directory temporary file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)
