#!/usr/bin/env python3
"""Enable the tracked Git hook that blocks dataset and model artifacts."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def install(repo: Path) -> None:
    hook = repo / ".githooks" / "pre-commit"
    if not hook.is_file():
        raise FileNotFoundError(f"Tracked pre-commit hook is missing: {hook}")
    subprocess.run(["git", "config", "core.hooksPath", ".githooks"], cwd=repo, check=True)
    configured = subprocess.run(
        ["git", "config", "--get", "core.hooksPath"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if configured != ".githooks":
        raise RuntimeError(f"Unexpected core.hooksPath value: {configured!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    install(args.repo.resolve())
    print("Git dataset/model safety hook enabled (.githooks/pre-commit)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
