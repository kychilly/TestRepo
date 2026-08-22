#!/usr/bin/env python3
"""Download and verify the official scGPT pan-cancer checkpoint folder."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gbm_study.plain_english import write_json_with_explanation

PAN_CANCER_FOLDER = (
    "https://drive.google.com/drive/folders/13QzLHilYUd0v3HTwa_9n4G4yEF-hdkqa?usp=sharing"
)
REQUIRED = ("best_model.pt", "vocab.json", "args.json")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default=PAN_CANCER_FOLDER)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [sys.executable, "-m", "gdown", "--folder", args.url, "-O", str(args.output)],
        check=False,
    )
    found = {name: next(args.output.rglob(name), None) for name in REQUIRED}
    missing = [name for name, path in found.items() if path is None]
    manifest = {
        "status": "completed" if result.returncode == 0 and not missing else "blocked",
        "official_folder": args.url,
        "files": {
            name: {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}
            for name, path in found.items()
            if path is not None
        },
        "missing": missing,
    }
    manifest_path = args.output / "asset_manifest.json"
    write_json_with_explanation(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
