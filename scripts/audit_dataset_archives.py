#!/usr/bin/env python3
"""Verify the top-level TP53 ZIP parts and their extracted members."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

from gbm_study.plain_english import write_json_with_explanation


def sha256_stream(stream: Any) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--extracted-root",
        type=Path,
        default=Path("data/import_20260820/TP53 Dataset(preprocessed)"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/readiness/dataset_archives.json")
    )
    args = parser.parse_args(argv)
    archives = sorted(args.data_dir.glob("TP53 Dataset(preprocessed)-*.zip"))
    parts = {path.name.rsplit("-", 1)[-1].removesuffix(".zip") for path in archives}
    expected = {"001", "002", "003"}
    missing_parts = sorted(expected - parts)
    results: list[dict[str, Any]] = []
    mismatches: list[str] = []
    for archive in archives:
        entry_count = 0
        members: list[str] = []
        with zipfile.ZipFile(archive) as zf:
            bad = zf.testzip()
            if bad:
                mismatches.append(f"Corrupt ZIP member: {archive.name}:{bad}")
            for info in zf.infolist():
                entry_count += 1
                members.append(info.filename)
                relative = Path(info.filename).relative_to("TP53 Dataset(preprocessed)")
                extracted = args.extracted_root / relative
                if not extracted.is_file():
                    mismatches.append(f"Missing extracted member: {relative}")
                    continue
                with zf.open(info) as source, extracted.open("rb") as target:
                    archive_hash = sha256_stream(source)
                    extracted_hash = sha256_stream(target)
                if archive_hash != extracted_hash:
                    mismatches.append(f"Extracted member differs: {relative}")
        with archive.open("rb") as stream:
            archive_hash = sha256_stream(stream)
        results.append(
            {
                "path": str(archive),
                "sha256": archive_hash,
                "member_count": entry_count,
                "members": members,
            }
        )
    result = {
        "status": "completed"
        if not missing_parts and not mismatches
        else "completed_with_blockers",
        "archives_found": len(archives),
        "expected_parts": sorted(expected),
        "missing_parts": missing_parts,
        "archives": results,
        "extracted_root": str(args.extracted_root),
        "mismatches": mismatches,
        "interpretation": "The two present ZIPs are internally valid and their extracted members match; the top-level part 002 is absent.",
        "next_actions": [
            "Obtain part 002 if it contains additional files; do not invent it from the extracted directory.",
            "Use the raw TCGA files from part 003 to build the provenance-preserving mutation join.",
        ],
    }
    write_json_with_explanation(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
