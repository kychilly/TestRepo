#!/usr/bin/env python3
"""Manifest, verify, and optionally package the exact Week 3/4 run assets."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any

from gbm_study.plain_english import write_json_with_explanation


ROOT = Path(__file__).resolve().parents[1]
ASSET_PATHS = (
    "data/import_20260820/TP53 Dataset(preprocessed)/processed/neftel_analysis_cohort.h5ad",
    "data/import_20260820/TP53 Dataset(preprocessed)/processed/analysis_ready_combined.h5ad",
    "data/import_20260820/TP53 Dataset(preprocessed)/pilot/tcga_pilot_subsample.h5ad",
    "artifacts/models/scGPT_pancancer/best_model.pt",
    "artifacts/models/scGPT_pancancer/vocab.json",
    "artifacts/models/scGPT_pancancer/args.json",
    "data/pilot/internal_candidate_universe.jsonl",
    "data/pilot/week3_validator_outcomes.jsonl",
    "data/pilot/stage34_verdicts_current.csv",
    "data/import_20260820/TP53 Dataset(preprocessed)/prior/grn_pilot_train_prior.csv",
    "data/import_20260820/TP53 Dataset(preprocessed)/prior/grn_pilot_adit_holdout_check.csv",
    "splits/combined_full_cohort_neftel_patient_splits.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_assets(root: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    missing: list[str] = []
    for relative in ASSET_PATHS:
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        files[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    return {
        "status": "completed" if not missing else "completed_with_blockers",
        "scope": "exact local assets required by the Adit Week 3 A100 and Week 4 IDH runs",
        "root": str(root),
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(int(value["bytes"]) for value in files.values()),
        "missing": missing,
        "limitations": [
            "These assets do not supply CGGA AC/MES/NPC/OPC truth.",
            "The current validator outcomes contain zero confirmed genes.",
            "The GRN holdout is a one-positive-edge software check, not a paper-ready benchmark.",
        ],
        "next_actions": [
            "Transfer the bundle or these exact files to the same relative paths on JupyterHub.",
            "Run this script with --verify-manifest on JupyterHub before starting the A100 job.",
        ],
    }


def verify_assets(root: Path, manifest_path: Path) -> dict[str, Any]:
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_files = expected.get("files")
    if not isinstance(expected_files, dict):
        raise ValueError("Asset manifest must contain a files object")
    required_paths = set(ASSET_PATHS)
    manifest_paths = {str(path) for path in expected_files}
    manifest_missing_entries = sorted(required_paths - manifest_paths)
    manifest_unexpected_entries = sorted(manifest_paths - required_paths)
    missing: list[str] = []
    mismatches: list[str] = []
    verified: list[str] = []
    for relative, metadata in expected_files.items():
        path = root / str(relative)
        if not path.is_file():
            missing.append(str(relative))
            continue
        if not isinstance(metadata, dict):
            mismatches.append(f"{relative}: invalid manifest metadata")
            continue
        actual_size = path.stat().st_size
        expected_size = int(metadata.get("bytes", -1))
        actual_hash = sha256(path)
        expected_hash = str(metadata.get("sha256", ""))
        if actual_size != expected_size or actual_hash != expected_hash:
            mismatches.append(
                f"{relative}: expected {expected_size} bytes/{expected_hash}, "
                f"found {actual_size} bytes/{actual_hash}"
            )
        else:
            verified.append(str(relative))
    return {
        "status": (
            "passed"
            if not missing
            and not mismatches
            and not manifest_missing_entries
            and not manifest_unexpected_entries
            else "blocked"
        ),
        "scope": "byte-for-byte A100 asset verification",
        "manifest": str(manifest_path),
        "required_file_count": len(ASSET_PATHS),
        "manifest_file_count": len(manifest_paths),
        "manifest_missing_entries": manifest_missing_entries,
        "manifest_unexpected_entries": manifest_unexpected_entries,
        "verified_files": verified,
        "verified_file_count": len(verified),
        "missing": missing,
        "mismatches": mismatches,
        "next_actions": (
            ["Proceed to scripts/a100_preflight.py."]
            if not missing
            and not mismatches
            and not manifest_missing_entries
            and not manifest_unexpected_entries
            else [
                "Restore the trusted 12-file manifest, re-transfer every missing or mismatched file, then verify again."
            ]
        ),
    }


def write_bundle(root: Path, output: Path, manifest: dict[str, Any]) -> None:
    if output.suffix != ".tar":
        raise ValueError("Bundle output must end in .tar (uncompressed for predictable creation)")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w") as archive:
        for relative in ASSET_PATHS:
            path = root / relative
            if path.is_file():
                archive.add(path, arcname=relative, recursive=False)
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
        info = tarfile.TarInfo("A100_ASSET_MANIFEST.json")
        info.size = len(manifest_bytes)
        archive.addfile(info, io.BytesIO(manifest_bytes))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("reports/readiness/a100_asset_manifest.json"),
    )
    parser.add_argument("--bundle", type=Path, default=None)
    parser.add_argument("--verify-manifest", type=Path, default=None)
    parser.add_argument(
        "--verification-output",
        type=Path,
        default=Path("reports/readiness/a100_asset_verification.json"),
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.verify_manifest is not None:
        result = verify_assets(root, args.verify_manifest)
        write_json_with_explanation(args.verification_output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "passed" else 2
    result = inspect_assets(root)
    if result["status"] != "completed":
        # Never overwrite the trusted complete manifest with an incomplete
        # inventory. Otherwise a later verification can incorrectly pass by
        # checking only the files that happened to exist on the broken host.
        failure_path = args.manifest.with_name(f"{args.manifest.stem}_generation_error.json")
        result["manifest_not_overwritten"] = str(args.manifest)
        result["failure_report"] = str(failure_path)
        result["next_actions"] = [
            "Obtain the complete bundle from a source machine that has all 12 assets.",
            "Do not use this incomplete machine to generate the trusted manifest.",
        ]
        write_json_with_explanation(failure_path, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2
    write_json_with_explanation(args.manifest, result)
    if args.bundle is not None:
        write_bundle(root, args.bundle, result)
        result["bundle"] = {
            "path": str(args.bundle),
            "bytes": args.bundle.stat().st_size,
            "sha256": sha256(args.bundle),
        }
        write_json_with_explanation(args.manifest, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
