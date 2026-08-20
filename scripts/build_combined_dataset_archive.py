#!/usr/bin/env python3
"""Package the useful old and new TP53 assets without overwriting sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

from gbm_study.plain_english import write_json_with_explanation


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/TP53_combined_old_new_20260820.zip"))
    parser.add_argument("--report", type=Path, default=Path("reports/readiness/combined_dataset.json"))
    args = parser.parse_args(argv)
    root = Path.cwd()
    old = root / "TP53 Dataset(preprocessed) 2"
    state_file = root / "data/neftel_qc-002.h5ad"
    merged = root / "data/import_20260820/TP53 Dataset(preprocessed)/processed/full_cohort_with_states.h5ad"
    analysis_ready = root / "data/import_20260820/TP53 Dataset(preprocessed)/processed/analysis_ready_combined.h5ad"
    clean_cgga = root / "data/import_20260820/TP53 Dataset(preprocessed)/processed/cgga_bulk_clean.h5ad"
    mutation_join = root / "data/import_20260820/TP53 Dataset(preprocessed)/pilot/patient_gene_mutation_join.csv"
    required = [old, state_file, merged, analysis_ready, clean_cgga, mutation_join]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing required combined-dataset input: " + ", ".join(missing))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "completed",
        "scope": "deduplicated useful old/new TP53 dataset bundle",
        "contents": {
            "older_dataset": "older/TP53 Dataset(preprocessed) 2/",
            "older_neftel_full": "older/data/neftel_qc-002.h5ad",
            "new_full_cohort_with_exact_neftel_states": "combined/full_cohort_with_states.h5ad",
            "analysis_ready_neftel_plus_clean_cgga": "combined/analysis_ready_combined.h5ad",
            "clean_cgga_bulk": "combined/cgga_bulk_clean.h5ad",
            "new_tcga_provenance_join": "combined/patient_gene_mutation_join.csv",
        },
        "why_combined": [
            "The older Neftel file supplies 6,576 exact cell-state labels.",
            "The new full cohort supplies the newer cohort annotations and CGGA rows.",
            "All 6,576 Neftel cell IDs matched exactly before the merged H5AD was created.",
        ],
        "not_combined_blindly": [
            "The old and new pilot H5ADs differ in shape and gene panel, so both are preserved rather than concatenated.",
            "CGGA rows remain Unknown for cell state and are not used as state truth.",
        ],
    }
    temporary_manifest = args.output.with_suffix(".manifest.json")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        archive.write(temporary_manifest, "MANIFEST.json")
        for path in old.rglob("*"):
            if path.is_file():
                archive.write(path, "older/" + str(path.relative_to(old)))
        archive.write(state_file, "older/data/neftel_qc-002.h5ad")
        archive.write(merged, "combined/full_cohort_with_states.h5ad")
        archive.write(analysis_ready, "combined/analysis_ready_combined.h5ad")
        archive.write(clean_cgga, "combined/cgga_bulk_clean.h5ad")
        archive.write(mutation_join, "combined/patient_gene_mutation_join.csv")
    temporary_manifest.unlink()
    manifest.update({
        "archive": {"path": str(args.output), "sha256": sha256(args.output), "bytes": args.output.stat().st_size},
        "status": "completed",
    })
    write_json_with_explanation(args.report, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
