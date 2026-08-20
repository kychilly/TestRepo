#!/usr/bin/env python3
"""Build the real unranked internal candidate universe from the pilot H5AD."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gbm_study.plain_english import write_jsonl_explanation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import anndata as ad  # type: ignore[import-not-found]

    data = ad.read_h5ad(args.adata, backed="r")
    digest = hashlib.sha256()
    with args.adata.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    source_hash = digest.hexdigest()
    genes = sorted({str(gene).upper() for gene in data.var_names})
    rows = [
        {
            "gene": gene,
            "source": "internal_h5ad_gene_panel",
            "source_h5ad_sha256": source_hash,
            "ranked": False,
        }
        for gene in genes
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    write_jsonl_explanation(
        args.output,
        row_count=len(rows),
        description="These are the real genes present in the internal pilot panel. They are not ranked yet.",
        next_actions=["Rank these genes with the real scGPT run, then join real validator evidence."],
    )
    print(json.dumps({"status": "completed", "genes": len(rows), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
