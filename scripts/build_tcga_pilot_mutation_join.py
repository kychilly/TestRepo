#!/usr/bin/env python3
"""Build a provenance-preserving TCGA pilot mutation/CNA join from ZIP data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import anndata as ad  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]

from gbm_study.plain_english import write_json_with_explanation


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mutation_rows(path: Path, patients: set[str], chunksize: int) -> list[pd.DataFrame]:
    keep = {
        "Hugo_Symbol", "NCBI_Build", "Tumor_Sample_Barcode", "Variant_Classification",
        "Variant_Type", "Consequence", "HGVSc", "HGVSp", "HGVSp_Short",
        "Transcript_ID", "Protein_Change", "protein_change", "IMPACT", "build",
    }
    out: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, sep="\t", comment="#", usecols=lambda c: c in keep,
                             chunksize=chunksize, low_memory=False):
        chunk["patient_id"] = chunk["Tumor_Sample_Barcode"].astype(str).str[:12]
        chunk = chunk[chunk["patient_id"].isin(patients)].copy()
        chunk = chunk[chunk["Variant_Classification"].eq("Missense_Mutation")]
        if chunk.empty:
            continue
        chunk["gene_symbol"] = chunk["Hugo_Symbol"].astype(str).str.upper()
        chunk["alteration_type"] = "missense"
        chunk["genome_build"] = chunk.get("NCBI_Build", chunk.get("build", "")).astype(str)
        chunk["source_file"] = str(path)
        out.append(chunk[[
            "patient_id", "gene_symbol", "alteration_type", "Variant_Classification",
            "Variant_Type", "Consequence", "HGVSc", "HGVSp", "HGVSp_Short",
            "Transcript_ID", "Protein_Change", "protein_change", "IMPACT",
            "genome_build", "source_file",
        ]].rename(columns={"Variant_Classification": "variant_classification"}))
    return out


def cna_rows(path: Path, patients: set[str], threshold: float) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t", low_memory=False)
    gene_col = "Hugo_Symbol"
    rows: list[dict[str, Any]] = []
    sample_columns = [column for column in table.columns if str(column)[:12] in patients]
    for _, row in table[[gene_col, *sample_columns]].iterrows():
        gene = str(row[gene_col]).upper()
        if not gene or gene == "NAN":
            continue
        for column in sample_columns:
            value = pd.to_numeric(row[column], errors="coerce")
            if pd.isna(value) or abs(float(value)) < threshold:
                continue
            rows.append({
                "patient_id": str(column)[:12],
                "gene_symbol": gene,
                "alteration_type": "amplification" if float(value) >= threshold else "deletion",
                "variant_classification": None,
                "Variant_Type": None,
                "Consequence": None,
                "HGVSc": None,
                "HGVSp": None,
                "HGVSp_Short": None,
                "Transcript_ID": None,
                "Protein_Change": None,
                "protein_change": None,
                "IMPACT": None,
                "genome_build": "TCGA CNA matrix",
                "source_file": str(path),
            })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, default=Path("data/import_20260820/TP53 Dataset(preprocessed)/pilot/tcga_pilot_subsample.h5ad"))
    parser.add_argument("--mutations", type=Path, default=Path("data/import_20260820/TP53 Dataset(preprocessed)/raw/tcga/data_mutations.txt"))
    parser.add_argument("--cna", type=Path, default=Path("data/import_20260820/TP53 Dataset(preprocessed)/raw/tcga/data_cna.txt"))
    parser.add_argument("--output", type=Path, default=Path("data/import_20260820/TP53 Dataset(preprocessed)/pilot/patient_gene_mutation_join.csv"))
    parser.add_argument("--chunksize", type=int, default=50000)
    parser.add_argument("--cna-threshold", type=float, default=2.0)
    args = parser.parse_args(argv)
    data = ad.read_h5ad(args.pilot, backed="r")
    patients = {str(value)[:12] for value in data.obs["Sample"].astype(str)}
    mutation_parts = mutation_rows(args.mutations, patients, args.chunksize)
    missense = pd.concat(mutation_parts, ignore_index=True) if mutation_parts else pd.DataFrame()
    cna = cna_rows(args.cna, patients, args.cna_threshold)
    joined = pd.concat([missense, cna], ignore_index=True)
    if joined.empty:
        raise SystemExit("No pilot patient rows were found in the raw TCGA files")
    joined = joined.drop_duplicates(subset=["patient_id", "gene_symbol", "alteration_type", "HGVSp_Short", "source_file"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(args.output, index=False)
    result = {
        "status": "completed",
        "scope": "TCGA pilot mutation and CNA join built from raw ZIP members",
        "pilot_patients": len(patients),
        "joined_rows": int(len(joined)),
        "joined_patients": int(joined["patient_id"].nunique()),
        "genes": int(joined["gene_symbol"].nunique()),
        "alteration_type_counts": {str(k): int(v) for k, v in joined["alteration_type"].value_counts().items()},
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "sources": {
            "pilot": {"path": str(args.pilot), "sha256": sha256(args.pilot)},
            "mutations": {"path": str(args.mutations), "sha256": sha256(args.mutations)},
            "cna": {"path": str(args.cna), "sha256": sha256(args.cna)},
        },
        "limitations": [
            "This join contains raw-mutation missense and CNA amplification/deletion calls.",
            "No silencing call is created because the supplied ZIP files contain no methylation/silencing source.",
            "CNA amplification threshold is absolute value >= the configured threshold of 2.0.",
        ],
        "next_actions": [
            "Add a methylation or validated expression-silencing source before using silencing as a label.",
            "Use the provenance columns in this join for Stage 3 variant/abstention evaluation.",
        ],
    }
    write_json_with_explanation(args.output.with_suffix(".json"), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
