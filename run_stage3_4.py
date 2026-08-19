"""
Stage 3 + Stage 4 driver.

Stage 3: join Jeffrey's per-patient mutation-to-gene mapping with the
precomputed protein evidence table (currently only populated for the four
gate genes: TP53, IDH1, EGFR, RPRM).

Stage 4: run validator.classify() (unmodified) over every joined record and
tally outcomes into the bucket-count table.

IMPORTANT CAVEAT (read before trusting the numbers below):
Protein evidence (pLDDT / ESM1b / ddG) has only been assembled for the four
gate genes so far. Every other candidate gene in the mutation table has no
evidence, so classify() will correctly route any of its missense mutations
to DATA_DEFICIENT (Step 2 of the decision tree: "no ESM1b score means we
cannot run the test, full stop"). That is not a bug in this script -- it is
an accurate reflection of the current pipeline state. The bucket counts
below should be read as "what Stage 4 outputs today, given evidence that
exists today," not as a final validated result.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from validator import (  # noqa: E402
    GeneRecord,
    Outcome,
    Thresholds,
    classify,
)

CONFIG_PATH = Path(__file__).parent / "config" / "validator.yaml"

# Protein evidence assembled so far -- gate genes only (per Ishaan's Week 1 table).
PROTEIN_EVIDENCE: dict[str, dict[str, float | None]] = {
    "TP53": {"plddt": 96.62, "esm1b": -10.12, "ddg": 3.01},
    "IDH1": {"plddt": 96.56, "esm1b": -13.77, "ddg": 0.4},
    "EGFR": {"plddt": 51.22, "esm1b": -11.78, "ddg": None},
    "RPRM": {"plddt": 63.16, "esm1b": -5.22, "ddg": None},
}

# Priority used only when collapsing multiple per-patient instances of the
# same gene down to a single gene-level bucket (see note in main()).
_GENE_LEVEL_PRIORITY = [
    Outcome.DESTABILIZING_DRIVER,
    Outcome.FUNCTIONAL_DRIVER,
    Outcome.UNCONFIRMED,
    Outcome.DATA_DEFICIENT,
    Outcome.ABSTAIN,
]


def load_altered_records(mutation_csv: Path) -> list[GeneRecord]:
    """Stage 3: read Jeffrey's table, keep only real alterations, join evidence."""
    records: list[GeneRecord] = []
    with open(mutation_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["alteration_type"] == "none":
                continue  # background/unaltered gene-patient pair, not a candidate
            gene = row["gene"]
            evidence = PROTEIN_EVIDENCE.get(gene, {})
            records.append(
                GeneRecord(
                    gene=gene,
                    mutation=f"{row['patient']}:{row['impact']}",
                    alteration_type=row["alteration_type"],
                    plddt=evidence.get("plddt"),
                    esm1b=evidence.get("esm1b"),
                    ddg=evidence.get("ddg"),
                )
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 3 + Stage 4 over the full pilot candidate list.")
    parser.add_argument(
        "--mutations",
        type=Path,
        default=Path("data") / "pilot" / "patient_gene_mutation_table.csv",
        help="Path to Jeffrey's per-patient mutation-to-gene mapping CSV.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to validator.yaml.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="If given, also write bucket_counts.csv and verdicts.csv here (e.g. reports/stage34).",
    )
    args = parser.parse_args()

    thresholds = Thresholds.from_yaml(args.config)
    records = load_altered_records(args.mutations)

    # Stage 4: classify every (gene, patient) altered instance.
    verdicts = [classify(r, thresholds) for r in records]

    # --- Instance-level bucket counts (one row per gene-patient alteration) ---
    instance_counts = Counter(v.outcome for v in verdicts)

    # --- Gene-level bucket counts ---
    # A candidate gene can appear altered in more than one patient, sometimes
    # with different alteration_types (12 of 1222 genes here). There's no
    # spec yet for how to collapse per-patient verdicts into one per-gene
    # bucket, so this uses the most permissive rule: a gene counts as its
    # single best outcome across all patients it was altered in, using
    # DESTABILIZING_DRIVER > FUNCTIONAL_DRIVER > UNCONFIRMED > DATA_DEFICIENT
    # > ABSTAIN. Flag this assumption to the team -- a stricter or
    # per-patient-level rule may be what Stage 5 actually needs.
    best_per_gene: dict[str, Outcome] = {}
    for v in verdicts:
        current = best_per_gene.get(v.gene)
        if current is None or _GENE_LEVEL_PRIORITY.index(
            v.outcome
        ) < _GENE_LEVEL_PRIORITY.index(current):
            best_per_gene[v.gene] = v.outcome
    gene_counts = Counter(best_per_gene.values())

    print("=" * 70)
    print("BUCKET-COUNT TABLE -- gene-patient instances (Stage 4 raw output)")
    print("=" * 70)
    total_instances = sum(instance_counts.values())
    for outcome in Outcome:
        n = instance_counts.get(outcome, 0)
        print(f"  {outcome.value:22s} {n:6d}   ({n / total_instances:.1%})")
    print(f"  {'TOTAL':22s} {total_instances:6d}")

    print()
    print("=" * 70)
    print(f"BUCKET-COUNT TABLE -- unique genes ({len(best_per_gene)} candidate genes)")
    print("=" * 70)
    total_genes = sum(gene_counts.values())
    for outcome in Outcome:
        n = gene_counts.get(outcome, 0)
        print(f"  {outcome.value:22s} {n:6d}   ({n / total_genes:.1%})")
    print(f"  {'TOTAL':22s} {total_genes:6d}")

    print()
    print("Genes that reached DESTABILIZING_DRIVER or FUNCTIONAL_DRIVER"
          " (the only ones eligible for Stage 5):")
    drivers = sorted(
        g for g, o in best_per_gene.items()
        if o in (Outcome.DESTABILIZING_DRIVER, Outcome.FUNCTIONAL_DRIVER)
    )
    print(f"  {drivers}")

    if args.report_dir is not None:
        args.report_dir.mkdir(parents=True, exist_ok=True)

        with open(args.report_dir / "bucket_counts.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["level", "outcome", "count", "total", "fraction"])
            for outcome in Outcome:
                n = instance_counts.get(outcome, 0)
                writer.writerow(["gene_patient_instance", outcome.value, n, total_instances, round(n / total_instances, 4)])
            for outcome in Outcome:
                n = gene_counts.get(outcome, 0)
                writer.writerow(["unique_gene", outcome.value, n, total_genes, round(n / total_genes, 4)])

        with open(args.report_dir / "verdicts.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["gene", "mutation", "outcome", "reason"])
            for v in verdicts:
                writer.writerow([v.gene, v.mutation, v.outcome.value, v.reason])

        print(f"\nWrote {args.report_dir / 'bucket_counts.csv'} and {args.report_dir / 'verdicts.csv'}")


if __name__ == "__main__":
    main()