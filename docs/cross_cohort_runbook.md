# Cross-cohort IDH evaluation

## What the supplied data can measure

The new archive supports a patient-level **IDH mutation-status** experiment:
fit on a stratified TCGA training split plus 27 patient-balanced Neftel
IDH-wildtype pseudobulks, evaluate internally on never-fit TCGA patients, and
evaluate externally on 624 never-fit labeled CGGA patients. TCGA, Neftel, and
CGGA are converted to within-sample gene percentile ranks because their source
files use incompatible numerical units. This transform is fixed without using
any outcome labels.

This is not an AC/MES/NPC/OPC external experiment. Every one of the 1,018 rows
in `cgga_pilot_subsample.h5ad` has `derived_state=Unknown`. Calling an IDH
result a cell-state result would be incorrect.

## Run or resume

```sh
PYTHONPATH=src:. .venv-scgpt/bin/python scripts/run_cross_cohort.py \
  --config config/cross_cohort.yaml \
  --output reports/cross_cohort_20260820
```

The command runs four feature arms (`all_genes`, `confirmed_genes`,
`unconfirmed_genes`, and a seeded size-matched shuffled control) with seeds 17,
42, and 101. `runs.jsonl` is an append-only checkpoint. Re-running the exact
command skips completed arm/seed pairs. Each machine-readable JSON or JSONL has
a same-name plain-English `.txt` companion.

## Metric definitions

- Macro-F1 is binary IDH WT-versus-mutant macro-F1 at a fixed 0.5 threshold.
- `idh_mutation_auroc` is patient-level IDH mutation-status AUROC. It is not
  variant-effect AUROC and is named differently to prevent that mix-up.
- Internal-to-external drop is internal TCGA macro-F1 minus external CGGA
  macro-F1. Positive means performance became worse externally.
- The headline drop advantage is unconfirmed-gene drop minus confirmed-gene
  drop. Positive means the confirmed model lost less performance externally.
- GRN AUROC is the existing prior-confidence sanity check. The supplied holdout
  collapses to one unique positive edge, so it is not publication evidence.

## What remains genuinely blocked

- External four-state macro-F1 needs authoritative CGGA AC/MES/NPC/OPC labels.
- Variant-effect AUROC needs independent benign and pathogenic variants plus a
  continuous score for each variant. The supplied table labels all 1,200
  missense calls pathogenic and therefore has no negative class.
- Abstention accuracy needs independently adjudicated gold outcomes. Scoring a
  validator against labels produced by that validator would be circular.
- A publishable GRN edge AUROC needs a larger frozen held-out positive set and
  prespecified matched negatives.
- A publishable confirmed/unconfirmed headline needs substantially more than
  the current two confirmed genes (TP53 and IDH1), and it must beat the shuffled
  size-matched control.

The completed run's confirmed pair did **not** beat that shuffled control, so
the machine-readable and plain-English reports explicitly call the current
headline `no_result`. The full scGPT all-on/validator-off/GRN-off/MC-off matrix
is a separate CUDA run and must not be confused with these CPU feature arms.
