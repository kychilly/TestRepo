# Dataset acceptance for the Week 2/3 handoff

## Current decision

The new Neftel pilot is accepted with limitations for the PCA/logistic
cell-state baseline. The new CGGA pilot is rejected in its current form. The
combined dataset is therefore not ready for every paper task, although the
Neftel baseline branch is runnable now.

The measured audit can be reproduced locally without loading data onto the
A100:

```bash
PYTHONPATH=src:. python scripts/audit_week2_datasets.py \
  --neftel "TP53 Dataset(preprocessed) 2/pilot/pilot_subsample.h5ad" \
  --cgga "TP53 Dataset(preprocessed) 2/pilot/cgga_pilot_subsample.h5ad" \
  --tcga-clinical \
    "TP53 Dataset(preprocessed) 2/tcga(raw data)/gbm_tcga_clinical_data.tsv" \
  --neftel-split splits/neftel_pilot_patient_splits.json \
  --output /tmp/tp53-week2-dataset-audit.json
```

The audit returns exit code 2 while either contract is rejected. This is an
intentional paper-readiness gate, not a GPU failure.

## Neftel single-cell pilot

- Shape: 4,982 cells by 2,503 genes across 21 patients.
- The matrix contains no non-finite values.
- Train/validation/test are patient-disjoint and contain 13/4/4 patients.
- All four states occur in every partition.
- PCA/logistic training and test evaluation complete with the dedicated split.
- The test set has only 18 OPC cells, and the measured pilot model had OPC F1
  of 0. This limits the strength of any paper claim.
- `CrossSection` has only one value, so Harmony has no batch variable to
  correct.
- There is no raw integer `counts` layer, so scVI cannot be fit correctly.

To enable Harmony, Jeffrey must provide a biologically valid batch column with
at least two represented training batches. To enable scVI, Jeffrey must provide
the same cells and genes with an integer raw-count layer named `counts`, plus
the preprocessing provenance that links it to the normalized matrix.

## CGGA bulk-patient pilot

- Shape: 1,018 patients by 1,752 genes, with one bulk-expression row per
  patient.
- All 1,018 `derived_state` values are `Unknown`; these are not cell-state
  labels.
- The expression matrix contains 15,270 non-finite values affecting all 1,018
  patients.
- No authoritative IDH-status column is present in the H5AD.

CGGA cannot be used as an AC/MES/NPC/OPC external test set. To enable a
separate patient-level IDH evaluation, Jeffrey must provide:

1. an authoritative per-patient IDH label with documented encoding;
2. a deterministic rule for duplicate/mismatched clinical identifiers;
3. a training-only fitted non-finite-value imputation policy, or a documented
   gene-filtering rule applied without using test labels;
4. a final gene identifier convention and mapping provenance; and
5. cohort/version identifiers and source checksums.

## TCGA clinical table

The supplied TCGA directory contains one 619-row clinical TSV, but no TCGA
expression matrix and no authoritative IDH-status field. It is useful as
clinical metadata only; it cannot drive the expression model or the IDH task.

Jeffrey must provide a sample-by-gene TCGA expression source, its normalization
level and gene-ID convention, an authoritative patient/sample join, IDH labels
if TCGA is used for that endpoint, and immutable source/version checksums. Only
the exact split/subset required for the experiment should be streamed.

## Data movement and Git safety

Do not upload either complete dataset directory, raw TSV ZIPs, the 2.6 GB
Neftel file, or model collections to shared GPU storage. Publish only the
required immutable Hugging Face subset/split and stream it at runtime. The
repository ignores dataset trees, H5AD files, archives, and model weights; its
pre-commit hook also blocks forced staging of those artifacts and staged files
larger than 50 MiB.

Git hooks can be deliberately bypassed, so all collaborators must still place
datasets under `data/` or `TP53 Dataset(preprocessed)*/` and must not use
`--no-verify` or change `core.hooksPath`.
