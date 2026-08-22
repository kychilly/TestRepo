# GBM ML Research Infrastructure

This repository contains validation-first Week 1 infrastructure for a leakage-resistant glioblastoma study.

## Scope

Week 1 and Week 2 support environment/provenance validation, patient-aware split validation, conventional-model integration points, pilot scGPT candidate serialization, a pluggable GRN recovery branch, MC-dropout uncertainty plumbing, and a validator masking seam. They do not access CGGA, produce manuscript metrics from real assets, or claim model-ranked genes are causal drivers.

The repository now contains the downloaded Neftel preprocessed H5AD under
`data/raw/neftel/`. TCGA/GBM and both CGGA cohorts, the canonical four-state
label column, the scGPT checkpoint, vocabulary, and CUDA runtime are still not
present. The exact data hash and observed fields are recorded in `data/README.md`.

## Data contracts

- `config/week1.example.json` is JSON configuration, not a runnable study configuration.
- A split file must be a JSON object with exactly `train`, `validation`, and `test` keys. Each value is a non-empty list of patient IDs.
- A data manifest and vocabulary are opaque files whose SHA-256 hashes are recorded; their contents must be validated by the eventual modality loader.
- Neftel observations are cells nested within patients. TCGA/CGGA observations are patients, never cells.

## Commands

Create a copy of the example configuration and replace every `null` with supplied Data Lead values. Then run:

```sh
python -m pip install -e '.[test]'
PYTHONPATH=src python -m gbm_study.cli validate-config --config path/to/week1.json
PYTHONPATH=src python -m gbm_study.cli validate-split --config path/to/week1.json
python -m pytest
```

The top-level entry points are also available as `python baselines.py` and
`python eval.py`. `baselines.py` selects one of `pca_logreg`, `scvi_probe`, or
`harmony_knn`, or uses `--method all` to run each arm into a separate
subdirectory; every invocation receives the same patient split file.
`eval.py` writes the validated JSON/manifest outputs described in
`docs/evaluation_protocol.md`.

Successful validation writes a machine-readable JSON result under the configured output directory. Results include the Git commit (or `null` for this initial uncommitted repository), configuration/data/split/vocabulary hashes, random seeds, and runtime information. Writes are atomic.

## Current limitations

The Neftel artifact is real but is already log-normalized and has no raw
`counts` layer, so scVI cannot run from it. It contains `CellAssignment`, not
the agreed `AC/MES/NPC/OPC` state field. TCGA/CGGA are absent, so the combined
train/validation/CGGA-test run and scientific metrics remain blocked.

The contract proposal in `docs/interfaces.md` requires Validator Lead sign-off before production records are emitted.

Ishaan's validator decision tree and runnable four-gene gate are now present.
The software classifications match the requested outcomes, but the authoritative
publication gate is blocked because exact source/version metadata is incomplete
and the IDH1 ΔΔG value is explicitly estimated.

Week 2 additions are software-tested only on synthetic fixtures. The pilot remains
blocked by missing real cell data, checkpoint, vocabulary, and CUDA; the GRN branch
remains blocked until the Data Lead supplies an edge list; and Stage 5 remains
blocked from scientific use until the validator decision tree is signed off.


# Data Registration Log

**Access Date:** 2026-08-09

## Cohort Overview & Sample Counts

| Cohort                    | Accession / Source        | Sample / Patient Count       | Local Directory |
|:--------------------------|:--------------------------|:-----------------------------| :--- |
| **Neftel et al. (2019)**  | SCP393 / GSE131928        | 28 Patients (7,930 cells)    | data/raw/neftel/ |
| **TCGA-GBM**              | cBioPortal                | 528 Patients (592 samples)   | data/raw/tcga/ |
| **TCGA-cna**              | cBioPortal                | 1084 Patients (1084 samples) | data/raw/tcga/ |
| **TCGA-Mutations**        | cBioPortal                | 812 Patients (812 samples)   | data/raw/tcga/ |
| **TCGA-mrna_seq_v2_rsem** | cBioPortal                | 155 Patients (160 samples)   | data/raw/tcga/ |
| **CGGA (325)**            | CGGA Portal (mRNAseq_325) | 325 Patients                 | data/raw/cgga/ |
| **CGGA (693)**            | CGGA Portal (mRNAseq_693) | 693 Patients                 | data/raw/cgga/ |
| **CGGA (286)**            | CGGA Portal (WESeq_286)   | 286 Patients                 | data/raw/cgga/ |

## File Inventory & SHA-256 Hashes

| Relative File Path | Size (MB) | SHA-256 Checksum |
| :--- | :--- | :--- |
