# GBM ML Research Infrastructure

This repository contains validation-first Week 1 infrastructure for a leakage-resistant glioblastoma study.

## Scope

Week 1 supports environment/provenance validation, patient-aware split validation, conventional-model integration points, and a future scGPT checkpoint smoke test. It does not access CGGA, produce manuscript metrics, or claim model-ranked genes are causal drivers.

The repository currently has no study data. The Data Lead must provide the real manifest, patient split, vocabulary, and checkpoint values. The example configuration intentionally leaves these values `null`.

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

No real data, split, labels, checkpoint, environment lock, GPU, or trained model exists in this repository. The evaluation pathway and transcriptomic-to-protein contract are implemented and tested, but no scientific metric, model ranking, or checkpoint inference result is reported until the Data Lead supplies those assets.

The contract proposal in `docs/interfaces.md` requires Validator Lead sign-off before production records are emitted.


# Data Registration Log

**Access Date:** 2026-08-09

## Cohort Overview & Sample Counts

| Cohort | Accession / Source | Sample / Patient Count | Local Directory |
| :--- | :--- | :--- | :--- |
| **Neftel et al. (2019)** | SCP393 / GSE131928 | 28 Patients (7,930 cells) | data/raw/neftel/ |
| **TCGA-GBM** | cBioPortal | 528 Patients (592 samples) | data/raw/tcga/ |
| **CGGA (325)** | CGGA Portal (mRNAseq_325) | 325 Patients | data/raw/cgga/ |
| **CGGA (693)** | CGGA Portal (mRNAseq_693) | 693 Patients | data/raw/cgga/ |

## File Inventory & SHA-256 Hashes

| Relative File Path | Size (MB) | SHA-256 Checksum |
| :--- | :--- | :--- |
