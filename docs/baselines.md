# Neftel conventional baselines

The baseline track treats cells as observations nested within patients. The only legal split is the supplied patient split file; the loader rejects overlap, missing patients, duplicate cell IDs, unknown states, and cells that cannot be assigned to exactly one partition. CGGA paths and patient identifiers are prohibited.

## Input contract

The CLI accepts Jeffrey's processed `.h5ad` output or the intermediate NumPy
archive used by the synthetic tests. For `.h5ad`, the configured observation
columns (`patient_id_key`, `state_key`, optional `cell_id_key`, and optional
`batch_key`) and gene column (`gene_id_key`, or `var_names`) are used; cell IDs
fall back to `obs_names`. For `.npz`, the archive must contain:

- `X`: finite cell-by-frozen-gene numeric matrix;
- `patient_id`, `cell_id`, and `state`: one-dimensional arrays with one value per cell;
- `gene_ids`: one-dimensional frozen gene identifier array;
- optional `batch` for a prespecified covariate.

The split loader accepts both the repository names `train`, `validation`,
`test` and Jeffrey's generated names `train`, `val`, `test_cgga`, normalizing
the latter in memory without changing the split hash. `test_cgga` remains the
held-out test partition.

The four state labels are fixed as `AC`, `MES`, `NPC`, `OPC`, and every probability file uses `probability_AC`, `probability_MES`, `probability_NPC`, `probability_OPC` in that order.

## Methods

`pca_logreg` fits `StandardScaler`, PCA, and multinomial logistic regression on training patients only. It does not perform HVG selection. The selected frozen gene list must exactly match the configured preprocessing gene list. The runner selects `C` from the configured candidates using validation-cell accuracy only, then refits the selected model on training patients and records the selection.

`scvi_probe` trains scVI on training cells only, obtains a latent representation,
and fits the shared logistic probe on that latent space. Query cells use
scVI's query-data pathway and never participate in training. It requires
non-negative integer-like raw counts and records count-layer/latent/epoch/batch
settings. A Jeffrey-preprocessed H5AD is log-normalized in `X`; it is rejected
for scVI unless a raw integer `layers["counts"]` is supplied. Missing scvi-tools or invalid data produces a structured
non-applicability record.

`harmony_knn` fits Harmony on training cells only, then learns a frozen Ridge
projection from training PCA/batch features to the Harmony embedding. That
projection transforms validation/test cells before kNN. This is explicitly
recorded as `harmony_train_only_frozen_query_projection_knn`; fitting Harmony
on all cells remains prohibited because it would leak information.

## Run

```sh
PYTHONPATH=src python scripts/run_baseline.py \
  --method pca_logreg \
  --adata PATH_TO_NEFTEL_NPZ \
  --splits splits/patient_splits.json \
  --fold 0 \
  --seed 17 \
  --config config/baselines.yaml \
  --output baseline_results/baselines/pca_logreg/fold0_seed17
```

Successful PCA runs write `predictions.jsonl`, `patient_summary.json`, and `run_metadata.json`. Every cell row contains the requested run/method/fold/seed/patient/cell/true/predicted/probability/split/hash fields. Failed and non-applicable methods write `run_error.json` with status, method, fold, seed, reason, config hash, runtime, and Git metadata. No method is declared scientifically successful merely because the process exits cleanly.

This repository has no real Neftel dataset or split file, so no baseline result is reported here. The first reproducible run must use the Data Lead's actual files and preserve their hashes. Harmony also requires `batch_key: batch` (or another explicitly populated covariate) in the configuration; a missing prespecified covariate is a deliberate failure.
