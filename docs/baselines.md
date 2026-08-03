# Neftel conventional baselines

The baseline track treats cells as observations nested within patients. The only legal split is the supplied patient split file; the loader rejects overlap, missing patients, duplicate cell IDs, unknown states, and cells that cannot be assigned to exactly one partition. CGGA paths and patient identifiers are prohibited.

## Input contract

The CLI accepts an intermediate NumPy archive because this repository does not contain an AnnData object. The archive must contain:

- `X`: finite cell-by-frozen-gene numeric matrix;
- `patient_id`, `cell_id`, and `state`: one-dimensional arrays with one value per cell;
- `gene_ids`: one-dimensional frozen gene identifier array;
- optional `batch` for a prespecified covariate.

The four state labels are fixed as `AC`, `MES`, `NPC`, `OPC`, and every probability file uses `probability_AC`, `probability_MES`, `probability_NPC`, `probability_OPC` in that order.

## Methods

`pca_logreg` fits `StandardScaler`, PCA, and multinomial logistic regression on training patients only. It does not perform HVG selection. The selected frozen gene list must exactly match the configured preprocessing gene list. The runner selects `C` from the configured candidates using validation-cell accuracy only, then refits the selected model on training patients and records the selection.

`scvi_probe` is a typed arm for a future real AnnData/scVI run. It requires non-negative integer-like raw counts, records count-layer/latent/epoch/batch settings, and uses a separate logistic probe rather than the scVI classifier. Without scvi-tools and an explicit AnnData/model loader, it produces a structured non-applicability record.

`harmony_knn` is currently classified as non-applicable. The selected Harmony API does not provide a valid transform for unseen validation/test cells in this contract. Fitting Harmony on all cells would leak information, so the implementation stops and records the scientific reason instead of silently running a transductive analysis.

## Run

```sh
PYTHONPATH=src python scripts/run_baseline.py \
  --method pca_logreg \
  --adata PATH_TO_NEFTEL_NPZ \
  --splits splits/patient_splits.json \
  --fold 0 \
  --seed 17 \
  --config config/baselines.yaml \
  --output results/baselines/pca_logreg/fold0_seed17
```

Successful PCA runs write `predictions.jsonl`, `patient_summary.json`, and `run_metadata.json`. Every cell row contains the requested run/method/fold/seed/patient/cell/true/predicted/probability/split/hash fields. Failed and non-applicable methods write `run_error.json` with status, method, fold, seed, reason, config hash, runtime, and Git metadata. No method is declared scientifically successful merely because the process exits cleanly.

This repository has no real Neftel dataset or split file, so no baseline result is reported here. The installed host also lacks scVI/Harmony dependencies. The first reproducible run must use the Data Lead's actual files and preserve their hashes.
