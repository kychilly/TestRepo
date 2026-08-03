# Manuscript evaluation protocol

This is the only approved pathway for manuscript numbers. It accepts a validated prediction file and the exact patient split file used for the run. It never imports or executes a model.

## Input validation

Prediction files may be JSONL, CSV, or Parquet. Cell-state rows require the run, method, fold, seed, patient/cell IDs, true and predicted state, four fixed probability columns, split, split hash, configuration hash, and model hash. Patient binary rows require an explicit `task` (`IDH` or `TP53`), one row per patient, binary `true_label`, and `probability_positive`. Patient labels are never inferred from cell predictions or majority vote.

The evaluator rejects duplicate keys, unknown states/tasks, probabilities outside `[0, 1]`, probability rows that do not sum to one, split membership errors, unknown patients, inconsistent patient-level rows, missing model metadata, unrecognized metric units, and split-hash mismatch.

## Metrics

For `metric_units: cell_state`, the evaluator reports macro-F1, balanced accuracy, per-state precision/recall/F1, confusion matrix, multiclass log loss, and multiclass Brier score in fixed `AC`, `MES`, `NPC`, `OPC` order.

For `metric_units: patient_binary`, it reports AUROC, AUPRC, balanced accuracy, sensitivity, specificity, Brier score, and calibration slope/intercept when the configured minimum sample size and both classes are available.

Non-estimable metrics are structured objects with `status: non_estimable`, a reason, and the metric name. They are never replaced with zero or unexplained NaN.

## Patient bootstrap

Every bootstrap replicate samples patient IDs with replacement and includes all rows for each sampled patient. Cells are never sampled independently. The distribution records the sampled patient IDs, sampled cell count, whether every required class was represented, validity, and the replicate estimate. Reports include the point estimate, bootstrap median, 2.5th and 97.5th percentiles, valid replicate count, and non-estimable replicate count.

## Run

```sh
PYTHONPATH=src python scripts/run_evaluation.py \
  --predictions results/baselines/pca_logreg/fold0_seed17/predictions.jsonl \
  --splits splits/patient_splits.json \
  --config config/evaluation.yaml \
  --output results/evaluation/pca_logreg/fold0_seed17
```

The command writes `metrics.json`, `bootstrap_distribution.parquet`, `confusion_matrix.csv`, `per_patient_metrics.parquet`, `warnings.json`, and `evaluation_manifest.json`. The manifest records evaluator Git commit, prediction/split hashes, split hash, configuration hash, bootstrap seed and replicate count, UTC timestamp, and package versions.

No notebook output is a manuscript source. Manuscript values must be traced to these files and their manifest.
