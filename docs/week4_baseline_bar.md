# Week 4 baseline bar

This is the bar the full system must beat. Numbers are pulled directly from
`results/evaluation/<method>/fold0_seed17/metrics.json` — never retyped or
estimated by hand. If a number changes, rerun the pipeline and repaste; do not
hand-edit this table.

## Evidence commands

```text
PYTHONPATH=src python baselines.py --method pca_logreg --adata data/pilot/pilot_subsample.h5ad --splits splits/patient_splits.json --fold 0 --seed 17 --config config/baselines.yaml --output results/baselines/pca_logreg/fold0_seed17
PYTHONPATH=src python baselines.py --method scvi_probe --adata data/pilot/pilot_subsample.h5ad --splits splits/patient_splits.json --fold 0 --seed 17 --config config/baselines.yaml --output results/baselines/scvi_probe/fold0_seed17
PYTHONPATH=src python baselines.py --method harmony_knn --adata data/pilot/pilot_subsample.h5ad --splits splits/patient_splits.json --fold 0 --seed 17 --config config/baselines.yaml --output results/baselines/harmony_knn/fold0_seed17

PYTHONPATH=src python eval.py --cell-predictions results/baselines/pca_logreg/fold0_seed17/predictions.jsonl --splits splits/patient_splits.json --config config/evaluation.yaml --output results/evaluation/pca_logreg/fold0_seed17
PYTHONPATH=src python eval.py --cell-predictions results/baselines/harmony_knn/fold0_seed17/predictions.jsonl --splits splits/patient_splits.json --config config/evaluation.yaml --output results/evaluation/harmony_knn/fold0_seed17
```

## Results (fold 0, seed 17)

| Method       | Status         | Macro-F1 | 95% CI (macro-F1) | Balanced accuracy | 95% CI (balanced accuracy) | Notes |
| ------------ | -------------- | -------- | ------------------ | ------------------ | ---------------------------- | ----- |
| harmony_knn  | completed      | 0.8533   | 0.7413 – 0.9456     | 0.8468              | 0.7380 – 0.9439               | Best-performing baseline. Non-overlapping CI with pca_logreg — a genuine, not marginal, gap. |
| pca_logreg   | completed      | 0.6256   | 0.5539 – 0.6676     | 0.6328              | 0.5689 – 0.6746               |       |
| scvi_probe   | not_applicable | —        | —                   | —                   | —                             | Requires non-negative integer raw counts; pilot data is logTPM-only (`IDHwtGBM.processed.SS2.logTPM.txt.gz`), no raw counts file available from this source. Not a code defect — see `docs/baselines.md`. |

### Per-state F1 (for diagnosing where the full system needs to improve)

| State | harmony_knn F1 | pca_logreg F1 |
| ----- | --------------- | -------------- |
| AC    | 0.8185          | 0.5946         |
| MES   | 0.8070          | 0.5848         |
| NPC   | 0.9270          | 0.7412         |
| OPC   | 0.8607          | 0.5816         |

## Provenance

- `split_hash` (both runs, identical): `ef61ed260a489e530f6fffe315f6b04e658f3f254831717364cab51209e0c450`
- `metric_config_hash` (evaluation config, both runs, identical): `2e5e912ddb42620b6aac63317963e2455a774b8a41c5712669821b4cecf29264`
- `config_hash` (baselines.yaml): `<paste from results/baselines/<method>/fold0_seed17/run_metadata.json>`
- evaluator git commit: `a7e7609f2e804b41cbf77e848da7495021759cab`
- bootstrap: 1000 replicates, seed 17, 0 non-estimable replicates for either method
- date generated: 2026-08-17 (UTC timestamps: harmony_knn 09:11:18, pca_logreg 09:09:20)
- package versions: numpy 2.4.6, pandas 3.0.5, pyarrow 25.0.1, scikit-learn 1.9.0, scipy 1.18.0

## What "beating this bar" means

Any full-system claim must show macro-F1 and balanced accuracy on the same
fold/seed/split exceeding **harmony_knn** (macro-F1 0.8533, 95% CI
0.7413–0.9456) — the stronger of the two runnable baselines — with
non-overlapping (or clearly superior) bootstrap 95% CIs, not just a higher
point estimate. `scvi_probe` is excluded from the comparison until real
raw-count data is available; if raw counts are obtained before Week 4, rerun
`scvi_probe` and re-evaluate this bar before finalizing it.