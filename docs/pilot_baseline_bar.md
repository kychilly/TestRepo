# Week 4 pilot baseline bar

Run date: 2026-08-16 (local), fold 0, seed 17.

The pilot was run with `data/TP53 Dataset(preprocessed) 2/pilot/pilot_subsample.h5ad`,
`splits/neftel_pilot_patient_splits.json`, and `config/baselines_pilot.yaml`.
Evaluation used `eval.py` with `config/evaluation_pilot.yaml` on the held-out
`test` patients and 1,000 patient-bootstrap replicates.

| Baseline | Status | Test macro-F1 | Test balanced accuracy | Notes |
|---|---|---:|---:|---|
| PCA + LogReg | completed | 0.495020 | 0.528025 | Selected `C=1.0` using validation-cell accuracy |
| scVI + probe | not applicable | — | — | Configured `counts` layer is absent from the pilot H5AD |
| Harmony + kNN | not applicable | — | — | Pilot has only one training batch (`CrossSection=none`) |

PCA bootstrap 95% intervals are macro-F1 `[0.424857, 0.539069]` and balanced
accuracy `[0.452198, 0.559096]`.

Artifacts:

- Baseline outputs: `baseline_results/pilot_fold0_seed17/`
- Evaluation source of truth: `reports/pilot_baselines/pca_pilot_eval/metrics.json`
- Evaluation manifest: `reports/pilot_baselines/pca_pilot_eval/cell_state/evaluation_manifest.json`

The full system's Week 4 pilot target should therefore exceed macro-F1
`0.495020` (and preferably balanced accuracy `0.528025`) under the same split,
seed, and evaluation protocol. The two unavailable baselines remain explicit
coverage gaps, not zero-valued measurements.
