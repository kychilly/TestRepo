# Verified Week 4 baseline bar

Dataset: imported 6,576-cell Neftel analysis cohort. Split: 15 train, 6
validation, 6 test patients. Fold 0, seed 42, 1,000 patient-bootstrap
replicates.

| Baseline | Status | Test macro-F1 | Balanced accuracy |
|---|---|---:|---:|
| PCA + logistic regression | completed | 0.518509 | 0.534941 |
| Harmony + kNN | completed | 0.433855 | 0.460169 |
| scVI + probe | data-blocked | — | — |

scVI is blocked because a deterministic sample of the supplied
`layers['counts']` contains non-integer log-scale values. It requires the
original raw count matrix, not transformed logTPM.

The machine-readable bar and plain-English explanation are
`reports/pilot_baselines_verified/baseline_bar.json` and `.txt`. Reproduction
commands are in
[team_week2_to_week4_execution.md](team_week2_to_week4_execution.md), section 4.
