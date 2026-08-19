# Week 4 Baseline Bar — Fold 0, Seed 42

The full system must beat **harmony_knn: macro-F1 = 0.2714** on the held-out test split
(the stronger of the two applicable conventional baselines).

## Results (test split)

| Method       | Macro-F1 | Balanced Accuracy | Status |
|--------------|----------|--------------------|--------|
| harmony_knn  | 0.2714   | 0.2738             | completed — **bar to beat** |
| pca_logreg   | 0.1712   | 0.2197             | completed |
| scvi_probe   | n/a      | n/a                | not_applicable (no raw counts layer wired up) |

## Per-state F1 (harmony_knn)

| State | F1     | Precision | Recall |
|-------|--------|-----------|--------|
| AC    | 0.300  | 0.300     | 0.300  |
| MES   | 0.250  | 0.233     | 0.270  |
| NPC   | 0.286  | 0.278     | 0.294  |
| OPC   | 0.250  | 0.273     | 0.231  |

## Per-state F1 (pca_logreg)

| State | F1     | Precision | Recall |
|-------|--------|-----------|--------|
| AC    | 0.000  | 0.000     | 0.000  |
| MES   | 0.202  | 0.173     | 0.243  |
| NPC   | 0.105  | 0.500     | 0.059  |
| OPC   | 0.377  | 0.280     | 0.577  |

**Note:** pca_logreg never predicts "AC" (precision/recall = 0.0), collapsing most
predictions into OPC/MES. Chance-level macro-F1 for 4 balanced classes is ~0.25 —
both baselines are near or below chance, which is plausible at pilot scale (small
per-patient cell counts, few patients/fold) but should be called out, not glossed
over, when this bar is cited later.

**Known issue — do not cite CIs yet:** the bootstrap CI in both eval reports is
degenerate (`ci_2_5 == ci_97_5 == point_estimate` despite `valid_replicates: 1000`).
This looks like a bug in `evaluation/reporting.py`'s CI computation, not a genuine
zero-width interval. Point estimates above are likely fine; treat the CIs as
placeholder until that's fixed.

## Provenance

| Field | harmony_knn | pca_logreg |
|---|---|---|
| Data | `data/pilot/pilot_subsample_mgh_only.h5ad` (CGGA excluded) | same |
| Splits | `splits/patient_splits.json`, fold 0 | same |
| split_hash | `1a5c78c6a5ae6eebc6112aebf6b79c9e9e1e9000627c6cc3f17132b1fe7cf53f` | same |
| input_file_sha256 | `23ee40318c9a4fbc4d87e062a3ddd0d39aa20a2335ec3699ebb272ff65a78fa1` | `f52773d37320135e25671985fe9c6c813278a5888cfa05f1c06ba95e831909b1` |
| metric_config_hash | `f52c48c00056e6b78a151fde5b02a781b0f3e940d942335702cc7bb266ac6d14` | same |
| evaluator_git_commit | `2d1ea0a235d31d965d907b3097d62fb2a8dd6464` | same |
| eval timestamp (UTC) | 2026-08-19T05:48:09 | 2026-08-19T05:46:45 |

scvi_probe was excluded from this comparison — its configured count layer
(`counts`) is not a validated raw-integer-count artifact in the current pilot
subsample, so the method could not be scientifically applied. This should be
revisited once a proper raw-counts preprocessing artifact exists, rather than
left out permanently.

The CI collapse is telling you something true about your current pilot split (too few test patients to support resampling), and that's worth noting directly in the Week 4 write-up rather than papering over
(in other words pilot split 5-10 patients is somehow too small)