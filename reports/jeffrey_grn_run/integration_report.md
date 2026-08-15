# Jeffrey GRN integration run — 2026-08-15

## Scope

This run uses Jeffrey's supplied pilot prior files together with the existing
repository code. Only Jeffrey's GRN integration was added. Teammate-owned
missing deliverables were executed when an existing path supported them, but
were not implemented here.

## Jeffrey GRN result

Inputs:

- Train prior: `TP53 Dataset(preprocessed)/prior/grn_pilot_train_prior.csv`
- Held-out check: `TP53 Dataset(preprocessed)/prior/grn_pilot_adit_holdout_check.csv`
- Train SHA-256: `c734cbb6b0fb0c159b17beafa20947dd06f5f8c86a4a779955f1e713d08c3841`
- Held-out SHA-256: `1e597a3d371c77d15159553a5434300e11cc0f3cd4b79bd0368eacf032348ec7`

Result: **completed**. The duplicate TP53→RPRM evidence rows were collapsed
to one unique held-out edge. Confidence labels were mapped deterministically
(`A=1.0`, `B=0.8`, `C=0.6`, `D=0.4`, `E=0.2`).

- Held-out positives: 1
- Deterministic negatives: 144
- AUROC: **1.0**
- Output: `grn_sanity.json`
- Visualization: `grn_prior.png`

This is a one-positive sanity check, not a publication-grade GRN performance
estimate.

## Existing team paths run

- Four-gene validator classification gate: **passed**
  - TP53 → destabilizing_driver
  - IDH1 → functional_driver
  - EGFR → abstain
  - RPRM → abstain
- Validator publication gate: **blocked** because evidence versions are
  incomplete and the IDH1 ΔΔG value is estimated.
- Full repository tests: **70 passed** after installing the pinned `anndata`
  and `harmonypy` runtimes.
- Pilot PCA/logistic baseline: **completed** on the five-patient pilot test
  split. Test macro-F1: **0.6256**; balanced accuracy: **0.6488**.
- Pilot Harmony/kNN baseline: **completed** on the same test split. Test
  macro-F1: **0.5555**; balanced accuracy: **0.5524**.
- Pilot scVI baseline: **not applicable** — the H5AD has no raw integer
  `counts` layer, so normalized expression cannot be substituted.
- Test metrics use the repository evaluator with **1,000 patient-level
  bootstrap replicates**. They are pilot baselines, not full-system targets.
- Synthetic smoke remains a software check only; its metrics are not pilot or
  manuscript metrics.

Test-only bootstrap intervals from 1,000 patient-level replicates:

- PCA/logistic macro-F1 95% interval: **0.4613–0.6702**; 999 valid
  replicates and 1 non-estimable replicate.
- Harmony/kNN macro-F1 95% interval: **0.4542–0.5880**; 999 valid
  replicates and 1 non-estimable replicate.

Visualization: `../pilot_baselines/test_metrics.png`.

## Explicitly not implemented here

The following teammate-owned code/tasks do not currently have a complete
real-data execution path and were not added in this run:

- Jeffrey: TCGA/CGGA mutation mapping, pilot cohort/HVG construction, and
  three real baseline metrics.
- Ishaan: full Stage 3/4 bucket counts, pooling decision, shuffled validator,
  and data-deficient coverage audit.
- Alexis: independent baseline and GRN reruns beyond the shared existing paths.

## Dataset zip handling

The four CGGA source zip files are now ignored by the scoped rule
`TP53 Dataset(preprocessed)/cgga(raw data)/*.zip`. They were not deleted.
They remain available locally for a future approved extraction/processing run.

## Environment blockers

The current host has no CUDA GPU, no scGPT checkpoint/vocabulary, and no
real scGPT checkpoint/vocabulary, and no raw-count layer in the pilot H5AD.
`anndata` and `harmonypy` were installed locally for the completed PCA and
Harmony runs. No real scGPT or GPU MC-dropout timing was substituted or
inferred.
