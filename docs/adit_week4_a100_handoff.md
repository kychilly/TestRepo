# Adit Week 4 A100 handoff

This file is the authoritative handoff after merging `origin/jeffrey-week34`
into `adit-week34`. It separates code readiness from scientific completion.

## What Jeffrey's merge establishes

The donor/batch conclusion reproduces on the exact Neftel H5AD used by Adit:

| PCA space | Donor silhouette | State silhouette |
| --- | ---: | ---: |
| 2 dimensions | -0.2067 | 0.0497 |
| 20 dimensions | 0.1509 | 0.0777 |

The result is in
`results/full_data_audit/current_neftel/donor_batch_audit.json`. It supports
keeping `harmony_knn` as the batch-corrected baseline. It does not authorize
post-hoc Harmony correction inside the scGPT arms.

Do not use `splits/patient_splits_full.json` for scGPT. It mixes Neftel and
TCGA IDs. The scGPT split is
`splits/combined_full_cohort_neftel_patient_splits.json` (15 train, 6
validation, 6 test patients).

The legacy commands below now route to maintained, provenance-safe builders:

```bash
python preprocess_full.py
python build_pilot_mutation_table_full.py
python build_splits_full.py
```

The mutation builder does not invent Neftel patient mutations or convert
unknown CGGA calls to missense. The combined expression H5AD is for the
patient-level cross-cohort IDH analysis, not for scVI and not as a replacement
for the Neftel-only scGPT input.

## Current completion state

- Repository quality gate: complete (`98` tests pass).
- scGPT/MC-dropout implementation and resumable checkpoints: code complete.
- Local Adit Week 3 scientific runs: not complete. The existing manifest has
  `0/12` completed because it was run without CUDA.
- Runs executable with current evidence: the three `validator_off` seeds.
- Remaining nine validator-on runs: blocked because Stage 3/4 currently has
  zero confirmed genes (`2454` abstain, `49` data-deficient).
- Full Week 4 scGPT external state evaluation: blocked because CGGA has no
  AC/MES/NPC/OPC truth. The current cross-cohort script is explicitly a CPU
  patient-level IDH analysis, not a scGPT external-state result.

## Inputs that must be present on JupyterHub

The data and checkpoint files below are Git-ignored. A Git push or clone does
not transfer them. Before requesting the A100, create an exact transfer
manifest (and, if useful, one uncompressed tar bundle) on the source machine:

```bash
python scripts/build_a100_asset_bundle.py

# Optional single-file transfer bundle (about 3 GB before filesystem overhead):
python scripts/build_a100_asset_bundle.py \
  --bundle /path/with/enough/space/adit_week4_a100_assets.tar
```

Extract the optional bundle from the JupyterHub repository root:

```bash
tar -xf /path/to/adit_week4_a100_assets.tar
python scripts/build_a100_asset_bundle.py \
  --verify-manifest reports/readiness/a100_asset_manifest.json
```

Do not start the GPU job unless verification reports `status: passed`.

Copy or clone the repository so these paths exist relative to its root:

```text
data/import_20260820/TP53 Dataset(preprocessed)/processed/neftel_analysis_cohort.h5ad
data/import_20260820/TP53 Dataset(preprocessed)/processed/analysis_ready_combined.h5ad
data/import_20260820/TP53 Dataset(preprocessed)/pilot/tcga_pilot_subsample.h5ad
artifacts/models/scGPT_pancancer/best_model.pt
artifacts/models/scGPT_pancancer/vocab.json
artifacts/models/scGPT_pancancer/args.json
data/pilot/internal_candidate_universe.jsonl
data/pilot/week3_validator_outcomes.jsonl
data/import_20260820/TP53 Dataset(preprocessed)/prior/grn_pilot_train_prior.csv
splits/combined_full_cohort_neftel_patient_splits.json
```

Use persistent storage for results. The MC samples and checkpoints can occupy
several gigabytes; do not write final outputs only to an ephemeral notebook
filesystem.

## A100 commands

Run these in a Jupyter terminal from the repository root:

```bash
export GBM_A100_SCRATCH=/mnt/localssd/gbm-a100-scratch
export GBM_PERSISTENT_OUTPUT_DIR=/mnt/persistent/gbm-results
export PYTHON_BIN=python3.11

bash scripts/bootstrap_a100.sh
source "$GBM_A100_SCRATCH/venv-a100/bin/activate"
export PYTHONPATH="$PWD/src:."
export HF_HOME="$GBM_A100_SCRATCH/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"

nvidia-smi
# Phase A: preflight, environment capture, GPU plan, and benchmark only.
python scripts/run_a100_week3.py \
  --config config/model_a100_local.yaml \
  --scratch "$GBM_A100_SCRATCH" \
  --results "$GBM_PERSISTENT_OUTPUT_DIR" \
  --session-id adit-week3-a100

# Inspect adit-week3-a100/scgpt_benchmark.json before starting the long matrix.

# Phase B: same session and settings; completed setup stages are skipped.
python scripts/run_a100_week3.py \
  --config config/model_a100_local.yaml \
  --scratch "$GBM_A100_SCRATCH" \
  --results "$GBM_PERSISTENT_OUTPUT_DIR" \
  --session-id adit-week3-a100 \
  --run-week3 \
  --week3-config config/week3_adit.yaml \
  --week3-output "$GBM_PERSISTENT_OUTPUT_DIR/adit-week3-a100/experiments"
```

The first orchestrator command runs preflight, environment capture, GPU
planning, and the 1,000-cell benchmark. The second command reuses those stages
and starts the experiment matrix. Do not run the standalone preflight and
benchmark first unless diagnosing a failure.

Rerun the final command unchanged after interruption. The manifest skips
finished stages; each experiment reuses its embedding checkpoint and skips
genes already present in `ranking_checkpoint.jsonl`.

## Experiments that must receive a fair A100 run

Run these in this order; do not skip the benchmark and do not call a blocked
artifact a result:

1. The 1,000-cell scGPT benchmark. Record throughput, GPU time, peak memory,
   checkpoint/vocabulary hashes, and the projected full-matrix cost.
2. The full unmasked scGPT internal-cohort embedding pass for seeds 17, 42,
   and 101. Save embeddings and held-out cell-state predictions.
3. MC dropout with 20 inference passes for each runnable MC-enabled arm. Save
   embedding mean, variance, per-pass timings, and the measured multiplier.
4. Full candidate-gene mask ranking for every runnable arm. Mask inference is
   restricted to training patients; test patients are used only for final
   prediction metrics. Resume the same command until every candidate is in the
   ranking checkpoint.
5. The one-variable-at-a-time matrix on identical folds/seeds:
   `all_on`, `validator_off`, `grn_off`, and `mc_dropout_off`. With the current
   zero-confirmed-gene evidence, only `validator_off` can finish; rerun the
   same matrix after valid protein evidence makes validator-on arms runnable.
6. CellFM and Geneformer only if the measured timing selector chooses the
   three-backbone branch. With the present 2,503-gene masking workload it is
   expected to select the prespecified scGPT-only Phase 1 branch.

These tasks do not need scarce A100 time and should be run afterward on CPU:

- Evaluate completed prediction files with `eval.py` and preserve bootstrap
  intervals.
- Run Stage 3/4, the shuffled-validator comparison, coverage audit, and GRN
  sanity check after new evidence arrives.
- Run `scripts/run_cross_cohort.py` for the patient-level IDH endpoint.
- Summarize the confirmed-versus-unconfirmed internal-to-external drop. If the
  confirmed set is empty or does not beat the shuffled control, report
  `no_result` instead of a headline claim.

## Evidence needed from teammates before all arms can finish

Ishaan/protein side must provide a real table with one auditable missense
variant per gene and independent scores:

```text
gene,mutation,alteration_type,plddt,esm1b,ddg,evidence_source
TP53,R175H,missense,92.1,-8.2,2.4,AlphaFold_version_and_pipeline
```

At least 15-20 genes must pass the prespecified confirmation rule. Otherwise
the pooling branch must provide additional cancer-driver candidate files and
the branch/count must remain recorded.

Jeffrey/GRN side must provide a larger frozen holdout (at least 20 unique
positive edges, no train overlap, and matched negatives). The current one-edge
holdout can test software but cannot support a paper AUROC claim.

The evaluation owner must provide independent benign/pathogenic variant labels
and independent abstain/non-abstain truth. Validator outputs cannot grade
themselves.

For external four-state Week 4 evaluation, the team must provide authoritative
CGGA AC/MES/NPC/OPC labels. If those labels do not exist, the protocol must be
amended before fitting anything: use patient-level IDH as the external endpoint
and describe it as a different endpoint, not as external cell-state validation.

## Final checks

```bash
make test
python scripts/audit_execution_readiness.py \
  --output reports/readiness/execution_readiness_post_jeffrey.json
```

Do not call Week 3 complete until the A100 experiment manifest contains the
expected completed runs and their embeddings, rankings, predictions, seeds,
timings, checkpoint hashes, and plain-English companion files.
