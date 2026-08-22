# Week 2–4 execution and A100 handoff

This is the authoritative runbook for the current repository. Run commands from
the repository root. A command is complete only when its output status and the
adjacent plain-English `.txt` file agree.

## Current verified boundary

- Code contract: ready for CPU validation and A100 execution.
- Internal cohort: 6,576 Neftel cells, 27 patients, 17,268 genes, four states.
- Patient split: 15 train, 6 validation, 6 test, with no patient overlap.
- Stage 3/4: 2,503 genes; 2,454 abstain, 49 data-deficient, zero confirmed.
- Feasibility branch: pooling required because `0 < 15` confirmable genes.
- GRN sanity: AUROC 1.0, but only one held-out positive edge; this is a software
  check, not a publishable biological estimate.
- Baselines: PCA/logistic macro-F1 0.518509; Harmony/kNN macro-F1 0.433855.
  scVI is blocked because the supplied Neftel file contains log-scale values,
  including in `layers['counts']`, rather than raw non-negative integers.
- GPU matrix now: the three `validator_off` seeds can run. The nine
  validator-on runs correctly stop until independent protein evidence produces
  at least one confirmed gene.

## 1. Local environment and repository verification

```bash
cd /path/to/TP-53-Gblastoma-ML-Research
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
export PYTHONPATH="$PWD/src:."
make test
python scripts/audit_execution_readiness.py \
  --output reports/readiness/execution_readiness.json
```

Read `reports/readiness/execution_readiness.txt`. `code_contract_ready` must be
`true`; `full_scientific_matrix_ready` remains `false` until protein evidence is
added.

## 2. Jeffrey: datasets, mutation join, and GRN prior

Build the deterministic 20-patient, IDH-balanced, 2,500-gene pilot. This pilot
is for patient-level IDH work; its CGGA rows have no four-state cell labels.

```bash
python build_pilot_subsample.py
python scripts/build_tcga_pilot_mutation_join.py
python scripts/run_grn_sanity_check.py \
  --config config/week2_adit.yaml \
  --train-prior "data/import_20260820/TP53 Dataset(preprocessed)/prior/grn_pilot_train_prior.csv" \
  --held-out "data/import_20260820/TP53 Dataset(preprocessed)/prior/grn_pilot_adit_holdout_check.csv" \
  --output reports/jeffrey_grn_run/grn_sanity_current.json
```

Jeffrey still needs to deliver:

1. Original Neftel raw UMI/read counts, cells by genes, with stable cell and
   patient IDs matching the state-labeled cohort. Do not reconstruct counts by
   exponentiating logTPM. Store raw integers in `layers['counts']` and retain
   normalized values in `X`.
2. A larger frozen GRN holdout with at least dozens of unique positive TF-target
   edges, database/version/provenance columns, and no pair overlap with training.
3. External single-cell AC/MES/NPC/OPC labels if external four-state macro-F1 is
   required. Current CGGA is bulk and can support patient-level IDH only.

## 3. Ishaan: full Stage 3/4 and shuffled control

Without protein evidence, run the honest blocked-data path:

```bash
python scripts/build_internal_candidate_universe.py
python scripts/build_stage34_inputs_from_combined_data.py
python run_stage3_4.py \
  --output reports/stage34/combined_full_candidate_run.json
python scripts/build_week3_validator_outcomes.py
python scripts/build_cross_cohort_verdicts.py
python run_coverage_audit.py
```

To unblock it, create `data/pilot/protein_evidence.csv` with exactly one
preselected variant per candidate gene:

```text
gene,mutation,alteration_type,plddt,esm1b,ddg,evidence_source
TP53,R175H,missense,92.1,-8.2,2.4,AlphaFold_version_and_score_pipeline
```

Do not duplicate a gene. If several variants exist, freeze and document a
selection rule before validation. Then run:

```bash
python scripts/build_stage34_inputs_from_combined_data.py \
  --protein-evidence data/pilot/protein_evidence.csv
python run_stage3_4.py \
  --output reports/stage34/combined_full_candidate_run.json
python scripts/build_week3_validator_outcomes.py
python scripts/build_cross_cohort_verdicts.py
python run_coverage_audit.py
```

If `confirmable_count_primary < 15`, the report selects
`pool_additional_cancer_types`. Prepare candidate JSONL files from LGG,
melanoma, and LUAD with unique candidate IDs and genes absent from the primary
universe. Add their protein rows to the same evidence file, then pass the same
files to both commands:

```bash
python scripts/build_stage34_inputs_from_combined_data.py \
  --protein-evidence data/pilot/protein_evidence.csv \
  --pool-candidates data/pilot/lgg_candidates.jsonl \
  --pool-candidates data/pilot/melanoma_candidates.jsonl \
  --pool-candidates data/pilot/luad_candidates.jsonl
python run_stage3_4.py \
  --pool-candidates data/pilot/lgg_candidates.jsonl \
  --pool-candidates data/pilot/melanoma_candidates.jsonl \
  --pool-candidates data/pilot/luad_candidates.jsonl \
  --output reports/stage34/combined_full_candidate_run.json
```

The shuffled validator preserves the exact real bucket counts and permutes
assignments with seed 17. Bucket counts alone can never show superiority because
they are intentionally identical. Supply frozen independent outcome labels via
`--gold-outcomes PATH`; otherwise the comparison correctly reports `no_result`.

## 4. Alexis: conventional baseline bar

The correct state-labeled input is the Neftel analysis cohort, not the old
single-batch pilot. Run all arms with the same patient split:

```bash
python baselines.py \
  --method all \
  --adata "data/import_20260820/TP53 Dataset(preprocessed)/processed/neftel_analysis_cohort.h5ad" \
  --splits splits/combined_full_cohort_neftel_patient_splits.json \
  --fold 0 --seed 42 \
  --config config/baselines_pilot.yaml \
  --output reports/pilot_baselines_verified/seed42

for method in pca_logreg harmony_knn; do
  python eval.py \
    --cell-predictions "reports/pilot_baselines_verified/seed42/$method/predictions.jsonl" \
    --splits splits/combined_full_cohort_neftel_patient_splits.json \
    --config config/evaluation_pilot.yaml \
    --output "reports/pilot_baselines_verified/seed42/${method}_eval"
done

python scripts/summarize_combined_baselines.py \
  --root reports/pilot_baselines_verified \
  --output reports/pilot_baselines_verified/baseline_bar.json
```

The scVI arm writes `run_error.json` until true raw integer counts are supplied.
Installing scvi-tools does not fix invalid input data.

## 5. Adit: A100 scGPT, MC dropout, ablations, and checkpoints

Upload the current repository, the internal H5AD, split file, and the three
checkpoint files while preserving their repository-relative paths. In a
persistent Jupyter terminal:

```bash
cd /path/to/TP-53-Gblastoma-ML-Research
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
```

The bootstrap pins scGPT 0.2.4 to torch 2.3.x and torchtext 0.18.x in an
isolated environment. Run the fail-closed checks and timing benchmark:

```bash
python scripts/a100_preflight.py \
  --config config/model_a100_local.yaml \
  --scratch "$GBM_A100_SCRATCH" \
  --output "$GBM_PERSISTENT_OUTPUT_DIR/preflight.json"

python scripts/benchmark_scgpt.py \
  --config config/model_a100_local.yaml \
  --output "$GBM_PERSISTENT_OUTPUT_DIR/scgpt_benchmark.json"
```

Start/resume the matrix with the same session and output paths:

```bash
python scripts/run_a100_week3.py \
  --config config/model_a100_local.yaml \
  --scratch "$GBM_A100_SCRATCH" \
  --results "$GBM_PERSISTENT_OUTPUT_DIR" \
  --session-id adit-week3-a100 \
  --run-week3 \
  --week3-config config/week3_adit.yaml \
  --week3-output "$GBM_PERSISTENT_OUTPUT_DIR/adit-week3-a100/experiments"
```

Rerun exactly that command after interruption. Each run fingerprints the seed,
fold, checkpoint, vocabulary, genes, and ablation; it saves embeddings before
gene masking and appends each finished gene to `ranking_checkpoint.jsonl` with
an `fsync`. It refuses to resume if protected settings changed. MC dropout uses
20 inference passes and saves mean/variance and its measured compute multiplier.

Expected before protein evidence: 3 completed `validator_off` runs and 9
explicitly blocked validator-on runs. Expected after valid confirmed genes are
available: 12 scGPT runs. CellFM and Geneformer run only if the measured timing
fits the budget and real adapters/checkpoints are configured; otherwise the
manifest records the Phase-1 scGPT-only compute decision.

## 6. Week 4 internal-to-external analysis

After Stage 3/4 has non-empty confirmed and unconfirmed groups:

```bash
python scripts/run_cross_cohort.py \
  --config config/cross_cohort.yaml \
  --output reports/cross_cohort_current
```

Current CGGA evaluation is patient-level IDH. The command refuses to invent
confirmed-group metrics when there are zero confirmed genes. Four-state
external macro-F1, variant AUROC, and abstention accuracy require their own
independent external labels; GPU hardware cannot create those labels.

## 7. Final acceptance

```bash
make test
python scripts/audit_execution_readiness.py \
  --output reports/readiness/execution_readiness.json
git diff --check
```

Archive the run directory, JSON/JSONL artifacts, their `.txt` explanations,
environment export, hashes, and checkpoint files. Never report a blocked arm as
zero, a stale file as current, or a one-edge GRN AUROC as a paper result.
