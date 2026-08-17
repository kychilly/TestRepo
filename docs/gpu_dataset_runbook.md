# Real pilot completion runbook

The code is fail-closed, but the scientific run is not complete until real assets exist. Synthetic fixture reports are tests, not pilot results.

## Dataset audit

```bash
PYTHONPATH=src:. python scripts/readiness_audit.py \
  --pilot "data/TP53 Dataset(preprocessed) 2/pilot/pilot_subsample.h5ad" \
  --split splits/neftel_pilot_patient_splits.json \
  --mutations "data/TP53 Dataset(preprocessed) 2/pilot/patient_gene_mutation_long.csv" \
  --grn-train "data/TP53 Dataset(preprocessed) 2/prior/grn_pilot_train_prior.csv" \
  --grn-holdout "data/TP53 Dataset(preprocessed) 2/prior/grn_pilot_adit_holdout_check.csv" \
  --output reports/readiness/current.json
```

The current audit is expected to be `blocked`: the pilot is log-normalized, has no `layers['counts']`, no `derived_state` IDH labels, and the mutation CSV lacks transcript/protein/genome-build/source provenance. These are data limitations, not GPU failures.

Jeffrey/Ishaan must provide a versioned pilot package containing: cell-level patient IDs and both IDH statuses; a 2–3k HVG panel explicitly containing TP53, IDH1, EGFR, and RPRM; raw integer counts in `layers['counts']` if scVI is required; mutation calls with `patient_id`, `gene_symbol`, `variant_status`, `impact`, `transcript_id`, `protein_change`, `genome_build`, and `source_file`; AlphaFold accession/version, pLDDT, and residue mapping; and non-overlapping GRN train/holdout files. Do not convert logTPM back to counts, label Adult/Pediatric as IDH, or use patient ID as a batch.

## GPU/Jupyter setup

Keep data, checkpoints, and caches outside Git:

```bash
export GBM_A100_SCRATCH=/scratch/$USER/tp53-gbm
mkdir -p "$GBM_A100_SCRATCH"/{hf,models,outputs}
export HF_HOME="$GBM_A100_SCRATCH/hf"
export HF_DATASETS_CACHE="$GBM_A100_SCRATCH/hf/datasets"
export TRANSFORMERS_CACHE="$GBM_A100_SCRATCH/hf/transformers"
python -m venv "$GBM_A100_SCRATCH/venv"
source "$GBM_A100_SCRATCH/venv/bin/activate"
pip install -r requirements-a100.txt
```

Copy the checkpoint, matching vocabulary, and checkpoint `args.json` into `$GBM_A100_SCRATCH/models/`. Configure `config/model_shared_gpu.yaml` with immutable dataset ID/revision/split, checkpoint, vocabulary, and a real `mask_score_provider: module:function`.

```bash
make A100_CONFIG=config/model_shared_gpu.yaml GBM_A100_SCRATCH="$GBM_A100_SCRATCH" a100-preflight
PYTHONPATH=src:. python scripts/check_environment.py --config config/model_shared_gpu.yaml --json-out "$GBM_A100_SCRATCH/environment.json"
PYTHONPATH=src:. python scripts/run_pilot_scgpt.py --config config/pilot.yaml --output "$GBM_A100_SCRATCH/outputs/pilot_scgpt.json"
```

The provider must return real `patient_id`, `cell_id`, `gene_id`, `state`, `baseline_logit`, and `masked_logit` rows; fit only on training patients; and record checkpoint/vocabulary/config hashes. Missing CUDA, assets, provider, or rows is a hard stop.

## Validation and baselines

After real candidate records exist, run Stage 3/4 with `--records` and `--candidates`. The output must include all five bucket counts, data-deficient coverage, confirmable count, feasibility branch, and real-versus-shuffled comparison. A real validator that does not beat its fixed-seed shuffled control is `no_result`.

```bash
PYTHONPATH=src:. python scripts/run_stage34_validation.py \
  --records "$GBM_A100_SCRATCH/outputs/pilot_scgpt_records.jsonl" \
  --candidates "$GBM_A100_SCRATCH/outputs/pilot_candidates.jsonl" \
  --config config/stage34.yaml --seed 17 \
  --output "$GBM_A100_SCRATCH/outputs/stage34_real.json"
make grn-sanity-current
```

The current valid baseline bar is PCA/LogReg; scVI and Harmony may correctly be `not_applicable` until their data contracts are met:

```bash
python baselines.py --method all --adata "data/TP53 Dataset(preprocessed) 2/pilot/pilot_subsample.h5ad" --splits splits/neftel_pilot_patient_splits.json --fold 0 --seed 42 --config config/baselines_pilot.yaml --output baseline_results
python eval.py --cell-predictions baseline_results/pca_logreg/predictions.jsonl --splits splits/neftel_pilot_patient_splits.json --config config/evaluation_pilot.yaml --output reports/pilot_baselines/pca_pilot_eval_seed42
```

MC dropout remains blocked until a real checkpoint/model adapter is available. Then run 20–50 inference passes, report per-gene mean/variance and elapsed-time multiplier, and keep the same split.
