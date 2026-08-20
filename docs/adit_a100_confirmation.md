# Adit GPU confirmation checklist

This is the exact procedure for the A100 JupyterHub. It uses the verified local Neftel cohort:

`data/import_20260820/TP53 Dataset(preprocessed)/processed/neftel_analysis_cohort.h5ad`

The cohort has 6,576 cells, 27 patients, and the four states `AC`, `MES`, `NPC`, and `OPC`.

## 1. Start a persistent Jupyter terminal

```sh
cd /path/to/TP-53-Gblastoma-ML-Research
python -m venv .venv-a100
. .venv-a100/bin/activate
python -m pip install -r requirements-a100.txt
python -m pip install 'scGPT==0.2.4'
export GBM_A100_SCRATCH=/mnt/localssd/gbm-a100-scratch
export GBM_PERSISTENT_OUTPUT_DIR=/mnt/persistent/gbm-results
export HF_HOME=$GBM_A100_SCRATCH/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export MPLCONFIGDIR=$GBM_A100_SCRATCH/matplotlib
mkdir -p "$GBM_A100_SCRATCH" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$MPLCONFIGDIR" "$GBM_PERSISTENT_OUTPUT_DIR"
export PYTHONPATH="$PWD/src:."
```

## 2. Confirm the GPU and inputs

```sh
nvidia-smi
test -f 'data/import_20260820/TP53 Dataset(preprocessed)/processed/neftel_analysis_cohort.h5ad'
test -f splits/combined_full_cohort_neftel_patient_splits.json
test -f artifacts/models/scGPT_pancancer/best_model.pt
test -f artifacts/models/scGPT_pancancer/vocab.json
test -f artifacts/models/scGPT_pancancer/args.json
```

Run the fail-closed preflight. Use a future deadline for the reservation:

```sh
python scripts/a100_preflight.py \
  --config config/model_a100_local.yaml \
  --scratch "$GBM_A100_SCRATCH" \
  --output "$GBM_PERSISTENT_OUTPUT_DIR/preflight.json" \
  --deadline-utc 2099-01-01T00:00:00Z
```

Continue only if `preflight.json` says `"status": "passed"`. Read `preflight.txt` too.

## 3. Run the 1,000-cell timing benchmark

```sh
python scripts/benchmark_scgpt.py \
  --config config/model_a100_local.yaml \
  --output "$GBM_PERSISTENT_OUTPUT_DIR/week3_scgpt_benchmark.json"
```

Accept it only when exactly 1,000 real training-patient cells were used, GPU timing was synchronized, and `projected_gpu_seconds_per_10000_cells` is non-null.

## 4. Run all Adit Week 3 arms with checkpointing

This creates 12 scGPT runs: four one-variable-at-a-time conditions times three seeds (`17`, `42`, `101`). It saves embeddings, rankings, predictions, metrics, MC mean/variance, and provenance.

```sh
python scripts/run_a100_week3.py \
  --config config/model_a100_local.yaml \
  --scratch "$GBM_A100_SCRATCH" \
  --results "$GBM_PERSISTENT_OUTPUT_DIR" \
  --session-id adit-week3-a100 \
  --run-week3 \
  --week3-config config/week3_adit.yaml \
  --week3-output "$GBM_PERSISTENT_OUTPUT_DIR/adit-week3-a100/experiments" \
  --deadline-utc 2099-01-01T00:00:00Z
```

If it stops, rerun the exact same command and session ID. Do not change the seed, fold, checkpoint, vocabulary, candidate list, ablation, or MC pass count in that directory.

## 5. Confirm the matrix

```sh
python - <<'PY'
import json, pathlib
p = pathlib.Path("/mnt/persistent/gbm-results/adit-week3-a100/experiments/manifest.json")
r = json.loads(p.read_text())
print(r["status"], r["completed_runs"], r["blocked_runs"])
assert r["completed_runs"] == 12 and r["blocked_runs"] == 0
assert len(r["seeds"]) >= 3
PY
```

Inspect each run for `run.json`, `embeddings.npz`, `rankings.jsonl`, `predictions.jsonl`, `checkpoint_status.json`, and matching `.txt` explanations. CellFM and Geneformer remain out of scope unless the measured timing fits the configured budget and real checkpoint-specific adapters are supplied.

## 6. Confirm the scientific boundary

Run `python scripts/audit_adit_weeks.py --output "$GBM_PERSISTENT_OUTPUT_DIR/adit_week_audit.json"`. A real paper result still needs external single-cell AC/MES/NPC/OPC truth, independent variant-effect labels and scores, an abstention gold set, and a GRN holdout with many unique positives. Current CGGA is bulk IDH-labeled, so it cannot provide the requested external four-state macro-F1.
