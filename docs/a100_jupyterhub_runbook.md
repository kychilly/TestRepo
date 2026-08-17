# A100 JupyterHub launch runbook

The A100 host is temporary compute. Dataset archives, complete H5AD files, and
model collections must not be cloned into the repository or downloaded to the
shared home directory. This workflow streams one immutable Hugging Face split,
uses one selected checkpoint package, and stages evidence after every command.

The current allocation shuts down at `2026-08-18T06:41:00Z`. Everything on the
host is deleted at shutdown.

## 1. Transfer code without transferring data

No commit or push is required. From the repository on the local Mac, build an
archive containing tracked and untracked, non-ignored files:

```bash
git ls-files --cached --others --exclude-standard -z | \
  tar --null -czf /tmp/TP53-GBM-A100-code-bundle.tar.gz --files-from - \
    data/README.md results/compute/week1_scgpt_benchmark.json
```

Because dataset trees, H5AD files, archives, and model weights are ignored,
they are absent from this archive. The two explicit files are small text/JSON
test fixtures, not expression or clinical data. Upload only
`/tmp/TP53-GBM-A100-code-bundle.tar.gz` with the JupyterLab file browser. Do
not upload either TP53 dataset directory or its ZIP file.

If the code is committed and published later by the repository owner, cloning
that exact commit URL is an alternative. A clone of the current remote branch
will not contain the uncommitted A100 integration in this working tree.

## 2. Extract code and establish scratch storage

Open **File → New → Terminal** in JupyterLab and run:

```bash
mkdir -p TP-53-Gblastoma-ML-Research
tar -xzf TP53-GBM-A100-code-bundle.tar.gz \
  -C TP-53-Gblastoma-ML-Research
cd TP-53-Gblastoma-ML-Research
git init

export GBM_A100_SCRATCH=/tmp/gbm-a100-$USER
export PYTHON_BIN=python3.11
# If Google Drive or another persistent mount exists, point this outside Git:
# export GBM_PERSISTENT_OUTPUT_DIR=/path/to/persistent/week3-results

bash scripts/bootstrap_a100.sh
source .venv-a100/bin/activate
```

`git init` creates only local repository metadata so the dataset-protection
hook can be enabled. It does not create a commit and does not contact or push
to any remote.

The bootstrap reuses the host's CUDA PyTorch instead of downloading another
large CUDA wheel. It disables pip caching and enables the repository's tracked
dataset/model pre-commit blocker.

## 3. Provision only the required model package

Place exactly one selected checkpoint, its `vocab.json`, and matching
`args.json` under `$GBM_A100_SCRATCH/model/`. Do not download a model zoo. Record
the source and SHA-256 values outside the repository.

Do not upload the local 2.6 GB Neftel H5AD or the full dataset ZIP. Jeffrey must
provide the immutable Hugging Face dataset ID, revision, split, and row-column
contract. The benchmark uses `load_dataset(..., streaming=True)`.

## 4. Render an untracked runtime configuration

```bash
python scripts/render_a100_config.py \
  --output "$GBM_A100_SCRATCH/model_a100_runtime.yaml" \
  --hf-dataset-id OWNER/DATASET \
  --hf-revision IMMUTABLE_COMMIT_SHA \
  --hf-split SPLIT \
  --hf-expression-column EXPRESSION_COLUMN \
  --hf-gene-ids-column GENE_IDS_COLUMN \
  --patient-id-column PATIENT_ID_COLUMN \
  --checkpoint "$GBM_A100_SCRATCH/model/best_model.pt" \
  --vocabulary "$GBM_A100_SCRATCH/model/vocab.json" \
  --model-args "$GBM_A100_SCRATCH/model/args.json" \
  --split-file splits/neftel_pilot_patient_splits.json \
  --precision bfloat16 --batch-size 32
```

The command refuses to write the runtime configuration inside the repository.

## 5. Run preflight and measured benchmark

```bash
python scripts/run_a100_week3.py \
  --config "$GBM_A100_SCRATCH/model_a100_runtime.yaml" \
  --scratch "$GBM_A100_SCRATCH"
```

The runner stops on the first failed stage and writes:

- `preflight.json`;
- environment and package exports;
- GPU plan;
- synchronized 1,000-cell scGPT benchmark;
- stdout/stderr for every stage;
- `run_manifest.json`; and
- a `.tar.gz` evidence archive for immediate download or persistent copying.

Do not report Week 3 timing unless the manifest and benchmark both say
`completed` and the selection reports exactly 1,000 training-patient cells.

## 6. Continuous preservation

After every completed run, download the generated evidence archive from the
Jupyter file browser or copy it to `GBM_PERSISTENT_OUTPUT_DIR`. No push or
commit is performed by this workflow. Never stage datasets, archives, model
weights, Hugging Face caches, virtual environments, or credentials.

The Git protections can be verified with:

```bash
python scripts/install_repo_safety.py
git check-ignore --no-index "data/example.h5ad"
git check-ignore --no-index "TP53 Dataset(preprocessed) 2/pilot/example.h5ad"
```

## Scientific limits that the A100 does not remove

The GPU does not supply missing labels, raw counts, a corrected cohort split,
the four-state model head, `mask_score_provider`, `mc_dropout_runner`, Ishaan's
sign-off, or Alexis's independent replication. Those inputs remain required by
their corresponding Week 2/3 branches. The existing CGGA hard-coded baseline
behavior is intentionally unchanged.

The accepted Neftel pilot path is PCA/logistic regression with
`splits/neftel_pilot_patient_splits.json`. Harmony is not applicable because
`CrossSection` has only one value, and scVI is not applicable because the H5AD
has no raw integer `counts` layer. CGGA must remain a separate bulk-patient IDH
evaluation and is not an AC/MES/NPC/OPC test cohort. Run
`scripts/audit_week2_datasets.py` locally before transferring any replacement
dataset.
