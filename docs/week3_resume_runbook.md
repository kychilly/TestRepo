# Week 3 resume runbook

This runbook is the restart point for the blocked external work. It records
what must be supplied, what commands to run, and what evidence is required
before a result is allowed into the paper or group-chat compute estimate.

## Preconditions

The shared GPU storage policy and Jeffrey delivery requirements are defined in
`docs/shared_gpu_and_jeffrey_inputs.md`. The host must use Hugging Face
streaming or an approved compact split; do not download full cohort archives or
model zoos to the shared machine.

The run host must have:

1. A CUDA GPU and a CUDA-compatible PyTorch wheel for Python 3.11.
2. The pinned packages in `requirements.txt`, plus the exact `pip freeze`
   export saved under `results/compute/`.
3. The real AnnData file, patient-level split JSON, scGPT checkpoint, and
   checkpoint-matched vocabulary.
4. A verified scGPT loader for that exact checkpoint. The benchmark must not
   infer an architecture from tensor names or substitute another checkpoint.
5. Ishaan/Validator Lead's written approval of the candidate-gene contract
   below before candidate records are produced.

Before requesting hardware, a provider-neutral plan can be generated locally:

```sh
PYTHONPATH=src python scripts/plan_gpu.py \
  --token-length TOKEN_LENGTH --cells 10000 --batch-size 32 \
  --output results/compute/week3_gpu_plan.json
```

The planner inspects actual visible CUDA devices, current free memory, and
bf16 support, then assigns weighted shards. It fails with `status: blocked` when
CUDA is unavailable; it does not simulate a GPU or use CPU throughput.

## Ordered execution

Run from the repository root on the provisioned host:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_environment.py \
  --config config/model.yaml \
  --json-out results/compute/environment.json \
  --environment-export results/compute/environment_export.json
```

Populate `config/model.yaml` only with the Data Lead's paths and hashes. The
environment check must report `status: passed`, `cuda_available: true`, a
non-null scGPT version, checkpoint/vocabulary SHA-256 values, and no shape or
special-token failures.

Then run the exact 1,000-cell training-only smoke benchmark:

```sh
python scripts/benchmark_scgpt.py \
  --config config/model.yaml \
  --output results/compute/week3_scgpt_benchmark.json
```

The accepted benchmark evidence is `status: completed`, exactly 1,000 cells,
training patients only, a successful finite forward pass, checkpoint and
vocabulary hashes, and non-null synchronized GPU timing. The compute number to
post is:

```text
projected_gpu_seconds_per_10000_cells
  = 10 * gpu_seconds_per_1000_cells
```

No projection may be posted when the benchmark is `blocked` or when the
forward pass used fewer than 1,000 real cells.

## Baseline and evaluation flow

Using the same split file for every arm, run `baselines.py` once per method:

```sh
for method in pca_logreg scvi_probe harmony_knn; do
  PYTHONPATH=src python baselines.py \
    --method "$method" --adata DATA.npz --splits SPLITS.json \
    --fold 0 --seed 17 --config config/baselines.yaml \
    --output "results/baselines/$method/fold0_seed17"
done
```

`pca_logreg` may complete. scVI and Harmony must either complete with a
validated unseen-cell path or emit their structured non-applicability record;
silently fitting on all cells is a leakage failure.

Evaluate the approved prediction file through `eval.py`:

```sh
PYTHONPATH=src python eval.py \
  --predictions results/baselines/pca_logreg/fold0_seed17/predictions.jsonl \
  --splits SPLITS.json --config config/evaluation.yaml \
  --output results/evaluation/pca_logreg/fold0_seed17
```

The JSON under `results/evaluation/` is the only source for manuscript
macro-F1, patient-level IDH AUROC, and bootstrap confidence intervals. A
separate patient-level IDH prediction file is required; IDH labels must never be
inferred from cell-state rows. ClinVar/DMS variant-effect AUROC remains a
separate optional analysis.

## Candidate-gene agreement gate

The proposed transport view is:

```json
{ "gene": "TP53", "state": "MES", "score": 2.31, "rank": 1, "seed": 17 }
```

Before implementation, Ishaan must reply with an explicit approval or edits to
these decisions:

- `gene` is canonical HGNC symbol after an explicit alias map;
- `state` is exactly one of `AC`, `MES`, `NPC`, `OPC`;
- `score` is the configured scGPT score method, not an unnamed importance;
- `rank` is one-based and scoped to `(run_id, state)`;
- `seed` is the ranking seed; and
- production records also carry the provenance fields in
  `schemas/candidate_gene.schema.json`.

Until this response is recorded, the producer and validator must not emit or
consume new production candidate rows.

## Group-chat message template

After the completed benchmark is independently checked, post only the measured
value and provenance:

```text
Week 3 scGPT compute budget: <projected_gpu_seconds_per_10000_cells>
GPU-seconds per 10k cells, measured from 1,000 real training-patient cells.
GPU=<name>; checkpoint_sha256=<hash>; vocabulary_sha256=<hash>;
seed=<seed>; batch_size=<batch>; token_length=<length>;
benchmark=<results/compute/week3_scgpt_benchmark.json>.
```

The current macOS host has no CUDA device, so this message must remain
unposted until the provisioned run produces the required artifact.
