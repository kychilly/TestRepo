# Stage 3/4, validator-control, GRN, and Adit completion audit

Checked on 2026-08-16 against the local pilot assets. “Code complete” and
“scientifically complete” are reported separately.

## Outcome

| Requirement | Code status | Scientific status | Evidence |
|---|---|---|---|
| Stage 3 candidate producer | Interface and validated candidate serialization complete | Blocked | Real scGPT run lacks checkpoint, vocabulary, gene metadata, CUDA run, and configured mask-score provider |
| Stage 4 five-bucket validator | Complete, candidate-aligned, and tested | Blocked for full pilot | No real full-pilot Stage 3 candidate JSONL or complete protein evidence exists |
| Feasibility/pooling branch | Complete and tested | Triggered on four-gene fixture; real decision blocked | Fixture has 2 confirmable records, below minimum 15; no additional-cancer candidate files are present |
| Fixed-seed shuffled control | Complete and tested | Control assignments generated; superiority comparison blocked | Bucket counts are preserved exactly with seed 17, but there is no independent truth label or locked downstream metric |
| GRN branch | Complete | Sanity check completed | AUROC 1.0 from 1 held-out positive and 144 unknown negatives |
| Adit Stage 5 masking | Complete and tested | Available | Only `destabilizing_driver` and `functional_driver` pass when validator is on |
| Adit MC-dropout | Complete as an integration seam | Blocked | scGPT checkpoint/vocabulary and a CUDA measurement are absent |

The full scientific task is **not complete**. The repository can prove the
software behavior and the GRN sanity check, but it cannot truthfully produce a
full-pilot five-bucket table or claim that the real validator beats the
shuffled control.

## Available fixture result (not a full-pilot result)

The candidate-aligned fixture uses the three Stage 3 fixture candidates (TP53,
EGFR, and RPRM). IDH1 evidence is correctly excluded because IDH1 is not a
Stage 3 candidate. It produces:

| Outcome | Count |
|---|---:|
| destabilizing_driver | 1 |
| functional_driver | 0 |
| abstain | 2 |
| unconfirmed | 0 |
| data_deficient | 0 |

Confirmable is defined as missense plus pLDDT at or above the configured floor
(70). The candidate-aligned fixture count is 1. Since 1 is below the frozen minimum of 15 (with
20 preferred), the branch is `pool_additional_cancer_types`. Pooling is blocked
because no schema-valid candidate/evidence files for additional cancer types
are present.

The shuffled control preserves the same five counts exactly and randomly
permutes assignments with seed 17. On this three-candidate fixture, that frozen
permutation happens to reproduce the original assignments. Both real and
shuffled outcome accuracy are 1.0 against the synthetic frozen labels, so the
required call is `no_result`. This is expected null-control behavior, not a
scientific result. A real comparison requires independently curated outcomes
or a locked downstream prediction metric.

## Local reproduction

Run the currently available audit:

```bash
make audit-current
```

Or run each component explicitly:

```bash
PYTHONPATH=src:. python scripts/run_stage34_validation.py \
  --records examples/validator_gate_input.jsonl \
  --candidates results/contracts/tp53/scgpt_candidate_output.jsonl \
  --gold-outcomes examples/validator_gold.synthetic.jsonl \
  --config config/stage34.yaml \
  --seed 17 \
  --output reports/stage34/fixture_feasibility.json

PYTHONPATH=src:. python scripts/run_grn_sanity_check.py \
  --config config/week2_adit.yaml \
  --train-prior "data/TP53 Dataset(preprocessed) 2/prior/grn_pilot_train_prior.csv" \
  --held-out "data/TP53 Dataset(preprocessed) 2/prior/grn_pilot_adit_holdout_check.csv" \
  --output reports/jeffrey_grn_run/grn_sanity_current.json

PYTHONPATH=src:. python scripts/run_adit_week2.py \
  --config config/week2_adit.yaml \
  --output reports/week2_adit/report_current.json

PYTHONPATH=src:. python -m pytest -q
```

For the real Stage 3/4 run, replace both fixture files with the real full-pilot
candidate list and matching validator records:

```bash
PYTHONPATH=src:. python scripts/run_stage34_validation.py \
  --records results/stage34/pilot_validator_records.jsonl \
  --candidates results/stage3/pilot_candidates.jsonl \
  --pool-candidates results/stage3/lower_grade_glioma_candidates.jsonl \
  --pool-candidates results/stage3/melanoma_candidates.jsonl \
  --pool-candidates results/stage3/lung_adenocarcinoma_candidates.jsonl \
  --config config/stage34.yaml \
  --seed 17 \
  --output reports/stage34/pilot_full.json
```

Only pass `--pool-candidates` files from prespecified cohorts. If the
confirmable count is at least 15, omit them and the runner records the
`gbm_only` branch.

## Jupyter/CUDA runbook for Stage 3 and MC-dropout

1. Start a Linux Jupyter environment with an NVIDIA CUDA GPU (24 GB VRAM
   minimum recommended), clone the repository, and install
   `requirements-a100.txt` without replacing the host CUDA-enabled Torch.
2. Place the official scGPT assets outside Git, for example:
   `/scratch/tp53/scgpt/best_model.pt`, `/scratch/tp53/scgpt/vocab.json`, and
   checkpoint-adjacent `args.json`.
3. Copy `config/model_shared_gpu.yaml` and fill `checkpoint_path`,
   `vocabulary_path`, the pilot H5AD path, `splits/neftel_pilot_patient_splits.json`,
   and scratch/output paths. Do not commit model weights or the 2.6 GB H5AD.
4. Run the preflight and benchmark:

   ```bash
   export GBM_A100_SCRATCH=/scratch/tp53/run1
   export A100_CONFIG=/scratch/tp53/model.yaml
   make a100-preflight
   PYTHONPATH=src:. python scripts/benchmark_scgpt.py \
     --config "$A100_CONFIG" \
     --output "$GBM_A100_SCRATCH/scgpt_benchmark.json"
   ```

5. Configure `config/pilot.yaml` with the same checkpoint/vocabulary, the
   pilot split, a complete HGNC-to-Ensembl gene metadata JSON, and the
   checkpoint-specific `mask_score_provider`. Run `scripts/run_pilot_scgpt.py`.
   A missing provider is a code/model integration blocker, not something the
   GPU can infer automatically.
6. Generate the rich validator join with `scripts/validate_contracts.py`, then
   transform the joined evidence to `GeneRecord` JSONL and run the real command
   above. Record checkpoint, vocabulary, split, config, and input hashes.
7. Configure `config/week2_adit.yaml` with the checkpoint, vocabulary, CUDA
   device, and checkpoint-specific MC-dropout runner; rerun Adit’s report.

## Jeffrey data handoff required

Jeffrey should deliver these immutable files with checksums and source/version
metadata:

1. A full-pilot alteration file conforming to
   `schemas/variant_record.schema.json`, including actual amino-acid changes,
   transcript/isoform, protein accession, mapping status, genome build,
   patient ID, cohort, and alteration type. The current four-column mutation
   table lacks the variant identity and mapping fields required by Stage 4.
2. A larger GRN held-out set, separated before model use, with at least dozens
   of known positives. The current one-positive AUROC is only a smoke/sanity
   check.
3. Prespecified additional-cancer candidate/variant files for the pooling
   cohorts, using the same schemas and no overlap with evaluation patients.
4. Cohort manifests and split files proving patient disjointness and immutable
   hashes.

The Validator Lead must separately deliver complete AlphaFold per-residue
pLDDT, ESM1b, and measured ΔΔG evidence conforming to
`schemas/protein_evidence.schema.json`. The current note covers only four genes,
omits exact software/database versions, and marks IDH1 ΔΔG as an estimate.
