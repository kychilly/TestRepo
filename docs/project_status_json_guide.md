# Project status, JSON guide, limitations, and paper plan

Generated/checked: 2026-08-15. This is the interpretation layer for the
repository artifacts. The authoritative evidence remains the JSON outputs and
their hashes; this document explains what each file means and what can be said
in a paper.

## Headline: where we are

The repository is software-ready but not scientifically complete. The current
audit scores software readiness 8/10, scientific readiness 3/10, and
publication readiness 4/10. The only real model results currently safe to call
measured are the Neftel pilot baseline results and the donor/batch audit.

The strongest current result is the PCA/logistic test baseline: macro-F1
0.6256 and balanced accuracy 0.6488, with patient-clustered bootstrap intervals
in its evaluator artifact. Harmony/kNN is lower (macro-F1 0.5555, balanced
accuracy 0.5524). These are a 26-patient Neftel pilot, not a TCGA/CGGA,
two-IDH-status, multi-cohort result.

The GRN AUROC of 1.0 is a sanity check with one held-out positive and 144
negatives. It is not evidence of general GRN recovery. The validator’s
classification gate matches the expected four fixture outcomes, but its
publication gate is blocked by incomplete source/version metadata and an
explicitly estimated IDH1 ΔΔG value.

## What Ishaan’s Week 2 code would add

Ishaan’s branch is not missing as software: the validator decision tree and
four-gene gate are present, tested, and wired into Stage 5. What is missing is
the real candidate list, real protein evidence, complete provenance, and
sign-off. Once supplied, it would add the paper’s feasibility layer:

1. Five-outcome counts across candidates (`destabilizing_driver`,
   `functional_driver`, `abstain`, `unconfirmed`, `data_deficient`).
2. Confirmable-gene count and the fraction of candidates that can actually
   enter Stage 5.
3. Evidence pooling/feasibility branch, fixed-seed shuffled control, and
   data-deficient coverage.
4. A validator-on versus validator-off ablation, showing whether the gate
   changes rankings or only masks unsupported claims.
5. Complete AlphaFold, ESM1b, and ΔΔG source/version/access-date provenance.

The likely benefit is stronger claim discipline and a quantifiable estimate of
how much of the candidate list is biologically confirmable. It will not by
itself improve predictive accuracy, prove causality, or replace TCGA/CGGA
validation. The first real run should be treated as a coverage/feasibility
analysis, not as a positive-effect test.

## JSON files: what each one means

### Configuration and split JSON

| File | Meaning | Current implication |
|---|---|---|
| `config/week1.example.json` | Non-runnable template for modality, assets, output, seed, and week. | Contains nulls; never use for a scientific run. |
| `config/model.yaml` | Main scGPT/model settings; included here because it controls the JSON outputs. | Requests CUDA but has no checkpoint or vocabulary. |
| `config/pilot.yaml` | Pilot scGPT run settings and provenance placeholders. | `n_cells: 1` and null hashes make it a configuration stub. |
| `config/week2_adit.yaml` | Adit’s GRN, MC-dropout, and validator settings. | GRN prior exists; checkpoint/GPU/validator production evidence do not. |
| `config/evaluation*.yaml` | Metric units, bootstrap seed/replicates, and metrics. | The protocol is specified; real multi-cohort clinical evaluation is not. |
| `config/baselines*.yaml` | Baseline model inputs and hyperparameters. | Pilot config correctly uses `derived_state`; generic config expects fields not present in Neftel. |
| `splits/patient_splits.json` | Canonical train/validation/test patient IDs. | Large generic split exists, but the supplied Neftel pilot uses its own 16/5/5 split. |
| `TP53 Dataset(preprocessed)/pilot/patient_splits.json` | Actual 26-patient pilot split. | This is the split used by current pilot baseline artifacts. |

### Schema JSON

These are contracts, not results. They define required fields, types,
allowed values, and sometimes conditional requirements.

| File | Headline |
|---|---|
| `schemas/variant_record.schema.json` | Patient alteration identity, genomic/protein mapping, source/version, and mapping status. Resolved variants must carry transcript and protein fields. |
| `schemas/protein_evidence.schema.json` | Evidence records for AlphaFold/pLDDT, ESM1b, ΔΔG, provenance, and evidence status. |
| `schemas/grn_edge.schema.json` | Directed TF-to-target edge, sign, confidence, database, and provenance. |
| `schemas/candidate_gene.schema.json` | Ranked candidate/suspect gene output. Rank and score do not mean causality. |
| `schemas/prediction.schema.json` | Prediction record union for allowed prediction shapes. |
| `schemas/evaluation_result.schema.json` | Metric output, units, estimates, uncertainty, and provenance. |
| `schemas/validator_input.schema.json` | Candidate plus variant/evidence joins, preserving zero/one/multiple join cardinality. |
| `schemas/validator_payload.schema.json` | Versioned payload passed into the validator. |
| `schemas/validator_gate_record.schema.json` | Validator outcome, eligibility, evidence provenance, and gate metadata. |

### Example JSON/JSONL

| File | Meaning |
|---|---|
| `examples/aliases.example.json` | Gene-alias normalization example (`ERBB1` → `EGFR`). |
| `examples/gene_metadata.synthetic.json` | Synthetic HGNC-to-Ensembl metadata fixture; not study data. |
| `examples/*.jsonl` | Synthetic contract fixtures for candidate genes, variants, protein evidence, GRN edges, predictions, and validator inputs. They test serialization and schema behavior only. |

### Results and reports

| File/group | Meaning and current reading |
|---|---|
| `results/week1_audit.json` | Structured Week 1 gate: contracts and selected checks pass; real scGPT, CUDA, cohorts, and canonical labels are blocked. |
| `results/week1_manifest.json` | Week 1 audit/report/reproduction pointers and status. |
| `results/week1_data_audit/donor_batch_audit.json` | Neftel donor/cell-assignment audit: 7,930 cells, 28 donors, 2,000 HVGs; cell-assignment separation exceeds donor separation, but CellAssignment is not the agreed four-state label. |
| `results/synthetic_smoke_report.json` | Synthetic end-to-end smoke results. Never cite as biological performance. |
| `results/validator_gate.json` | Copy of the validator gate result, including passing fixture classifications and blocked publication status. |
| `results/compute/current_environment.json` and `week1_environment_check.json` | Runtime inventory. macOS arm64, Torch 2.9.1, no CUDA, no scGPT/AnnData/scVI/Harmony, no checkpoint/vocabulary. |
| `results/compute/*benchmark*.json` | scGPT benchmark contracts. All current timing/mapping fields are null because real assets are absent. |
| `results/compute/gpu_plan.json` | GPU planning result. Blocked; no CPU timing may be promoted to GPU cost. |
| `results/compute/week2_grn_sanity.json` | Older blocked compute record saying no real edge list was configured. The newer Jeffrey report contains the pilot GRN sanity result. |
| `results/compute/week2_pilot_scgpt*.json` | Blocked pilot scGPT outputs with suspect-ranking interpretation explicitly preserved. |
| `reports/pilot_baselines/*/metrics.json` | Evaluated metric and bootstrap JSON. `pca_test_eval` and `harmony_test_eval` are the current measured test artifacts. |
| `reports/pilot_baselines/*/evaluation_manifest.json` | Hashes, split identity, evaluator commit, package versions, and bootstrap settings. |
| `reports/pilot_baselines/*/warnings.json` | Evaluator warnings. Read alongside metrics before citing a result. |
| `reports/pilot_baselines/*/run_metadata.json` | Baseline run identity, seed, model/config/split hashes, and status. |
| `reports/pilot_baselines/*/run_error.json` | Structured failures. PCA generic and scVI are not applicable to the current inputs; Harmony has a loader/runtime failure in one path. |
| `reports/jeffrey_grn_run/grn_sanity.json` | GRN held-out sanity AUROC and edge counts with input hashes. |
| `reports/jeffrey_grn_run/validator_gate.json` | Full four-gene gate decisions, thresholds, provenance warnings, and publication gate. |
| `reports/week2_adit/report.json` | Adit integration status: GRN completed, Stage 5 implementation completed, MC-dropout blocked by missing checkpoint. |

There are also JSONL prediction/payload outputs under `results/contracts/`;
they are row-oriented contract artifacts rather than aggregate performance
reports. They should be retained for audit, but not summarized as a model
result without a locked evaluation manifest.

## Complete limitation list

### Data and labels

- TCGA-GBM is not supplied locally; CGGA 325/693 expression files are present
  in a preprocessed folder, but their integrated study loader/evaluation is not
  complete.
- The current pilot is Neftel-only, IDH-wildtype, 26 patients, and therefore
  cannot support the requested two-IDH-status, multi-cohort claim.
- Neftel has `CellAssignment` (`Macrophage`, `Malignant`, `Oligodendrocyte`,
  `T-cell`), not canonical AC/MES/NPC/OPC. The current pilot’s derived labels
  include 1,351 `Unknown` cells excluded from evaluation.
- The supplied mutation table is heuristic four-gene metadata, not
  authoritative TCGA/CGGA mutation calls with build/transcript/source fields.
- CGGA clinical IDH counts exist in the audit, but per-variant missense,
  amplification, and silencing calls are absent.
- The H5AD is log-normalized and lacks a raw integer `counts` layer, so scVI
  cannot be scientifically run from it.

### Modeling and evaluation

- scGPT checkpoint and vocabulary are absent; real forward-pass and MC-dropout
  timing are unmeasured.
- CUDA is unavailable on the current macOS arm64 runtime; all GPU fields are
  correctly null.
- No real Stage 3/4 candidate list exists, so validator bucket counts, pooling,
  shuffled control, and coverage are not computable.
- The GRN sanity set has one held-out positive; AUROC is therefore unstable and
  cannot establish generalization.
- No independent Alexis replication is present.
- The baseline pilot is small and cell-level results are clustered within only
  26 patients; cell counts must not be mistaken for independent biological
  replicates.
- Harmony/scVI integration paths are blocked or failed on current inputs.
- Clinical IDH/TP53 patient-level metrics are not available and must not be
  inferred from cell majority votes.

### Provenance and publication

- AlphaFold/ESM1b/ΔΔG sources and versions are incomplete; IDH1 ΔΔG is
  explicitly estimated.
- Validator thresholds are frozen in the config, but the validator lead’s
  sign-off is still required.
- Synthetic fixtures prove software behavior only.
- A model-ranked gene remains a candidate/suspect, never a confirmed causal
  driver, unless the signed evidence gate supports that wording.

## Figures added

Run:

```sh
python scripts/generate_paper_figures.py --output reports/paper_figures
```

This generates seven evidence-labeled PNGs and a manifest:

- readiness score;
- data/asset presence;
- baseline performance with bootstrap intervals;
- fixed-order confusion matrices;
- batch-risk and split audit;
- GRN/validator status;
- compute blockers.

The larger manuscript figure set still required after real assets arrive is:
cohort flow/QC, donor/batch/state embeddings, split leakage table, full
baseline comparison, calibration, patient-level IDH/TP53 ROC and calibration,
scGPT cost/uncertainty, candidate rank stability, GRN stratification, validator
on/off ablation, and a provenance appendix. The current figure generator is
designed to be extended with those real artifacts rather than fabricate them.

## Compute recommendation

Do not buy hardware based on the current null timing fields. First obtain the
checkpoint, vocabulary, raw-count decision, and a fixed 1,000-cell benchmark.
For scGPT plus 20–50 MC-dropout passes, use a Linux CUDA host with an NVIDIA
GPU having at least 24 GB VRAM, 64–128 GB system RAM, fast local NVMe, and
CUDA/PyTorch versions pinned in the run manifest. A 40–48 GB GPU is preferable
if token length or batch size grows. Keep the Mac for orchestration and review,
not the scientific GPU benchmark.

Operationally, measure peak allocated/reserved VRAM, wall time, synchronized GPU
time, cells/sec, and pass multiplier at batch sizes 8/16/32/64. Use mixed
precision only after numerical equivalence checks, persist checkpoint and
vocabulary hashes, and rerun the final locked analysis on the same image or
environment. If purchasing is not necessary, a short-lived cloud GPU is
adequate for benchmarking and inference; raw counts and complete provenance
are higher-priority dependencies than a larger GPU.
