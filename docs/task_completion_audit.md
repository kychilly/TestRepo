# Task completion audit

This is an evidence report, not a completion claim. It records the current
state after the Week 1 data integration and Week 2 implementation audit on 2026-08-10.

## Evidence commands

```text
PYTHONPATH=src python -m pytest -q
52 passed, 9 warnings

PYTHONPATH=src python scripts/generate_candidates.py \
  --scores examples/scgpt_mask_delta_scores.synthetic.jsonl \
  --gene-metadata examples/gene_metadata.synthetic.json \
  --output results/contracts/tp53/scgpt_candidate_output.jsonl \
  --state MES --run-id run17 --checkpoint-hash sha256:synthetic-checkpoint \
  --vocabulary-hash sha256:synthetic-vocabulary --cohort Neftel --fold 0 --seed 17 \
  --split-hash sha256:synthetic-split --config-hash sha256:synthetic-config
status=completed; candidates=3

PYTHONPATH=src python scripts/validate_contracts.py \
  --candidates examples/candidate_gene.example.jsonl \
  --variants examples/variant_record.example.jsonl \
  --evidence examples/protein_evidence.synthetic.jsonl \
  --validator-config-version 1.0.0 \
  --output results/contracts/validator_input.jsonl \
  --join-output results/contracts/validator_join.jsonl
status=completed; payload_shape=simplified_validator_payload; inputs_written=2

PYTHONPATH=src python scripts/synthetic_smoke.py \
  --output results/synthetic_smoke_report.json
status=completed for candidate producer and PCA/IDH synthetic checks;
scVI, Harmony, and CUDA status=blocked where prerequisites are absent

PYTHONPATH=src python scripts/check_environment.py --config config/model.yaml \
  --json-out results/compute/current_environment.json
exit code: 2

PYTHONPATH=src python scripts/benchmark_scgpt.py --config config/model.yaml \
  --output results/compute/current_scgpt_benchmark.json
exit code: 2
```

The authoritative machine-readable evidence is in
`results/compute/current_environment.json` and
`results/compute/current_scgpt_benchmark.json`.

## Requirement-by-requirement result

| Requirement                                     | Current result                                                                       | Proof / reason                                                                                                                                                                                                                                                                                                               |
| ----------------------------------------------- | ------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CUDA GPU provisioned                            | NOT COMPLETE                                                                         | Environment report: `cuda_available=false`, `gpu_count=0`.                                                                                                                                                                                                                                                                   |
| scGPT installed                                 | NOT COMPLETE                                                                         | Environment report: `scgpt=null`; local import check is false.                                                                                                                                                                                                                                                               |
| 1,000-cell checkpoint forward pass              | NOT COMPLETE                                                                         | Benchmark status is `blocked`; all timing fields are null; checkpoint and vocabulary are unset.                                                                                                                                                                                                                              |
| GPU-seconds per 10,000 cells posted             | NOT COMPLETE                                                                         | `results/compute/week3_chat_report.txt` says `ACTION=DO_NOT_POST`; the actual CUDA planner returns `blocked` on this host.                                                                                                                                                                                                   |
| Pinned requirements                             | PARTIAL                                                                              | Exact package pins exist in `requirements.txt`, but no successful target-host install/export exists.                                                                                                                                                                                                                         |
| PCA + logistic regression                       | SOFTWARE COMPLETE                                                                    | `tests/test_baselines.py` verifies training-only fit, fixed probability order, reproducibility, and persistence.                                                                                                                                                                                                             |
| scVI baseline                                   | CODE COMPLETE, REAL RUN MISSING                                                      | `src/baselines/scvi_probe.py` now trains scVI on train cells and uses the query pathway; synthetic smoke reports `blocked` because scvi-tools is absent.                                                                                                                                                                     |
| Harmony + kNN baseline                          | CODE COMPLETE, REAL RUN MISSING                                                      | `src/baselines/harmony_knn.py` now fits Harmony on train cells and applies a frozen query projection; synthetic smoke reports `blocked` because harmonypy is absent.                                                                                                                                                         |
| Macro-F1 / bootstrap evaluator                  | SOFTWARE COMPLETE                                                                    | `tests/test_evaluation.py` plus `tests/test_eval_entrypoint.py`; combined output is tested.                                                                                                                                                                                                                                  |
| IDH AUROC                                       | SOFTWARE COMPLETE, SCIENTIFIC RUN MISSING                                            | `patient_binary` requires a separate patient-level IDH file; no real IDH prediction file exists. ClinVar/DMS is kept separate as an optional variant analysis.                                                                                                                                                               |
| Candidate scoring, schema producer, and handoff | SCORING/SERIALIZATION CODE COMPLETE, REAL scGPT RUN BLOCKED, VALIDATOR CONSUMER HELD | `src/models/candidate_scoring.py` and `scripts/generate_candidates.py` convert patient-aware mask-logit rows into ranked candidates; the TP53 synthetic flow writes candidate, rich join, and simplified payload artifacts. A real checkpoint has not produced the score rows. Ishaan’s decision-tree consumer remains held. |
| AlphaFold evidence                              | SYNTHETIC TEST ONLY                                                                  | `tests/test_alphafold_evidence_fixture.py` checks shape/provenance only; no evidence was downloaded or computed.                                                                                                                                                                                                             |
| Jeffrey cohort tasks                            | NOT DONE BY THIS WORK                                                                | No cohort registration, preprocessing, split construction, or donor/batch audit was performed.                                                                                                                                                                                                                               |
| Shared Neftel H5AD registration                 | CODE COMPLETE, PARTIAL DATA DELIVERY                                                | `data/README.md` records access date, SHA-256, 7,930 cells, 28 `Sample` donors, and observed metadata. TCGA and both CGGA artifacts were not present in the supplied folder. |
| Frozen Neftel QC/HVG contract                   | DATA VALIDATED, REPROCESSING ENVIRONMENT MISSING                                    | H5AD contains QC metrics and exactly 2,000 HVGs matching `config/qc.yaml`; `preprocess.py` now validates existing H5AD input. `scanpy/anndata` are not installed in the current host. |
| Canonical four-state labels                      | BLOCKED                                                                              | The downloaded H5AD contains `CellAssignment` (`Macrophage`, `Malignant`, `Oligodendrocyte`, `T-cell`), not `AC/MES/NPC/OPC`; no mapping is inferred. |
| Patient split integration                        | NOT COMPLETE                                                                         | The split file is now canonicalized to `train/validation/test`, but it references TCGA/CGGA patients absent from the downloaded Neftel-only artifact; combined evaluation remains blocked until those cohorts arrive. |
| Donor/batch visual audit                         | CODE COMPLETE, PROXY RESULT                                                          | `results/week1_data_audit/donor_batch_audit.json` and `donor_vs_cellassignment_pca.png` report donor silhouette `-0.0168` versus CellAssignment proxy `0.4323`; donor does not explain more separation in this proxy audit. |
| Week 2 pilot scGPT embeddings and candidates   | CODE COMPLETE, REAL RUN BLOCKED                                                      | `scripts/run_pilot_scgpt.py` emits structured `status=blocked` output for missing data/checkpoint/vocabulary/CUDA; synthetic candidate construction is tested and carries the fixed suspect-ranking interpretation label. |
| Week 2 GRN held-out edge recovery              | CODE COMPLETE, REAL RUN BLOCKED                                                      | `src/models/grn.py` validates hashed edge records, performs deterministic edge splitting, asserts against transitive training-prior leakage, and reuses evaluator AUROC; the CLI blocks when `grn_edge_list_path` is null. |
| Week 2 MC dropout                              | CODE COMPLETE, REAL RUN BLOCKED                                                      | `src/models/mc_dropout.py` computes per-cell/per-dimension mean and variance and pass-count timing; blocked output leaves timings null and marks them non-representative until real CUDA assets exist. |
| Week 2 Stage 5 masking                        | CODE COMPLETE, VALIDATOR CONSUMER HELD                                               | `src/gbm_study/stage5_masking.py` provides the outcome-source protocol, non-scientific stub, on/off masking, and provenance; real outcomes remain blocked pending Ishaan/Validator Lead sign-off. |
| Ishaan validator decision tree and four-gene gate | CODE COMPLETE, CLASSIFICATION PASSED, PUBLICATION GATE BLOCKED                   | `gate.py` classifies TP53=`destabilizing_driver`, IDH1=`functional_driver`, EGFR=`abstain`, and RPRM=`abstain`; the authoritative gate is blocked until exact source/version metadata and a measured IDH1 ΔΔG replace the supplied note. |

## Correction to prior handoff

The earlier statement that the requested work was “completed and verified” was
too broad. The passing tests prove software contracts and synthetic behavior;
they do not prove GPU availability, checkpoint inference, real cohort results,
or scientific baseline performance. Those claims remain unmade until the
required assets and execution host exist.
