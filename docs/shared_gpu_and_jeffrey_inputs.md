# Shared GPU policy and Jeffrey delivery contract

This contract applies to the Adit model-integration branch and intentionally
does not change Ishaan's validator schemas, thresholds, or decision tree.

## Shared GPU rule

The GPU host is compute, not bulk storage. Do not clone or download complete
cohort archives, large model zoos, or broad Hugging Face caches onto it.

- Preferred input: `datasets.load_dataset(..., streaming=True)` with an exact
  dataset revision and split.
- Acceptable local input: only the compact analysis split needed for the run,
  plus the single selected checkpoint and its vocabulary/args files.
- Default preflight limits: 2 GiB per local input and 5 GiB total. Any larger
  transfer requires explicit approval before the run.
- Set `HF_HOME`, `HF_DATASETS_CACHE`, and `TRANSFORMERS_CACHE` to the assigned
  scratch location, not a shared home directory. Record that location in the
  environment manifest.
- Never download several alternative checkpoints to compare on the GPU host.
  Select and hash one checkpoint before provisioning the final run.
- Do not present T4/L4/A100 timing interchangeably. Benchmark the exact final
  GPU, batch size, precision, token length, checkpoint, and 1,000-cell split.

Use `make scgpt-shared-gpu`; its dedicated configuration fails closed until the
immutable Hugging Face fields are populated. `scripts/benchmark_scgpt.py` uses
reservoir sampling over training patients and calls
`load_dataset(..., streaming=True)` and retains only the requested benchmark
cells in memory. A local H5AD remains supported for an approved compact pilot.

## GPU choice statement

An A100 is requested for production throughput and memory headroom, not because
40 GB is a demonstrated minimum. Validate first on an L4 24 GB when available.
A T4 16 GB can be used for a reduced-batch smoke test. MC-dropout passes are
sequential: they multiply time, not VRAM. The final paper records measured
timing from the exact GPU actually used.

## What Jeffrey must deliver beyond Week 3/4

### Compact expression transport

1. A Hugging Face dataset ID, immutable revision/commit, config name, and split
   name for the Neftel analysis cells, or an explicitly approved compact H5AD.
2. One row per cell with `patient_id`, `cell_id`, and a fixed-length expression
   vector. For streaming, provide the exact expression and gene-ID column names.
3. A single ordered gene list with namespace and release (`HGNC symbol` or
   `Ensembl`, including version policy). Expression columns must match this list
   exactly.
4. A manifest with source file hashes, transformation steps, dimensions,
   sparsity, dtype, normalization, and access date.

### Labels and inclusion rules

5. A signed malignant-cell inclusion rule. Immune/oligodendrocyte cells and the
   current 1,351 `Unknown` rows must not silently enter four-state training.
6. A canonical AC/MES/NPC/OPC label per included cell, with the method and
   threshold used to derive it. `GBMType` is adult/pediatric and must not be
   represented as IDH status.
7. A patient table containing cohort, IDH status, TP53 status, missingness,
   and all stratification variables used to build splits.

### Frozen split and cohort contract

8. The exact train/validation/test JSON committed or deposited with a SHA-256.
   It must state whether CGGA-325 and CGGA-693 are separate external tests or
   one pooled external test.
9. Per-split patient, cell, state, cohort, and IDH counts, plus zero-overlap and
   missing-patient checks.
10. A decision on TCGA: either provide TCGA expression and authoritative
    variants, or remove TCGA modeling claims from the paper design.

### scGPT checkpoint package

11. Exactly one selected checkpoint file, its matching `vocab.json`, and its
    `args.json`; provide SHA-256 hashes and the upstream model/release URL.
12. The intended value representation: raw counts, normalized expression, or
    rank-binned values; token length; number of bins; required special tokens;
    and checkpoint-specific preprocessing instructions.
13. If candidate scoring uses state logits, provide the trained four-state head
    and a checkpoint-specific `mask_score_provider` implementation. A generic
    embedding checkpoint cannot produce `baseline_logit - masked_logit` rows by
    itself.

### Baseline and bulk-cohort inputs

14. Raw integer Neftel counts in a `counts` layer if scVI remains a required
    baseline. LogTPM/log1p values cannot be reverse-labeled as counts.
15. CGGA-325 and CGGA-693 normalization metadata and the frozen 23,271-gene
    overlap or another prespecified feature panel.
16. An explicit missing-IDH policy (currently 1 missing in CGGA-325 and 51 in
    CGGA-693) and exact clinical/expression join reports.

### Variant and GRN evidence

17. Authoritative patient-level variant/CNV/silencing calls with genome build,
    transcript/protein mapping, source version, and access date. The heuristic
    pilot mutation table is not acceptable for publication.
18. A larger GRN prior and held-out set with unique biological edges, database
    versions, confidence definitions, and a frozen holdout seed. One held-out
    positive is only a software sanity check.

### Reproducibility

19. Data-use/license confirmation for every cohort and permission to stream the
    selected subset from the chosen host.
20. A compact expected-output fixture: retained gene count, mapped vocabulary
    count, selected training-cell count, and one non-sensitive checksum or
    summary that confirms the GPU loader sees the intended data.

No final GPU run should begin until items 1–13 are complete. Items 14–20 gate
the corresponding baseline, external-validation, variant, GRN, and publication
claims.
