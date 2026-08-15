# Week 2 ownership and merge contract

## Jeffrey — data and priors

- Deliver 5–10 patients per cohort, balanced across four states and both IDH
  statuses, with a 2–3k HVG panel containing TP53, IDH1, EGFR, and RPRM.
- Deliver authoritative TCGA/CGGA variant calls and the per-gene alteration
  table: missense, amplification, silencing, mapping status, build, and source.
- Deliver the GRN train prior and held-out edge slice with provenance.
- Run or hand off the three baseline inputs and cohort/split manifest.

## Adit — model integration

- `scripts/run_adit_week2.py` is the merge entry point.
- GRN branch: held-out AUROC plus input hashes and edge counts.
- MC dropout: active inference, 20–50 passes, per-gene mean/variance, and
  measured multiplier once the checkpoint, vocabulary, and CUDA host exist.
- Stage 5: validator on/off switch; only confirmed driver buckets feed the
  final prediction.
- Outputs and tests must retain seed, split hash, model hashes, and blocked
  status when real assets are absent.

## Ishaan — validator and feasibility

- Run the full candidate list through Stages 3–4.
- Produce five-outcome counts, confirmable-gene count, feasibility/pooling
  branch, shuffled fixed-seed control, and data-deficient coverage.
- Supply signed-off validator thresholds and complete protein-evidence metadata.

## Alexis — replication and audit

- Repeat the baseline/GRN paths independently using the same frozen inputs.
- Report coverage and baseline/GRN numbers with provenance.

## Merge gate

The Adit merge command is:

```sh
PYTHONPATH=src python scripts/run_adit_week2.py \
  --config config/week2_adit.yaml \
  --output results/week2_adit/report.json
```

`completed_with_blockers` is an honest integration result, not a scientific
pass. It becomes a scientific completion only when the listed external assets
are present and the report contains measured rather than blocked values.
