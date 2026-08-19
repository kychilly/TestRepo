# Transcriptomic-to-protein data contract

**Contract status:** version `1.0.0`, candidate producer implemented; validator consumer remains awaiting Ishaan/Validator Lead sign-off. The field meanings below are the interface agreement encoded by the schemas.

## Candidate-gene interchange view

The minimum human-readable interchange requested for each scGPT candidate is:

```json
{ "gene": "TP53", "state": "MES", "score": 2.31, "rank": 1, "seed": 17 }
```

This is a view of the production record, not a replacement for it. The checked-in
`schemas/candidate_gene.schema.json` intentionally requires provenance needed to
make paper numbers reproducible: run/backbone/checkpoint, vocabulary and split
hashes, patient scope, contributing patients, score method, gene namespace, and
configuration hash. `gene` is the canonical gene symbol; `score` is the
expression-derived candidate score; `state` is one of `AC`, `MES`, `NPC`, or `OPC`; and `rank` is
one-based within a declared run/state. `seed` is the integer random seed used
for the ranking run.

The scoring implementation is `src/models/candidate_scoring.py`. It accepts
scGPT mask perturbation rows containing `patient_id`, `cell_id`, `gene`,
`baseline_logit`, and `masked_logit`; computes `baseline_logit - masked_logit`;
averages cells within each patient; then computes the mean and patient-level SD
per gene. `src/models/candidate_generation.py` consumes those scores, sorts
deterministically by descending score and gene name, emits one-based ranks, and
validates every full record against `candidate_gene.schema.json`. The runnable
entry point is `scripts/generate_candidates.py`. None of these stages adds
AlphaFold, ESM1b, ΔΔG, or validator outcomes.

The model-specific scGPT adapter still has to supply the baseline and masked
state logits. The repository now implements the scoring/serialization stage
and tests it with synthetic logits; it does not claim that a real checkpoint
has produced these rows until the CUDA/checkpoint prerequisites are available.

### Agreement gate

Before validator-consumer implementation changes, Ishaan must confirm:

1. whether the five-field view is sufficient for transport or must include the
   provenance fields in every exchanged row;
2. whether `score` means the configured `score_method` output (recommended) and
   which score method is selected for the first scGPT run;
3. whether ranking is per state and run (recommended) or global across states;
4. the canonical gene namespace and alias policy; and
5. the handling of missing/duplicate candidates and patient scope.

The producer is implemented and emits schema-valid records for Ishaan to review.
The validator decision tree and threshold semantics remain held until Ishaan
signs off; this is a consumer gate, not a reason to leave the producer undefined.

## Two-stage validator handoff

The implementation deliberately keeps both representations:

1. `schemas/validator_input.schema.json` is the rich audit record. It preserves
   the full candidate provenance, every joined variant, mapping status, join key,
   cardinality, ambiguity, and precomputed protein evidence.
2. `schemas/validator_payload.schema.json` is the simplified consumer record.
   Its `candidate` object contains exactly `gene`, `state`, `score`, `rank`, and
   `seed`; `variants` and `protein_evidence` carry the evidence joined to that
   candidate. This is the JSONL handed to the validator.

`scripts/validate_contracts.py` writes the simplified payload with `--output`.
Pass `--join-output` when the rich provenance artifact is also required:

```sh
PYTHONPATH=src python scripts/validate_contracts.py \
  --candidates examples/candidate_gene.example.jsonl \
  --variants examples/variant_record.example.jsonl \
  --evidence examples/protein_evidence.synthetic.jsonl \
  --validator-config-version 1.0.0 \
  --output baseline_results/contracts/validator_input.jsonl \
  --join-output baseline_results/contracts/validator_join.jsonl
```

This makes the conversion explicit and reversible: Ishaan receives the compact
payload, while the paper/audit trail retains the complete join.

### Simplified payload fields

`schema_version` and `validator_config_version` identify the contract and the
threshold configuration. `input_id` links the payload to the immutable
candidate row. `candidate.gene` is the normalized HGNC symbol; `state` is the
one of four Neftel states; `score` is the configured scGPT expression score;
`rank` is one-based within the declared run/state scope; and `seed` identifies
the stochastic run. `variants` contains every matched patient-level alteration
without collapsing transcripts or events. `protein_evidence` contains only
precomputed evidence keyed by `variant_id`, with source/version and nullable
measurements for missing data. `join_cardinality` records zero/one/multiple
matches, and `validator_eligibility` records the data-availability gate; it is
not a final biological outcome.

### Protein-evidence fields

Each `protein_evidence` row has `evidence_id`, `variant_id`, and canonical
`gene` identity; `protein_accession`; `evidence_status` (`complete`, `partial`,
or `missing`); AlphaFold source/version plus `plddt_score`; ESM1b source/version
plus `esm1b_score`; and stability source/version plus `delta_delta_g`. Null
scores are allowed only because the row explicitly records incomplete or
missing evidence. The pipeline stores these values and provenance; it does not
compute AlphaFold, ESM1b, or ΔΔG.

## Semantic decisions

A candidate record is a ranked gene-level result from one transcriptomic run. `candidate_id` identifies that immutable ranking row. `contributing_patient_ids` is mandatory because a scope label alone cannot prove which patients contributed. `patient_scope=train_only` means only the listed training patients contributed and requires `training_only=true`; `internal_refit` means an explicitly documented train-plus-internal-refit population and is not a fold-level training-only ranking; `external_forbidden` records that external patients were excluded from the scoring population. Validation or test patients can be supplied as a forbidden set to the contract validator and cause a hard failure.

`score_method` is controlled and deliberately excludes the ambiguous term `importance`. `checkpoint_hash`, `vocabulary_hash`, `split_hash`, and `config_hash` are required provenance values. `gene` is stored only after alias normalization, with the mapping maintained outside the record and passed explicitly to the join builder. Unknown aliases fail closed.

A variant record is one patient-specific alteration. `variant_id` is required so multiple variants, transcripts, and isoforms remain independently addressable. Protein fields are populated only for `mapping_status=resolved`; every unresolved mapping has a non-empty `mapping_notes`. Amplifications, deletions, silencing, and other non-protein alterations remain present and produce abstention rather than disappearing.

A validator input is never a candidate record alone. `build_validator_inputs` requires both candidate and variant collections and performs an explicit join on candidate patient scope, cohort, normalized gene identity, and Ensembl gene ID. One input is emitted per candidate. `variant_provenance` preserves zero, one, or multiple matching variants, and `protein_mapping` preserves one mapping entry per variant. Multiple variants are never collapsed and produce `abstain_ambiguous`. Zero matches produce `abstain_missing_evidence`.

## Versioning

All three schemas use `schema_version=1.0.0`. Patch versions may clarify documentation without changing accepted semantics; a minor version may add backward-compatible optional fields; any removal, type change, controlled-vocabulary change, or semantic reinterpretation requires a new major version. Producers and consumers must reject unsupported major versions.

## Files and command

- `schemas/candidate_gene.schema.json`: ranked transcriptomic candidate.
- `src/models/candidate_scoring.py`: patient-aware mask-delta score aggregation.
- `scripts/generate_candidates.py`: score rows to candidate JSONL entry point.
- `schemas/variant_record.schema.json`: patient-specific alteration and mapping evidence.
- `schemas/validator_input.schema.json`: explicit joined validator payload.
- `schemas/protein_evidence.schema.json`: precomputed AlphaFold/ESM1b/ΔΔG evidence.
- `schemas/validator_payload.schema.json`: simplified validator-consumer payload.
- `src/schemas/records.py`: typed validation and join implementation.
- `scripts/validate_contracts.py`: JSONL validation and input generation.

```sh
PYTHONPATH=src python scripts/validate_contracts.py \
  --candidates examples/candidate_gene.example.jsonl \
  --variants examples/variant_record.example.jsonl \
  --evidence examples/protein_evidence.synthetic.jsonl \
  --validator-config-version 1.0.0 \
  --output baseline_results/contracts/validator_input.jsonl
```

The synthetic EGFR missense and amplification rows intentionally produce a multiple-variant abstention for EGFR. The RPRM silencing row remains present and produces non-protein abstention. This is expected and demonstrates why a gene-level candidate cannot be sent directly to a missense validator.

## Validator Lead sign-off checklist

1. Confirm the three patient-scope meanings and whether `external_forbidden` is retained as a first-class scope.
2. Confirm that `contributing_patient_ids`, `candidate_id`, and `variant_id` are required interface additions.
3. Confirm the controlled score-method vocabulary and the `eligible_missense`/abstention states.
4. Confirm the join key and whether cohort plus Ensembl identity are sufficient for production joins.
5. Confirm the validator configuration versioning policy.
