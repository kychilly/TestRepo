# Transcriptomic-to-protein data contract

**Contract status:** version `1.0.0`, proposed for Validator Lead sign-off. The field meanings below are the interface agreement encoded by the schemas and must be reviewed before production records are emitted.

## Semantic decisions

A candidate record is a ranked gene-level result from one transcriptomic run. `candidate_id` identifies that immutable ranking row. `contributing_patient_ids` is mandatory because a scope label alone cannot prove which patients contributed. `patient_scope=train_only` means only the listed training patients contributed and requires `training_only=true`; `internal_refit` means an explicitly documented train-plus-internal-refit population and is not a fold-level training-only ranking; `external_forbidden` records that external patients were excluded from the scoring population. Validation or test patients can be supplied as a forbidden set to the contract validator and cause a hard failure.

`score_method` is controlled and deliberately excludes the ambiguous term `importance`. `checkpoint_hash`, `vocabulary_hash`, `split_hash`, and `config_hash` are required provenance values. `gene_symbol` is stored only after alias normalization, with the mapping maintained outside the record and passed explicitly to the join builder. Unknown aliases fail closed.

A variant record is one patient-specific alteration. `variant_id` is required so multiple variants, transcripts, and isoforms remain independently addressable. Protein fields are populated only for `mapping_status=resolved`; every unresolved mapping has a non-empty `mapping_notes`. Amplifications, deletions, silencing, and other non-protein alterations remain present and produce abstention rather than disappearing.

A validator input is never a candidate record alone. `build_validator_inputs` requires both candidate and variant collections and performs an explicit join on candidate patient scope, cohort, normalized gene identity, and Ensembl gene ID. One input is emitted per candidate. `variant_provenance` preserves zero, one, or multiple matching variants, and `protein_mapping` preserves one mapping entry per variant. Multiple variants are never collapsed and produce `abstain_ambiguous`. Zero matches produce `abstain_missing_evidence`.

## Versioning

All three schemas use `schema_version=1.0.0`. Patch versions may clarify documentation without changing accepted semantics; a minor version may add backward-compatible optional fields; any removal, type change, controlled-vocabulary change, or semantic reinterpretation requires a new major version. Producers and consumers must reject unsupported major versions.

## Files and command

- `schemas/candidate_gene.schema.json`: ranked transcriptomic candidate.
- `schemas/variant_record.schema.json`: patient-specific alteration and mapping evidence.
- `schemas/validator_input.schema.json`: explicit joined validator payload.
- `src/schemas/records.py`: typed validation and join implementation.
- `scripts/validate_contracts.py`: JSONL validation and input generation.

```sh
PYTHONPATH=src python scripts/validate_contracts.py \
  --candidates examples/candidate_gene.example.jsonl \
  --variants examples/variant_record.example.jsonl \
  --validator-config-version 1.0.0 \
  --output /tmp/validator_input.jsonl
```

The synthetic EGFR missense and amplification rows intentionally produce a multiple-variant abstention for EGFR. The RPRM silencing row remains present and produces non-protein abstention. This is expected and demonstrates why a gene-level candidate cannot be sent directly to a missense validator.

## Validator Lead sign-off checklist

1. Confirm the three patient-scope meanings and whether `external_forbidden` is retained as a first-class scope.
2. Confirm that `contributing_patient_ids`, `candidate_id`, and `variant_id` are required interface additions.
3. Confirm the controlled score-method vocabulary and the `eligible_missense`/abstention states.
4. Confirm the join key and whether cohort plus Ensembl identity are sufficient for production joins.
5. Confirm the validator configuration versioning policy.
