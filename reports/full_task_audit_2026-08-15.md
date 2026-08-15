# Full task audit — 2026-08-15

## Bottom line

Software readiness: **8/10**. Scientific/pilot readiness: **3/10**.
Overall publication readiness: **4/10**. The repository is now runnable for
the supplied Neftel pilot and Jeffrey GRN prior, but it is not yet a valid
full four-state, two-IDH-status, TCGA/CGGA study.

No numeric team acceptance thresholds were supplied beyond the requested
bootstrap/provenance/gate contracts, so baseline scores cannot honestly be
called “above target.” They are recorded as the Week 4 comparison bar.

## Completed with evidence

- Pilot H5AD: 7,302 cells, 2,503 genes, 26 patients.
- Four canonical state labels are present as derived `*-like` labels and are
  normalized in the pilot baseline loader. 1,351 cells remain explicit
  `Unknown` and are excluded from model evaluation.
- Required genes are present in the 2,503-gene panel; the panel is within the
  requested 2–3k range.
- PCA/logistic test baseline: macro-F1 0.6256, balanced accuracy 0.6488.
- Harmony/kNN test baseline: macro-F1 0.5555, balanced accuracy 0.5524.
- Both baselines use the same 26-patient pilot split and 1,000 patient-level
  bootstrap replicates.
- GRN sanity check: AUROC 1.0, one unique held-out positive, 144 negatives,
  input hashes recorded.
- Four-gene classification gate: passed expected classifications.
- Stage 5 masking, MC-dropout wrapper, validator on/off flag, provenance, and
  tests: implemented; full suite is 70 passed.
- CGGA archives: extracted to `TP53 Dataset(preprocessed)/cgga(processed)/`;
  source zips are ignored and retained.

## Requirements not met and why

- The pilot is Neftel-only and IDH-wildtype; it does not satisfy both IDH
  statuses or the multi-cohort requirement.
- The supplied mutation table is heuristic four-gene metadata, not TCGA/CGGA
  variant calls. No TCGA MAF/VCF or CGGA mutation-call file is present.
- CGGA clinical data provide IDH status (325: 175 mutant/149 wildtype/1
  missing; 693: 356 mutant/286 wildtype/51 missing), but not per-variant
  missense/amplification/silencing calls.
- scVI is blocked because the H5AD has no raw integer `counts` layer.
- scGPT and real MC-dropout timing are blocked by missing checkpoint,
  vocabulary, and CUDA GPU.
- No real AlphaFold/ESM1b/ΔΔG evidence with complete source/version metadata
  is present; the publication validator gate remains blocked.
- No real Stage 3/4 candidate list exists, so bucket counts, feasibility,
  pooling, shuffled control, and data-deficient coverage cannot be computed
  without fabricating inputs. The existing validator fixtures are synthetic.
- No independent Alexis run exists; the shared baseline/GRN paths were run
  once and are not a separate replication.

## Required next inputs

1. TCGA and CGGA variant calls with patient IDs, alteration type, genome build,
   transcript/protein mapping, and source/version metadata.
2. A true two-IDH-status, multi-cohort pilot expression/label contract.
3. Raw counts if scVI is required.
4. scGPT checkpoint, vocabulary, and CUDA host for Stage 3 and MC-dropout
   timing.
5. Complete AlphaFold/ESM1b/ΔΔG evidence provenance.
6. A real candidate list and protein-evidence join to run Ishaan’s bucket,
   pooling, shuffled-control, and coverage audits.
