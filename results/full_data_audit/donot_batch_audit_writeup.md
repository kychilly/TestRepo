# Donor/Batch Audit — Full Scale

**Run:** `scripts/run_full_donor_batch_audit.py`
**Data:** `data/processed/full_cohort_neftel_tcga.h5ad` (8,074 cells/samples: 6,576 Neftel, 1,338 CGGA, 160 TCGA)

## Headline finding: donor explains more separation than cell state (20D), but not in the 2D figure

| Dimensionality | Donor silhouette | State silhouette |
| 20 PCs (full audit space) | **0.158** | 0.079 |
| 2 PCs (matches the plotted figure) | -0.213 | **0.072** |

Two different, both-correct results depending on how many principal
components are considered. In the 2D projection actually shown in
`neftel_donor_vs_state_pca.png`, cell state visibly separates (NPC/OPC
cluster left, AC/MES cluster right) while donor shows no visible structure
(silhouette is negative — donor labels do worse than random at explaining
the 2D layout). But across the full 20-dimensional PCA space used for the
audit's summary statistic, donor identity explains more separation overall
than cell state does (0.158 vs. 0.079).

**Interpretation:** cell state creates strong separation concentrated in
the first 1-2 principal components (visible by eye). Donor/batch effects
are real but weaker and more diffuse, spread across many of the remaining
components — invisible in a 2D scatter plot, but present when measured
across the full PCA space. This is a real, if visually subtle, patient-level
batch effect, and it confirms the risk flagged in the Week 1 audit plan:
*"If donor explains more variance than state, say so now — that is the
batch-effect risk from our limitations section and it changes the plan."*

**Calibration note:** all four scores are modest in absolute terms
(silhouette ranges -1 to 1; none indicate strong, cleanly-separated
clustering). The finding is that donor identity's influence, while not
visible in a simple 2D plot, is measurably larger than cell state's when
the full embedding is considered — not that cells cluster cleanly by donor
and not at all by biology.

*"While a 2D PCA projection shows visible clustering by cell state (silhouette = 0.072) rather than donor
(silhouette = -0.213), a more complete 20-dimensional analysis reveals donor
identity explains more separation overall (silhouette = 0.158 vs. 0.079 for
state) — indicating a real, if visually subtle, patient-level batch effect
that motivates our use of Harmony batch correction (harmony_knn) as a
baseline."*

## Methodological response already in place

This finding is the direct motivation for including `harmony_knn` in the
baseline suite — Harmony batch-correction on `patient_id` (per
`config/baselines.yaml`) is the concrete response to this donor-separation
risk, not an arbitrary baseline choice.

## Secondary finding: cohort-level separation (assay-type effect)

| Comparison | Silhouette score |
| Data source (Neftel / CGGA / TCGA) | 0.448 |

Substantially higher separation than the donor/state comparison, but this
reflects Neftel being single-cell versus CGGA/TCGA being bulk RNA-seq —
different assay types, not a within-assay technical batch effect. Reported
separately so it isn't conflated with the donor finding above.

## Figures

- `results/full_data_audit/neftel_donor_vs_state_pca.png` — donor vs. state, within Neftel
- `results/full_data_audit/cohort_level_pca.png` — data source, full cohort
- Full numeric output: `results/full_data_audit/donor_batch_audit_full.json`