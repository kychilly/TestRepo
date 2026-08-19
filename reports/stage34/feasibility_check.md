# Feasibility Call — Confirmable Gene Count

**Decision date:** 2026-08-19
**Rule:** if fewer than 15–20 genes are confirmable (missense + AlphaFold-confident,
pLDDT >= 70.0), trigger the pooling plan.

## Deciding count

**2 confirmable genes** — well below the 15–20 threshold.
TP53 and IDH1, though protein evidence only exists for 4 genes total, ONLY ABLE TO TEST 4 CANDIDATES, so obv this falls below the threshold

| Gene | Missense in pilot cohort? | pLDDT | AlphaFold-confident (>=70.0)? | Confirmable? |
|------|---------------------------|-------|--------------------------------|--------------|
| TP53 | yes | 96.62 | yes | yes |
| IDH1 | yes | 96.56 | yes | yes |
| EGFR | no (alteration is amplification, not missense) | 51.22 | no | no |
| RPRM | no (no alteration recorded for RPRM in this cohort) | 63.16 | no | no |
| all other genes with a missense alteration (1,136 genes) | yes | not assembled | unknown | no (no evidence to test) |

## Branch taken

**Pooling plan triggered.** 2 < 15, so the pilot cohort alone does not supply enough
missense + AlphaFold-confident genes to proceed on pilot data only.

## Why the count landed at 2, not higher

- Of the 4 gate genes, only TP53 and IDH1 are both missense-altered in this cohort
  and have a trustworthy AlphaFold structure prediction.
- EGFR's alteration in the pilot cohort is amplification, not a point mutation, so
  it can never pass a missense-based confirmability test here regardless of its
  pLDDT.
- RPRM has no recorded alteration in any pilot patient at all.
- 1,138 unique genes carry a missense alteration somewhere in the pilot cohort, but
  protein evidence (pLDDT / ESM1b / ddG) has only been assembled for the 4 gate
  genes so far — so 1,136 of those missense genes cannot be evaluated for
  AlphaFold confidence at all, not because they failed the test, but because the
  test hasn't been run on them yet.

## Next step

Per the pooling plan: pull candidate drivers from additional cancer types to widen
the confirmable-gene pool beyond the pilot cohort's 8 MGH + CGGA patients, rather
than proceeding on pilot data alone.