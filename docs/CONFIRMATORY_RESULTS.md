# cfcap confirmatory results (one page; research lead)

**Status: PENDING.** Scoring runs once at ~03:10Z, after ops posts "final pull complete". This page is filled from
`results/confirmatory/verdicts.csv` only. Bars are as committed in `docs/CONFIRMATORY.md`, including amendments
ac543b8, b3af187, e176756, 727f995, 0dc1aec, c2b97cd, 03387d0 and ffc3921.

Legend: **HOLDS** = every bar of the claim passes (P-bars after Holm) · **FAILS** = a bar fails on a powered cell ·
**PARTIAL** = some bars pass, others fail or are missing · **UNDERPOWERED** = the power rule fails, so only E_ep is
reported · **MISSING** = no confirmatory rows.

| claim | bars (CONFIRMATORY §) | verdict | key numbers [95 % CI] | note |
|---|---|---|---|---|
| C-MAIN validity (ours, T1 / T2) | §1 bar 1 | — | | |
| P1 vs capxfer, lone-faulted | §1 bar 2 | — | | |
| P2 vs best trivial / non-learned | §1 bar 3 | — | | |
| P3 calibration vs raw (T2) | §1 bar 4 | — | | |
| cfBH (descriptive) | §1 bar 5 | — | | |
| C-MERGE δ_comp / validity / P4 | §2 | — | | |
| C-GATE H12 / P5 (≥ 3-module) | §3 bar 1 | — | | |
| C-GATE P6′ robust vs par; lawcal held-out s3 | §3b | — | | |
| C-ENV validity / P7 / generalisation | §4, §4b | — | | |
| C-SCALE validity / comp vs direct | §5 | — | | |
| C-HET validity / degradation (H1 faulted block) | §6 | — | | |
| C-EXPLAIN (secondary) | §6b | — | | |

## Holm table (P1–P7, m = 7)
(filled from verdicts.csv)

## What this means for the paper
(filled after review)

## Missing / underpowered cells → tomorrow on free workers
See `docs/board/research.md` "TOMORROW (free workers)".
