# cfcap confirmatory results (one page; research-lead review)

Scored **once**, at 03:17Z on 11-26, by `scripts/analysis/confirmatory.py` (scorer fb51ccb; run 77fc39b) against
`docs/CONFIRMATORY.md` @ 986730d. The full table is `results/confirmatory/CONFIRMATORY_RESULTS.md` and the rows are
in `verdicts.csv` / `cells.csv`. This page gives the research lead's reading. **The verdicts stand as scored.**
Anomalies are flagged, not re-scored.

Sample that landed:
- C-MAIN 1,050 model episodes (1,220 clusters in all; baselines only 210);
- C-MERGE 975 pairs;
- C-GATE 165–220 episodes per cell (of 600);
- C-SCALE N 7–10 only;
- C-HET 250–500 per split.

**Every E_dec cell failed the power rule.**

| claim | verdict | key numbers [95 % CI] | reading |
|---|---|---|---|
| **C-MAIN validity, E_ep (the guarantee), T1** | **HOLDS** | 0.95 % [0.39, 1.61] vs α = 10 % | The per-episode simultaneous guarantee holds with a wide margin |
| C-MAIN E_dec at matched U* = 1 %, T1 | **FAILS** (underpowered, exceeds) | 3.2 % [2.6, 3.9] | Matched-point transfer failed (diagnosis below) |
| C-MAIN 1b per kind (matched point) | **FAILS** | merge 4.2 %, multi-child 6.7 %, lone 2.9 % | Every kind is elevated about 3×, so this is not merge-specific. At the *own* bound: merge 0.35 %, multi-child 1.0 %, lone 0.12 % |
| **P1 vs capxfer** (lone-faulted CF@U*) | **HOLDS** (Holm) | +38.5 pts [34.5, 42.7] | |
| **P2 vs best trivial** (ghat_physics) | **HOLDS** (Holm) | +35.3 pts [34.6, 36.1] | Baselines cover 210 episodes. Check that the pairing used the common episodes only |
| C-MAIN T2 rows, P3 | **MISSING** | — | |
| **C-MERGE** δ_comp, validity, P4 | **HOLDS** (Holm on P4) | flips 0.07 % (T1) / 0.13 % (T2); per-pair miscov 0.4 % / 2.9 %; CF 90.8 %; +55.2 pts vs copy-parent | Strongest result. Merge E_dec is underpowered (389 incapable), with 0 observed |
| C-GATE P5 (H12) | **FAILS** | raw 97.1 % unsafe on undominated (fails silently); par 0.49 % | The intersection–union test needs both. Calibrated physics did *not* fail |
| C-GATE P6′ (robust vs par) | **FAILS** | par − robust E_ep −6.1 pts; robust E_ep at s = 3 is 7.3 % (upper 10.8 %) | robust is not valid at s = 3. par is "safer" only because it barely certifies |
| C-GATE robust validity, s = 0–2 | HOLDS (E_ep 0 %) | | E_dec underpowered |
| C-GATE price of robustness | "PASS" by definition, **ANOMALY** | CF(par) − CF(robust) = −56.7 pts | Not a sign bug: CF(par, s = 0) = 0.31 vs robust 0.88. par is degenerate in T2 (CF 0.33 even on *dominated* legs at s = 0, where pi certifies 0.999), and lawcal ≈ par to 4 decimals. Reading: a construction issue in the T2 par/lawcal margin, so this row carries no evidence |
| C-GATE lawcal held-out s = 3 | PASS, but same anomaly as par | 0.49 % | |
| C-ENV validity, P7 | **MISSING** | (KS prerequisite PASS; 33 % of oracle plans reconfigure ≥ 20 %) | Missions were scored before the model bounds existed |
| C-SCALE | **PARTIAL** | E_ep 1.7 % (N 7–8, 9–10) PASS; coverage 0.86 PASS / **0.81 FAIL** (N 9–10); H4b and comp-vs-direct MISSING | Coverage drops with size, as Prop. 2b predicts |
| C-HET | **PARTIAL** | validity PASS (E_ep 0.4 % H1, 3.2 % H2); coverage drop ≤ 0.2 pts PASS; **CF drop ID → H2 19.8 pts FAIL**; H1 CF direct − ours −3.5 FAIL | Coverage transfers to new types; usefulness does not, on mixed bodies |
| C-EXPLAIN | **PARTIAL** | wheel precision 0.953 PASS; informative 1.00 PASS; **deficit precision 0.77 FAIL**; **false-reason 13 % FAIL** | The margin/deficit fix is not enough: posterior medians sit below τ on truly capable bodies |

**Holm (m = 7):** P1, P2 and P4 reject (p = 0.0005). P5 and P6′ do not. P3 and P7 are missing.

## Scorer anomalies (documented; the run stands)
1. The price-of-robustness row passes a bar whose premise (par more useful than robust) is inverted. It is
   arithmetic-correct. The anomaly is par's T2 construction (see the table).
2. The "cal-size sensitivity" rows (cal 0–999 vs 0–1,999) are identical (k = 0.791), because the model's C-MAIN
   margin came from the **train-run cal of 300 episodes, 100 of them composed**, not from the generator cal split.
   The A-POWER cal trim therefore never touched the model rows.
3. amortized_comp and credible_lower have identical matched-point rows. That is expected: same point/scale and
   only k differs. They differ at their own bounds (E_ep 0.95 % vs 7.1 %).
4. **Confirmed scorer defect, P1/P2 pairing.** `_paired_diff` computes each method's ratio over the union
   bootstrap index (1,220 clusters): ours over 1,050 episodes, capxfer/ghat_physics over 210. It is therefore not a
   paired test on common episodes.
   - The effect sizes (+35–38 pts) make a flip implausible.
   - The correct rescore (intersection of eids, same bootstrap weights) is reported as a **post-hoc correction**
     (paper role). Tonight's verdict stays in the record.
   - Until the correction exists, cite P1/P2 as "unpaired, same stream".

## Diagnosis of the C-MAIN matched-point and A-KIND failures (exploratory reasoning)
- **Likely cause: the 100-composed-episode calibration.** About 900 incapable cal decisions put only ~9 events at
  the 1 % quantile, and k was chosen on that same set. That is high variance and optimistically selected. A
  transfer to 3.2 % is what one expects from that.
- **The guaranteed quantity holds** (E_ep 0.95 %, and own-bound E_dec ≤ 1 % in every kind). The failure is in
  *matched-point reporting*, not in the certificate.

## Honest thesis given these verdicts
*A docked modular robot can certify, with a per-episode guarantee that held at confirmatory grade (unsafe episodes
1 % at α = 10 %), which bodies it could become can do which job. Those bodies include merges of modules that were
never docked together, certified at 91 % with flips of 0.1 %. It approves 38 points more faulted bodies than the
prior certificate, and 55 points more merges than a copy-the-parent rule. Plug-in physics fails silently under a
wrong motor law (97 % unsafe on unvisited operating points), whereas calibration keeps it safe, at a large cost in
usefulness. Our model-free robust alternative is not yet valid at the strongest misspecification.*

The paper must not claim:
- matched-1 % unsafe rates;
- the robust/gate certificate as a validated method;
- explanation faithfulness (≥ 90 %);
- usefulness on mixed module types;
- anything on env missions or N > 10.

## Tomorrow (free workers): a NEW pre-declared run on unread seeds
See `docs/board/research.md` "TOMORROW (final)". Tonight's verdicts are final for the cells scored. Tomorrow's run
is a separate, pre-declared confirmatory run on seeds no role has read, with a calibration declared now.
