# Calibrating the fit-quality guards

**Date:** 2026-08-12
**Scope:** two experiments behind the fit-robustness work. (1) What are
physically plausible bounds for a fitted wind scalar, judged against every fit in
the archive rather than intuition? (2) Does `min_cluster_size` actually fix
degenerate fits, and what does it cost?

Both were prompted by Chile shipping a k=10 correction containing a wind scalar
of 80.2 and an offset that never converged, while scoring as a corrected win at
monthly resolution (`method-hourly-resolution.md`).

## Part 1: the scalar bounds, from 373,179 real fits

Every `factors_*.csv` under `output/validation` was surveyed: 585 files, 373,179
fitted scalars, 19 regions.

| region | scalars | min | median | max | <0.3 | >3.0 | failed offsets |
|---|---|---|---|---|---|---|---|
| DK | 329,699 | 0.090 | 0.762 | 2.679 | 2125 | 0 | 0 |
| US | 14,200 | 0.003 | 1.000 | 66.388 | 203 | 357 | 2 |
| DE | 12,025 | 0.199 | 0.858 | 7.386 | 11 | 32 | 0 |
| UK | 8,990 | 0.096 | 0.853 | 2.864 | 69 | 0 | 0 |
| BR | 4,775 | 0.314 | 1.118 | 8.198 | 0 | 270 | 0 |
| AR | 785 | 0.465 | 1.004 | 25.726 | 0 | 50 | 0 |
| CL | 735 | 0.202 | 1.195 | **150.407** | 10 | 158 | **56** |
| AU-NEM | 670 | 0.256 | 0.968 | 2.645 | 2 | 0 | 0 |
| NZ | 320 | 0.679 | 1.454 | 2.182 | 0 | 0 | 0 |
| SE | 200 | 0.359 | 0.652 | 0.871 | 0 | 0 | 0 |
| NL | 60 | 0.126 | 0.202 | 0.364 | 57 | 0 | 0 |
| BE | 60 | 0.314 | 0.439 | 0.527 | 0 | 0 | 0 |

Pooled: 0.66% of scalars fall below 0.3, 0.23% exceed 3.0, and 58 offsets failed
to converge.

**The two bounds are not equally informative, so they are set asymmetrically.**

- **The ceiling discriminates sharply.** Healthy regions top out just below 3.0
  (UK 2.86, DK 2.68, IT 2.65, AU-NEM 2.65, NZ 2.18); regions with known
  pathologies run far past it (CL 150, US 66, AR 26, BR 8.2, DE 7.4). A ceiling
  of **3.0** sits in the gap between two clearly separated populations. Kept.
- **The floor discriminates weakly**, because the country-level path legitimately
  fits low scalars when compensating for the cube-law overshoot. A floor of 0.3
  produced NO false positives on this data, so the original worry that it would
  cry wolf was wrong. But Belgium's minimum is 0.314, only 4.7% clear, which is
  too fine a margin to rely on across other years. The floor is therefore set at
  **0.2**: it still catches the unambiguous (US 0.003, DK 0.090, UK 0.096) and
  stays quiet on legitimate fits. A warning that fires on good fits is one people
  learn to ignore.

An unplanned corroboration: the floor independently flags 57 of 60 Netherlands
scalars. The Netherlands was already excluded for an unrelated ENTSO-E coverage
defect, so two independent diagnostics agree on the same region.

### The result that undercut the premise of Part 2

Cluster size does **not** predict whether a fit goes degenerate. Pairing every
factors table with its training fleet:

| training sites in cluster | n | median scalar | max scalar | % above 3.0 |
|---|---|---|---|---|
| 1 site | 182,275 | 0.766 | 126.4 | 0.217 |
| 2 | 45,194 | 0.770 | 98.7 | 0.323 |
| 3 to 5 | 71,586 | 0.776 | 150.4 | 0.201 |
| 6 to 20 | 52,975 | 0.777 | 11.9 | 0.187 |
| above 20 | 21,149 | 0.799 | 7.1 | **0.416** |

The RATE of implausible scalars is flat in cluster size, and is highest for the
largest clusters. What size predicts is SEVERITY: the worst case is 126 to 150
for clusters of five sites or fewer, against 7 to 12 above that.

So "degenerate fits come from small clusters" is an inference from three Chilean
cases that 373,179 cases do not support. Part 2's hypothesis was revised to
"`min_cluster_size` should cap how bad a fit gets, not eliminate bad fits" BEFORE
the experiment was run.

## Part 2: what min_cluster_size buys, and what it costs

Chile, k=10, fixed slice, held-out 2024. Uncorrected baseline RMSE 0.1226.

| min_cluster_size | clusters kept | smallest | max scalar | implausible | failed offsets | corrected RMSE | corrected MBE | corrected r |
|---|---|---|---|---|---|---|---|---|
| 1 (current) | 10 | 1 | **80.23** | 3 | 1 | **0.1048** | -0.0017 | 0.433 |
| 3 | 7 | 3 | 39.28 | 2 | **0** | 0.1069 | 0.0000 | 0.376 |
| 5 | 4 | 5 | **5.50** | 1 | 0 | **0.2366** | +0.0753 | 0.183 |

Gates, evaluated on the best-RMSE merged variant (min=3):

- **G1** severity capped below 10: 39.3. **FAIL**
- **G2** still beats uncorrected: 0.1069 against 0.1226. **PASS**
- **G3** within 10% of baseline: 0.1069 against 0.1153 allowed. **PASS**

**No setting satisfies all three gates.** Severity does fall monotonically
(80.2, 39.3, 5.50), and G1 would pass at min=5, but min=5 collapses the
correction: RMSE 0.2366 against an uncorrected 0.1226, so the correction becomes
roughly twice as bad as doing nothing, and correlation falls from 0.43 to 0.18.
Merging Chile to four clusters destroys the spatial resolution the correction
depends on.

### Reading

- **`min_cluster_size` is a useful guard, not a fix for degeneracy.** At min=3 it
  costs 2% of RMSE and buys a clean sheet on failed offsets (1 to 0), a mean bias
  of exactly zero, and one fewer implausible scalar. That is a fair trade for
  robustness. But 39.3 is still an absurd scalar, so the fit is not honest, only
  less dishonest.
- **The prediction from Part 1 held.** Size caps severity and does not remove
  frequency. The feature behaves as the survey said it would, not as the original
  three-case intuition said.
- **Chile's degeneracy is not a clustering problem.** It is ERA5 being wrong in
  the Atacama, and no constraint on the partition repairs that. This is the same
  conclusion the matched-curve work, the cluster sweep and the ML transfer retest
  all reached independently: the lever is a higher-resolution wind product.

### Recommendation

Do NOT enable `min_cluster_size` for Chile on these numbers. It does not deliver
an honest fit, and the setting that would (min=5) makes the correction worse than
no correction at all. Keep it available as an opt-in guard for regions where a
one-site cluster appears and skill is not resolution-critical, and rely on the
`fit_quality` warning plus the exported `degenerate` layer to make the remaining
bad fits visible rather than pretending they are fixed.

## Caveats

- Part 2 is one region, one cluster count, one held-out year. Chile is the
  hardest case in the set and may not generalise; a healthy region would likely
  show a smaller skill cost.
- The survey pools every run in `output/validation`, including diagnostic and
  superseded runs, so the per-region counts reflect how often a region was
  re-run, not its importance.
- The `>20 sites` band is the smallest (21,149 scalars) and carries the widest
  uncertainty on its 0.416% rate.

Scripts: `scripts/analysis/min_cluster_size_tradeoff.py`.
Data: `output/min_cluster_size/cl_tradeoff.csv`.
