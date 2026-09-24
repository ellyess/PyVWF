# New Zealand: first run (the first new-climate region since the survey)

**Date:** 2026-07-23
**Scope:** New Zealand's first train and evaluate run, the first region in a new climate
since the European set.

**Correction notice, 2026-09-24: every corrected figure in this document was
simulated on other units' power curves and capacities.** `correct_wind_speed`
rebuilt the turbine axis in sorted ID order and then attached each unit's model
key and capacity by position, in the fleet's own order. The NZ fleet's IDs are
not sorted and its eight training farms use seven model keys, so the corrected
simulation of this run gave farms other farms' curves and capacities. Fixed in
`a94670c`. The factors and the uncorrected row are unaffected: training and the
uncorrected simulation build the turbine axis in fleet order.

The same training run (`output/validation/NZ/train-k147`) was re-evaluated twice
from a clean tree on the licensed library as input root (`power_curves.csv`
sha256 `689cfee7…`, the same file as the original run), with every unit
resolved to a curve of `open` origin. Before the fix, at `b6bdfe4`, the
re-evaluation reproduced the original run's RMSE, MAE, MBE and r exactly for
every variant. After the fix, at `462f37b`. Training years 2019-23, test year
2024, 12 farms and 137 farm-months scored:

| Variant | RMSE before | RMSE after | MAE before | MAE after | MBE before | MBE after | r before | r after |
|---|---|---|---|---|---|---|---|---|
| uncorrected | 0.1568 | 0.1568 | 0.1426 | 0.1426 | -0.0617 | -0.0617 | 0.639 | 0.639 |
| affine k=1 fixed | 0.1430 | **0.1509** | 0.1276 | **0.1357** | -0.0307 | **-0.0295** | 0.622 | **0.651** |
| affine k=4 fixed | 0.1108 | **0.1070** | 0.0814 | **0.0770** | +0.0250 | **+0.0309** | 0.603 | **0.705** |
| affine k=7 fixed | 0.1063 | **0.1046** | 0.0777 | **0.0761** | +0.0206 | **+0.0253** | 0.663 | **0.742** |
| affine k=1 season | 0.1436 | **0.1516** | 0.1282 | **0.1360** | -0.0309 | **-0.0297** | 0.610 | **0.640** |
| affine k=4 season | 0.1112 | **0.1078** | 0.0812 | **0.0774** | +0.0238 | **+0.0298** | 0.587 | **0.686** |
| affine k=7 season | 0.1084 | **0.1068** | 0.0789 | **0.0766** | +0.0207 | **+0.0253** | 0.644 | **0.722** |

The headline table below lists five of these seven variants and is left in
place as published; its corrected rows are withdrawn and replaced by the after
columns here. The sentences below that quote a withdrawn figure are marked
where they stand. What those sentences claim has not been re-assessed against
the after figures.

The resampled gain in the notice of 2026-09-11 below was also computed on
frames with this defect. Re-resampled after the fix, on the scorecard row's run
(`refresh_2026-09-20`, `k=7` fixed), the RMSE gain is 0.052 with a 95% interval
of -0.032 to 0.115; the interval still includes zero (`scorecard.md`, notice of
2026-09-24). The MAE interval and the single-farm shares were not recomputed.

Data: `output/c1_turbine_order_2026-09-24/region_nz/` (`before/` and `after/`).

**Correction notice, 2026-09-11: the headline is not resolved.** The headline
says the affine correction wins on every metric, and places NZ with DK, DE and
UK among the regions where the correction earns its keep. When the test year's
12 farms are resampled (1,000 paired draws, on the scorecard row's run), the
`k=7` fixed RMSE gain is 0.051 with a 95% interval of -0.032 to 0.114, and the
MAE gain is 0.065, with an interval of -0.018 to 0.119. Neither excludes zero.
One farm, 12% of capacity, goes from an RMSE of 0.05 uncorrected to 0.24
corrected, and carries 61% of the corrected squared error. The UK is not
resolved either (`scorecard.md`, correction notice of the same date). The
figures in the table stand. The intervals are conditional on the single test
year and treat farms as independent, so they understate the uncertainty.

Trained 2019–2023, evaluated on held-out **2024**, EMI per-farm monthly CF,
external combined curve library, k-means++ defaults. Region config
`configs/regions/nz.toml`, source `emi-nz`. Run:
`output/validation/NZ/{train-k147, evaluate-2024-k147}`.

## Headline: the affine correction wins on every metric; NZ behaves like Denmark, not Australia

| variant | MBE | MAE | RMSE | Pearson r |
|---|---|---|---|---|
| uncorrected | −0.062 | 0.143 | 0.157 | 0.639 |
| affine k=1 fixed | −0.031 | 0.128 | 0.143 | 0.622 |
| affine k=4 fixed | +0.025 | 0.081 | 0.111 | 0.603 |
| **affine k=7 fixed** | **+0.021** | **0.0777** | **0.106** | 0.663 |
| affine k=7 season | +0.021 | 0.0789 | 0.108 | 0.644 |

12 farms, 137 farm-months in the test year. *[Withdrawn 2026-09-24: every
corrected row of this table; the uncorrected row stands. The figures after the
fix are in the notice of that date.]*

**Best config (k=7, fixed): MAE −45% (0.143→0.078), RMSE −32%
(0.157→0.106).** *[Withdrawn 2026-09-24: after the fix, 0.143→0.076 and
0.157→0.105.]* This is a level-and-scale win of the same character as
Denmark (D2: level-dominated regions gain broadly), and the opposite of
Australia (near-unbiased, where the correction added farm-level noise
*[withdrawn 2026-09-24 in `region-au-nem.md`: that document's corrected
figures rest on the same defect]*). NZ is
the fourth region to land clearly in the "correction earns its keep" camp
(DK, DE, UK, now NZ), and the first Southern-Hemisphere one that does.

## Bias structure (the D2 diagnosis, run first)

- **ERA5 under-predicts NZ wind: uncorrected MBE −0.062** against a fleet-mean
  observed CF near 0.36. Negative bias in a high-wind, complex-terrain fleet
  (Manawatu Gorge, Cook Strait / West Wind funnelling, Te Uku ridge) is
  physically what you expect: 0.25° reanalysis smooths the orographic
  acceleration these sites are sited on. This is the same *kind* of gap the ML
  re-test flagged as globally under-represented (Tehachapi-type terrain), now
  with the sign confirmed on real data.
- **The bias is spatially structured, not uniform.** k=1 barely helps (MAE
  0.143→0.128); the jump is k=1→k=4 (0.128→0.081). *[Withdrawn 2026-09-24:
  after the fix, k=1 MAE is 0.136 and k=4 MAE 0.077.]* A single fleet-wide factor
  leaves most of the error on the table: different farms need different
  corrections, and clustering is what captures it. Consistent with the
  us_br finding that `cluster_list=[1]` leaves 20–50% of achievable reduction
  unclaimed.
- **Correlation is near-flat** (0.64→0.66): the win is in level and scale, not
  timing. *[Withdrawn 2026-09-24: after the fix, r goes from 0.64 to 0.74 for
  k=7 fixed.]* That is exactly what an affine-in-wind correction can fix and is the
  signature of a resource-magnitude bias rather than a phase error.
- **Fixed ≈ season here.** Seasonal slicing does not beat fixed (k=7: 0.0777
  vs 0.0789 *[withdrawn 2026-09-24: 0.0761 vs 0.0766 after the fix]*): NZ's year-round westerly regime has modest seasonal amplitude,
  so there is little seasonal shape for the correction to exploit, unlike the
  trade-wind or monsoon regions.

## The k ceiling was lower than estimated (recorded so it is not re-tripped)

The config first shipped `cluster_list=[1,5,10]` on the estimate of ~11 unique
coordinates. **k=10 crashed**: k-means requires `n_clusters ≤ n_samples`, and
only **8 farms reach the clusterer** in the 2019–2023 window. 13 farms − 2
post-window commissions (Harapaki 2023-11, Kaiwera Downs 2 2026) = 11 with
training obs; three more drop inside `train_set` (sparse coverage / sim-obs
merge) → 8. The sweep is now `[1,4,7]`, safely under that ceiling; k near 8
would be one-farm-per-cluster (the fake plateau). The **test** year evaluates
all 12 farms present in 2024; the training ceiling does not limit evaluation.

Provenance note: the first (aborted) `[1,5,10]` train wrote k=1 and k=5 before
failing at k=10; reusing the run-name left a stale `k=5` in the metrics. The
committed run (`train-k147`) was produced fresh after deleting the stale
directory, so its factors are exactly `[1,4,7]`.

## Caveats carried into this result

- **Te Rere Hau is degraded** (5 turbines stopped, 2 derated late in the
  window): its observed CF understates the resource, so its farm-level
  correction partly absorbs a mechanical fault as if it were reanalysis bias.
  An exclusion-run robustness check is the obvious next probe (the AU
  far-north pattern).
- **Hub heights are hand-compiled**, three unverified (Tararua III, Mill
  Creek, Kaiwera Downs 2); `height_source` flags them in the farm table.
- **No curtailment screen.** NZ's hydro-dominated system curtails little wind
  over this window, but metered injection is net of it, a standing caveat,
  unlike BR.
- **Single seed, single test year.** The handoff's noise-floor concern applies:
  a second test year would make the fixed-vs-season near-tie conclusive rather
  than suggestive.
