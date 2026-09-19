# Roughness temporal treatment: the per-timestep derivation is adopted

**Reproduction record, added 2026-09-18.** Driver:
`scripts/studies/method-roughness-treatment/roughness_treatment_study.py`.
Until 2026-09-18 it was in `scripts/analysis/`, the path any command below
uses; `scripts/studies/README.md` maps each old path to its new one. Numbers:
the R0 and R1 run manifests under `output/roughness_treatment_2026-09-12/`
record commits `48192c5` and `1ff1d54`, both clean trees. The comparison under
its `analysis/` records none; the driver's last commit before it was written is
`1ff1d54`, and rerun at `51807f8` the driver reproduces those files byte for
byte (`tests/test_pin_bootstrap_reproduction.py`).

**Date:** 2026-09-12
**Scope:** which temporal treatment of the derived surface roughness PyVWF
uses. Pre-registered in `method-roughness-treatment-prereg.md`; the background
is `../design/roughness-temporal-treatment.md`. Terms follow `CONTEXT.md`.

**The per-timestep derivation is adopted as the method. It is adopted on
method fidelity and comparability, not on accuracy.** The accuracy effect was
measured on Denmark, the registered gate resolved, and it is 0.0002 in
corrected RMSE. That is too small to matter, and it is not the reason for the
change.

## The conditions

Every region derives the roughness length by inverting the log wind profile
between the 10 m and 100 m winds. The difference between the rows is not the
formula, it is whether the result varies in time. **At the time of this
comparison** eleven rows applied a single annual mean of it and six applied it
hour by hour. That split is history: the per-timestep derivation was adopted
below and rolled out, and every scorecard row now reads `per timestep`.

| Condition | Roughness applied |
|---|---|
| R0 | the stored annual mean in the European files, which is what these rows run on today |
| R1 | derived from the same files' 10 m and 100 m winds per timestep, then averaged to daily with the winds |

No new data: both conditions read the same European files, and everything else
is held at each row's scorecard configuration.

- **DK.** Turbine-level, k=100, `season`, trained 2015 to 2019, tested on the
  single held-out year 2020, 5,410 units scored. Combined library at
  `input/combined/reference/power_curves.csv`
  (sha256 `689cfee7…`, 236 curves): 87.9% of fleet capacity runs on curves of
  `external` origin and 12.1% on curves matching the open library, so the row
  is not third-party reproducible. Substituted share 0.0.
- **FR.** Country-level, N=10, `fixed`, trained 2015 to 2021, tested on the
  single held-out year 2023. Bundled open library (sha256 `56314f39…`), and
  the substituted share is 1.00: every grid point's model key is absent from
  that library, so every one is simulated on the table's first column, a
  100 kW distributed-wind curve. That is a standing property of the FR row and
  is identical in both conditions.

Both rows opted in to extrapolation, in both conditions, under deviation D2 of
the pre-registration. DK: 0.597% of test-fleet capacity and 0.515% of
training-fleet capacity, recorded identically in all four runs. FR: zero.

Runs: `output/roughness_treatment_2026-09-12/`, all four at commit `48192c5`
with `git_dirty: false`, each manifest recording the treatment it applied.

## The full metrics tables

Denmark, `evaluate-2020-R0/metrics.csv` and `evaluate-2020-R1/metrics.csv`,
fleet scope, every row of each file:

| Condition | Variant | num_clu | time_res | MBE | MAE | RMSE | r | EMD | Units | Samples |
|---|---|---|---|---|---|---|---|---|---|---|
| R0 | uncorrected | 1 | none | +0.1097 | 0.1118 | 0.1467 | 0.793 | 0.1097 | 5410 | 64090 |
| R0 | affine-wind | 100 | season | +0.0231 | 0.0552 | 0.0855 | 0.833 | 0.0266 | 5410 | 64090 |
| R1 | uncorrected | 1 | none | +0.1118 | 0.1137 | 0.1482 | 0.790 | 0.1118 | 5410 | 64090 |
| R1 | affine-wind | 100 | season | +0.0223 | 0.0551 | 0.0853 | 0.833 | 0.0260 | 5410 | 64090 |

France, `evaluate-2023-R0/metrics.csv` and `evaluate-2023-R1/metrics.csv`,
national scope, every row of each file. The row the scorecard reports is
`fixed_10`:

| Condition | Variant | num_clu | time_res | MBE | MAE | RMSE | r |
|---|---|---|---|---|---|---|---|
| R0 | uncorrected | 1 | none | +0.16484 | 0.16484 | 0.17122 | 0.9876 |
| R0 | affine-wind | 1 | fixed | +0.00730 | 0.00918 | 0.01245 | 0.9945 |
| R0 | affine-wind | 10 | fixed | +0.00611 | 0.00918 | 0.01220 | 0.9937 |
| R0 | affine-wind | 1 | season | +0.00682 | 0.00926 | 0.01256 | 0.9932 |
| R0 | affine-wind | 10 | season | +0.00630 | 0.00970 | 0.01296 | 0.9921 |
| R1 | uncorrected | 1 | none | +0.16478 | 0.16478 | 0.17111 | 0.9876 |
| R1 | affine-wind | 1 | fixed | +0.00738 | 0.00924 | 0.01249 | 0.9945 |
| R1 | affine-wind | 10 | fixed | +0.00620 | 0.00918 | 0.01223 | 0.9938 |
| R1 | affine-wind | 1 | season | +0.00689 | 0.00930 | 0.01258 | 0.9933 |
| R1 | affine-wind | 10 | season | +0.00637 | 0.00972 | 0.01298 | 0.9922 |

`excluded_share` is zero in all four runs, so every variant of every run is
scored on all of its rows.

**Fit quality.** No offset failed anywhere. Neither DK run is degenerate:
scalars span 0.538 to 1.152 under R0 and 0.526 to 1.135 under R1, inside the
0.2 to 3.0 bounds. The FR row reported here, `fixed_10`, is not degenerate
either (scalars 0.516 to 2.022 under R0, 0.516 to 2.027 under R1). **FR's
unreported `season_10` variant is degenerate in both conditions**, carrying one
implausible scalar (3.070 under R0, 3.062 under R1) in cluster 7. It is listed
in the table above because the pre-registration commits to reporting every row
of the cited files, and it enters the common-row scoring of both conditions
equally; nothing was excluded on its account.

## The paired comparison

Procedure B of the curve library study: 1,000 paired draws, seed 20260911, 95%
percentile intervals, resampling units for DK and months for FR, with all four
frames of a row scored on the rows common to all of them. Each condition's
rebuilt point metrics reproduce its own `metrics.csv` to 1e-12 before anything
is resampled. Data: `output/roughness_treatment_2026-09-12/analysis/`.

| Row | Quantity | Estimate | 95% interval | Width |
|---|---|---|---|---|
| DK | corrected RMSE, R1 minus R0 | **-0.00019** | -0.00031 to -0.00007 | 0.00024 |
| DK | uncorrected RMSE, R1 minus R0 | +0.00149 | +0.00140 to +0.00159 | 0.00019 |
| DK | R0 correction gain | 0.0613 | 0.0593 to 0.0632 | 0.0038 |
| DK | R1 correction gain | 0.0629 | 0.0610 to 0.0649 | 0.0039 |
| FR | corrected RMSE, R1 minus R0 | +0.000032 | -0.000041 to +0.000088 | 0.00013 |
| FR | uncorrected RMSE, R1 minus R0 | -0.00011 | -0.00026 to +0.00008 | 0.00034 |
| FR | R0 correction gain | 0.1590 | 0.1381 to 0.1788 | 0.0407 |
| FR | R1 correction gain | 0.1589 | 0.1379 to 0.1785 | 0.0407 |

**G1 resolved: R1 better.** The DK interval excludes zero.

**G2: consistent with zero, and uninformative.** The FR interval is narrow, so
this is not an underpowered null, but France cannot show the treatment at all;
the section below says why.

**G3 was withdrawn on 2026-09-12, before any DK result existed** (deviation D1
of the pre-registration).

**P1 is refuted.** It predicted that DK's difference would be
indistinguishable under G1. The interval excludes zero, so the gate resolved
instead. The prediction was wrong and this record says so.

## Why R1 is adopted, and it is not the RMSE

None of the three reasons is accuracy.

1. **Per-timestep is the published method.** The method derives z0 inside the
   simulation sequence, and six of the seventeen rows already do that. The
   annual mean is an undocumented divergence that nobody chose: it arrived
   with the pre-combined European files and was never recorded as a decision.
   Restoring the derivation removes a permanent caveat from every future
   paper, which otherwise has to state that the equation it gives is not the
   one that ran.
2. **What was measured is the effect at the most exposed row there is, and
   most rows cannot show it.** DK's median unit stands at 45 m, the lowest of
   any row, and even there the effect is 0.0002. Four rows sit at exactly
   100 m, where the treatment is arithmetically inert. So 0.0002 is an upper
   bound drawn from the one place the question can be asked, not a typical
   effect. A future region whose fleet sits at 30 m is outside everything
   measured here, and should inherit the principled method by default rather
   than the accidental one.
3. **Uniformity.** Eleven rows on one treatment and six on another confounds
   every comparison between regions in this repository, including the transfer
   work and the physics-informed study. Removing that split is worth more than
   0.0002 in either direction.

## What 0.0002 is, and what it is not

It is a real difference, resolved by the registered gate, and it is not a
material one.

- The turbine-level screen used elsewhere in this project is 0.002, ten times
  larger. A difference below it would not be called material anywhere else
  here.
- **That screen was never established as a limit either.** It rests on ten
  null clustering A/B pairs whose largest absolute difference in MAE was
  0.0018, with a second test year and seed repeats named as prerequisites and
  never done. So the comparison above is between a number too small to matter
  and a threshold that was never validated.
- The gate resolved because a paired design on 64,090 rows resolves almost
  anything. **That is power, not importance.** The same machinery would
  resolve a difference an order of magnitude smaller. Nothing about a resolved
  interval says the difference is worth acting on, and the decision to act
  here rests on the three reasons above, not on this number.

A reader must not take "resolved" for "material".

## Most rows cannot show this at all

The roughness reaches a simulated capacity factor by one route, the hub-height
profile `w(h) = w100 ln(h / z0) / ln(100 / z0)`. At 100 m the factor is exactly
1 whatever z0 is, so the roughness cancels, and the sensitivity grows with the
distance from that reference. The full table is in
`../design/roughness-temporal-treatment.md`.

Four rows put every unit at exactly 100 m: AU-NEM and BR at turbine level, and
the BE and SE country grids. For those the two treatments are arithmetically
identical. Every country-level region gives its grid points one uniform height
between 80 and 100 m, where the whole span from smooth water to broken forest
is worth at most 1.3% of the speed, so no country-level row can test the
question. That is what makes France's null uninformative rather than
supporting, and it is a property of the grids, not of France.

**The split between the treatments does not line up with which rows can show
it.** AU-NEM and BR already apply the per-timestep treatment and are inert;
DK, which carries nearly all of the scorecard's exposure, is on the annual
mean. Any
attempt to read the scorecard for the effect of the treatment will therefore
find nothing, whichever rows it compares.

## What the correction does with a time-varying roughness

This is the more interesting result, and it is not the gate.

| DK | R0, annual mean | R1, per timestep | R1 minus R0 |
|---|---|---|---|
| Uncorrected RMSE | 0.1467 | 0.1482 | **+0.0015** |
| Corrected RMSE | 0.0855 | 0.0853 | **-0.0002** |
| Correction gain | 0.0613 | 0.0629 | +0.0017 |

R1's raw simulated capacity factors are **worse**, by 0.0015 and with the
interval excluding zero, and its uncorrected MBE is higher too (+0.1118
against +0.1097). Its corrected ones are better, and the correction gain
rises. The sign flips between the two lines.

So the fit is not absorbing input noise, which is what P4 anticipated. It is
exploiting temporal structure that the annual mean destroys: a roughness that
varies through the year gives the affine pair, fitted per cluster and season, a
signal that co-varies with what it is correcting, and the fit converts a worse
starting point into a better finish. That is an observation about what the
correction actually does, and it is the strongest reason in this document to
think the treatment is more than bookkeeping.

**Say plainly what this is not.** It is not evidence that R1's raw winds are
better. They are measurably worse. Anything that consumes uncorrected PyVWF
output, rather than corrected output, gets a worse input from this change on
the only row where it was measured.

## What R1 costs, in scored steps

The hourly roughness is undefined under vanishing or inverted shear, while an
annual mean always has a value, so R1 was expected to lose steps R0 keeps
(P3). It does, barely, and only at DK:

| Row | Variant | Condition | Unit-months partly scored | Wholly missing | Off-curve below share |
|---|---|---|---|---|---|
| DK | uncorrected | R0 | 47 | 0 | 1.630e-05 |
| DK | uncorrected | R1 | 49 | 0 | 1.632e-05 |
| DK | affine-wind season_100 | R0 | 0 | 0 | 0.0 |
| DK | affine-wind season_100 | R1 | 0 | 0 | 0.0 |
| FR | uncorrected | both | 0 | 0 | 0.0 |
| FR | affine-wind fixed_10 | R0 | 9 | 0 | 3.440e-05 |
| FR | affine-wind fixed_10 | R1 | 9 | 0 | 3.440e-05 |

Two unit-months out of a fleet of 5,410, and no excluded rows in either
condition. The loss is real and negligible. It would not be negligible in
complex terrain, where the estimator is undefined for whole months
(`../design/undefined-roughness-in-complex-terrain.md`), and no row tested here
sits in such terrain.

**P3 held for the wrong reason.** It predicted that R1 would lose steps R0
keeps, because the hourly roughness is undefined in some hours, and DK's count
did rise. But the losses sit in the **uncorrected** variant, and DK's corrected
variant loses nothing at all, so the extra steps never reach the score the gate
is computed on. At FR it is the other way round: the uncorrected variant loses
nothing and the corrected `fixed_10` loses nine unit-months, identically in
both conditions, which is the affine pair pushing speeds below the curve and
has nothing to do with the roughness. The prediction's number came out right
and its mechanism did not.

## There are three routes to the roughness, and the record sees two

The comparison above is between routes A and B of three:

| Route | How z0 is produced | What the file carries | Rows |
|---|---|---|---|
| A, annual mean | hourly z0 averaged over the year into one static field | hourly winds and a stored `z0` | DE, DK, UK and the eight country rows: 11 |
| B, per timestep at load | derived hour by hour in `prep_era5`, averaged to daily with the winds | hourly winds only | AU-NEM, NZ, CL, AR: 4 |
| C, per timestep, stored daily | derived hour by hour in `scripts/era5/combine.py` and averaged to daily there | daily winds and roughness, no 10 m winds | US, BR: 2 |

B and C are the same treatment computed at different stages, so the adopted
method is what routes B and C already produce. Two consequences of C had not
been named before this work:

- **The manifest cannot tell A from C.** `era5_roughness.applied` says
  `stored` whenever the file carries a roughness field, so the US and Brazilian
  rows, which are on the per-timestep treatment, carry the same label as the
  European rows, which are not.
- **Route C cannot answer this question at all.** `combine.py` drops the 10 m
  winds, so `roughness = "derived"` raises on those files. This comparison
  could not have been run on the US or Brazil whatever their hub heights, and
  neither could any future one, on that input.

Both are logged as candidate work and neither is started: whether the manifest
should record which kind of stored roughness a run applied, and whether the US
and Brazilian rows should move to the raw-monthly route so they can answer the
question. The design note carries the detail
(`../design/roughness-temporal-treatment.md`).

## Consequences

- **The extended ERA5 download generates the per-timestep treatment, by
  carrying no roughness at all.** The new files hold the hourly 10 m and 100 m
  winds and nothing else, so `prep_era5` derives z0 per timestep at load and
  averages it to daily with the winds, exactly as it does for AU-NEM, NZ, CL
  and AR. There is no combine step: the raw monthly download is what the
  configurations read. Storing a daily roughness instead, as
  `scripts/era5/combine.py` does for the US and Brazil, would produce the same
  numbers under a label the manifest cannot distinguish from the annual mean,
  and would drop the 10 m winds that any future comparison needs.
- **The eleven European rows are re-run on it after the download.** Each
  becomes a new row with its own configuration, and the scorecard states the
  treatment per row. The three suspended rows return in the same pass, since
  they need the same wider box.
- **The new files are `era5/EU_2026-09`**: hourly winds and no stored
  roughness at all, so `prep_era5` derives per timestep and the manifest
  records `derived` rather than a stored field that is not an annual mean.
- **`era5/EU` is kept, unchanged**, so the rows published today stay
  reproducible against the input they were produced from. The scorecard states
  the treatment per row, and the design note maps each ERA5 directory to its
  route.
- **The thesis disclosure stands as history.** Its method section gives the
  equation and is silent on the temporal treatment; the runs behind it used
  the annual mean, and that does not change retrospectively.

## Caveats

- Every result here rests on a single held-out test year per row, 2020 for DK
  and 2023 for FR, and everything is screening-level.
- **One turbine-level row decides the method.** DK is the only row in the
  scorecard with the hub heights to show the effect, so there is no
  independent replication of the direction, and no turbine-level row tested
  the treatment in complex terrain.
- DK's figures in both conditions include 0.6% of test capacity simulated from
  extrapolated winds, identically in both, so the paired difference is
  unaffected and the absolute figures are not.
- The choice of `k=100` and `season` for DK, and of `fixed_10` for FR, sits
  outside the resampling: the interval covers the sampling of units and
  months, not the choice of configuration.
- No figure in the scorecard changes on this comparison. The re-run does that,
  and it reports its own numbers.
- Nothing here is evidence about running at native hourly resolution. Both
  conditions average to daily, which is the published resolution.
