# The eleven European rows on the per-timestep roughness and the wider ERA5 box

**Reproduction record, added 2026-09-18.** Driver:
`scripts/studies/method-eu-rerun/era5_overlap_check.py`. Until 2026-09-18 it
was in `scripts/analysis/`, the path any command below uses;
`scripts/studies/README.md` maps each old path to its new one. Numbers: the
re-run manifests under `output/eu_rerun_2026-09-12/` record commit `b5d47d0`,
all 26 with a clean tree. The overlap check's output under
`output/era5_overlap_2026-09-12/` records no commit; the driver's last commit
before it was written is `3898bdc`. The paired comparisons come from
`scripts/analysis/eu_rerun_compare.py`, a tool that stays in place and
reproduces them byte for byte.

**Date:** 2026-09-13
**Scope:** what changed when the eleven scorecard rows reading `era5/EU` were
re-run on `era5/EU_2026-09`. Pre-registered in `method-eu-rerun-prereg.md`; the
method decision it implements is `method-roughness-treatment.md`. Terms follow
`CONTEXT.md`.

**Correction notice, 2026-09-25: the country-level fits with more than one
cluster used the per-cluster solver, not the joint national fit.**
`country_obs_is_per_cluster` read the rounding differences between clusters'
capacity-weighted means of one national observation as distinct zonal
observations, so every national fit with N greater than 1 fitted each cluster's
offset alone against the national series (fixed in `ac26f6a`;
`docs/findings/scorecard.md`, notice of the same date, has the details and the
joint-fit figures of the eight scorecard rows). Scalars, uncorrected figures and
N=1 fits are unaffected. Affected here, not re-measured: the headline, the
table's FR, BE, SE, NO, ES and IT rows, the failed-offset and below-zero shares
in the fit quality (the scalar ranges stand), the FR and BE treatment
differences, the SE and NO decomposition, the ES and IT rows and the Italy
section's N=3 rows and fitted pairs, gates G2 and G5, predictions P3 to P5 and
the related caveat. Standing: PT and every N=1 row, and the extent result the
wider download resolved.

**Two things changed, and the plan measured them apart. The roughness treatment
moved every row by less than 0.0002 in corrected RMSE. The wider box returned
Spain, Italy and Portugal from suspension, and their figures improve by up to
two thirds, because their winds are now real rather than extrapolated up to
five degrees past the data.** The three rows return to the scorecard, none of
them daggered. One registered prediction is refuted and its reasoning was
wrong in a way worth recording.

## What changed, and what did not

Each new row's configuration differs from the row it replaces in exactly three
settings: the ERA5 path, the file tag, and `roughness = "derived"`. Denmark
changes a fourth, `allow_extrapolation`, registered in advance: its box stops
at 13.5°E and Bornholm lies near 14.9°E, so the wider download does not reach
it and the limit is the box rather than the data.

Everything else is held, and was checked rather than assumed. For the three
returning rows and Norway, published against new: identical grid points by id,
identical timestamps, identical fleet capacity, coordinates and hub heights,
the same cluster counts, the same curve library by sha256, the same
substituted share, and twelve scored months on both sides.

**G0 passed on its first branch before any row was re-run.** The old and new
downloads are bit-identical where they overlap: twelve samples across 2015,
2019 and 2023, 288 cells by up to 744 hours each, maximum absolute difference
0.000e+00 in all four wind components, coordinates matching exactly
(`scripts/analysis/era5_overlap_check.py`, data in
`output/era5_overlap_2026-09-12/`). So a difference between a published row and
its re-run is the treatment or the extent, and nothing else.

Runs: `output/eu_rerun_2026-09-12/`, all thirteen at commit `b5d47d0` with
`git_dirty: false`, every manifest recording the roughness requested and
applied as `derived`, and `excluded_share` zero throughout.

## Where the improvement is

The suspended rows improve because their winds are real, and the evidence is
that nothing else moved. Splitting each fleet by whether a grid point lay
outside the extent its published configuration loaded:

| Row | Points outside the old extent | Mean uncorrected CF there | Mean uncorrected CF at points always inside | Uncorrected CF missing at the outside points |
|---|---|---|---|---|
| IT | 19 of 28 | 0.3556 to **0.2071** | 0.1358 to 0.1357 | 22.3% to **0** |
| ES | 34 of 52 | 0.4705 to **0.2019** | 0.2136 to 0.2134 | 19.0% to **0** |
| PT | 23 of 26 | 0.2083 to **0.2419** | 0.2560 to 0.2561 | 0% to 0 |
| NO | 3 of 26 | 0.5264 to **0.5487** | 0.3410 to 0.3420 | 5.6% to **0** |

Points that were always inside move in the fourth decimal, which is the
roughness treatment. Points that were outside move in the first. The published
runs also produced no capacity factor at all for a fifth of the steps at those
points, because extrapolated speeds landed off the power curve; that is now
zero everywhere.

## The full metrics tables

Every re-run row, its reported configuration and its uncorrected baseline,
from each run's `metrics.csv` under `output/eu_rerun_2026-09-12/new/`. Each
result is a single held-out test year: 2019 for DE and UK, 2020 for DK, 2023
for the country-level rows. Curve library by sha256: `689cfee7…`, the combined
library, for DE, DK and UK; `56314f39…`, the bundled open library, for the
eight country-level rows, whose substituted share is 1.00 as it has always
been, every grid point falling back to the first column of `power_curves.csv`.

| Row | Config | Uncorr RMSE | Corr RMSE | Uncorr MBE | Corr MBE | Corr r | Extrapolated share |
|---|---|---|---|---|---|---|---|
| DE | fixed_100 | 0.0861 | 0.0572 | +0.0426 | +0.0007 | 0.855 | 0 |
| DK | season_100 | 0.1482 | 0.0853 | +0.1118 | +0.0223 | 0.833 | **0.60%** |
| UK | fixed_50 | 0.1458 | 0.1147 | +0.0376 | -0.0383 | 0.704 | 0 |
| FR | fixed_10 | 0.1711 | 0.0122 | +0.1648 | +0.0062 | 0.994 | 0 |
| BE | season_3 | 0.3399 | 0.0201 | +0.3367 | -0.0024 | 0.983 | 0 |
| IE | season_1 | 0.1721 | 0.0212 | +0.1680 | +0.0092 | 0.985 | 0 |
| SE | fixed_4 | 0.0876 | 0.0298 | +0.0844 | -0.0272 | 0.990 | 0 |
| NO | fixed_4 | 0.0350 | 0.0363 | +0.0271 | -0.0279 | 0.968 | 0 |
| ES | fixed_4 | 0.0281 | 0.0262 | +0.0128 | +0.0112 | 0.990 | 0 |
| IT | season_3 | 0.0703 | 0.0168 | -0.0692 | -0.0031 | 0.971 | 0 |
| PT | season_1 | 0.0893 | 0.0274 | -0.0847 | +0.0176 | 0.966 | 0 |

Fit quality, from the same files. **No row has an implausible scalar and no
offset failed anywhere**, so no row carries a dagger:

| Row | Scalar range | Worst share of one cluster's training days sent below 0 m/s | Off-curve below share, reported variant | Unit-months partly scored |
|---|---|---|---|---|
| DE | 0.534 to 2.798 | 0.3426 | 4.29e-04 | 84 |
| DK | 0.526 to 1.135 | 0.0025 | 0 | 0 |
| UK | 0.255 to 1.865 | 0.0429 | 6.01e-04 | 1077 |
| FR | 0.516 to 2.027 | 0.0313 | 3.44e-05 | 9 |
| BE | 0.314 to 0.527 | 0 | 0 | 0 |
| IE | 0.626 to 0.704 | 0 | 0 | 0 |
| SE | 0.492 to 0.766 | 0 | 0 | 1 |
| NO | 0.755 to 1.315 | 0.0002 | 0 | 0 |
| ES | 0.930 to 1.199 | 0 | 0 | 0 |
| IT | 1.041 to 2.649 | 0.2435 | 1.03e-02 | 69 |
| PT | 1.485 to 1.655 | 0.0029 | 1.72e-03 | 11 |

## The treatment, measured

Published against new, scored on the rows common to both, 1,000 paired draws
at the seed of the curve library study, units resampled for the turbine-level
rows and months for the country-level ones. Both sides reproduce their own
`metrics.csv` to 1e-12 before anything is resampled.

| Row | Rows scored | Corrected RMSE published | new | Difference | 95% interval | Excludes zero |
|---|---|---|---|---|---|---|
| DE | 54,188 | 0.057164 | 0.057174 | +0.000010 | -0.000005 to +0.000026 | no |
| DK | 64,090 | 0.085486 | 0.085299 | **-0.000187** | -0.000315 to -0.000071 | **yes** |
| UK | 4,159 | 0.114619 | 0.114684 | +0.000065 | -0.000125 to +0.000271 | no |
| FR | 12 | 0.012202 | 0.012234 | +0.000032 | -0.000041 to +0.000088 | no |
| BE | 12 | 0.020081 | 0.020081 | 0.000000 | -0.000000 to +0.000000 | no |
| IE | 12 | 0.021177 | 0.021234 | +0.000057 | -0.000117 to +0.000206 | no |

**Denmark is the only row that resolves, and the hub-height geometry predicted
exactly that.** The roughness reaches a capacity factor only through
`ln(h/z0) / ln(100/z0)`, which is 1 at 100 m for any z0, so the treatment can
move a speed only in proportion to the distance from that reference
(`../design/roughness-temporal-treatment.md`). DK's median unit stands at 45 m,
the lowest fleet in the scorecard. Of the five rows that do not resolve, three
are country grids at a uniform 90, 100 and 85 m, where the treatment is inert
or nearly so; Germany's capacity sits at 93 m on average. The United Kingdom is
the exception, and it is a limitation rather than a tidy story: its fleet
averages 75 m, low enough that the geometry says it should show the treatment,
and its interval is the widest of the six, 0.000396 against Germany's 0.000030.
So the data cannot yet say. The prediction that the effect lives where hub
heights sit far from 100 m held, with the one row that could have tested it
hardest unable to.

**Where a future test would have power.** The difference between the two is
sample size, not geometry: Denmark resolved on 64,090 scored rows from 5,410
units, and the United Kingdom has 4,159 rows from 348 farms, fifteen times
fewer. If the treatment question reopens, the United Kingdom is where to ask
it, with more units or a second test year rather than a different region. On
the usual square-root scaling, the United Kingdom at Denmark's unit count would
carry an interval near 0.0001, which would resolve a difference the size of
Denmark's. That is a projection from the widths in the table above, not a
measurement.

**Belgium's difference is zero to every digit the bootstrap carries, and that
is not a coincidence.** All 44 of its grid points sit at exactly 100 m, where
the hub-height factor is 1 whatever the roughness is, and Belgium lay wholly
inside the old extent, so neither change can reach it. The underlying
uncorrected capacity factors are not literally bit-identical: they differ by at
most 6.9e-08, floating-point noise of a kind consistent with array shape
changing an accumulation order. **That residual has not been traced to its
source.** Tracing it would mean comparing the hub-height wind series at
Belgium's grid points between the two runs, and if those differ, the daily mean
taken over differently shaped sliced arrays is the first suspect, since the new
files carry a larger box and one variable fewer. **It is not worth tracing.**
It is 6.9e-08 in capacity factor, around 3e-07 of the values it sits in, and
the smallest difference this document reports, Sweden's treatment term at
3e-06, is forty times larger. Nothing here rests on it.

### Sweden and Norway, decomposed

Both rows change treatment and extent at once, so each was also run on the old
files with the derived roughness. All three conditions are scored on the rows
common to all three, so the two terms add to the total by construction.

| Row | Term | Estimate | 95% interval | Excludes zero |
|---|---|---|---|---|
| SE | treatment | -0.000003 | -0.000010 to -0.000000 | yes |
| SE | extent | +0.000041 | +0.000005 to +0.000079 | yes |
| SE | total | +0.000038 | +0.000001 to +0.000076 | yes |
| NO | treatment | -0.000039 | -0.000307 to +0.000279 | no |
| NO | extent | **-0.002551** | -0.004060 to -0.000876 | yes |
| NO | total | -0.002590 | -0.004012 to -0.000875 | yes |

**In both rows the extent term is the larger: fourteen times for Sweden,
sixty-five times for Norway.** Norway's corrected RMSE falls from 0.0389 to
0.0363 and its correction gain moves from -0.0051 to -0.0013. The correction
still does not help Norway, and the interval still includes zero, so the honest
reading is that it went from actively harmful to merely useless.

## Spain, Italy and Portugal: no treatment claim

Their published figures are not results, so there is nothing to difference
against, and the plan said so before any of this ran. Their figures are
reported for the record, against the published ones they supersede:

| Row | Uncorr RMSE published | new | Corr RMSE published | new | Corr MBE published | new |
|---|---|---|---|---|---|---|
| ES | 0.135 | **0.0281** | 0.026 | 0.0262 | +0.016 | +0.0112 |
| IT | 0.066 | 0.0703 | 0.034 | **0.0168** | -0.020 | -0.0031 |
| PT | 0.110 | 0.0893 | 0.074 | **0.0274** | +0.029 | +0.0176 |

Italy's corrected RMSE halves and Portugal's falls by two thirds. Spain's
corrected figure barely moves while its uncorrected RMSE falls by a factor of
five, from 0.135 to 0.028: the correction was absorbing a bias that the
fabricated winds had created.

## An ordinary affine fit drops calm days, and `fit_quality` does not see it

This is the finding that does not come from the treatment or the extent.

Italy's re-run passes every check this project has. No implausible scalar, no
failed offset, no dagger, real winds, a fleet wholly inside the loaded extent.
It still drops **1.0% of its capacity-weighted steps** and leaves 69 of its 336
grid-point-months scored on only part of their steps. The uncorrected variant
loses nothing at all: every loss is in the corrected variant, and every one is
below the curve, with no step above it and none missing a speed.

The cause is in the factors. All twelve of Italy's fitted pairs carry a
negative offset, so each has a zero-crossing speed below which the corrected
speed goes negative and falls off the power curve. The worst is cluster 0 in
summer, scalar 2.649 and offset -5.001 m/s, crossing at 1.89 m/s; the mildest
is cluster 1 in winter at 0.10 m/s. The dropped steps are calm days.

**So the mechanism behind the daggered-rows notice is not confined to
degenerate fits.** It was first seen where a fit was pathological, and it was
described there as what a degenerate pair does. It is what an ordinary affine
correction does at low wind speeds, and `fit_quality` cannot see it: it bounds
the scalar and checks that each offset converged, and every Italian pair passes
both.

**The drop rate is partly a property of the configuration, not of the data.**
The same Italian run, same winds, same fleet, across its four variants:

| Variant | Off-curve below share | Unit-months partly scored |
|---|---|---|
| fixed_1 | 4.47e-04 | 1 |
| season_1 | 2.80e-03 | 14 |
| fixed_3 | 3.02e-03 | 42 |
| season_3 | 1.03e-02 | 69 |

More clusters and more time slices means more fitted pairs, and each is another
chance to cross zero. A reader comparing two configurations of the same region
is therefore comparing two different rates of silent loss as well as two fits.

It is not confined to Italy either: Germany's reported variant drops 84
unit-months and the United Kingdom's 1,077, both on fleets wholly inside their
extents, both undaggered.

## Gates

| Gate | Outcome |
|---|---|
| **G0** | **Passed, first branch.** Bit-identical where the downloads overlap |
| **G1** | **Passed.** Every re-run row reports an extrapolated share of zero except DK's 0.60%, whose box is unchanged by design |
| **G2** | **Passed.** Every run clean at `b5d47d0`, no failed offset, substituted shares and curve libraries identical to the rows they replace |
| **G3** | Reported above. One of six resolves, DK, in favour of the new treatment; the rest are consistent with zero |
| **G4** | Reported above. The extent term is the larger in both rows, and the scorecard says so |
| **G5** | **Passed.** No cluster sends a majority of its capacity-weighted training days below 0 m/s: the worst shares are ES 0.0000, IT 0.2435, PT 0.0029, against a registered threshold of a half |

## Predictions

| # | Prediction | Outcome |
|---|---|---|
| P1 | G0 agrees | **Held.** Bit-identical, not merely within tolerance |
| P2 | Every row but DK reports a zero extrapolated share | **Held** |
| P3 | The treatment-only rows move by less than 0.002 in corrected RMSE | **Held.** The largest is DK at 0.000187 |
| P4 | ES and IT move the most, and their corrected RMSE gets **worse** | **Refuted.** They moved the most and got better |
| P5 | At least one returning row still carries a dagger, on fit quality | **Refuted.** None does |

**P4 was refuted because its reasoning was wrong, not because its number
missed.** The reasoning was mine. I predicted that the Spanish and Italian
figures would worsen because their published values were flattered by days
dropping out of the score, so scoring them properly would cost them. That
reasoned from the symptom. Those rows were not merely scored on a biased
sample: they were simulated from fabricated winds, extrapolated up to five
degrees past the data. Replacing fabricated input with real input improves
results. The dropped days were real and second-order, and the table at the top
of this document measures which of the two dominated. Inverting symptom and
cause is the same error the suspension notice was written to correct, made
again in the document that planned the correction.

**P5's refutation is the outcome nobody predicted.** Three rows suspended for
an input defect return with no implausible scalar, no failed offset and no
dagger between them. The pre-registration assumed at least one would still be
visibly broken once the input was sound. None is.

## Consequences

- The eleven new rows replace the published ones in the scorecard, which
  carries `per timestep` in the Roughness column for all of them.
- Spain, Italy and Portugal leave the suspended table and enter the
  country-level table. Their suspension notice gains a dated resolution rather
  than being deleted.
- The published rows move to a Superseded section with their figures, their
  input, their configuration and the date they were superseded.
- Sweden and Norway lose the § marker, since their fleets are now inside the
  loaded extent. Denmark keeps it, with its 0.60%.
- Whether `fit_quality` should bound the share of training steps a pair sends
  below zero is candidate work, and Italy is its motivating case: an ordinary
  fit, not a pathological one.

## Caveats

- Every result rests on a single held-out test year per row, and everything is
  screening-level.
- The treatment comparison is six rows, of which one resolves. It measures what
  the change did to these fleets, not what it would do to a fleet at 30 m.
- Norway's correction still does not help. The interval on its gain includes
  zero in every condition.
- The country-level rows still fit under-determined offsets against one
  national series per month, and still run every grid point on a fallback
  curve. Nothing here changes either.
- The returning rows have one sound test year each. Their published history
  under fabricated winds is kept in the Superseded section, and no trend should
  be read across the two.
