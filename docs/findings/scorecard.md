# Multi-region validation scorecard

**Date:** 2026-08-24 (refreshed; first recorded 2026-07-24)
**What this is:** one place that states, per region, how much PyVWF's affine
wind-speed correction reduces the error between ERA5-simulated and observed wind
capacity factors, on a held-out year. Every number is read from a
`metrics.csv` under `output/validation/`; source paths are given so each is
auditable. This is screening-level validation, one test year per region, not an
accredited yield assessment.

> **Refreshed 2026-08-24.** Every row in both tables below was re-run on
> current HEAD (v0.4.0, commit `41462e9`, clean tree) against the same
> observations, the same byte-identical curve libraries and the same
> configuration as the original runs. The figures here are the refreshed ones.
>
> **Why it needed doing.** The original runs were made at commit `df2e81f` with
> `git_dirty: true`, so they were not reproducible from any commit, and they
> predated four commits that touch fitting, including `53d667b`, whose own note
> says it "changes fitted scalars on any fleet with incomplete reporting".
>
> **What the refresh changed: almost nothing in aggregate.** Seven of the nine
> turbine rows and seven of the eight country rows reproduced to four decimal
> places. Three figures moved at the reported precision: Chile's corrected RMSE
> 0.104 to 0.105, the United States' 0.098 to 0.097, and Norway's uncorrected
> baseline 0.025 to 0.034. That last one is the largest single change and is
> consistent with the region-shape and country-estimator fixes that landed
> after the original runs; Norway's qualitative finding is unchanged, and the
> correction still makes it worse.
>
> **What the refresh did change, and it matters:** the United States' worst
> fitted wind scalar more than doubled, from 20.54 to 46.39, while its headline
> RMSE improved by 0.0005. That is exactly the failure mode `53d667b` predicted
> for a fleet with incomplete reporting, and it is invisible in the skill
> metric. Four rows rest on degenerate fits, marked with a dagger below.
>
> Refreshed runs, with manifests: `output/validation/refresh_2026-08-24/`.

## Turbine / plant-level (observed capacity factor per farm)

Matched real turbine curves and hub heights; k-swept affine fit; best held-out
`affine-wind` row (fleet scope). Source: `refresh_2026-08-24/<CODE>/`, which
carries the region config used and a manifest per run. The originals these
replace were `cluster_sweep_2026-07-24/<CODE>/`, except CL
(`cl_matched_2026-07-24`) and AR (`ar_capfix3_2026-07-24`). Curve library per
region matches the original: external (licensed) for all but CL and AR, which
use the bundled open library, verified by sha256 on both sides.

| Region | Fleet (test) | Train → test | Uncorr RMSE | Corr RMSE | Uncorr MBE | Corr MBE | Corr r | Best cfg |
|---|---|---|---|---|---|---|---|---|
| Germany (DE) | 4814 turbines | 2015-18 → 2019 | 0.086 | **0.057** | +0.042 | +0.001 | 0.85 | k100 fixed |
| Denmark (DK) | 5410 turbines | 2015-19 → 2020 | 0.147 | **0.085** | +0.110 | +0.023 | 0.83 | k100 season |
| Brazil (BR) | 151 complexes | 2021-23 → 2024 | 0.139 | **0.105** | -0.046 | -0.015 | 0.72 | k60 fixed † |
| United States (US) | 520 plants | 2019-21 → 2022 | 0.110 | **0.097** | +0.022 | +0.024 | 0.79 | k250 fixed † |
| Australia (AU-NEM) | 77 farms | 2020-22 → 2023 | 0.115 | **0.094** | +0.009 | -0.006 | 0.61 | k45 season |
| United Kingdom (UK) | 348 turbines | 2015-18 → 2019 | 0.145 | **0.115** | +0.037 | -0.038 | 0.70 | k50 fixed |
| New Zealand (NZ) | 12 farms | 2019-23 → 2024 | 0.157 | **0.106** | -0.062 | +0.021 | 0.66 | k7 fixed |
| Chile (CL) | 59 plants | 2021-23 → 2024 | 0.123 | **0.105** | -0.026 | -0.002 | 0.43 | k10 fixed † |
| Argentina (AR) | 59 plants | 2021-23 → 2024 | 0.151 | **0.133** | +0.010 | +0.001 | 0.44 | k10 fixed † |

**† The fit behind this row is degenerate.** `fit_quality` run against the exact
factors file each row reports, with the calibrated bounds (scalar in 0.2 to 3.0,
offsets required to converge):

| Region | Config | Max scalar | Implausible scalars | Failed offsets |
|---|---|---|---|---|
| Chile (CL) | k10 fixed | **80.23** | 3 | **1** |
| United States (US) | k250 fixed | **46.39** | 5 | 0 |
| Argentina (AR) | k10 fixed | **15.53** | 1 | 0 |
| Brazil (BR) | k60 fixed | **4.82** | 2 | 0 |

The other five are clean: DE 2.79, AU-NEM 2.64, UK 1.86, NZ 1.81, DK 1.15, all
inside the ceiling with no failed offsets. Every country-level fit in the table
below is also clean. These figures come from the refreshed runs and now travel
in `metrics.csv` automatically, so a future run cannot hide them.

Chile is the worst and is documented in `method-scalar-bounds.md`, which
shows no `min_cluster_size` setting satisfies all three of its gates: raising it
to 3 clears the failed offset and still beats uncorrected (0.1069 against
0.1226) but leaves a scalar of 39.3, and raising it to 5 collapses the
correction entirely (0.2366 against an uncorrected 0.1226). The same document
reports the exported Chile field flagging 717 of 2,697 grid cells, 26.6%, as
degenerate. A corrected RMSE of 0.104 that contains a wind scalar of 80 is not a
result to quote without this context.

Read this as: across four continents, on fleets the model never saw, the
correction lowers capacity-factor RMSE and drives the systematic bias (MBE)
toward zero in every case. It is strongest where the raw ERA5 bias is a
clusterable wind-speed offset (DE, DK, BR), and weakest where the residual is
not a wind-speed error at all (CL, AR; see caveats).

That reading holds for the five clean rows. For the four daggered ones the
aggregate improved while individual clusters did not: an RMSE that improves
while the fit contains a scalar of 80 is telling you the fleet average absorbed
the damage, not that the correction is sound everywhere it was applied. The
per-cluster factors from those four rows should not be used, and the gridded
exports built from them carry a `degenerate` layer for exactly this reason.

## Country-level (ENTSO-E national aggregate, 2023)

Capacity-weighted national monthly CF, held-out 2023.
Source: `refresh_2026-08-24/<CODE>/`, replacing `country_baseline_2026-07-23/`.
Bundled open curve library throughout.

| Region | Uncorr RMSE | Corr RMSE | Uncorr MBE | Corr MBE | Best cfg |
|---|---|---|---|---|---|
| France (FR) | 0.171 | **0.012** | +0.165 | +0.006 | N=10 fixed |
| Belgium (BE) | 0.340 | **0.020** | +0.337 | -0.002 | N=3 season |
| Spain (ES) | 0.135 | **0.026** | +0.130 | +0.016 | N=4 fixed |
| Ireland (IE) | 0.172 | **0.021** | +0.168 | +0.009 | N=1 season |
| Sweden (SE) | 0.088 | **0.030** | +0.084 | -0.027 | N=4 fixed |
| Italy (IT) | 0.066 | **0.034** | +0.062 | -0.020 | N=3 season |
| Portugal (PT) | 0.110 | **0.074** | -0.097 | +0.029 | N=1 season |
| Norway (NO) | 0.034 | 0.039 | +0.024 | -0.030 | correction does not help |

The country-level fit removes very large mean biases (FR, BE, ES, IE all from
0.13-0.34 down to ~0.01-0.03). Two honest notes: NO is already close to
unbiased uncorrected (RMSE 0.034) and the correction makes it worse (0.039); and the country method
fits under-determined offsets against one national series per month, so the
offsets largely repair the scalar's cube-law overshoot rather than a genuine
additive spatial bias (`method-country-level.md`).

## What must NOT be overclaimed

- **Four of the nine turbine-level rows rest on degenerate fits** (CL, US, AR,
  BR). The aggregate metric is real; the underlying per-cluster factors are not
  usable. Chile carries a fitted wind scalar of 80.23 with one offset that never
  converged, and the United States carries 46.39. Neither is visible in the
  skill metric, which is the point.
- **An aggregate that barely moves does not mean the fit did not move.** The
  2026-08-24 refresh changed the United States' headline RMSE by 0.0005 while
  more than doubling its worst fitted scalar. Check `fit_quality`, not the
  metric, when judging whether a correction is sound.
- **One test year per region.** Orderings of two close configs are not
  meaningful; the headline is the uncorrected-to-corrected drop, not the exact k.
- **CL and AR are caveated, not headline wins.** ERA5 exaggerates the
  north-south wind gradient in both (Atacama too calm, Patagonia too windy). The
  correction removes the mean bias but adds limited skill; for CL it only helps
  once real turbine curves are used, and CL is unstable at low k. The k=10 row
  reported above is itself degenerate, so "use k=1 or k>=8" is not sufficient
  guidance: check `fit_quality` rather than the cluster count. AR is usable
  only after its capacity denominators were rebuilt; a northern cluster still
  fits an extreme scalar that a higher-resolution wind product, not more data,
  would fix (`region-south-america.md`).
- **NO gets worse; NL is excluded** (an ENTSO-E coverage defect makes its CF
  series unusable). Reporting either as a corrected region would be false.
- **US carries an unscreened curtailment confound** (ERCOT/SPP); its near-zero
  fleet MBE is partly an aggregation artefact.
- **AU-NEM's "does correction help?" is config-dependent** (it improves the
  fleet seasonal cycle but can worsen absolute farm RMSE in curtailed South
  Australia); the table row is the matched-curve k-swept result.
- Screening-level throughout: not MEASNET/DNV-accredited, not investment advice.

## Data provenance

Turbine/plant observations by region: DK Danish Energy Agency; DE public
turbine register; UK REPD/Ofgem; US EIA-923; BR ONS; AU-NEM AEMO; NZ EMI; CL
Coordinador (CEN); AR CAMMESA (capacities rebuilt from turbine specs).
Country-level: ENTSO-E Transparency. All correction operates on ERA5 at 0.25deg.
Confidential inputs (WindStats DE/ES, Ofgem certificate warehouse, licensed
curve library) are not redistributed and are not required to reproduce the open
rows above, which run on the bundled open curve library.
