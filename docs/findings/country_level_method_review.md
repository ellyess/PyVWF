# Country-level method review

**Date:** 2026-07-23
**Branch:** multi-region-validation
**Scope:** the `obs_level = "country"` path, compared against the turbine-level
path, plus an audit of the nine ENTSO-E observation series on disk.

## Summary

The country-level path is wired consistently with the turbine-level path, but it
fits a different estimator under the same name, and its observations were never
checked. Two of nine regions could not load at all, and four had capacity-factor
series that are physically impossible. Fixes for the loading, the checking and
the spatial weighting are in this branch. The identifiability of the joint offset
fit is unchanged and remains open.

## 1. What each path fits

| | Turbine level | Country level |
|---|---|---|
| Scalar `α` | `calculate_scalar`: obs/sim over the same turbines, capacity-weighted, restricted to rows that reported | `cluster_train_set` country branch: obs is the one national CF, sim is the cluster mean |
| Offset `β` | `find_offset`: one observation, one `β`, solved exactly per (cluster, slice) | `find_offsets_country_level`: N cluster offsets against one national number, joint L-BFGS-B from `x0 = 0` |
| Constraints vs parameters | 1 observation to 1 free `β` per cluster. Determined. | 1 observation to N free `β`. Under-determined by N-1. |
| Clustering | `cluster_turbines(num_clu, ...)` runs | `clus_info = turb_info.copy()`; `num_clusters` is ignored |

The scalar definition is the substantive divergence. At turbine level `α` is a
local bias ratio. At country level

```
α_c = CF_national_observed / CF_c_simulated
```

so a cluster the reanalysis says is windier than the national mean receives
`α < 1` and a calmer cluster receives `α > 1`. That flattens the simulated
spatial field toward the national mean rather than correcting bias at each
location. Verified on the real NL grid: all five clusters receive the identical
observation and the scalars (0.785, 0.815, 0.807, 0.791, 0.803) track only the
spread in the simulation.

This compounds with something both paths share. `α` is a capacity-factor ratio
applied to a wind speed, and power goes roughly as `w³`. At turbine level that is
harmless because `find_offset` then drives `β` until the simulated CF equals the
observation exactly, so the pair is self-correcting. At country level nothing
re-fits per cluster, so the cube-law overshoot survives into the factors. It is
the mechanism behind the NL scalars of 0.22 to 0.50 in
[TURBINE_GRID_EVALUATION_ANALYSIS.md](TURBINE_GRID_EVALUATION_ANALYSIS.md).

`num_clusters` being ignored is verified directly: `cluster_train_set` with
`num_clu = 5` and `num_clu = 37` returns byte-identical frames. The output is
still written as `factors_<slice>_<n>.csv` and the manifest still records
`num_clu`, so the label can disagree with the content. A smoke train of ES writes
`factors_fixed_5.csv` containing 4 clusters.

## 2. Observation audit

A national fleet's sub-daily capacity factor must at some point approach the
fleet peak, and can never exceed 1. Both tests are source-free.

| Region | step | mean CF | peak CF | clipped | verdict |
|---|---|---|---|---|---|
| FR train | 1 h | 0.231 | 0.936 | 0 | pass |
| FR test | 1 h | 0.257 | 0.809 | 0 | pass |
| ES train | 1 h | 0.246 | 0.795 | 0 | pass |
| ES test | 15 min | 0.238 | 0.710 | 0 | pass |
| IT train | 1 h | 0.219 | 0.775 | 0 | pass |
| IT test | 1 h | 0.237 | 0.783 | 0 | pass |
| SE train | 1 h | 0.255 | 0.952 | 0 | pass |
| SE test | 1 h | 0.264 | 0.856 | 0 | pass |
| BE test | 1 h | 0.234 | 0.794 | 0 | pass |
| NO test | 1 h | 0.309 | 0.787 | 0 | pass |
| PT test | 1 h | 0.275 | 0.900 | 0 | pass |
| BE train | 1 h | 0.209 | **1.295** | 0 | CF above 1 |
| PT train | 1 h | 0.301 | **1.032** | 0 | CF above 1 |
| NO train | 1 h | 0.313 | **1.500** | 1 row | one clipped spike |
| **NL train** | 15 min | **0.075** | **0.569** | 0 | **not usable** |
| **NL test** | 15 min | 0.141 | **0.380** | 0 | **not usable** |
| **IE train** | 30 min | **0.480** | **1.500** | 500 rows | **not usable** |
| **IE test** | 30 min | **0.669** | **1.500** | 999 rows | **not usable** |

NL's generation peaks at 1528 MW against a stated 2646 to 7674 MW of capacity
across seven years, and implies 2.8 TWh/yr against a real Dutch wind output of
roughly 11 to 12 TWh/yr. IE has capacity frozen at 1919 MW across both splits,
which inflates every derived capacity factor and drives 5.7% of the test rows
onto the fetcher's `clip(0, 1.5)` ceiling.

BE, PT and NO fail on isolated spikes above 1 rather than on a systematic error.
Their means are plausible and their test splits are clean.

The correlation with the recorded results matters: the regions the earlier
evaluation called good (FR, ES, BE, NO) are the ones whose series pass, and NL,
the catastrophic one, has the worst series. **The country-level failures observed
so far are more attributable to the observation construction than to the
correction maths.** That has to be separated before any methodological conclusion
is drawn from them.

## 3. Spatial weighting

Three places aggregate to "country" and they did not agree. `cluster_train_set`
took an unweighted mean, while `find_offsets_country_level` and the harness's
`_country_skill` both capacity-weighted. On a grid whose points all carry the
same synthetic capacity the two agree, which is why the inconsistency stayed
invisible.

The deeper problem is that the uniform grid makes "capacity-weighted" mean
point-count weighted, that is **land-area weighted**, while the observation is
national generation over national installed capacity, that is **fleet weighted**.
Real fleets are concentrated, so the gap is absorbed into `α` and `β` as if it
were reanalysis bias. The turbine path has no equivalent gap: observation and
simulation sit on the same machines.

Measured against the Global Wind Power Tracker fleet as of 2021:

| Region | grid points | with no fleet | largest cluster reweighting |
|---|---|---|---|
| NO | 338 | 313 | 66.4 pp |
| BE | 105 | 64 | 38.5 pp |
| PT | 60 | 34 | 35.6 pp |
| IT | 110 | 83 | 25.6 pp |
| FR | 468 | 302 | 24.7 pp |
| IE | 70 | 34 | 23.5 pp |
| SE | 72 | 35 | 23.4 pp |
| NL | 165 | 116 | 19.7 pp |
| ES | 84 | 33 | 13.3 pp |

Half to nine tenths of every grid carried no wind capacity at all, so the
simulated country average was mostly an average over empty countryside. In
France, one cluster holds 34.9% of the fleet on 10.3% of the area while another
holds 0.1% of the fleet on 10.5% of the area. In Norway two entire clusters
contain no operating wind capacity.

## 3a. What NL and IE actually need

Both are fetch-side and neither is repairable by rescaling what is on disk.

**NL.** The annual mean capacity factor climbs from 0.043 in 2015 to 0.188 in
2021 while installed capacity climbs 2646 to 7674 MW. Generation rises 12.6x
against a 2.9x rise in the denominator:

| year | mean gen (MW) | max cap (MW) | CF |
|---|---|---|---|
| 2015 | 115 | 2646 | 0.043 |
| 2017 | 209 | 3300 | 0.063 |
| 2019 | 370 | 3675 | 0.101 |
| 2021 | 1439 | 7674 | 0.187 |

That is a generation series covering a growing share of the fleet the capacity
counts, not a constant scale error. Multiplying the series by a constant cannot
fix it, which rules out the simplest reading of the peak-CF failure.

**IE.** Generation looks sane (746 to 1086 MW, growing 1.45x). Installed
capacity takes exactly two values across seven years, 1907.13 and 1918.73 MW,
a total movement of 0.6%, while Ireland's real wind fleet went from roughly
2.5 GW to 4.3 GW. Every derived capacity factor is inflated by the ratio, which
is why 5.7% of the test rows sit on the fetcher's clip ceiling.

### Refetch and repair

Both were refetched with `--psr-type all` (the stored files were onshore-only),
which changed NL substantially and IE not at all. The consistency guard added to
`calculate_capacity_factor` did not fire, so the two queries did cover the same
wind kinds; the coverage-mismatch hypothesis is wrong.

**IE: repaired.** The refetch returned identical generation and the same frozen
1907.13 MW capacity, confirming the fault is entirely in ENTSO-E's
installed-capacity endpoint. Generation itself is sound: 2032 MW peak in 2015
against a claimed 1907 MW capacity is by itself proof the denominator is wrong.
Substituting the Global Wind Power Tracker's annual register
(`scripts/region_tools/repair_country_capacity.py`) fixes it:

| | before | after |
|---|---|---|
| capacity | 1907 to 1919 MW | 1943 to 4118 MW |
| mean CF (train) | 0.499 | 0.332 |
| peak CF (test 2023) | 1.500 (clipped) | 0.934 |
| annual peak CF 2017-2021 | n/a, saturated | 0.96, 1.00, 0.94, 0.93, 0.97 |

Those annual peaks are exactly what a national fleet looks like. GWPT
undercounts Ireland before 2017 (peak CF still 1.05 in 2015 and 1.08 in 2016),
so `ie.toml` now trains on 2017-2021. The test split passes every gate.

The effect on the model is large. Evaluated on 2023:

| | RMSE before | RMSE after | MBE before | MBE after |
|---|---|---|---|---|
| uncorrected | 0.1967 | 0.1721 | -0.186 | +0.168 |
| N=1 fixed | 0.1598 | **0.0230** | -0.146 | +0.011 |
| N=3 fixed | 0.1608 | **0.0230** | -0.147 | +0.010 |

IE moves from the worst region to one of the best, and the earlier "correction
barely helps IE" reading was an artefact of the denominator. Note the single-
and three-cluster fits are now indistinguishable there.

**NL: still unusable.** The refetch roughly doubled the reported generation
(mean CF 0.075 to 0.129 on the training window), but the peak capacity factor
still never exceeds 0.57 over seven years and 0.50 in 2023, and the annual mean
still climbs 3.4x across the record:

| year | gen mean (MW) | gen max (MW) | capacity (MW) | peak CF |
|---|---|---|---|---|
| 2015 | 193 | 649 | 2874 | 0.23 |
| 2017 | 476 | 1453 | 4257 | 0.34 |
| 2019 | 772 | 2367 | 4632 | 0.51 |
| 2021 | 1439 | 4374 | 7674 | 0.57 |

A national fleet reaches 0.85 or higher at some point in any year. The covered
share of the fleet is still growing at the end of the window, so no sub-window
inside 2015-2023 is stable, and no constant rescaling helps. This looks like
genuine ENTSO-E reporting coverage rather than a code fault: Dutch units below
the reporting threshold entered "Actual Aggregated Generation per Type"
progressively. **NL should not be used for country-level bias correction from
this source.** Its post-repair numbers are in the run directory but should not
be reported; the earlier NL results in
[TURBINE_GRID_EVALUATION_ANALYSIS.md](TURBINE_GRID_EVALUATION_ANALYSIS.md) are
explained by this and not by fleet-composition drift alone.

## 4. Changes made in this branch

- `vwf/loaders/country_obs_checks.py`: physical-bound gates on a country CF
  series (peak, ceiling saturation, mean band, frozen capacity register, missing
  fraction). `EntsoeFileSource.load_observations` runs them on every load and
  warns; `check_country_cf(..., strict=True)` raises.
- `scripts/analysis/audit_country_observations.py`: the table in section 2, over
  every country region and both splits.
- `EntsoeFileSource` accepts the `_aggregated` filename suffix, which is how
  `generate_country_level_training_data` writes the national series for the
  zonal regions. NO and SE previously raised `FileNotFoundError` for both splits
  and could not be trained at all.
- `cluster_train_set`'s country branch is now capacity-weighted, matching the
  offset fit and the skill metric. Output is unchanged on a uniform grid, and a
  grid without a `capacity` column still falls back to equal weights.
- `scripts/region_tools/weight_country_grid_points.py`: replaces the synthetic
  uniform capacities with real GWPT capacity summed onto the nearest grid point,
  drops points carrying no fleet, and reports the area-versus-fleet shift.
  Applied to all nine grids, with the uniform grids kept alongside as
  `<c>_grid_points.uniform.bak.csv`.

GWPT coordinate coverage is 100% of operating projects in all nine countries, so
nothing is lost to missing geolocation.

**Design note.** [HARNESS_DESIGN.md](../design/HARNESS_DESIGN.md) records that
the country-level joint-offset optimiser is studied by RQ4 and changed by
nothing. It is unchanged: `find_offsets_country_level` is untouched, and every
`entsoe-country` region still routes to it, because a single national
observation is not per-cluster. The new per-cluster route is reached only by the
new `entsoe-zonal` source. Other changes in this section (weighting, monthly
aggregation, cluster validation, year-specific fleets) do move the numbers for
existing country regions, so results from before this branch are not comparable
with results after it.

### Second pass

- **Year-specific fleets.** `--per-year 2015 2024` writes one grid per year on a
  point set fixed across the range, so the cluster set is stable between splits
  and only the weights move. `EntsoeFileSource` prefers the year matching its
  split (the test year for test, the end of the window for train), falling back
  to the static grid. NL's fleet goes 2,939 to 12,022 MW over that range and SE's
  4,226 to 15,373 MW; a single snapshot cannot represent either.
- **`num_clu` now means something.** `assign_country_clusters` accepts `1`
  (collapse the country to one cluster) or the grid's own cluster count, and
  raises otherwise. Every ENTSO-E config previously asked for 5 clusters; only
  NL's grid actually had 5, so seven of nine regions had been writing
  `factors_<slice>_5.csv` files holding a different number of rows, with
  manifests recording 5. Configs are corrected and now carry
  `cluster_list = [1, <n>]`.
- **`prepare_country_fleet`** is called by both `train_set` and `val_set`, so
  evaluation can no longer score a fleet training never saw.
- **`country_cf_to_monthly`** computes `Σ(gen · Δt) / Σ(cap · Δt)` when the
  generation and capacity columns are present, falling back to the mean of
  ratios otherwise. Interval length is taken per row, so a file whose resolution
  changes partway through is still weighted correctly. On a synthetic month
  where capacity doubles halfway, the mean of ratios reads 0.25 against a true
  0.20.
- **`entsoe-zonal` source** plus `configs/regions/se_zonal.toml` (code `SE-BZ`).
  See section 5.
- **Fetcher consistency guard.** `calculate_capacity_factor` records which wind
  kinds each query covered and refuses to divide one by the other when they
  differ.
- **Two more gates**: annual-mean-CF drift beyond 1.6x across the record, and a
  capacity register with too few distinct values *and* negligible total
  movement. The second is deliberately conjunctive: PT's register is two numbers
  over seven years but the fleet genuinely only grew 15%, while IE's two numbers
  are 0.6% apart.

## 5. RQ4: per-zone observations

`entsoe-zonal` reads the per-bidding-zone observation files
`generate_country_level_training_data` has always written alongside the national
aggregate. The grid points' clusters are already numbered from the zone id
(`SE_1` to cluster 0, and so on), so zone and cluster are the same partition and
nothing had pointed the loader at those files.

With one observation per cluster the fit is exactly determined, so
`AffineWindCorrection.fit` routes to the turbine-level per-cluster
`find_offset` instead of the joint L-BFGS-B. The routing is decided by
`country_obs_is_per_cluster`, which reads off whether the observations actually
differ across clusters within a period, because that is precisely the condition
under which per-cluster offsets are estimable.

A zonal run is scored on the same capacity-weighted national aggregate a
national run is, so the two are directly comparable.

**SE is the working case.** All four Swedish zones have files for train
2015-2019 and test 2023.

**NO is blocked, and refetching did not unblock it.** Zones NO_1 to NO_4 now
have 2015-2021 training and 2023 test files with plausible capacity factors
(18%, 37%, 30%, 32%). NO_5 does not: **ENTSO-E reports 0 MW of installed wind
capacity for NO_5**, so its capacity factor comes out at the 1.5 clip ceiling
for every row, and it has no 2023 generation at all. Nothing was written for it.
A partial zone set shifts the national aggregate, so `EntsoeZonalFileSource`
refuses it rather than using it quietly, and Norway has no zonal run.

The NO_5 gap also contradicts the grid: GWPT puts about 1 GW of operating wind
inside the NO_5 box while ENTSO-E registers none there. One of the two is wrong
about where Norwegian wind is, and the zone boxes being approximations makes
either possible. Resolving it needs a source outside both.

**Zone geometry.** The boxes in `generate_country_level_training_data` are
approximations of the market boundaries, not the boundaries themselves. They are
used by `assign_country_zones.py` because they are what the observations were
fetched against, so grid and observations at least agree. Substituting real
bidding-zone geometry is a further improvement.

## 5a. Results

Every country region, trained and evaluated on 2023 with the changes above.
Full table in `output/validation/country_baseline_2026-07-23/all_metrics.csv`.
RMSE on the monthly capacity-weighted national aggregate, `fixed` slice:

| region | uncorrected | N=1 baseline | N clusters | N | better |
|---|---|---|---|---|---|
| FR | 0.1712 | 0.0125 | 0.0122 | 10 | clusters (marginal) |
| BE | 0.3399 | 0.0243 | 0.0214 | 3 | clusters |
| SE | 0.0880 | 0.0320 | 0.0264 | 4 | clusters |
| ES | 0.1349 | 0.0331 | 0.0260 | 4 | clusters |
| NO | 0.0252 | 0.0388 | 0.0290 | 4 | clusters |
| IT | 0.0661 | 0.0376 | 0.0360 | 3 | clusters (marginal) |
| IE (repaired) | 0.1721 | 0.0230 | 0.0230 | 3 | tie |
| PT | 0.1103 | 0.0767 | 0.0831 | 2 | **baseline** |
| NL | not usable, see 3a | | | | |

Read conservatively. The multi-cluster fit beats the identifiable single-cluster
baseline in six of eight usable regions, ties in one and loses in one, so it is
extracting something real rather than fitting noise. But **this does not show the offsets are meaningful**: an
N-cluster run has N *identified* scalars as well as N under-determined offsets,
and the improvement could come entirely from the scalars. The test that
separates them is an N-cluster run with offsets pinned to zero. That has not
been run.

Two further caveats. This is one test year per region, so the ordering of two
close variants is not meaningful (FR and IT differ by under 0.002 RMSE, and IE's
two variants are identical to four decimal places). And NO's correction is worse
than no correction at all under both variants (uncorrected 0.0252), so "clusters
beat baseline" there compares two things that should not be used.

### Zonal versus national, matched window

**Two earlier versions of this comparison were wrong and are retracted.** The
first paired every cluster with the wrong zone's observations (the shipped
cluster labels were not the zones). The second used the approximate bounding
boxes in `generate_country_level_training_data`, which misplace the NO3/NO5
boundary and are a hand-drawn approximation for Sweden. Both are now replaced by
real zone polygons, vendored under `configs/curation/zones/` with sources and
licences; the numbers below are on that geometry and supersede everything
earlier.

Both fits are scored two ways: on the national aggregate, and zone by zone
against each zone's own observation (`scope` column in `metrics.csv`). The
second exists because the first is the joint optimiser's own training objective,
so scoring a zonal fit on it flatters the national fit by construction.
`run_evaluate(..., score_zones=...)` scores a nationally trained run per zone,
which is what makes the two comparable.

Sweden, trained 2015-2019, evaluated 2023, `fixed` slice:

| | national RMSE | per-zone RMSE | per-zone r |
|---|---|---|---|
| uncorrected | 0.0875 | 0.1240 | 0.883 |
| national fit, N=1 | 0.0548 | 0.0595 | 0.848 |
| **national fit, N=4** | **0.0446** | **0.0528** | **0.941** |
| zonal fit, N=1 | 0.0566 | 0.0609 | 0.848 |
| zonal fit, N=4 | 0.0581 | 0.0611 | 0.910 |

**The national fit wins on the zonal fit's own metric**: per-zone RMSE 0.0528
against 0.0611, per-zone r 0.941 against 0.910. That disposes of the "the
comparison is biased toward the national metric" explanation. The
exactly-determined per-zone fit is genuinely worse at predicting the zones.

Two things the per-zone metric revealed that the national one had hidden:

- **Four clusters beat one** on the national fit (0.0528 against 0.0595), and
  per-zone correlation rises from 0.848 to 0.941. On the national metric alone
  the zonal run's N=1 and N=4 were indistinguishable, which had suggested Sweden
  has no usable zone-level detail. It does; errors in opposite directions across
  zones cancel in the aggregate, so the national metric cannot see it. A test in
  `tests/test_rq4_controls.py` pins the cancellation: two zones wrong by 0.20 in
  opposite directions score a perfect national RMSE.
- **Uncorrected per-zone error (0.1240) exceeds uncorrected national error
  (0.0875)**, the same cancellation in the raw reanalysis.

Caveat: one country, one test year, four zones, and Sweden is the weakest of the
vendored geometries (a redraw of a proprietary map, and Swedish zone membership
is defined per grid area rather than by a geographic line). A hint about the
mechanism, not an established result. Section 5b is the better explanation for
why the identifiable fit did not help.

### Norway: the NO5 conflict resolved

Real geometry resolves it, and **the bounding box was wrong while ENTSO-E was
right**. Three independent lines agree that NO5 has no operating wind:

- ENTSO-E reports 0 MW installed wind capacity and no 2023 generation for NO5.
- NVE's own built-wind register (`Vindkraft2/MapServer/0`, 61 plants, 5198 MW)
  places 0 MW inside NVE's own NO5 polygon.
- Norway's large western clusters sit in NO3, not NO5: the NO3/NO5 boundary runs
  south of Sunnfjord rather than along the Vestland county line, so Guleslettene,
  Lutelandet, Hennoy and Mehuken are all NO3, and Fosen Vind (Storheia, Roan,
  Harbaksfjellet, Kvenndalsfjellet) is unambiguously NO3.

After reassignment, one GWPT record still landed in NO5: "Hordavind wind farm",
682 MW, marked operating with no start year, no operator and an approximate
location. At that size it would be Norway's largest wind farm, and it appears in
no register. It is excluded via `configs/curation/gwpt_exclusions.csv`, with the
evidence recorded there. Norway's fleet then totals 5121 MW against NVE's 5198
MW, agreeing to 1.5%, and NO5 is correctly empty. `no.toml` drops to
`cluster_list = [1, 4]`.

Norway's national run on real geometry: uncorrected RMSE 0.0338, N=1 0.0420,
N=4 0.0389. **The correction remains worse than no correction**, which is now a
clean result rather than a geometry artefact and needs explaining.

## 5b. What the offsets are actually doing

The zero-offset control (`scalar-only`, offsets pinned to 0, scalars
bit-identical to the affine model's) was meant to test whether the
under-determined country offsets carry information. It answered a different and
more useful question.

Country-level RMSE on 2023, `fixed` slice:

| region | uncorrected | affine (N) | scalar-only (N) |
|---|---|---|---|
| BE | 0.3399 | **0.0214** | 0.1633 |
| FR | 0.1712 | **0.0122** | 0.1083 |
| IE | 0.1721 | **0.0230** | 0.0970 |
| SE | 0.0880 | **0.0296** | 0.1222 |
| PT | 0.1103 | **0.0831** | 0.2399 |
| NO | 0.0252 | 0.0373 | 0.0525 |
| ES | 0.1349 | **0.0260** | 0.0432 |
| IT | 0.0661 | 0.0360 | **0.0309** |

The offsets do enormous work, often a factor of four to seven. But the reason is
not that they capture a spatial additive wind-speed bias. **The offset is mostly
repairing the scalar.**

The scalar is a capacity-factor ratio applied to a wind speed. Power goes
roughly as `w³`, so a CF ratio of 0.4 corresponds to a wind-speed factor near
`0.4^(1/3) = 0.74`, not 0.4. Applied directly it over-corrects badly, and the
offset then pulls the result back. The evidence:

- Across 320 fitted (cluster, slice) pairs, **corr(scalar, offset) = -0.79**.
  Per region it is below -0.86 in nine of ten, and below -0.97 in six: FR -0.999,
  NO -0.993, SE -0.989, NZ -0.988, PT -0.987, DK -0.975. Only ES is positive
  (+0.456) and that is unexplained.
- Scalars are systematically small (median 0.723, 76% below 1) and offsets
  systematically positive (median +0.64 m/s, 71% positive), in the direction that
  compensates.
- The effective wind-speed factor `(scalar·w + offset)/w` at 8 m/s has median
  0.835 and tracks `cbrt(scalar)` (median 0.898) with **correlation 0.921**.
- The zero-offset runs overshoot in exactly the predicted way. DK's mean bias
  goes from +0.124 uncorrected to **-0.061** with the scalar alone, crossing
  zero; NZ's goes from -0.062 to **+0.140**.

So the affine pair is largely acting as one effective wind-speed multiplier that
approximates the cube root of the CF ratio, using two nearly collinear
parameters to do it. This holds at turbine level too (DK -0.975, NZ -0.988),
where the offsets are exactly identified, so **it is a property of the affine
parameterisation, not of the country-level identifiability problem**.

That reframes RQ4. The question was whether the under-determined offsets mean
anything. The answer appears to be that they mostly encode a deterministic
function of the scalar, which would explain both why the joint fit works despite
being under-determined (the solution set is nearly one-dimensional in the
direction that matters) and why making it identifiable did not help.

Testable predictions this generates, none yet run:

1. A one-parameter model `cor_ws = k · unc_ws` with `k` fitted directly to wind
   speed, or `k = cbrt(obs_CF/sim_CF)`, should approach affine performance. If it
   does, the second parameter is close to redundant.
2. The residual after removing the `cbrt` relationship is the part of the offset
   that is genuinely additive bias. That residual, not the raw offset, is what
   RQ4 should be asking about.
3. If (1) holds, the country-level identifiability problem largely dissolves: one
   parameter per cluster against one national observation is still
   under-determined for N > 1, but the parameter count halves and a pooled fit
   across periods becomes far better conditioned.

Caveats: 8 m/s is a chosen reference speed and the relationship is not exact;
correlation is not proof of mechanism; ES contradicts the pattern and has not
been explained. This is a strong lead, not a settled result.

Cost note: the joint fit's runtime scales badly with cluster count. NO's
4-cluster run took 235 s and FR's 10-cluster run 136 s, against 19 s for PT's
2-cluster run.

## 6. Still open

**Pooling across periods.** Fitting one `β_c` per cluster jointly across all T
training periods turns 1-vs-N into T-vs-N, identified whenever the cluster CF
time series are not collinear, and needs no data that is not already on disk.
The catch is conditioning: cluster capacity factors in a small country are
highly correlated, so the problem can be identified in principle and useless in
practice. It needs a conditioning diagnostic (singular values of the Jacobian at
the solution) reported alongside the fit, not assumed. Not implemented.

**Regularisation** as a fallback where pooling is ill-conditioned, with the
penalty stated and its strength reported. Not implemented.

**NL, IE and NO observations.** All three need refetching; blocked on an
ENTSO-E API key. Section 3a records what to look for and the guard that now
catches the NL mode at source.

**Per-year fleets within the training window.** The fleet is now split-specific
but still static inside a training window. `train_set`'s country branch
simulates once over the whole window with one `turb_info`, so making the weights
vary year by year inside it needs restructuring.

**IT and DK zones.** Both have market zones and would extend the RQ4 comparison,
but neither has per-zone observation files fetched yet.

**The one-parameter test.** Section 5b's predictions: fit `cor_ws = k · unc_ws`
with `k` on the wind-speed scale, and compare against affine. If it matches,
the offset is largely redundant and the RQ4 parameter count halves. This is now
the highest-value open item.

**Why NO and PT get worse under correction**, on clean geometry and clean
observations. NO's uncorrected RMSE of 0.0338 is already better than any
corrected variant.

**Denmark's DK2 polygon is missing Bornholm**, and the Swedish polygons are the
weakest of the vendored set. Neither blocks current work; both would need
attention before a Danish zonal run or a published Swedish zonal result.
