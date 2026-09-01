# Country-level method review

**Date:** 2026-07-23
**Scope:** the `obs_level = "country"` path against the turbine-level path, plus
an audit of the nine ENTSO-E observation series on disk.

The country-level path is wired consistently with the turbine-level path but
fits a different estimator under the same name, and its observations had never
been checked. Two of nine regions could not load at all and four carried
physically impossible capacity factors. Loading, checking and spatial weighting
are fixed. The identifiability of the joint offset fit is unchanged, and section
6 argues the question itself was probably the wrong one.

## 1. What each path fits

| | Turbine level | Country level |
|---|---|---|
| Scalar `a` | obs/sim over the same turbines, capacity-weighted, restricted to rows that reported | obs is the one national CF, sim is the cluster mean |
| Offset `b` | one observation, one `b`, solved exactly per (cluster, slice) | N cluster offsets against one national number, joint L-BFGS-B from `x0 = 0` |
| Constraints vs parameters | determined | **under-determined by N-1** |
| Clustering | `cluster_turbines` runs | `num_clusters` was ignored |

The scalar definition is the substantive divergence. At country level
`a_c = CF_national_observed / CF_c_simulated`, so a cluster the reanalysis says
is windier than the national mean receives `a < 1` and a calmer one `a > 1`.
That flattens the simulated spatial field toward the national mean rather than
correcting bias at each location. Verified on the real NL grid: all five
clusters receive the identical observation and the scalars (0.785, 0.815, 0.807,
0.791, 0.803) track only the spread in the simulation.

This compounds with something both paths share. `a` is a capacity-factor ratio
applied to a wind speed, and power goes roughly as `w^3`. At turbine level that
is harmless because `find_offset` then drives `b` until simulated CF equals the
observation exactly, so the pair is self-correcting. At country level nothing
re-fits per cluster, so the cube-law overshoot survives into the factors.

`num_clusters` being ignored was verified directly: `cluster_train_set` with
`num_clu = 5` and `num_clu = 37` returned byte-identical frames, while the
output was still written as `factors_<slice>_<n>.csv` and the manifest still
recorded `num_clu`. A smoke train of ES wrote `factors_fixed_5.csv` holding 4
clusters.

## 2. Observation audit

A national fleet's sub-daily capacity factor must at some point approach the
fleet peak and can never exceed 1. Both tests are source-free.

| Region | step | mean CF | peak CF | clipped | verdict |
|---|---|---|---|---|---|
| FR train / test | 1 h | 0.231 / 0.257 | 0.936 / 0.809 | 0 | pass |
| ES train / test | 1 h / 15 min | 0.246 / 0.238 | 0.795 / 0.710 | 0 | pass |
| IT train / test | 1 h | 0.219 / 0.237 | 0.775 / 0.783 | 0 | pass |
| SE train / test | 1 h | 0.255 / 0.264 | 0.952 / 0.856 | 0 | pass |
| BE test | 1 h | 0.234 | 0.794 | 0 | pass |
| NO test | 1 h | 0.309 | 0.787 | 0 | pass |
| PT test | 1 h | 0.275 | 0.900 | 0 | pass |
| BE train | 1 h | 0.209 | **1.295** | 0 | CF above 1 |
| PT train | 1 h | 0.301 | **1.032** | 0 | CF above 1 |
| NO train | 1 h | 0.313 | **1.500** | 1 row | one clipped spike |
| **NL train / test** | 15 min | **0.075** / 0.141 | **0.569** / **0.380** | 0 | **not usable** |
| **IE train / test** | 30 min | **0.480** / **0.669** | **1.500** | 500 / 999 rows | **not usable** |

NL's generation peaks at 1528 MW against a stated 2646 to 7674 MW of capacity
across seven years, implying 2.8 TWh/yr against a real Dutch output near 11 to
12 TWh/yr. IE has capacity frozen at 1919 MW across both splits, inflating every
derived capacity factor and driving 5.7% of test rows onto the fetcher's
`clip(0, 1.5)` ceiling. BE, PT and NO fail on isolated spikes rather than a
systematic error; their means are plausible and their test splits are clean.

The correlation with the recorded results matters: the regions the earlier
evaluation called good (FR, ES, BE, NO) are the ones whose series pass, and NL,
the catastrophic one, has the worst series. **The country-level failures observed
so far are more attributable to the observation construction than to the
correction maths**, and that has to be separated before any methodological
conclusion is drawn from them.

## 3. Spatial weighting

Three places aggregated to "country" and disagreed: `cluster_train_set` took an
unweighted mean while the offset fit and the harness skill metric both
capacity-weighted. On a grid whose points all carry the same synthetic capacity
the two agree, which is why it stayed invisible.

The deeper problem is that a uniform grid makes "capacity-weighted" mean
point-count weighted, that is **land-area weighted**, while the observation is
national generation over national installed capacity, that is **fleet
weighted**. Real fleets are concentrated, so the gap is absorbed into `a` and
`b` as if it were reanalysis bias. The turbine path has no equivalent gap, since
observation and simulation sit on the same machines.

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

Half to nine tenths of every grid carried no wind capacity, so the simulated
country average was mostly an average over empty countryside. In France one
cluster holds 34.9% of the fleet on 10.3% of the area while another holds 0.1%
of the fleet on 10.5%. In Norway two entire clusters contain no wind at all.

## 4. NL and IE: what they need

Both are fetch-side, and neither is repairable by rescaling what is on disk.
Refetching with `--psr-type all` (the stored files were onshore-only) changed NL
substantially and IE not at all. The consistency guard did not fire, so the two
queries covered the same wind kinds and the coverage-mismatch hypothesis is
wrong.

**IE: repaired.** The refetch returned identical generation and the same frozen
1907.13 MW capacity, so the fault is entirely in ENTSO-E's installed-capacity
endpoint. A 2032 MW generation peak in 2015 against a claimed 1907 MW capacity
is by itself proof the denominator is wrong. Substituting the GWPT annual
register (`scripts/region_tools/repair_country_capacity.py`) fixes it:

| | before | after |
|---|---|---|
| capacity | 1907 to 1919 MW | 1943 to 4118 MW |
| mean CF (train) | 0.499 | 0.332 |
| peak CF (test 2023) | 1.500, clipped | 0.934 |
| annual peak CF 2017-2021 | saturated | 0.96, 1.00, 0.94, 0.93, 0.97 |

Those annual peaks are what a national fleet looks like. GWPT undercounts
Ireland before 2017 (peak CF 1.05 in 2015, 1.08 in 2016), so `ie.toml` trains on
2017-2021. Evaluated on 2023 the effect is large: uncorrected RMSE 0.1967 to
0.1721, and the N=1 fixed fit 0.1598 to **0.0230**. IE moves from the worst
region to one of the best, and the earlier "correction barely helps IE" reading
was an artefact of the denominator.

**NL: still unusable.** The refetch roughly doubled reported generation (mean CF
0.075 to 0.129 on the training window), but peak capacity factor still never
exceeds 0.57 over seven years, and 0.50 in 2023, while the annual mean climbs
3.4x across the record:

| year | gen mean (MW) | gen max (MW) | capacity (MW) | peak CF |
|---|---|---|---|---|
| 2015 | 193 | 649 | 2874 | 0.23 |
| 2017 | 476 | 1453 | 4257 | 0.34 |
| 2019 | 772 | 2367 | 4632 | 0.51 |
| 2021 | 1439 | 4374 | 7674 | 0.57 |

A national fleet reaches 0.85 or higher at some point in any year. The covered
share of the fleet is still growing at the end of the window, so no sub-window
is stable and no constant rescaling helps. This is ENTSO-E reporting coverage,
not a code fault: Dutch units below the reporting threshold entered "Actual
Aggregated Generation per Type" progressively. **NL should not be used for
country-level bias correction from this source.**

## 5. Fixes with consequences for the numbers

Loading, checking and weighting are all changed, so **country-level results
produced before these changes are not comparable with results produced after
them**. The joint offset optimiser itself is untouched.

- **Weighting.** `cluster_train_set`'s country branch is now capacity-weighted,
  matching the offset fit and the skill metric, and
  `weight_country_grid_points.py` replaces the synthetic uniform capacities with
  real GWPT capacity summed onto the nearest grid point, dropping points that
  carry no fleet. GWPT coordinate coverage is 100% of operating projects in all
  nine countries.
- **`num_clu` now means something.** It accepts 1 (collapse to one cluster) or
  the grid's own cluster count, and raises otherwise. Every ENTSO-E config had
  asked for 5; only NL's grid had 5, so seven of nine regions had been writing
  factor files whose row count disagreed with their own manifest.
- **Monthly aggregation.** `country_cf_to_monthly` computes
  `sum(gen*dt) / sum(cap*dt)` where the columns allow, per-row interval length,
  falling back to the mean of ratios. On a synthetic month where capacity
  doubles halfway, the mean of ratios reads 0.25 against a true 0.20.
- **Year-specific fleets.** One grid per year on a point set fixed across the
  range, so the cluster set is stable between splits and only the weights move.
  NL's fleet runs 2,939 to 12,022 MW over 2015-2024 and SE's 4,226 to 15,373 MW;
  a single snapshot represents neither.
- **Observation gates**, run on every load: peak, ceiling saturation, mean band,
  missing fraction, annual-mean drift beyond 1.6x, and a frozen capacity
  register. The last is deliberately conjunctive, requiring both too few
  distinct values and negligible movement, because PT's register is two numbers
  over seven years while the fleet genuinely grew 15%, and IE's two numbers are
  0.6% apart.
- **Loading.** `EntsoeFileSource` accepts the `_aggregated` suffix; NO and SE
  previously raised `FileNotFoundError` on both splits and could not train.
- **Fetcher guard.** `calculate_capacity_factor` records which wind kinds each
  query covered and refuses to divide one by the other when they differ.

## 6. Results

Trained and evaluated on 2023, RMSE on the monthly capacity-weighted national
aggregate, `fixed` slice. Full table in
`output/validation/country_baseline_2026-07-23/all_metrics.csv`.

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
| NL | not usable, see section 4 | | | | |

Read conservatively. The multi-cluster fit beats the identifiable
single-cluster baseline in six of eight usable regions, ties in one and loses in
one, so it extracts something real rather than fitting noise. But **this does
not show the offsets are meaningful**: an N-cluster run has N *identified*
scalars as well as N under-determined offsets, and the improvement could come
entirely from the scalars.

Two further caveats. One test year per region, so the ordering of two close
variants is not meaningful (FR and IT differ by under 0.002, and IE's two
variants agree to four decimals). And NO's correction is worse than no
correction at all under both variants, so "clusters beat baseline" there
compares two things that should not be used.

### Zonal against national

Scored two ways: on the national aggregate, and zone by zone against each zone's
own observation. The second exists because the first is the joint optimiser's
own training objective, so scoring a zonal fit on it flatters the national fit
by construction. Numbers are on real zone polygons vendored under
`configs/curation/zones/` with sources and licences; earlier bounding-box
approximations misplaced the NO3/NO5 boundary and are superseded.

With one observation per cluster the fit is exactly determined, so
`AffineWindCorrection.fit` routes to the per-cluster `find_offset` rather than
the joint optimiser. `country_obs_is_per_cluster` decides by reading whether the
observations actually differ across clusters within a period, which is precisely
the condition under which per-cluster offsets are estimable.

Sweden, trained 2015-2019, evaluated 2023, `fixed` slice:

| | national RMSE | per-zone RMSE | per-zone r |
|---|---|---|---|
| uncorrected | 0.0875 | 0.1240 | 0.883 |
| national fit, N=1 | 0.0548 | 0.0595 | 0.848 |
| **national fit, N=4** | **0.0446** | **0.0528** | **0.941** |
| zonal fit, N=1 | 0.0566 | 0.0609 | 0.848 |
| zonal fit, N=4 | 0.0581 | 0.0611 | 0.910 |

**The national fit wins on the zonal fit's own metric.** That disposes of the
"biased toward the national metric" explanation: the exactly-determined per-zone
fit is genuinely worse at predicting the zones.

The per-zone metric also reveals what the national one hides. Four clusters beat
one (0.0528 against 0.0595, per-zone r 0.848 to 0.941) where on the national
metric alone the zonal N=1 and N=4 were indistinguishable, which had suggested
Sweden has no usable zone-level detail. Errors in opposite directions across
zones cancel in the aggregate; `tests/test_rq4_controls.py` pins this, with two
zones wrong by 0.20 in opposite directions scoring a perfect national RMSE. The
same cancellation is in the raw reanalysis: uncorrected per-zone error 0.1240
exceeds uncorrected national error 0.0875.

Caveat: one country, one test year, four zones, and Sweden is the weakest of the
vendored geometries, a redraw of a proprietary map where zone membership is
defined per grid area rather than by a geographic line. A hint about the
mechanism, not an established result.

**Norway's NO5, resolved: the bounding box was wrong and ENTSO-E was right.**
Three independent lines agree NO5 has no operating wind. ENTSO-E reports 0 MW
installed and no 2023 generation; NVE's own built-wind register (61 plants,
5198 MW) places 0 MW inside NVE's own NO5 polygon; and the large western
clusters sit in NO3, because the NO3/NO5 boundary runs south of Sunnfjord rather
than along the Vestland county line, putting Guleslettene, Lutelandet, Hennoy,
Mehuken and all of Fosen Vind in NO3. One GWPT record still landed in NO5,
"Hordavind wind farm" at 682 MW, marked operating with no start year, no
operator and an approximate location; at that size it would be Norway's largest
wind farm and it appears in no register, so it is excluded via
`configs/curation/gwpt_exclusions.csv`. Norway's fleet then totals 5121 MW
against NVE's 5198 MW, agreeing to 1.5%. On real geometry: uncorrected RMSE
0.0338, N=1 0.0420, N=4 0.0389, so **the correction remains worse than no
correction**, now as a clean result rather than a geometry artefact.

## 7. What the offsets are actually doing

The zero-offset control (`scalar-only`, offsets pinned to 0, scalars
bit-identical to the affine model's) was meant to test whether the
under-determined country offsets carry information. It answered a more useful
question. RMSE on 2023, `fixed` slice:

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

The offsets do enormous work, often a factor of four to seven. But not because
they capture a spatial additive wind-speed bias. **The offset is mostly
repairing the scalar.** A CF ratio of 0.4 corresponds to a wind-speed factor
near `0.4^(1/3) = 0.74`, not 0.4; applied directly the scalar over-corrects and
the offset pulls it back. Four lines of evidence:

- Across 320 fitted (cluster, slice) pairs, **corr(scalar, offset) = -0.79**,
  below -0.86 in nine regions of ten and below -0.97 in six: FR -0.999,
  NO -0.993, SE -0.989, NZ -0.988, PT -0.987, DK -0.975. Only ES is positive
  (+0.456), and that is unexplained.
- Scalars are systematically small (median 0.723, 76% below 1) and offsets
  systematically positive (median +0.64 m/s, 71% positive), in the compensating
  direction.
- The effective wind-speed factor `(scalar*w + offset)/w` at 8 m/s has median
  0.835 and tracks `cbrt(scalar)` (median 0.898) with **correlation 0.921**.
- The zero-offset runs overshoot as predicted: DK's mean bias goes from +0.124
  uncorrected to **-0.061** with the scalar alone, crossing zero, and NZ's from
  -0.062 to **+0.140**.

So the affine pair largely acts as one effective wind-speed multiplier
approximating the cube root of the CF ratio, using two nearly collinear
parameters to do it. This holds at turbine level too (DK -0.975, NZ -0.988),
where the offsets are exactly identified, so **it is a property of the affine
parameterisation, not of the country-level identifiability problem**.

That reframes the identifiability question. It was whether the under-determined
offsets mean anything; the answer appears to be that they mostly encode a
deterministic function of the scalar, which would explain both why the joint fit
works despite being under-determined (the solution set is nearly
one-dimensional in the direction that matters) and why making it identifiable
did not help.

Caveats: 8 m/s is a chosen reference speed, the relationship is not exact,
correlation is not proof of mechanism, and ES contradicts the pattern
unexplained. A strong lead, not a settled result.

Cost note: the joint fit scales badly with cluster count. NO's 4-cluster run
took 235 s and FR's 10-cluster run 136 s, against 19 s for PT's 2-cluster run.

## 8. Open

**The one-parameter test, highest value.** Fit `cor_ws = k * unc_ws` with `k` on
the wind-speed scale, or `k = cbrt(obs_CF/sim_CF)`. If it approaches affine
performance the second parameter is close to redundant, the parameter count
halves, and the country-level identifiability problem largely dissolves. The
residual after removing the cube-root relationship is the part of the offset
that is genuinely additive bias, and that residual, not the raw offset, is what
the identifiability question should ask about.

**Pooling across periods.** One `b_c` per cluster fitted jointly across all T
training periods turns 1-vs-N into T-vs-N, identified whenever the cluster CF
series are not collinear, and needs no data that is not on disk. The catch is
conditioning: cluster capacity factors in a small country are highly correlated,
so the problem can be identified in principle and useless in practice. It needs
a conditioning diagnostic (singular values of the Jacobian at the solution)
reported alongside the fit, not assumed. Regularisation is the fallback where
pooling is ill-conditioned, with the penalty stated and its strength reported.

**Why NO and PT get worse under correction**, now on clean geometry and clean
observations. NO's uncorrected RMSE of 0.0338 beats every corrected variant.

**NL, IE and NO observations** all need refetching, blocked on an ENTSO-E key.

**Per-year fleets inside a training window.** The fleet is split-specific but
still static within a window, because the country branch simulates once over the
whole window with one `turb_info`.

**IT and DK zones** would extend the per-zone comparison; neither has per-zone
observation files yet. Denmark's DK2 polygon is missing Bornholm, and the
Swedish polygons are the weakest of the vendored set.
