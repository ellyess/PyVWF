# US and Brazil: first real runs, and a scalar-dilution bug

**Date:** 2026-07-22
**Scope:** the first real train and evaluate runs for the United States and
Brazil, and the defects they exposed.

**Question.** Both regions are staged against real data (EIA-923/860 plus
USWTDB; ONS plus ANEEL) and real ERA5. Does the `affine-wind` correction improve
held-out skill as it does for DK, DE and UK, and how many spatial clusters does
it need?

**Verdict: PASS, after fixing a bug this run exposed.** Brazil worked
immediately. The US initially came out *worse than no correction at all*, which
traced to a real defect in `calculate_scalar`. DK, DE and UK were never affected
because their monthly reporting is near-complete, which is why three regions of
prior work never surfaced it.

Setup: `us.toml` (train 2019-2021, test 2022; 1284 plants, 356 in the training
fleet after the complete-reporting filter) and `br.toml` (train 2021-2023, test
2024; 193 complexes, 125 training). ERA5 as 48 monthly CDS files per region
reduced to yearly daily files. Curves are the bundled open library merged with
the licensed one (236 curves), matched per plant for the US and uniform for
Brazil, which carries no turbine data.

## The bug: observed means diluted by the reporting fraction

`calculate_scalar` aggregates observed and simulated CF per (cluster, year,
slice) and takes `scalar = obs / sim`. Its `weighted_avg` computed
`(v * w).sum(min_count=1) / w.sum()`: the numerator skips NaN observations, the
denominator summed **every** unit's capacity including plants that reported
nothing. `sim` is never NaN, so only `obs` was scaled down, by exactly the
reporting fraction.

> `scalar_computed = scalar_true * (fraction of capacity reporting)`

Verified on the real US fleet, 2019:

| quantity | value |
| --- | --- |
| capacity fraction reporting | **0.431** |
| obs weighted, as computed | 0.1594 |
| obs weighted, correct | 0.3702 |
| ratio | **0.4306**, the reporting fraction |
| sim weighted | 0.3387 |
| **scalar as computed** | **0.4708** |
| **scalar correct** | **1.0931** |

The US correction was pushing wind speed **down 53%** where it should nudge it
**up 9%**.

**The decisive symptom** was corrected bias near -0.15 *in sample as well as
out* (2019 -0.159, 2020 -0.160, 2021 -0.147, 2022 -0.145). A correction that
optimises CF bias cannot leave -0.15 on the data it was fitted to, which ruled
out overfitting and poor transfer and pointed upstream of fit and apply.

The fix masks to rows that have a value in numerator and denominator both,
preserving `min_count=1` semantics so an all-missing group still returns NaN.

**Why the suite missed it.** `test_correction.py` covered an all-present group
and an all-missing group but never a *partly* reporting one, the only
configuration where the two denominators differ. Two tests added; the second
uses lopsided capacities so a naive "count reporting rows" fix still fails.

## The US, before and after

Held-out 2022, 520 units, 6078 samples, k=10.

| variant | MAE | RMSE | MBE | r |
| --- | --- | --- | --- | --- |
| uncorrected | 0.0796 | 0.1098 | +0.022 | 0.733 |
| affine-wind, **before fix** | 0.1527 | 0.1729 | -0.145 | 0.725 |
| affine-wind, **after fix** | **0.0741** | **0.1020** | +0.025 | **0.744** |

The fitted factors change character completely: scalars go from a near-uniform
0.31 to 0.66 across all clusters to 0.88 to 1.58, straddling 1.0 with genuine
cluster-to-cluster spread, and offsets shrink to small values of both signs. The
flat scalars near 0.5 were the bug's signature.

## Brazil, unaffected, and it already worked

Held-out 2024, 151 units, 389 samples.

| variant | k | slice | MAE | MBE | r |
| --- | --- | --- | --- | --- | --- |
| uncorrected | - | - | 0.1102 | -0.046 | 0.525 |
| affine-wind | 8 | fixed | 0.0905 | -0.023 | 0.602 |
| affine-wind | 8 | season | 0.0901 | +0.008 | 0.593 |
| affine-wind | 50 | fixed | 0.0796 | -0.016 | 0.700 |
| affine-wind | 120 | fixed | 0.0781 | -0.013 | 0.732 |
| affine-wind | 120 | season | **0.0702** | +0.022 | **0.743** |

Brazil's factors are identical to three decimals before and after the fix
(1.0385, 0.7981), because its training fleet is already filtered to
complete-reporting complexes, so the dilution factor is about 1. That makes
Brazil a useful control: the fix is not a free knob that moves everything, it
corrects exactly the regions with reporting gaps.

Explicit Southern-Hemisphere `season` slicing beats `fixed`, which retroactively
justifies stating seasons per region rather than inheriting the Northern default.

## Cluster count, and why the first sweep was measuring luck

Held-out MAE, best slice, measured through the fixed scalar and `k-means++`
init, curve library pinned to open plus licensed for all five regions.

| region (test year) | uncorrected | k=1 | k=10 | k=100 | k=500+ | best |
| --- | --- | --- | --- | --- | --- | --- |
| US (2022) | 0.0796 | 0.0889 | 0.0737 | 0.0730 | 0.0675 (300) | **0.0624** (700) |
| BR (2024) | 0.1102 | 0.1201 | 0.0843 (8) | - | - | **0.0708** (120) |
| DK (2020) | 0.1118 | 0.0634 | 0.0608 | 0.0552 | **0.0511** (500) | 0.0671 (1000) |
| DE (2019) | 0.0594 | 0.0468 | 0.0436 | 0.0396 | 0.0366 (500) | **0.0364** (1000) |
| UK (2019) | 0.1066 | 0.1009 | 0.0875 | 0.0755 | **0.0679** (500) | 0.0679 (1000) |

Improvement over uncorrected at best k: US -22%, BR -36%, DK -54%, DE -39%,
UK -36%. **Every region is far better at high k than at the shipped k=10.**

**k=1 is worse than uncorrected in the US and Brazil** (0.0889 against 0.0796;
0.1201 against 0.1102) though not in Europe. A single global scalar and offset
is actively harmful: spatial resolution is what makes the method work, not a
refinement on top of it. This result carries no seed variance, because at k=1
there is no partition to be lucky about, so it survives everything below.

### The k-curve was partition lottery, not k

Extending DK past k=500 gave a non-monotonic curve (season slice, held-out
2020): MAE 0.0623, 0.0562, 0.0683, **0.0516**, 0.0684 at k = 10, 100, 500, 700,
1000. k=700 being the best DK result of any run rules out overfitting past an
optimum, which degrades monotonically.

Ruled out in turn: overfitting; an evaluate-time re-clustering mismatch (the
KMeans refit in `run_evaluate` reproduces the training partition with 100%
identical labels at every k); cross-k state leakage (DK k=500 run alone is
bit-identical to k=500 run last of [10, 100, 500]); and degenerate factors (no
identity fallbacks at any k, scalar p50 0.72 to 0.74 across k=100 to 1000).

**Cause: `init="random"` in `cluster_turbines`.** Holding k=500 and varying only
the seed:

| seed | 0 | 1 | 7 | 42 |
| --- | --- | --- | --- | --- |
| MAE, `init="random"` | 0.0514 | 0.0842 | 0.0628 | 0.0683 |
| MAE, `init="k-means++"` | 0.0513 | 0.0511 | 0.0507 | 0.0509 |

At one k, random init spans 0.0328 in MAE, **larger than the entire k=10 to
k=1000 spread at fixed seed** (0.0516 to 0.0684). k=700's apparent win was
partition luck; seed 0 reaches the same score at k=500. `k-means++` collapses
the spread to 0.0006, a 55-fold reduction, and beats the luckiest random draw
(0.0507 against 0.0514). It is sklearn's default precisely because random init
falls into bad local optima, and `n_init=10` selects among those by inertia,
which is not held-out skill.

`init="k-means++"` is applied, with `tests/test_clustering.py` pinning
seed-stability; three of its four tests fail on the old init. The init-only
effect at the shipped k, curve library held fixed:

| region | random init | k-means++ | delta | points/cluster |
| --- | --- | --- | --- | --- |
| DE k=10 | 0.0479 | 0.0479 | 0.0000 | ~1000 |
| DK k=10 | 0.0623 | 0.0622 | -0.0001 | ~370 |
| UK k=10 | 0.0888 | 0.0885 | -0.0003 | ~600 |
| US k=10 | 0.0741 | 0.0737 | -0.0004 | ~36 |
| BR k=8 | 0.0901 | 0.0843 | **-0.0058** | ~16 |
| DK k=500 | 0.0514-0.0842 | 0.0507-0.0513 | **64% to 1% spread** | ~7 |

The effect scales with how hard the partition is, measured as points per
cluster: negligible where clusters are large and unambiguous, dominant at high
k. **Prior European results at k=10 are unaffected**, so nothing published needs
revising. The change matters for sweeps, where it is the difference between
measuring k and measuring luck.

A related trap cost one comparison and is worth stating: the curve library is
selected by the `PYVWF_INPUT` environment variable, so an early comparison ran
Europe on the bundled library and US/BR on open plus licensed without noticing.
The tell was that *uncorrected* baselines moved, which a clustering change
cannot cause. This is why the resolved curve-library path and hash now go into
every run manifest.

### The ceiling on k is unique coordinates, not fleet size

| region | training rows | unique (lat, lon) | clusters at k=1000 |
| --- | --- | --- | --- |
| DK | 3707 | **3706** | 1000 (genuine) |
| DE | 4288 | **579** | 579 (saturated) |
| UK | 5621 | **332** | 332 (saturated) |
| US | 1091 | 1071 | 700 at k=700 (genuine) |
| BR | 125 | 119 | 119 at k=120 (saturated) |

UK k=500 and k=1000 return bit-identical metrics because both collapse to the
same 332 clusters: UK turbine records are grouped at farm coordinates. Requesting
k above the distinct-location count silently wastes compute and produces a fake
plateau that reads like convergence. **Bound k by unique coordinates, not row
count.** The practical ceiling is the *training* fleet, not the metadata fleet:
Brazil's is 125 complexes, not 193, so k=150 raises `n_samples=125 should be >=
n_clusters=150`.

DK is the only genuine interior optimum, and it is also the only region whose
coordinates are per-turbine (3706 distinct positions for 3707 rows), so k=1000
means about 3.7 turbines per cluster. The others saturate before reaching that
sparsity. **No universal k exists**: US and Brazil were still improving at their
ceilings, DK has an optimum well below its, DE and UK plateau at saturation.

Two caveats. The pre-fix US sweep (k=1/100/300, MAE 0.153, 0.143, 0.110) was
measured through the broken scalar and must not be cited. And the harness DK
curve disagrees in shape with the legacy DK run (`output/runs/turbine_dk_research`,
onshore/bimonth: 0.0703, 0.0636, 0.0607 at k=10/100/500, improving
monotonically) where the harness degrades at k=500. The runs differ in mode (all
against onshore) and slicing, so they are not like-for-like, but the
disagreement should be resolved before publishing a DK k recommendation.

## Two clustering improvements that landed inside the noise floor

**Capacity-weighted clustering: no evidence it helps.** Unweighted, a 5 MW site
pulls a centroid as hard as a 500 MW one while skill is scored
capacity-weighted, so weighting the KMeans fit looked obvious. A/B at fixed k,
same library, same seed:

| region | k | unweighted | weighted | delta |
| --- | --- | --- | --- | --- |
| US | 100 | 0.0730 | **0.0722** | -1.1% |
| UK | 100 | 0.0755 | **0.0737** | -2.4% |
| DE | 100 | 0.0403 | 0.0403 | 0.0% |
| DK | 100 | **0.0567** | 0.0574 | +1.2% |
| BR | 60 | **0.0763** | 0.0778 | +2.0% |

Two better, two worse, one unchanged, all within 3%, and the season slice
agrees. `weight_col` defaults to None. The choice is a modelling one rather than
an optimisation: unweighted says clusters represent **meteorology**, since
reanalysis bias is a property of location and a small site samples the same wind
field as a large one, while weighted says they should track where the energy is.
The data favours neither, so the default keeps the interpretation that matches
what a correction factor is for.

**Geographic against degree-space distance: also within noise.** Clustering raw
degrees is geometrically wrong, a degree of longitude being 111 km at the equator
and 71 km at 50N, so `geographic=True` clusters on unit-sphere Cartesian
coordinates where Euclidean distance is monotone in great-circle distance.

| region | k | degree-space | geographic | delta |
| --- | --- | --- | --- | --- |
| BR | 60 | 0.0763 | **0.0752** | -1.4% |
| DE | 100 | 0.0403 | **0.0402** | -0.2% |
| UK | 100 | 0.0755 | 0.0754 | -0.1% |
| US | 100 | **0.0730** | 0.0737 | +1.0% |
| DK | 100 | **0.0567** | 0.0576 | +1.6% |

A prediction registered before running, that the US would move most because it
has the widest latitude span (cos(lat) 0.91 to 0.64), was **wrong**: the US got
slightly worse while DK, with the narrowest band and least distortion to correct,
moved most. **The effect does not track the distortion magnitude**, which is
itself evidence these differences are noise. `geographic` defaults to False. The
correctness argument stands on its own, since degree-space distance *is* the
wrong metric, but it should not be sold as a skill improvement.

**Both obvious improvements landed inside the noise floor.** With one seed and
one held-out year per region, effects below about 3% are not measurable.
Establishing that floor, through a second held-out year or seed repeats now the
partition is deterministic, is a prerequisite for any further clustering
experiment; otherwise each one produces another uninterpretable swing.

## What was ruled out for the US

Recorded because the negative results stand on their own.

1. **Power-curve mis-assignment.** The uniform curve (226 W/m2) sat near the
   fleet's 25th percentile, not its median (281 W/m2). Fixed by matching per
   plant, but uncorrected sim CF moved 0.333 to 0.304 against observed 0.351, so
   the gap *widened* and cluster factors barely moved. Not the cause.
2. **Anomalous fitted factors.** US k=1 gave scalar 0.5614 and offset 1.6160;
   the UK's *working* k=1 gives 0.5637 and 1.6621. The near-identity was a
   coincidence of two different reporting fractions and briefly sent the
   investigation the wrong way.
3. **Train/apply asymmetry.** Ruled out by the same code path improving DK, DE,
   UK and Brazil.
4. **Non-stationarity.** Uncorrected MBE per year is stable and small (2019
   +0.0053, 2020 +0.0033, 2021 +0.0107, 2022 +0.0219).
5. **Unscreened curtailment.** Real, but not the cause; see below.

## The curtailment confound

The US is the only region with unscreened curtailment (Brazil masks 2610
complex-months). Splitting the 2022 held-out set by balancing area:

| subset | n | uncorrected MBE | uncorrected MAE |
| --- | --- | --- | --- |
| ERCOT + SPP | 2697 | **+0.0391** | 0.0807 |
| excluding ERCOT + SPP | 3398 | **-0.0241** | 0.0861 |
| all | 6095 | +0.0039 | 0.0837 |

**The near-zero fleet-wide uncorrected bias is an aggregation artefact.** Bias is
positive inside the curtailed regions, the direction suppressed observed output
produces, and negative outside, nearly cancelling. ERA5 is **not** unbiased for
the US; it is regionally biased in opposite directions and the aggregate hides
it. Any claim of the form "ERA5 needs no correction in the US" is unsupported.

This did not explain the correction failure, since pre-fix the correction lost on
the clean subset too (0.0861 to 0.1215), but it remains a live data-quality issue
for interpreting US results.

*Caveat.* These subset numbers are an independent unweighted recomputation over
matched plant-months and do not reproduce `metrics.csv` exactly (full-fleet
uncorrected MAE 0.0837 here against 0.0796 in the harness, which weights
differently). Absolute values should come from `metrics.csv`; only the
between-subset contrast is claimed.

## Power curves: per-plant matching, and a trap

`plant_hub_heights_from_uswtdb` already computed `rotor_diameter_m` and
`build_us_metadata` discarded it, so every US plant took one uniform curve while
AU-NEM and Europe match per site. Diameter is now carried through.

**The obvious implementation is wrong.** `vwf.data.add_models` matches on
specific power alone, which is safe against a utility-only catalogue but not
against the bundled open library, which carries distributed machines down to
1 kW. A 1.5 kW Pika sits near 212 W/m2, indistinguishable from a modern utility
turbine. On the real fleet, plain `add_models` gave **28% of plants a sub-100 kW
curve** and paired a 1 kW SWIFT with 4.2 MW machines. `assign_curves_from_library`
therefore constrains rated capacity to 0.5 to 2.0 times the plant's per-turbine
kW *first*, then picks nearest specific power within that band.

On the merged 236-curve library: 95 distinct curves assigned, 1181 of 1284
matched, rating ratio p5 0.60 / p50 1.00 / p95 1.94, and the top assignments are
the actual US workhorses (Vestas V100/V110, GE 1.6, REpower MM92).

**Reproducibility.** Open-library-only gives uncorrected sim CF 0.309 against
0.304 for open plus licensed, with cluster factors agreeing to about three
decimals. **The open-library result reproduces the licensed one**, so published
results need not depend on non-redistributable curves.

## Two silent traps

Both bite only at scale, which is why smaller regions never surfaced them.

- **Unbounded nearest-cell `argmin`.** `aggregate_turbines_to_grid` assigns each
  turbine its nearest grid cell with no distance limit, so 6 Hawaii plants
  (198.6 MW) were snapped to the westernmost CONUS cell about 2000 km away and
  simulated with Californian wind. Screened at staging by `filter_to_bbox`; the
  guard belongs in `aggregate_turbines_to_grid` for every region.
- **One height level per unique hub height.** `interpolate_wind` builds
  `time x lat x lon x height` with a level per *distinct* hub height. USWTDB's
  continuous heights gave the US 237 of them: a 51 GB array, and the first US
  train run was SIGKILLed. Binning to 10 m, the granularity
  `aggregate_turbines_to_grid` already uses, leaves 12 levels, 2.6 GB and 79 s.
  Brazil survived only because ONS has no hub-height data. The underlying
  scaling, O(grid x unique_heights) for an O(turbines) result, is unfixed.

**DK, DE and UK cannot have been touched by the scalar bug.** Their reporting
fraction is *exactly* 1.000 (DK 5588 plants, DE 10477, UK 6345, zero missing
observations), so the `present` mask is all-True and the fixed and unfixed
`weighted_avg` execute identical arithmetic. That is why the method "always
worked" on those three and only the US, at 43.1% reporting, broke. All three were
re-run end to end and the correction improves every one (DK -55% MAE, DE -37%,
UK -37%).

## Open

1. **Sweep `cluster_list` per region.** 8 to 10 is below the optimum everywhere,
   k=1 is actively harmful, and DK shows the optimum is interior. No universal
   value exists.
2. **Resolve the DK harness-versus-legacy disagreement at high k** before
   publishing a DK cluster-count recommendation.
3. **Screen US curtailment** into the shared `pyvwf.qc` module, with
   `ons_br.constrained_off_account` as the working reference. Needed to interpret
   the +0.039 / -0.024 regional bias split.
4. **Move the domain guard into `aggregate_turbines_to_grid`**, so no region can
   silently simulate a site outside its reanalysis box.
5. **Fix `interpolate_wind` scaling** by interpolating to points then applying
   the log law. The log law is nonlinear in z0, so reordering shifts results for
   every existing region and needs a deliberate decision.
