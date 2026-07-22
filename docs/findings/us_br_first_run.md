# US + Brazil — first real train/evaluate, and a scalar-dilution bug

**Question.** Both new regions are staged against real data (EIA-923/860 +
USWTDB; ONS + ANEEL) and real ERA5. Does the `affine-wind` correction improve
held-out skill, as it does for DK/DE/UK — and how many spatial clusters does it
need?

**Verdict: PASS, after fixing a bug this run exposed.** Brazil worked
immediately (held-out MAE -18% at k=8, -36% at k=120). The US initially came
out *worse than no correction at all*, which traced to a real defect in
`calculate_scalar`: the observed mean was diluted by the fraction of capacity
that reported, so any region with sparse reporting got a systematically
under-scaled correction. Fixed; the US correction now beats uncorrected on MAE,
RMSE and correlation. DK/DE/UK were never affected because their monthly
reporting is near-complete — which is why three regions of prior work never
exposed it.

## Method

- Regions: `configs/regions/us.toml` (train 2019-2021, test 2022; 1284 plants,
  356 in the training fleet after the complete-reporting filter) and `br.toml`
  (train 2021-2023, test 2024; 193 complexes, 125 in the training fleet).
- ERA5: 48 monthly CDS files per region reduced to yearly daily files
  (`combine_era5_*_daily.py`), 1461 days each, loaded via `prep_era5`.
- Curves: bundled open library merged with the licensed library (236 curves),
  matched per plant for the US; Brazil uniform (ONS carries no turbine data).
- Skill: `scripts/validate_region.py evaluate` on the held-out year,
  `metrics.csv`.

## The bug: observed means diluted by the reporting fraction

`correction.calculate_scalar` aggregates observed and simulated CF per
(cluster, year, time-slice) and takes `scalar = obs / sim`. Its `weighted_avg`
computed:

```python
(v * w).sum(min_count=1) / w.sum()
```

The numerator skips NaN observations; the denominator summed **every** unit's
capacity, including plants that reported nothing. `sim` is never NaN, so only
`obs` was scaled down — by exactly the reporting fraction:

> **`scalar_computed = scalar_true x (fraction of capacity reporting)`**

Verified on the real US fleet, 2019:

| quantity | value |
| --- | --- |
| capacity fraction reporting | **0.431** |
| obs weighted, as computed | 0.1594 |
| obs weighted, correct | 0.3702 |
| ratio (0.1594/0.3702) | **0.4306** = the reporting fraction |
| sim weighted | 0.3387 |
| **scalar as computed** | **0.4708** |
| **scalar correct** | **1.0931** |

The US correction was therefore pushing wind speed **down 53%** where it should
nudge it **up 9%**.

**The decisive symptom** was that corrected bias was ~-0.15 *in-sample as well
as out* (2019 -0.159, 2020 -0.160, 2021 -0.147, 2022 -0.145). A correction that
optimises CF bias cannot leave -0.15 on the data it was fitted to; that ruled
out overfitting and poor transfer, and pointed upstream of fit/apply.

Fix: mask to rows that have a value, in numerator and denominator both. The
`min_count=1` semantics are preserved so an all-missing group still returns NaN.

**Why the test suite missed it.** `test_correction.py` covered an all-present
group and an all-missing group, but never a *partly* reporting one — the only
configuration where the two denominators differ. Two tests added; the second
uses lopsided capacities so a naive "count reporting rows" fix still fails.

## Result 1 — the US, before and after the fix

Held-out 2022, 520 units, 6078 samples, k=10.

| variant | MAE | RMSE | MBE | r |
| --- | --- | --- | --- | --- |
| uncorrected | 0.0796 | 0.1098 | +0.022 | 0.733 |
| affine-wind, **before fix** | 0.1527 | 0.1729 | -0.145 | 0.725 |
| affine-wind, **after fix** | **0.0741** | **0.1020** | +0.025 | **0.744** |

The correction now beats uncorrected on MAE (-6.9%), RMSE (-7.1%) and
correlation. The fitted factors change character completely: scalars go from a
near-uniform 0.31-0.66 across all clusters to 0.88-1.58, straddling 1.0 with
genuine cluster-to-cluster spread, and offsets shrink to small values of both
signs. The flat ~0.5 scalars were the bug's signature.

## Result 2 — Brazil: unaffected by the fix, and it already worked

Held-out 2024, 151 units, 389 samples.

| variant | k | slice | MAE | MBE | r |
| --- | --- | --- | --- | --- | --- |
| uncorrected | - | - | 0.1102 | -0.046 | 0.525 |
| affine-wind | 8 | fixed | 0.0905 | -0.023 | 0.602 |
| affine-wind | 8 | season | 0.0901 | +0.008 | 0.593 |
| affine-wind | 50 | fixed | 0.0796 | -0.016 | 0.700 |
| affine-wind | 120 | fixed | 0.0781 | -0.013 | 0.732 |
| affine-wind | 120 | season | **0.0702** | +0.022 | **0.743** |

Brazil's factors are identical to three decimal places before and after the fix
(1.0385, 0.7981 unchanged), because its training fleet is already filtered to
complete-reporting complexes — dilution factor ~1. That makes Brazil a useful
control: the fix is not a free knob that moves everything, it corrects exactly
the regions with reporting gaps.

Explicit Southern-Hemisphere `season` slicing beats `fixed`, which retroactively
justifies stating the seasons rather than inheriting the NH default.

## Result 3 — cluster count matters, and the optimum is region-specific

All five regions, held-out MAE, best slice, measured through the FIXED scalar:

| region (test year) | uncorrected | k=1 | k~10 | k~100 | k~200-500 |
| --- | --- | --- | --- | --- | --- |
| US (2022) | 0.0796 | 0.0889 | 0.0741 | 0.0729 (k=50) | **0.0712** (k=200) |
| BR (2024) | 0.1102 | 0.1148 | 0.0901 (k=8) | **0.0708** (k=120) | - (fleet caps at 125) |
| DK (2020) | 0.1255 | - | 0.0623 | **0.0562** | 0.0683 (k=500) |
| DE (2019) | 0.0655 | - | 0.0479 | 0.0446 | **0.0414** (k=500) |
| UK (2019) | 0.1112 | - | 0.0877 | 0.0762 | **0.0696** (k=500) |

Three things, none of which the shipped configs reflect:

1. **k=1 is WORSE than not correcting**, in both regions where it was tested
   (US 0.0889 vs 0.0796; BR 0.1148 vs 0.1102). A single global scalar/offset is
   actively harmful. Spatial resolution is what makes the method work, not a
   refinement on top of it.
2. **k=10 is well below the optimum everywhere.** Every region improves
   materially from k=10 to k=100.
3. **Skill is NOT monotonic in k, and not smooth either.** Extending DK to
   k=700 and k=1000 (season slice, held-out 2020):

   | k | 10 | 100 | 500 | **700** | 1000 |
   | --- | --- | --- | --- | --- | --- |
   | MAE | 0.0623 | 0.0562 | 0.0683 | **0.0516** | 0.0684 |
   | r | 0.809 | 0.832 | 0.773 | **0.843** | 0.775 |
   | MBE | +0.023 | +0.019 | -0.003 | +0.018 | -0.006 |

   k=700 is the best DK result of any run, so this is **not** overfitting past
   an optimum (that degrades monotonically). An earlier draft of this document
   claimed DK "peaks near k=100 and overfits past it"; that was wrong and is
   retracted. The good and bad runs split cleanly: the bad ones (500, 1000)
   are exactly those where MBE collapses to ~0 while MAE, RMSE and r all
   worsen.

   Ruled out as causes: **overfitting** (k=700 beats k=100); **evaluate-time
   re-clustering mismatch** (the KMeans refit in `run_evaluate` reproduces the
   training partition with 100% identical labels at every k); **cross-k state
   leakage in the driver loop** (DK k=500 run alone is BIT-IDENTICAL to k=500
   run as the last of [10, 100, 500] — so multi-k sweeps are trustworthy); and
   **degenerate factors** (no identity fallbacks at any k; scalar p50 is
   0.72-0.74 across k=100/500/700/1000).

   **Cause found: `init="random"` in `cluster_turbines`.** Holding k=500 fixed
   and varying only the KMeans seed:

   | seed | 0 | 1 | 7 | 42 |
   | --- | --- | --- | --- | --- |
   | MAE (`init="random"`, current) | 0.0514 | 0.0842 | 0.0628 | 0.0683 |
   | MAE (`init="k-means++"`) | 0.0513 | 0.0511 | 0.0507 | 0.0509 |

   With random init, MAE at ONE k spans 0.0514-0.0842 (range 0.0328) — larger
   than the entire k=10..1000 spread at fixed seed (0.0516-0.0684). **The
   k-curve was noise.** k=700's apparent win was partition luck; seed 0 reaches
   the same score at k=500.

   `k-means++` collapses the spread to 0.0006, a ~55x reduction, and is also
   BETTER than the luckiest random draw (0.0507 vs 0.0514). It is sklearn's
   default precisely because random init falls into bad local optima, and
   `n_init=10` selects among those by INERTIA, which is not held-out skill.

   **Consequences.** Any single-run-per-k sweep measures partition lottery, not
   k, so "optimal k" cannot be read off one run per k under the current init.
   What survives unaffected: **k=1 is worse than uncorrected** (US 0.0889 vs
   0.0796; BR 0.1148 vs 0.1102) — at k=1 there is no partition to be lucky
   about, so that result carries no seed variance — and the direction that
   larger k beats k=1 by a wide margin.

   Switching to `k-means++` is a one-argument change that makes k-sweeps
   meaningful, but it moves results for EVERY region including DK/DE/UK, so it
   is a deliberate decision, not a silent fix. NOT APPLIED.

The practical ceiling is the *training* fleet, not the metadata fleet: Brazil's
is 125 complexes (not 193), so k=150 raises
`ValueError: n_samples=125 should be >= n_clusters=150`.

Two caveats on this table. The pre-fix US sweep (k=1/100/300, MAE 0.153 ->
0.143 -> 0.110) was measured through the broken scalar and must not be cited.
And the harness DK curve *disagrees in shape* with the legacy DK run
(`output/runs/turbine_dk_research`, onshore/bimonth: 0.0703 -> 0.0636 -> 0.0607
at k=10/100/500, improving monotonically) where the harness degrades at k=500.
The runs differ in mode (all vs onshore) and slicing (fixed/season vs bimonth),
so they are not like-for-like — but the disagreement should be resolved before
any k recommendation is published for DK.

## What was ruled out for the US, before the bug was found

Recorded because the negative results stand on their own.

1. **Power-curve mis-assignment.** The uniform curve
   (`2019COE_Market_Average_2.6MW_121`, 226 W/m2) sat near the fleet's 25th
   percentile, not its median (281 W/m2). Fixed by matching per plant (below).
   Uncorrected sim CF moved 0.333 -> 0.304 against observed 0.351 — the gap
   *widened* — and cluster factors barely moved. Not the cause.
2. **Anomalous fitted factors.** US k=1 gave scalar 0.5614 / offset 1.6160;
   UK's *working* k=1 gives 0.5637 / 1.6621. The near-identity was a
   coincidence of two different reporting fractions and briefly sent the
   investigation the wrong way.
3. **Train/apply asymmetry.** Ruled out by the same code path improving DK, DE,
   UK and Brazil.
4. **Non-stationarity.** Uncorrected MBE per year is stable and small
   (2019 +0.0053, 2020 +0.0033, 2021 +0.0107, 2022 +0.0219).
5. **Unscreened ERCOT/SPP curtailment.** Real, but not the cause — see below.

## The curtailment confound (real, and still open)

The US is the only region with unscreened curtailment (Brazil masks 2610
complex-months). Splitting the 2022 held-out set by balancing area — ERCOT (TX)
212 plants / 39.3 GW, SPP-ish (KS/OK/NE/SD/ND/NM) 235 / 34.9 GW, other 837 /
67.4 GW:

| subset | n | uncorrected MBE | uncorrected MAE |
| --- | --- | --- | --- |
| ERCOT + SPP | 2697 | **+0.0391** | 0.0807 |
| excluding ERCOT + SPP | 3398 | **-0.0241** | 0.0861 |
| all | 6095 | +0.0039 | 0.0837 |

**The near-zero fleet-wide uncorrected bias is an aggregation artifact.** Bias
is *positive* inside the curtailed regions (+0.039 — the direction that
suppressed observed output produces) and *negative* outside (-0.024), nearly
cancelling. ERA5 is **not** unbiased for the US; it is regionally biased in
opposite directions and the aggregate hides it. Any claim of the form "ERA5
needs no correction in the US" is unsupported.

This did not explain the correction failure (pre-fix, the correction lost on the
clean subset too: 0.0861 -> 0.1215), but it remains a live data-quality issue
for interpreting US results.

*Caveat.* These subset numbers are an independent unweighted recomputation over
matched plant-months and do not reproduce `metrics.csv` exactly (full-fleet
uncorrected MAE 0.0837 here vs 0.0796 in the harness, which weights
differently). Absolute values should come from `metrics.csv`; only the
*between-subset contrast* is claimed.

## Power curves: per-plant matching, and a trap

`plant_hub_heights_from_uswtdb` already computed `rotor_diameter_m` and
`build_us_metadata` discarded it, so every US plant took one uniform curve while
AU-NEM and Europe match per site. Diameter is now carried through.

**The obvious implementation is wrong.** `vwf.data.add_models` matches on
specific power alone — safe against a utility-only catalogue (AU-NEM's licensed
library) but not against the **bundled open library**, which carries distributed
machines down to 1 kW. A 1.5 kW Pika sits near 212 W/m2, indistinguishable from
a modern utility turbine. On the real fleet, plain `add_models` gave **28% of
plants a sub-100 kW curve** and paired a 1 kW SWIFT with 4.2 MW machines.
`assign_curves_from_library` therefore constrains rated capacity to 0.5-2.0x the
plant's per-turbine kW *first*, then picks nearest specific power within it.

Merged open + licensed libraries (236 curves): 95 distinct curves assigned,
1181/1284 matched, rating ratio p5 0.60 / p50 1.00 / p95 1.94. Top assignments
are the actual US workhorses (Vestas V100/V110, GE 1.6, REpower MM92).

**Reproducibility.** Open-library-only gives uncorrected sim CF 0.309 vs 0.304
for open+licensed, with cluster factors agreeing to ~3 decimal places. **The
open-library result reproduces the licensed-library result**, so published
results need not depend on non-redistributable curves.

## Two silent traps fixed along the way

Both bite only at scale, which is why smaller regions never surfaced them.

- **Unbounded nearest-cell `argmin`.** `aggregate_turbines_to_grid` assigns each
  turbine its nearest grid cell with no distance limit, so 6 Hawaii plants
  (198.6 MW) were snapped to the westernmost CONUS cell ~2000 km away and
  simulated with Californian wind. Screened at staging by `filter_to_bbox`
  (`--bbox`); the guard belongs in `aggregate_turbines_to_grid` for every region.
- **One height level per unique hub height.** `interpolate_wind` builds
  `time x lat x lon x height` with a level per *distinct* hub height. USWTDB's
  continuous heights gave the US 237 of them — a ~51 GB array, and the first US
  train run was SIGKILLed (exit 137). Binning to 10 m (the granularity
  `aggregate_turbines_to_grid` already uses) leaves 12 levels, ~2.6 GB, 79 s.
  Brazil survived only because ONS has no hub-height data. The underlying
  scaling — O(grid x unique_heights) for an O(turbines) result — is unfixed.

## Named follow-ups

1. ~~Re-check DK/DE/UK against the scalar fix.~~ **RESOLVED — no effect,
   provably.** Their reporting fraction is *exactly* 1.000 (DK 5588 plants, DE
   10477, UK 6345; zero missing observations). With no NaNs the `present` mask
   is all-True, so the fixed and unfixed `weighted_avg` execute identical
   arithmetic: the bug cannot have touched them. This also explains why the
   method "always worked" on those three, and why only the US (43.1% reporting)
   broke. All three were re-run end-to-end and the correction improves every
   one (DK -55% MAE, DE -37%, UK -37%).
2. **Sweep `cluster_list` per region and pick the optimum.** 8-10 is below the
   optimum everywhere, k=1 is actively harmful, and DK shows the optimum is
   interior (peaks ~100, degrades by 500). No universal value exists; each
   region needs its own sweep against the training-fleet ceiling.
3. **Resolve the DK harness-vs-legacy disagreement at high k** (see Result 3)
   before publishing a DK k recommendation.
3. **Screen US curtailment** into the shared `pyvwf.qc` module, with
   `ons_br.constrained_off_account` as the working reference. Needed to
   interpret the +0.039 / -0.024 regional bias split.
4. **Move the domain guard into `aggregate_turbines_to_grid`**, so no region can
   silently simulate a site outside its reanalysis box.
5. **Fix `interpolate_wind` scaling** (interpolate to points, then apply the log
   law) — but the log law is nonlinear in z0, so reordering shifts results for
   every existing region and needs a deliberate decision.
