# US + Brazil — first real train/evaluate, and where the correction stops helping

**Question.** Both new regions are staged against real data (EIA-923/860 +
USWTDB; ONS + ANEEL) and real ERA5. Does the `affine-wind` correction improve
held-out skill, as it does for DK/DE/UK — and how many spatial clusters does it
need?

**Verdict: SPLIT.** Brazil behaves like the European regions: the correction cuts
held-out MAE by 36% and raises correlation from 0.53 to 0.74. The **US does not**:
on every cluster count tested the correction is *worse than no correction at all*,
and it degrades correlation. The method is not at fault — US factors are
near-identical to UK's working ones — and the cause is not curtailment,
power-curve assignment, or cluster count, all of which were tested and
eliminated. It remains open.

A second, independent result: **`num_clusters` = 8-10 (the shipped configs) is far
too low for both regions.** Skill was still improving at the largest k tested.

## Method

- Regions: `configs/regions/us.toml` (train 2019-2021, test 2022, 1284 plants /
  523 monthly reporters) and `br.toml` (train 2021-2023, test 2024, 193
  complexes / 125 in the training fleet).
- ERA5: 48 monthly files per region from CDS, reduced to yearly daily files
  (`combine_era5_*_daily.py`), 1461 days each, loaded through `prep_era5`.
- Curves: the bundled open library merged with the licensed library
  (236 curves), matched per plant for the US; Brazil is uniform (ONS carries no
  turbine data). See "Power curves" below.
- Skill: `scripts/validate_region.py evaluate`, held-out year, `metrics.csv`.

## Result 1 — Brazil: the correction works

Held-out 2024, 151 units, 389 samples.

| variant | k | slice | MAE | MBE | r |
| --- | --- | --- | --- | --- | --- |
| uncorrected | - | - | 0.1102 | -0.046 | 0.525 |
| affine-wind | 50 | fixed | 0.0796 | -0.016 | 0.700 |
| affine-wind | 120 | fixed | 0.0781 | -0.013 | 0.732 |
| affine-wind | 50 | season | 0.0744 | +0.019 | 0.695 |
| affine-wind | 120 | season | **0.0702** | +0.022 | **0.743** |

Every metric improves: MAE -36%, |MBE| more than halved, r 0.525 -> 0.743.
The explicit Southern-Hemisphere `season` slicing beats `fixed`, which
retroactively justifies stating the seasons rather than inheriting the NH
default.

## Result 2 — the US: the correction hurts, at every k

Held-out 2022, 520 units, 6078 samples.

| variant | k | MAE | MBE | r |
| --- | --- | --- | --- | --- |
| **uncorrected** | - | **0.0796** | +0.022 | **0.733** |
| affine-wind | 1 | 0.1527 | -0.145 | 0.725 |
| affine-wind | 2 | 0.1537 | -0.146 | 0.713 |
| affine-wind | 100 | 0.1430 | -0.127 | 0.597 |
| affine-wind | 300 | 0.1099 | -0.078 | 0.633 |

MAE falls steadily with k (0.153 -> 0.110) and |MBE| halves, but **no
configuration beats doing nothing**, and correlation *drops* under correction
(0.733 -> 0.60-0.63). A pure level error would leave r untouched; losing r means
the correction injects noise as well as bias. At k=300 there are ~1.7 observed
plants per cluster, so high-k factors are fit on one or two plant-years each.

## Result 3 — cluster count: 8-10 is far too low

From the existing DK run (`output/runs/turbine_dk_research`, onshore, bimonth):

| k | 1 | 3 | 10 | 20 | 50 | 100 | 200 | 500 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MAE | 0.0717 | 0.0731 | 0.0703 | 0.0692 | 0.0671 | 0.0636 | 0.0624 | **0.0607** |

Flat from k=1-10, then improving steadily to -15% at k=500 and still falling.
Bias is nearly constant (+0.047 -> +0.040) while MAE drops, so extra clusters
resolve *local* structure rather than the overall level. Brazil agrees (k=50 ->
120 improves both slices) and the US MAE trend agrees. **The shipped
`cluster_list = [10]` / `[8]` sit in the flat zone where clustering buys
nothing.**

Practical ceiling: k cannot exceed the *training* fleet. Brazil's is 125
complexes (not 193), so k=150 raises
`ValueError: n_samples=125 should be >= n_clusters=150`.

## What was ruled out for the US

Each was a live hypothesis, tested and killed. Recorded because the negative
results are the useful part.

1. **Power-curve mis-assignment.** The uniform curve
   (`2019COE_Market_Average_2.6MW_121`, 226 W/m2) sat near the fleet's 25th
   percentile, not its median (281 W/m2). Fixed by matching per plant (see
   below). Uncorrected sim CF moved 0.333 -> 0.304 against observed 0.351 — the
   gap *widened* — and the cluster factors barely moved. Not the cause.
2. **Anomalous fitted factors.** US k=1 gives scalar 0.5614 / offset 1.6160;
   UK's *working* k=1 gives 0.5637 / 1.6621. Near-identical. The fit is
   behaving normally; these factors are simply right for a region with real
   bias and wrong for one without.
3. **Train/apply asymmetry in the correction code.** Ruled out by the same code
   path improving DK, DE, UK and (here) Brazil.
4. **Non-stationarity between train and test.** Uncorrected MBE per year:
   2019 +0.0053, 2020 +0.0033, 2021 +0.0107, 2022 +0.0219. Stable and small
   throughout; the test year is not anomalous.
5. **Unscreened ERCOT/SPP curtailment.** See below — real, but not the cause.

## The curtailment confound (real, but not the explanation)

The US is the only region with unscreened curtailment (Brazil masks 2610
complex-months; DK/DE/UK have little). Splitting the 2022 held-out set by
balancing area — ERCOT (TX) 212 plants / 39.3 GW, SPP-ish (KS/OK/NE/SD/ND/NM)
235 / 34.9 GW, other 837 / 67.4 GW:

| subset | n | uncorrected MBE | uncorrected MAE | corrected (k=300) MAE |
| --- | --- | --- | --- | --- |
| ERCOT + SPP | 2697 | **+0.0391** | 0.0807 | 0.1134 |
| excluding ERCOT + SPP | 3398 | **-0.0241** | 0.0861 | 0.1215 |
| all | 6095 | +0.0039 | 0.0837 | 0.1179 |

Two conclusions:

- **The "already unbiased" US baseline is an aggregation artifact.** Bias is
  *positive* in the curtailed regions (+0.039 — the direction curtailment
  suppressing observed output produces) and *negative* outside them (-0.024).
  They nearly cancel to +0.004 fleet-wide. ERA5 is **not** unbiased for the US;
  it is regionally biased in opposite directions, and the aggregate hides it.
  Any headline of the form "ERA5 needs no correction in the US" is unsupported.
- **Curtailment does not explain the correction's failure.** Excluding all 447
  ERCOT/SPP plants, the correction is still clearly worse (0.0861 -> 0.1215,
  r 0.739 -> 0.617). It fails on the clean subset too.

*Caveat.* These subset numbers are an independent unweighted recomputation over
matched plant-months and do not reproduce `metrics.csv` exactly (full-fleet
uncorrected MAE 0.0837 here vs 0.0796 in the harness, which weights
differently). Absolute values should be taken from `metrics.csv`; only the
*between-subset contrast* is claimed here.

## Power curves: per-plant matching, and a trap

`plant_hub_heights_from_uswtdb` already computed `rotor_diameter_m` and
`build_us_metadata` discarded it, so every US plant took one uniform curve while
AU-NEM and Europe match per site. Diameter is now carried through and each plant
gets a real curve.

**The obvious implementation is wrong.** `vwf.data.add_models` matches on
specific power alone. That is safe against a utility-only catalogue (AU-NEM's
licensed library) but not against the **bundled open library**, which also
carries distributed machines down to 1 kW — a 1.5 kW Pika sits near 212 W/m2,
indistinguishable from a modern utility turbine. Measured on the real fleet,
plain `add_models` gave **28% of plants a sub-100 kW curve** and paired a 1 kW
SWIFT with 4.2 MW machines. `assign_curves_from_library` therefore constrains
rated capacity to 0.5-2.0x the plant's per-turbine kW *first*, then picks
nearest specific power within that band.

Open + licensed libraries merged (236 curves): 95 distinct curves assigned,
1181/1284 matched, rating ratio p5 0.60 / p50 1.00 / p95 1.94. Top assignments
are the actual US fleet workhorses (Vestas V100/V110, GE 1.6, REpower MM92).

**Reproducibility note.** Matching against the open library alone gives
uncorrected sim CF 0.309 vs 0.304 for open+licensed — a 0.005 difference, with
cluster factors agreeing to ~3 decimal places. **The open-library result
reproduces the licensed-library result**, so published results need not depend
on non-redistributable curves.

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
  Brazil survived only because ONS has no hub-height data and every complex
  takes the uniform 100 m default (1 level). The underlying scaling —
  O(grid x unique_heights) for an O(turbines) result — is unfixed.

## Named follow-ups

1. **Explain the US correction failure.** It is not curves, curtailment,
   cluster count, non-stationarity, or the fit. Next: check whether the fitted
   scalar/offset magnitude is responsive to the size of the bias present at all,
   since identical factors appear for the US (small bias) and UK (large bias).
2. **Raise `cluster_list`.** 8-10 is in the flat zone; both regions improved to
   the largest k tested. Sweep against the training-fleet ceiling.
3. **Screen US curtailment** into the shared `pyvwf.qc` module, with
   `ons_br.constrained_off_account` as the working reference. Needed to
   interpret the regional bias split, even though it does not explain Result 2.
4. **Move the domain guard into `aggregate_turbines_to_grid`**, so no region can
   silently simulate a site outside its reanalysis box.
5. **Fix `interpolate_wind` scaling** (interpolate to points, then apply the log
   law) — but the log law is nonlinear in z0, so reordering shifts results for
   every existing region and needs a deliberate decision.
