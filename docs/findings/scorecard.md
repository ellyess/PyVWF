# Multi-region validation scorecard

Per region, how much PyVWF's affine wind-speed correction reduces the error
between ERA5-simulated and observed capacity factors on a held-out year. Every
number is read from a `metrics.csv` under `output/validation/`, with the source
path given so each is auditable. Screening-level validation, one test year per
region, not an accredited yield assessment.

**Correction notice, 2026-09-11: the UK and NZ rows do not show that the
correction improves those regions.** Both rows report a lower corrected RMSE,
and the reading under the turbine-level table counts both among the clean rows
where the correction lowers RMSE and drives MBE toward zero. Every figure in
both rows stands as measured. The claim does not: when the test year's units
are resampled, neither row's gain can be distinguished from zero, and in both a
few units that the correction makes worse decide the result.

| Row | Units | Capacity-effective units | RMSE gain [95% interval] | MAE gain [95% interval] | Corrected squared error: largest unit, largest five |
|---|---|---|---|---|---|
| UK k50 fixed | 348 farms | 104 | 0.031 [-0.001, 0.069] | 0.034 [0.014, 0.057] | 37%, 60% |
| NZ k7 fixed | 12 farms | 9.3 | 0.051 [-0.032, 0.114] | 0.065 [-0.018, 0.119] | 61%, 87% |

The gain is uncorrected minus corrected error, capacity-weighted as in
`metrics.csv`. Each interval comes from 1,000 paired draws of the test year's
units with replacement. Capacity-effective units is (sum of weights) squared
over the sum of squared weights. Every row's rebuilt metrics reproduced its
`metrics.csv` before resampling. Scripts: `scripts/analysis/baseline_bootstrap.py`
and `scripts/analysis/unit_concentration.py`. Data:
`output/curve_library_study_2026-09-11/baseline_bootstrap/`
(`all_concentration.csv`, `UK_top5_units.csv`, `NZ_top5_units.csv`), from the
`curve_resolution_backfill_2026-09-11` evaluate frames.

- **UK.** Four of the five farms that carry the most corrected error are worse
  corrected than uncorrected. The largest, 2.2% of capacity, goes from a farm
  RMSE of 0.31 to 0.47. MBE moves from +0.037 to -0.038, not toward zero. The
  MAE gain is resolved; the RMSE gain is not. The choice of `k=50` over `k=100`
  (0.115 against 0.123) cannot be resampled without a re-run, because the
  cluster sweep's corrected frames were not saved. It remains open.
- **NZ.** One farm, 12% of capacity, goes from a farm RMSE of 0.05 to 0.24 and
  carries 61% of the corrected squared error. Neither gain is resolved.
- **The NZ limit is structural.** With 12 farms, 9.3 of them
  capacity-effective, any one farm can decide the fleet result. Such a fleet
  resolves a gain only if nearly every farm improves by a similar amount, in
  any test year. The marker records a limit of the region, not an unlucky year.
- **The UK's 348 units are farms, not turbines.** They are ROC stations whose
  generation is spread equally over turbine-shaped rows (`docs/design/harness.md`).
  Until this notice, the table's Fleet column said 348 turbines.
- **The MBE reading is also false for the US**, whose MBE moves from +0.022
  to +0.024.

These intervals understate the uncertainty, for three reasons. They are
conditional on each row's single test year. They treat units as independent,
although neighbouring units share weather. And both rows' cluster counts were
chosen as the best of a sweep on that same test year (`method-cluster-count.md`),
and the resampling leaves that choice out.

The same check finds no such pattern in DE, DK or US. Their RMSE gain intervals
are 0.027 to 0.031, 0.059 to 0.063 and 0.008 to 0.017, and no single unit
carries more than 2.6% of corrected squared error in any of them. BR's gain is
also resolved, at 0.016 to 0.050.

Two rows are resolved only narrowly, and their claims are not withdrawn.
AU-NEM's RMSE gain is 0.021, with an interval of 0.001 to 0.040. AR's is 0.018,
with an interval of 0.002 to 0.031, and its MAE gain interval, -0.001 to 0.024,
includes zero. The interval is itself a lower bound on the uncertainty, so an
interval that excludes zero by 0.001 or 0.002 is not a clean result. CL is not
assessed in this notice.

**Correction notice, 2026-09-11: the daggered rows' worst damage never reached
their scores.** The reading under the turbine-level table says that, in the four
daggered rows, the fleet average absorbed the damage of a degenerate fit. It did
not absorb it. The corrected score never contained it.

A degenerate cluster loses corrected values in two ways. A failed offset leaves
the cluster with no factors to apply. An implausible scalar pushes corrected
speeds above 40 m/s, where the power curves end, and a large negative offset
pushes them below 0 m/s. Outside the curve, the interpolator returns no value
rather than zero output. Until 2026-09-11 a missing value dropped out of the
score, and a monthly mean still skips missing days. So the corrected score
left out exactly the units and days that a degenerate fit damaged most.

| Row | Degenerate clusters with missing values | Units (capacity) | Missing unit-days: failed offset / above 40 m/s / below 0 m/s | Unit-months wholly / partly missing | Corrected RMSE, as scored / with off-curve days as zero |
|---|---|---|---|---|---|
| CL k10 fixed | 6, 8 | 6 plants (26%) | 1,464 / 695 / 0 | 65 / 7 | 0.104 / 0.109 |
| AR k10 fixed | 7 | 2 plants (5.1%) | 0 / 308 / 0 | 0 / 24 | 0.133 / 0.132 |
| US k250 fixed | 15, 38, 102, 156, 183 | 28 plants (1.4%) | 0 / 727 / 2,569 | 21 / 244 | 0.097 / 0.096 |
| BR k60 fixed | 29 | 1 complex (0.3%) | 0 / 0 / 128 | 0 / 12 | 0.105 / 0.107 |

The last column scores the saved frames on common rows twice, the second time
with every off-curve corrected day counted as zero output. Zero is the physical
value below cut-in and above cut-out. This is a measurement, and the simulation
is unchanged. Scripts: `scripts/analysis/missing_value_audit.py`,
`off_curve_sensitivity.py` and `common_row_rescore.py`. Data:
`output/curve_library_study_2026-09-11/` (`missing_value_audit/`,
`off_curve_sensitivity/`, `common_row_rescore/`).

- **CL compared two different sets of plants.** Its published uncorrected
  score covered 59 plants and 677 plant-months, and its corrected score 55
  plants and 635. The rows missing from the corrected side are the worst
  uncorrected. So the published gain of 0.018 was mainly a comparison with a
  different denominator on each side, not an overstated improvement. On the
  same rows, the uncorrected RMSE itself falls from 0.123 to 0.110, which is
  more than the gain under either scoring. The harness now scores every
  variant of a run on the rows all of them can score (commit `513f57c`). For
  CL this excludes 49 plant-months, 16.0% of capacity, and six plants wholly:
  the four in cluster 6, and two that `fixed_10` cannot score in most months
  and the unreported `season_10` variant cannot score in the rest. The table
  now shows the re-run on the fixed harness
  (`output/validation/common_row_rerun_2026-09-11/CL/evaluate-2024-rerun/`):
  RMSE 0.110 uncorrected and 0.104 corrected, where 0.123 and 0.105 were
  published; MBE -0.015 and +0.001, where -0.026 and -0.002 were published;
  53 of 59 plants scored. On those 53 plants (35.8 capacity-effective), the
  gain of 0.006 has a 95% interval of -0.007 to 0.021 when plants are
  resampled, and the MAE gain of 0.004 has one of -0.009 to 0.017. The gain
  cannot be distinguished from zero, so the row carries the ‡ marker.
- **An unreported configuration moved AR's reported figures.** The common rows
  are shared by all variants of a run. AR's reported `fixed_10` row moved
  because its unreported `season_10` variant lacks two rows. The table now
  shows the re-run (`output/validation/common_row_rerun_2026-09-11/AR/evaluate-2024-rerun/`).
  Uncorrected RMSE went from 0.151 to 0.150, uncorrected MBE from +0.010 to
  +0.011, and correlation from 0.44 to 0.43. Corrected RMSE and MBE are
  unchanged at displayed precision. AR's resampled gain interval on the common
  rows is 0.002 to 0.030, against 0.002 to 0.031 in the notice above. The
  re-run does not reach AR's 24 partly missing plant-months, which are still
  scored on the days that remain.
- **The US** loses 11 of 6,078 plant-months to common-row scoring, and nothing
  moves at displayed precision. Every other row, all eight country-level rows
  included, reproduces its `metrics.csv` exactly.
- **The `min_cluster_size` comparisons under the degenerate-fit table** used
  different rows too. The pre-registered gates in `method-scalar-bounds.md`
  still pass on common rows, with smaller margins; that document carries the
  figures.
- **Below-zero days also occur in clusters that `fit_quality` calls clean**, in
  the US, BR, DE, UK, NZ, AU-NEM and FR. `fit_quality` bounds the scalar and
  checks that each offset converged. It does not check whether a scalar and
  offset pair goes negative over the observed speed range. Counting those days
  as zero moves no corrected RMSE in those rows by more than 0.002. ES and IT
  are a different case, covered by the suspension notice below.

The CL interval has the limits stated in the UK and NZ notice above, and the
choice of `k=10` on the test year also stays outside the resampling.

**Suspension notice, 2026-09-11: the IT, PT and ES rows are suspended.** They
are not simulations of those fleets' winds. The chain, in order:

1. **The ERA5 download never covered them.** The European ERA5 files cover
   12°W to 22°E and 42°N to 72°N. The IT, PT and ES configurations' boxes
   extend south of 42°N, and the download was never extended to match.
2. **The harness extrapolated silently.** `interpolate_wind` interpolates with
   `fill_value=None`, which extrapolates linearly outside the grid without a
   warning. Grid points up to 5° outside the data receive extrapolated winds,
   and uncorrected hub-height speeds reach -57.7 m/s.
3. **The fit fitted factors to that input.** In ES, clusters 0 and 3 lie
   wholly outside the data. They carry offsets of -5.64 and -4.46 m/s, at
   scalars near 0.5.
4. **Those factors push most days off the curve.** In the ES training years,
   clusters 0 and 3 put 54% and 53% of their capacity-weighted days below
   0 m/s. Those days drop out of the fit's objective and out of the score.
   This is the mechanism in the daggered-rows notice above, set off by
   fabricated input rather than by a degenerate fit.
5. **The corrected CF reads too high.** Counting the dropped days as zero
   lowers the mean corrected CF by 0.036 in ES and by 0.036 in IT. ES's
   corrected MBE goes from +0.016 to -0.020, and IT's corrected RMSE from
   0.034 to 0.064. In the ES training years, the fitted national CF matches
   the observed one only with the dropped days removed. Counted as zero, it is
   0.03 to 0.05 below observed in every year.

| Row | Capacity outside the ERA5 data | Grid points outside | Furthest outside |
|---|---|---|---|
| IT | 94.5% | 19 of 28 | 4.4° |
| PT | 89.9% | 23 of 26 | 4.5° |
| ES | 50.3% | 34 of 52 | 5.0° |

**PT shows why a stable result is not evidence.** 90% of PT's capacity is
extrapolated from a single row of data at 42°N, its northern border. Yet its
figures do not move when off-curve days count as zero, because few of its days
fall off the curve. Nothing in PT's metrics shows that its winds were never in
the input, and a reader comparing it with a sound row cannot tell the two
apart. The check that exposed ES and IT cannot see PT.

The three rows move to a table of suspended rows under the country-level
table, with their published figures kept for the record. They return once ERA5
is downloaded to cover their boxes and they are re-run. NO (4.4% of capacity,
3 of 26 grid points, up to 7.5° east of the data) and SE (0.8%, 1 of 40, up to
1.0°) stay in the table with those shares stated, and get the same download.
Script for the training-year figures: `scripts/analysis/training_objective_check.py`.
Data: `output/curve_library_study_2026-09-11/training_objective_check/`.

All rows were produced by PyVWF v0.4.0 at commit `41462e9` from a clean tree on
2026-08-24, one region per process. Runs are in
`output/validation/refresh_2026-08-24/<CODE>/`, outside the repository. The CL
and AR rows are the exception since 2026-09-11: they come from an
evaluate-only re-run of the same training directories on the common-row
harness, at commit `bbaf5b3` from a clean tree
(`output/validation/common_row_rerun_2026-09-11/`). Each row
was run from the single-configuration file committed under
`configs/regions/scorecard/` (`<code>_k<N>.toml` or `<code>_country.toml`), which
fixes the cluster count and time slice the row reports. For the eight
country-level regions that file is identical to the maintained
`configs/regions/<code>.toml`. For the nine turbine-level regions it is not: the
maintained configs carry a cluster sweep, and for seven of the nine (all but DK
and UK) that sweep does not contain the reported cluster count, so re-running the
maintained config does not reproduce the row.

Each evaluate manifest's `trained_from` records a temporary path that no longer
exists. The link to `train-refresh/` was verified on 2026-09-11: re-running every
evaluation at the same commit, against the surviving training directories,
reproduced all seventeen `metrics.csv` files byte for byte.

Seven rows (DE, DK, UK, US, BR, AU-NEM, NZ) were run with a licensed curve
library that is not redistributable, identified by sha256 in each manifest.
Four of them (DE, DK, UK, US) simulate 74% to 90% of their capacity on curves
only that library contains, and are not reproducible by a third party. The
other three (BR, AU-NEM, NZ) simulate only on curves the open library also
contains: re-run on the open library on 2026-09-11, each reproduced its
`metrics.csv` byte for byte (`output/validation/open_library_check_2026-09-11/`).
The remaining ten (CL, AR and the eight country-level regions) were run on the
bundled open library. The country-level grid points name Vestas models that the
open library does not contain, so every unit in those eight rows fell back to a
single default curve, the open library's first column:
`2019COE_DW100_100kW_27.6`, a 100 kW distributed-wind turbine.

**Other brand** is the share of each row's fitted training fleet, by capacity,
simulated on another manufacturer's curve. **Reference curve** is the share on a
research reference design or generic composite curve, which is never the unit's
own machine; on the open library it is often the only kind available, so read
it as a fact about the library rather than about the matching. **Unverifiable**
is the share where either side cannot be identified, so the match cannot be
checked in either direction. Unit counts and the largest mismatched pairs are in
`output/validation/curve_resolution_backfill_2026-09-11/cross_manufacturer_audit.csv`,
produced by `scripts/analysis/curve_match_audit.py`.

Every turbine-level row except BR is matched on specific power, by one of three
routes:

- **DE, DK and UK:** `add_models` at load time, a fuzzy manufacturer match then
  nearest specific power.
- **US, NZ, CL and AR:** `assign_curves_from_library` at processing time,
  nearest specific power within a rating band, with no manufacturer step.
- **AU-NEM:** a specific-power class.

BR assigns one uniform curve to every complex, and nothing records its
manufacturers, so its match shares are n/a rather than zero.

The two mismatch columns therefore measure how far each row rests on the
assumption that specific power fixes a curve's shape. Whether held-out skill
survives that assumption is not assessed here. Largest case: 13.4% of DE
capacity is Vestas turbines on Gamesa curves.

The rule is strict and names brands, not lineages. A Bonus turbine on a
Siemens curve counts as other brand. A GE plant on the DOE reference curve of a
GE 1.5 MW machine counts as a reference curve; that is 6.6% of US capacity.

Every model key in every turbine-level row has a curve in its
`power_curves.csv`, so nothing there was substituted. That is a different table
from `models.csv`, which the audit reads for each model's manufacturer: 11 AU-NEM
farms (14.3% of capacity) carry keys with a curve in `power_curves.csv` and no
row in `models.csv`, because the open library's normalized composites are
curve-only by design; the audit identifies them from the open library's
provenance file as reference curves. The country-level table's **Substituted**
column is the share of the evaluated fleet whose model key has no curve in
`power_curves.csv` at all. Per-row records are in each row's
`curve_resolution.csv` under the same folder.

## Turbine / plant-level (observed capacity factor per farm)

Matched real turbine curves and hub heights; k-swept affine fit; best held-out
`affine-wind` row, fleet scope.

| Region | Fleet (test) | Train → test | Uncorr RMSE | Corr RMSE | Uncorr MBE | Corr MBE | Corr r | Best cfg | Other brand | Reference curve | Unverifiable |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Germany (DE) | 4814 turbines | 2015-18 → 2019 | 0.086 | **0.057** | +0.042 | +0.001 | 0.85 | k100 fixed | 40.0% | 8.9% | 0.0% |
| Denmark (DK) | 5410 turbines | 2015-19 → 2020 | 0.147 | **0.085** | +0.110 | +0.023 | 0.83 | k100 season | 11.6% | 0.6% | 23.0% |
| Brazil (BR) | 151 complexes | 2021-23 → 2024 | 0.139 | **0.105** | -0.046 | -0.015 | 0.72 | k60 fixed † | n/a | n/a | 100.0% |
| United States (US) | 520 plants | 2019-21 → 2022 | 0.110 | **0.097** | +0.022 | +0.024 | 0.79 | k250 fixed † | 48.3% | 22.0% | 1.1% |
| Australia (AU-NEM) | 77 farms | 2020-22 → 2023 | 0.115 | **0.094** | +0.009 | -0.006 | 0.61 | k45 season | 2.8% | 84.5% | 4.5% |
| United Kingdom (UK) | 348 farms | 2015-18 → 2019 | 0.145 | **0.115** ‡ | +0.037 | -0.038 | 0.70 | k50 fixed | 21.8% | 7.5% | 0.0% |
| New Zealand (NZ) | 12 farms | 2019-23 → 2024 | 0.157 | **0.106** ‡ | -0.062 | +0.021 | 0.66 | k7 fixed | 41.9% | 47.3% | 0.0% |
| Chile (CL) | 59 plants (53 scored) | 2021-23 → 2024 | 0.110 | **0.104** ‡ | -0.015 | +0.001 | 0.43 | k10 fixed † | 3.5% | 91.6% | 0.0% |
| Argentina (AR) | 59 plants | 2021-23 → 2024 | 0.150 | **0.133** | +0.011 | +0.001 | 0.43 | k10 fixed † | 0.2% | 96.7% | 0.0% |

**‡ Gain not distinguishable from zero when the test year's units are resampled;
see the correction notices above.**

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
inside the ceiling with no failed offsets, as is every country-level fit below.
These figures travel in `metrics.csv` automatically, so a future run cannot hide
them.

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

Capacity-weighted national monthly CF, held-out 2023, bundled open curve library
throughout.

| Region | Uncorr RMSE | Corr RMSE | Uncorr MBE | Corr MBE | Best cfg | Substituted |
|---|---|---|---|---|---|---|
| France (FR) | 0.171 | **0.012** | +0.165 | +0.006 | N=10 fixed | 100% |
| Belgium (BE) | 0.340 | **0.020** | +0.337 | -0.002 | N=3 season | 100% |
| Ireland (IE) | 0.172 | **0.021** | +0.168 | +0.009 | N=1 season | 100% |
| Sweden (SE) | 0.088 | **0.030** | +0.084 | -0.027 | N=4 fixed | 100% |
| Norway (NO) | 0.034 | 0.039 | +0.024 | -0.030 | correction does not help | 100% |

NO has 4.4% and SE 0.8% of capacity outside the ERA5 data, which the harness
extrapolated (suspension notice above). Both get the same ERA5 download as the
suspended rows.

**Suspended rows.** Not results: most of their capacity was simulated from
winds extrapolated beyond the ERA5 data (suspension notice above). The
published figures are kept for the record only.

| Region | Capacity outside the ERA5 data | Published uncorr RMSE | Published corr RMSE | Published uncorr MBE | Published corr MBE | Best cfg |
|---|---|---|---|---|---|---|
| Italy (IT) | 94.5% | 0.066 | 0.034 | +0.062 | -0.020 | N=3 season |
| Portugal (PT) | 89.9% | 0.110 | 0.074 | -0.097 | +0.029 | N=1 season |
| Spain (ES) | 50.3% | 0.135 | 0.026 | +0.130 | +0.016 | N=4 fixed |

The country-level fit removes very large mean biases (FR, BE and IE all from
0.17-0.34 down to ~0.01-0.02). Two honest notes: NO is already close to
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
- **An aggregate that barely moves does not mean the fit did not move.** A
  change of fitting code moved the United States' headline RMSE by 0.0005 while
  more than doubling its worst fitted scalar, from 20.54 to 46.39. Check
  `fit_quality`, not the metric, when judging whether a correction is sound.
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
