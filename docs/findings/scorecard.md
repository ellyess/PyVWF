# Multi-region validation scorecard

**Reproduction record, added 2026-09-18.** Drivers:
`scripts/studies/scorecard/missing_value_audit.py`,
`scripts/studies/scorecard/off_curve_sensitivity.py`,
`scripts/studies/scorecard/training_objective_check.py`,
`scripts/studies/scorecard/unit_concentration.py`. Until 2026-09-18 they were
in `scripts/analysis/`, the path any command below uses;
`scripts/studies/README.md` maps each old path to its new one. Numbers:
`unit_concentration.py`'s output records no commit; the driver's last commit
before it was written is `4b6d143`, and rerun at `51807f8` it reproduces its
recorded files byte for byte (`tests/test_pin_bootstrap_reproduction.py`). The
outputs of `missing_value_audit.py`, `off_curve_sensitivity.py` and
`training_objective_check.py` record no commit either, and each was written on
2026-09-11 minutes before the driver's first commit (`bbaf5b3`, `b7826d3` and
`b7826d3`), so the exact code that produced them is not recorded. The runs they
read are pinned by their own manifests.

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

**Resolved, 2026-09-13.** The download was made and the three rows were
re-run: `era5/EU_2026-09` covers 12 W to 31.5 E and 36 to 72 N, and every unit
of all three fleets is now inside the extent its configuration loads. Their
extrapolated shares are zero. They are back in the country-level table above,
and none of them carries a dagger: no implausible scalar and no failed offset
between them. Their figures improved, Italy's corrected RMSE halving and
Portugal's falling by two thirds, because the winds are real rather than
extrapolated; the evidence that it is the winds and not something else is in
`method-eu-rerun.md`, which splits each fleet by whether a grid point lay
outside the old extent. This resolves the input defect only. Italy's fit still
sends 1.0% of its capacity-weighted steps off the curve on calm days, which is
an ordinary property of the affine correction rather than a symptom of bad
input, and is a finding in its own right in that document. Sweden and Norway
were re-run in the same pass and are no longer marked either: the sentence
above, that they stay in the table with their shares stated, describes what was
true until this download.

*[Correction, 2026-09-12: NO and SE are not the only rows left with
extrapolated winds. DK is one too, 0.6% of capacity and 47 of 5,446 units, and
now carries § in the turbine-level table above. This paragraph listed the
country-level rows only, and the check of 2026-09-11 behind it compared the
European rows against the extent of the ERA5 files rather than against each
row's own bbox-sliced extent, which is what a run loads. DK's box stops inside
the files. The commit message of c480f46, which added the extent guard, says
"the other twelve rows are inside their grids" on the same mistaken basis; it
is eleven, and DK is not one of them. A commit message is dated history, so it
is corrected here and not rewritten.

`scripts/analysis/extent_audit.py` now asks the second question for all
seventeen rows, so the count has a script behind it. Six rows carry
extrapolated winds: ES 50.3%, IT 94.5%, PT 89.9%, NO 4.4%, SE 0.8% and DK 0.6%.
The five published shares reproduce exactly, including the distances out, and
DK is the addition. The other eleven rows have no unit outside the extent their
own configuration loads. Data:
`output/extent_audit_2026-09-12/`.]*

**Correction notice, 2026-09-12: the rows do not share one roughness
treatment.** The difference between regions is not the formula, it is whether
the result of the formula varies in time. Every region derives the surface
roughness z0 by inverting the log wind profile from the 10 m and 100 m winds,
the equation the method describes. The European files carry a single annual
mean of that quantity, computed once per year in
`src/vwf/datasets/combine_era5_files.py`. Every other region derives z0 hour by
hour and averages it to daily along with the winds.

| Roughness applied | Route | Rows |
|---|---|---|
| Per timestep, averaged to daily | derived at load from the file's hourly winds | DE, DK, UK, the eight country-level rows, AU-NEM, NZ, CL and AR: 15 of 17 |
| Per timestep, averaged to daily | derived and stored as a daily field by `scripts/era5/combine.py` | US, BR: 2 of 17 |

*[Updated 2026-09-13: the eleven European rows were re-run on the per-timestep
treatment (`method-eu-rerun.md`), so no scorecard row applies an annual-mean
roughness any more and the split this notice recorded is closed. The published
rows that did are in the Superseded section below. The rows still differ by
route, and a manifest still cannot tell the second route from a stored annual
mean.]*

The Roughness column of the tables below carries these two values per row.
They are the same treatment computed at different stages. A run's manifest
cannot tell the second from a stored annual mean: it reports `stored` whenever
the file carries a roughness field, whatever that field is. The second route
also drops the 10 m winds, so the US and Brazilian rows cannot derive a
roughness at all (`docs/design/roughness-temporal-treatment.md`).

**Cross-region comparison is confounded.** This table invites reading rows
against each other, and the two halves differ in an input, not only in fleet,
observations and climate. Every statement in this repository that ranks or
contrasts regions inherits that, including the transfer and physics-informed
work. A reader cannot work this out from the rows.

**The per-timestep derivation was adopted as the method on 2026-09-12**
(`method-roughness-treatment.md`), on method fidelity and comparability rather
than on accuracy: the measured effect on Denmark is 0.0002 in corrected RMSE,
resolved by the pre-registered gate and far too small to carry a method change
on its own. The eleven European rows were re-run on it on 2026-09-13, and the
figures in the tables below are those re-runs. The treatment moved every one of
them by less than 0.0002 in corrected RMSE, and only Denmark's difference
excludes zero; what moved the returning rows was the wider box, not the
treatment (`method-eu-rerun.md`).

**What the dating evidence supports.** No PyVWF run output surviving in this
repository predates the combined European files of 11 February 2026; the
earliest surviving output is 13 February 2026. Run directories are pruned, so
this does not establish that no earlier run existed, only that none survives to
be checked.

**The thesis is silent, not wrong.** Its method section gives the equation and
states that the ERA5 input is hourly. It does not state how z0 is treated in
time, and a reader would naturally take the equation as applying per timestep.
Nothing in the thesis changes. Any paper drawn from those chapters has to state
the temporal treatment that actually ran.

**Two documentation errors, corrected on 2026-09-12.** They are not the
finding. [`src/vwf/datasets/COMBINED_ERA5_USAGE.md`](https://github.com/ellyess/PyVWF/blob/ecf0cc3172a875b27ef4a2fe2d9c2e2a25a42621/src/vwf/datasets/COMBINED_ERA5_USAGE.md) said the European z0 is
derived from terrain data; it is not, and the code has no terrain option. The
usage line in `combine_era5_files.py` offered `--roughness-source terrain`,
which does not exist.

All rows were produced by PyVWF v0.4.0 at commit `41462e9` from a clean tree on
2026-08-24, one region per process. Runs are in
`output/validation/refresh_2026-08-24/<CODE>/`, outside the repository. The AR
row is an exception since 2026-09-11: it comes from an evaluate-only re-run of
the same training directory on the common-row harness, at commit `bbaf5b3`
from a clean tree (`output/validation/common_row_rerun_2026-09-11/`). The CL
row is the exception since 2026-09-19: it was re-run, training and evaluation,
with the bracketed offset search (#18), at commit `201c62e` from a clean tree
(`output/validation/bracketed_2026-09-19/CL/`). The US row was re-run that way
too, then again the same day under the accepted-years rule (#28), at commit
`0fd6574` from a clean tree (`output/validation/accepted_years_2026-09-19/US/`),
which is the run it reports. The figures the
dated notices above quote for the US and CL were measured on the runs those
rows replaced, and are superseded rather than re-measured; that includes the
resampled gain intervals. Each row
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

Every turbine-level row except BR is matched on specific power, by one of four
routes:

- **DK and UK:** `add_models` at load time, a fuzzy manufacturer match then
  nearest specific power. Both registers also record a model designation, which
  `add_models` does not read.
- **DE:** the same `add_models` route, but its register carries no model
  designation, so the fuzzy manufacturer tier is all it can use. Grouping DE
  with DK and UK, as this list did until 2026-09-13, hid that difference.
- **US, NZ, CL and AR:** `assign_curves_from_library` at processing time,
  nearest specific power within a rating band, with no manufacturer step.
- **AU-NEM:** a specific-power class.

BR assigns one uniform curve to every complex, and nothing records its
manufacturers, so its match shares are n/a rather than zero.

The two mismatch columns therefore measure how far each row rests on the
assumption that specific power fixes a curve's shape. Whether held-out skill
survives that assumption is not assessed here. Largest case: 13.4% of DE
capacity is Vestas turbines on Gamesa curves.

*[Note, 2026-09-13: for Germany the columns measure something else. The German
register records a manufacturer, a rating, a rotor diameter and a date, and no
model designation at all, so no assignment better than the nearest specific
power is available from it: a brand-and-spec matcher places **0.0% of DE
capacity, 0 of 11,433 units** against the licensed library
(`scripts/analysis/curve_library_match.py`, coverage in
`method-curve-library-prereg.md`). The 13.4% above is therefore a limit of the
register rather than of the matching, and the two argue for different things:
one for a better register, the other for changing the method. The figure
stands; what it is evidence of does not.]*

The rule is strict and names brands, not lineages. A Bonus turbine on a
Siemens curve counts as other brand. A GE plant on the DOE reference curve of a
GE 1.5 MW machine counts as a reference curve; that is 6.6% of US capacity.

**Correction notice, 2026-09-13: Denmark's three audit shares were computed on
truncated manufacturers.** `load_turbine_metadata` truncates the Danish
manufacturer to its first word, so "NEG Micon" becomes "NEG", and the audit's
stop-word list contains "neg", correctly, as a word that never identifies a
brand on its own. Together they classify every NEG Micon unit as unverifiable:
994 units in the register, 1,034 in the fitted fleet, 19.9% of its capacity.
Read with the full manufacturer string the row reads:

| | Published | Corrected |
|---|---|---|
| Other brand | 11.6% | **15.0%** |
| Reference curve | 0.6% | **1.7%** |
| Unverifiable | 23.0% | **3.1%** |

The table above now carries the corrected figures. **The claim those numbers
supported was wrong in kind, not only in size:** Denmark's manufacturers are
recorded, and the pipeline discards part of each one. No other row is affected,
because no other loader branch truncates. The truncation itself is not changed
here: it is an input to a published row, and changing it is separate work.

**`add_models` reads a manufacturer, a capacity, a rotor diameter and a hub
height, and no model designation, for any region.** So the designations the
Danish and British registers do record are unused by curve assignment as it
stands. Whether reading them would assign better curves is the T1 condition of
the curve library study (`method-curve-library-prereg.md`), and it is untested
here.

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

| Region | Fleet (test) | Train → test | Uncorr RMSE | Corr RMSE | Uncorr MBE | Corr MBE | Corr r | Best cfg | Roughness | Other brand | Reference curve | Unverifiable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Germany (DE) | 4814 turbines | 2015-18 → 2019 | 0.086 | **0.057** | +0.043 | +0.001 | 0.86 | k100 fixed | per timestep | 40.0% | 8.9% | 0.0% |
| Denmark (DK) § 0.6% | 5410 turbines | 2015-19 → 2020 | 0.148 | **0.085** | +0.112 | +0.022 | 0.83 | k100 season | per timestep | 15.0% | 1.7% | 3.1% |
| Brazil (BR) | 151 complexes | 2021-23 → 2024 | 0.139 | **0.105** | -0.046 | -0.015 | 0.72 | k60 fixed † | per timestep, stored daily | n/a | n/a | 100.0% |
| United States (US) | 520 plants (512 scored) | 2019-21 → 2022 | 0.108 | **0.096** | +0.024 | +0.023 | 0.79 | k250 fixed † | per timestep, stored daily | 48.3% | 22.0% | 1.1% |
| Australia (AU-NEM) | 77 farms | 2020-22 → 2023 | 0.115 | **0.094** | +0.009 | -0.006 | 0.61 | k45 season | per timestep | 2.8% | 84.5% | 4.5% |
| United Kingdom (UK) | 348 farms | 2015-18 → 2019 | 0.146 | **0.115** ‡ | +0.038 | -0.038 | 0.70 | k50 fixed | per timestep | 21.8% | 7.5% | 0.0% |
| New Zealand (NZ) | 12 farms | 2019-23 → 2024 | 0.157 | **0.106** ‡ | -0.062 | +0.021 | 0.66 | k7 fixed | per timestep | 41.9% | 47.3% | 0.0% |
| Chile (CL) | 59 plants (53 scored) | 2021-23 → 2024 | 0.110 | **0.104** ‡ | -0.015 | +0.001 | 0.43 | k10 fixed † | per timestep | 3.5% | 91.6% | 0.0% |
| Argentina (AR) | 59 plants | 2021-23 → 2024 | 0.150 | **0.133** | +0.011 | +0.001 | 0.43 | k10 fixed † | per timestep | 0.2% | 96.7% | 0.0% |

**‡ Gain not distinguishable from zero when the test year's units are resampled;
see the correction notices above.**

**§ Part of the fleet lies outside the loaded ERA5 extent, and its winds were
extrapolated; the share of capacity follows the marker** (the rule is in
`docs/README.md`). DK's box stops at 13.5°E and Bornholm lies near 14.9°E, so
47 of the 5,446 units in its test fleet, 0.6% of capacity, sit up to 1.64°
beyond the data, as do 15 of the 3,707 in the fleet the row was trained on,
0.5% of that capacity and up to 1.55°. The row's own run records the share in
its `metrics.csv`, as the rule requires; the figure was first measured by the
extent audit of 2026-09-12 (`scripts/analysis/extent_audit.py`, data in
`output/extent_audit_2026-09-12/`), because the run published then predated
`extrapolated_capacity_share`. The marker says the figures were produced partly
from winds that were extrapolated rather than interpolated, and it survived the
re-run because the extended download does not reach Bornholm: DK's own box
stops first. The data covering Bornholm is already in the European files, so
this is a bounding-box error and not a missing download; widening the box is
logged as separate work, and produces a different DK row with its own
configuration.

**† The fit behind this row is degenerate.** `fit_quality` run against the exact
factors file each row reports, with the calibrated bounds (scalar in 0.2 to 3.0,
offsets required to converge):

| Region | Config | Max scalar | Implausible scalars | Failed offsets |
|---|---|---|---|---|
| Chile (CL) | k10 fixed | **80.23** | 3 | **2** |
| United States (US) | k250 fixed | 2.72 | 0 | **5** |
| Argentina (AR) | k10 fixed | **15.53** | 1 | 0 |
| Brazil (BR) | k60 fixed | **4.82** | 2 | 0 |

The other five are clean: DE 2.79, AU-NEM 2.64, UK 1.86, NZ 1.81, DK 1.15, all
inside the ceiling with no failed offsets, as is every country-level fit below.

Since 2026-09-19 (#28) a factor averages its scalar and offset over its
accepted years, and a factor whose accepted years are not a majority of its
training years is refused and carries no scalar. The US row is re-run under
that rule, so its maximum scalar covers applied factors only: the 46.39 it
showed before belonged to cluster 38, now refused, and all five of its
degenerate clusters are refused factors, counted as failed offsets, which is
why the row keeps its dagger with a maximum scalar inside the bounds. The CL,
AR and BR rows predate the rule; CL's 80.23 is a refused cluster's scalar
that a re-run would no longer show.
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

| Region | Uncorr RMSE | Corr RMSE | Uncorr MBE | Corr MBE | Best cfg | Roughness | Substituted |
|---|---|---|---|---|---|---|---|
| France (FR) | 0.171 | **0.012** | +0.165 | +0.006 | N=10 fixed | per timestep | 100% |
| Belgium (BE) | 0.340 | **0.020** | +0.337 | -0.002 | N=3 season | per timestep | 100% |
| Ireland (IE) | 0.172 | **0.021** | +0.168 | +0.009 | N=1 season | per timestep | 100% |
| Sweden (SE) | 0.088 | **0.030** | +0.084 | -0.027 | N=4 fixed | per timestep | 100% |
| Norway (NO) | 0.035 | 0.036 | +0.027 | -0.028 | correction does not help | per timestep | 100% |
| Spain (ES) | 0.028 | **0.026** | +0.013 | +0.011 | N=4 fixed | per timestep | 100% |
| Italy (IT) | 0.070 | **0.017** | -0.069 | -0.003 | N=3 season | per timestep | 100% |
| Portugal (PT) | 0.089 | **0.027** | -0.085 | +0.018 | N=1 season | per timestep | 100% |

**No country-level row carries § any more.** Sweden and Norway did, at 0.8%
and 4.4% of capacity, and the wider download of 2026-09-12 covers both fleets;
their re-runs record an extrapolated share of zero. Denmark is the only
scorecard row still marked, because its own bounding box, not the data, stops
short of Bornholm. The marker rule is in `docs/README.md`.

**Superseded rows, 2026-09-13.** The published figures of every European row,
kept for the record and replaced in the tables above. All eleven ran on
`era5/EU`, the annual-mean roughness, and the narrower ERA5 box; their
configurations are in `configs/regions/scorecard/superseded/2026-09-13/`, under
their own names and unchanged. The canonical name in
`configs/regions/scorecard/` points at the row standing today.
What moved, and why, is in `method-eu-rerun.md`.

| Region | Published uncorr RMSE | Published corr RMSE | Published uncorr MBE | Published corr MBE | Best cfg | Was |
|---|---|---|---|---|---|---|
| Germany (DE) | 0.086 | 0.057 | +0.042 | +0.001 | k100 fixed | in the table |
| Denmark (DK) | 0.147 | 0.085 | +0.110 | +0.023 | k100 season | in the table, § 0.6% |
| United Kingdom (UK) | 0.145 | 0.115 ‡ | +0.037 | -0.038 | k50 fixed | in the table |
| France (FR) | 0.171 | 0.012 | +0.165 | +0.006 | N=10 fixed | in the table |
| Belgium (BE) | 0.340 | 0.020 | +0.337 | -0.002 | N=3 season | in the table |
| Ireland (IE) | 0.172 | 0.021 | +0.168 | +0.009 | N=1 season | in the table |
| Sweden (SE) | 0.088 | 0.030 | +0.084 | -0.027 | N=4 fixed | in the table, § 0.8% |
| Norway (NO) | 0.034 | 0.039 | +0.024 | -0.030 | N=4 fixed | in the table, § 4.4% |
| Italy (IT) | 0.066 | 0.034 | +0.062 | -0.020 | N=3 season | suspended |
| Portugal (PT) | 0.110 | 0.074 | -0.097 | +0.029 | N=1 season | suspended |
| Spain (ES) | 0.135 | 0.026 | +0.130 | +0.016 | N=4 fixed | suspended |

A superseded row is not withdrawn. It states what was published, on what
input, and the date it was replaced. The three that were suspended were never
results at all, and their figures are kept only so the correction can be
checked.

The country-level fit removes very large mean biases (FR, BE and IE all from
0.17-0.34 down to ~0.01-0.02). Two honest notes: NO is already close to
unbiased uncorrected (RMSE 0.035) and the correction does not help (0.036, with
the interval on the difference including zero); and the country method
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
