# Output structure

The harness writes every run under `output/validation/<CODE>/`, one directory
per train or evaluate run, each self-contained and stamped (a UTC timestamp, or
the `--run-name` you pass).

```text
output/validation/<CODE>/
├── train-<stamp>/
│   ├── factors_<slice>_<k>.csv        # correction factors, one per (slice, cluster count)
│   ├── train_turb_info_<k>.csv        # training fleet with cluster assignments
│   ├── fit_diagnostics_<slice>_<k>.csv # where the fitted factors send the training speeds
│   ├── curve_resolution.csv           # which power curve each model key resolved to
│   └── run_manifest.json              # full provenance of the run
└── evaluate-<year>-<stamp>/
    ├── metrics.csv                    # skill table, one row per variant
    ├── unc_cf.csv                     # uncorrected capacity factor
    ├── cor_cf_<slice>_<k>.csv         # corrected CF, one per factors file
    ├── curve_resolution.csv
    ├── scoring_exclusions.csv         # rows left out so every variant is scored on the same rows
    └── run_manifest.json
```

## Factors (`factors_<slice>_<k>.csv`)

One row per `(cluster, time-slice)`, with the fitted parameters:

| Column | Meaning |
|---|---|
| `cluster` | Spatial cluster index |
| `<slice>` | Time-slice key (`fixed`, `season`, …) |
| `scalar` | Multiplicative wind correction |
| `offset` | Additive wind correction (m/s) |
| `avail` | Availability factor (`scaled-affine` only) |

`<slice>` is the time resolution and `<k>` the cluster count, matching a
`cluster_list` × `time_slices` entry in the config.

## Fit diagnostics (`fit_diagnostics_<slice>_<k>.csv`)

An affine pair with a negative offset sends every speed below
`-offset / scalar` to a negative corrected speed. That speed has no value on
the power curve, so the step drops out of the fit's own objective. The file
records, for each cluster, slice value and training year:

| Column | Meaning |
|---|---|
| `scalar`, `offset` | The fitted pair |
| `zero_crossing_speed` | `-offset / scalar` where the offset is negative, else empty |
| `unit_steps` | Training unit-steps in the group |
| `weight_steps` | The same, capacity-weighted |
| `weight_below_zero`, `weight_above_curve` | Capacity-weighted steps the pair sends below 0 m/s, and above the curve |

The weighted counts are kept, not shares, so that shares aggregate exactly.
`fit_quality` reduces them to three columns of `metrics.csv` (below), and the
train manifest's `fit_diagnostics` block holds them per factors file.

## Metrics (`metrics.csv`)

One row per variant (the uncorrected baseline plus each factors file), scored on
the test year:

| Column | Meaning |
|---|---|
| `variant` | `uncorrected` or the correction model name |
| `num_clu`, `time_res` | Which factors file it scores |
| `scope` | `fleet` (turbine) / `national` / `per-zone` (country) |
| `mbe`, `mae`, `rmse` | Mean bias, mean absolute, root-mean-square error |
| `pearson_r` | Correlation with observations |
| `emd` | Earth-mover distance of the CF distributions (turbine-level) |
| `n_units`, `n_samples` | Fleet size and paired observations scored, the same for every variant |
| `substituted_capacity_share` | Share of fleet capacity simulated on a curve other than the one its model key names (see below) |
| `excluded_share` | Share of scorable rows left out because some variant has no value for them (see below) |
| `extrapolated_capacity_share` | Share of fleet capacity outside the loaded ERA5 extent (see below). Zero unless the region opted in |
| `max_below_zero_share`, `max_above_curve_share` | From the fit diagnostics: the worst share of one cluster's training steps its pair sends below 0 m/s, and above the curve. Recorded beside the dagger; they do not set it |
| `max_period_dropped_share` | The worst share of one training period's capacity-weighted steps the fitted factors drop. For a country-level fit, the share of that period's objective computed on nothing |
| `off_curve_below_share`, `off_curve_above_share`, `no_speed_share` | Capacity-weighted shares of this variant's simulated unit-steps below the power curve, above it, and with no speed (see below) |
| `unit_months_wholly_missing`, `unit_months_partly_missing` | Unit-months with every step missing, and with some but not all (see below) |

Read the uncorrected row first: judge the correction against the bias structure
it starts from, not in isolation.

## Manifest (`run_manifest.json`)

Full provenance so any output is attributable: `pyvwf_version`, `git_commit` and
`git_dirty`, `created_utc`, the resolved `region`, `observations` (source, unit,
time convention), `correction`, `seasons`, the `curve_library` identity (whether
the open or a licensed library was used), the `curve_resolution` summary, and,
for evaluate runs, `evaluation_year` and `trained_from`. Design §6. Evaluate and
transfer runs also carry the `common_row_scoring` summary and the `off_curve`
record per variant. Every harness run carries the `era5_extent` block (see
below) and an `era5_roughness` block, which records the temporal treatment of
the roughness the run requested and the one it applied. The two differ when a
region asks for a stored field and its ERA5 files carry none
(`../design/roughness-temporal-treatment.md`).

## Curve resolution (`curve_resolution.csv`)

Which curve every unit was actually simulated on. A model key the curve table
lacks does not stop a run: the unit is simulated on the table's first column
(`vwf.wind.default_curve_key`, a 100 kW distributed-wind turbine in the bundled
library) with a one-off warning. This file is the record of that, one row per
model key the fleet requests:

| Column | Meaning |
|---|---|
| `requested` | The model key the units carry |
| `n_units`, `capacity`, `capacity_share` | How much of the fleet carries it |
| `assigned_by` | How the key got onto the fleet: the metadata's `model_source`, the `add_models` tier (`fuzzy-manufacturer+specific-power` or `specific-power-only`), or `as-given` |
| `status` | `resolved` (the table has the key) or `substituted` |
| `curve_used` | The key whose curve was used |
| `curve_sha256` | Hash of that curve's values |
| `origin` | `open` if the values match a curve in the bundled open library, else `external` (under `input/combined`, the licensed library) |

The manifest carries the summary (`substituted_capacity_share`, the
substitution map, the open and external shares of capacity, and capacity by
`assigned_by`), and evaluate and transfer runs copy `substituted_capacity_share`
into every row of `metrics.csv`. A non-zero share also raises a warning at run
time. Any result from a run with a non-zero share was not simulated on the
fleet's own curves, and should be read with that share beside it.

## Loaded ERA5 extent (`era5_extent`)

The loaded extent is the lon/lat range of the ERA5 grid a run loaded, after its
bbox slice. A unit inside it has its winds interpolated between grid cells. A
unit outside it would have its winds extrapolated from the edge of the grid.

- By default, a unit outside the loaded extent stops the run with an
  `ExtrapolationError`.
- `prep_era5` warns when the ERA5 files stop short of the requested bbox.
- A region can opt in with `[era5] allow_extrapolation = true`. The run then
  finishes, and records the share.

The manifest's `era5_extent` block records the loaded extent, the requested
bbox, the units outside and their capacity share, the furthest distance
outside, and whether the region opted in. It is written for every run, with
zeros when nothing is outside.

Inside the loaded extent is a statement about position only. It does not
check the data in those cells. A unit can sit inside the extent over cells
whose values are unusable. The off-curve counts below record that.

A scorecard row with a non-zero `extrapolated_capacity_share` carries the §
marker and the share (`docs/README.md`).

## Off-curve values (`off_curve`)

A power curve covers speeds from its table's first speed (0 m/s) to its last
(40 m/s). A simulated speed outside that range has no value on the curve. The
capacity factor there is missing, not zero. A missing speed also gives a
missing capacity factor, for example where the input roughness is undefined.

A monthly mean skips missing steps. So a unit-month can be scored on only some
of its steps, and the steps it loses are the ones the simulation could not
handle. The common-row scoring (below) excludes only unit-months with every
step missing.

Each variant of an evaluate or transfer run records:

| Field | Meaning |
|---|---|
| `off_curve_below_share` | Capacity-weighted share of unit-steps with a speed below the curve |
| `off_curve_above_share` | The same, above the curve |
| `no_speed_share` | The same, with no speed |
| `unit_months_wholly_missing` | Unit-months with every step missing |
| `unit_months_partly_missing` | Unit-months with some steps missing, still scored on the rest |

The fields are columns of `metrics.csv`, and the manifest's `off_curve` block
holds them per variant. A non-zero share raises a warning at run time.

## Scoring exclusions (`scoring_exclusions.csv`)

Every variant of an evaluate or transfer run is scored on the same rows. A row
is scored only if every variant has a simulated value, an observation and,
at turbine level, a capacity. A corrected variant has no value for units in a
cluster whose offset fit failed. Without this rule, its score would leave out
exactly those units, and the uncorrected score would keep them.

The common rows are shared by all variants of the run, not only by the pair
being compared. So a variant you do not report can change the score of one you
do report. In the AR scorecard row, the reported `fixed_10` score moved because
the unreported `season_10` variant lacked two rows. The variant set is part of
a run's design: adding or removing a factors file can change every score.

The file lists each row that at least one variant could score but another
could not. It has one row per excluded row:

| Column | Meaning |
|---|---|
| `scope` | `fleet`, `national` or `per-zone` |
| `ID`, `year`, `month` | The unit and month (turbine-level) |
| `ym`, `cluster` | The month, and the zone for `per-zone` (country-level) |
| `capacity` | The unit's capacity (turbine-level) |
| `missing_in` | The variants that had no value, joined by `;` |

The file is written for every run, with a header only when nothing is excluded.
The manifest's `common_row_scoring` block gives, per scope, the rows scorable,
scored and excluded, the excluded share and the units with every row excluded.
A non-zero share also raises a warning at run time.

**Known gaps.** Coverage is the harness's train, evaluate and transfer runs
only:

- `vwf.harness.hindcast.run_hindcast` returns frames and writes no run
  directory or manifest, so there is nowhere to put the record.
- The legacy `PyVWF` path shares the fallback but is not wired to the log.

In both, a substitution is still visible only as the run-time warning.

`resolved` means the table has the key, not that the key names the right
machine. `add_models` matches manufacturers fuzzily, and can assign a different
manufacturer's model at the same specific power (see its docstring). That shows
up in `assigned_by`, not in `status`.
