# Output structure

The harness writes every run under `output/validation/<CODE>/`, one directory
per train or evaluate run, each self-contained and stamped (a UTC timestamp, or
the `--run-name` you pass).

```text
output/validation/<CODE>/
├── train-<stamp>/
│   ├── factors_<slice>_<k>.csv        # correction factors, one per (slice, cluster count)
│   ├── train_turb_info_<k>.csv        # training fleet with cluster assignments
│   ├── curve_resolution.csv           # which power curve each model key resolved to
│   └── run_manifest.json              # full provenance of the run
└── evaluate-<year>-<stamp>/
    ├── metrics.csv                    # skill table, one row per variant
    ├── unc_cf.csv                     # uncorrected capacity factor
    ├── cor_cf_<slice>_<k>.csv         # corrected CF, one per factors file
    ├── curve_resolution.csv
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
| `n_units`, `n_samples` | Fleet size and paired observations scored |
| `substituted_capacity_share` | Share of fleet capacity simulated on a curve other than the one its model key names (see below) |

Read the uncorrected row first: judge the correction against the bias structure
it starts from, not in isolation.

## Manifest (`run_manifest.json`)

Full provenance so any output is attributable: `pyvwf_version`, `git_commit` and
`git_dirty`, `created_utc`, the resolved `region`, `observations` (source, unit,
time convention), `correction`, `seasons`, the `curve_library` identity (whether
the open or a licensed library was used), the `curve_resolution` summary, and,
for evaluate runs, `evaluation_year` and `trained_from`. Design §6.

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
