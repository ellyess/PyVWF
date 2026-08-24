# Output structure

The harness writes every run under `output/validation/<CODE>/`, one directory
per train or evaluate run, each self-contained and stamped (a UTC timestamp, or
the `--run-name` you pass).

```text
output/validation/<CODE>/
├── train-<stamp>/
│   ├── factors_<slice>_<k>.csv        # correction factors, one per (slice, cluster count)
│   ├── train_turb_info_<k>.csv        # training fleet with cluster assignments
│   └── run_manifest.json              # full provenance of the run
└── evaluate-<year>-<stamp>/
    ├── metrics.csv                    # skill table, one row per variant
    ├── unc_cf.csv                     # uncorrected capacity factor
    ├── cor_cf_<slice>_<k>.csv         # corrected CF, one per factors file
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

Read the uncorrected row first: judge the correction against the bias structure
it starts from, not in isolation.

## Manifest (`run_manifest.json`)

Full provenance so any output is attributable: `pyvwf_version`, `git_commit` and
`git_dirty`, `created_utc`, the resolved `region`, `observations` (source, unit,
time convention), `correction`, `seasons`, the `curve_library` identity (whether
the open or a licensed library was used), and, for evaluate runs,
`evaluation_year` and `trained_from`. Design §6.
