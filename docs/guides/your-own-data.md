# Running a correction on your own fleet data

PyVWF ships adapters for a number of public national datasets, but you can also
point it at your own fleet without writing any adapter code: supply two CSV files
and use [`ClientCsvTurbineSource`](../../src/vwf/sources/client_csv.py). Use this
when your observations are private, one-off, or otherwise not worth a dedicated
source module.

## What you provide

**1. Fleet metadata** (one row per site). Column names are remappable, so you can
keep your own headers and supply a `column_map`. Required after mapping:

| Standard column | Meaning |
| --- | --- |
| `ID` | Site identifier (string) |
| `lon`, `lat` | Location, degrees |
| `capacity` | Rated capacity (kW, or MW with `capacity_unit="mw"`) |
| `height` | Hub height, metres (> 1) |

Plus **one of**: a `model` power-curve key per site (used directly), or a rotor
`diameter`, in which case a curve is matched from the library by specific power
using the same routine the Danish and German fleets use (a `manufacturer` column
sharpens the match). Optional `type` (`onshore`/`offshore`) defaults to onshore.

**2. Monthly generation** (long form, one row per site-month), with columns `ID`,
`year`, `month`, and a value column that is either generated energy (default;
converted to capacity factor using the site capacity and the hours in the month)
or a capacity factor directly (`generation_is_cf=True`).

## Worked example

```python
import dataclasses
from pathlib import Path
from vwf.harness.regions import load_region
from vwf.harness.driver import run_train, run_evaluate
from vwf.sources.client_csv import ClientCsvTurbineSource

column_map = {
    "ID": "site_id", "lon": "longitude", "lat": "latitude",
    "capacity": "rated_mw", "height": "hub_m", "diameter": "rotor_m",
}

def make_source():  # a fresh instance per split keeps train/test clean
    return ClientCsvTurbineSource(
        "my_meta.csv", "my_gen.csv",
        country="DK",                       # selects the ERA5 subset
        column_map=column_map,
        capacity_unit="mw",
        generation_column="generation_mwh",
    )

# Start from any region config for its ERA5 box and seasons, then point it at
# your source and choose the split and resolution.
base = load_region(Path("configs/regions/dk.toml"))
spec = dataclasses.replace(
    base,
    source="client-csv-turbine",
    train_years=(2015, 2019),
    test_years=(2020,),
    cluster_list=(1, 50),
    time_slices=("fixed", "season"),
)

out = Path("output/my_run")
train_dir = run_train(spec, out, source=make_source())
eval_dir = run_evaluate(spec, train_dir, out, source=make_source())
# eval_dir/metrics.csv holds the uncorrected baseline and each corrected variant.
```

Two further outputs follow directly from `train_dir`:

- **Gridded correction field**: `scripts/analysis/export_correction_field.py`
  writes an atlite-ready NetCDF of the fitted corrections for the region.
- **Resource in context**: `scripts/analysis/run_hindcast.py` applies the
  correction across every ERA5 year available and ranks each month against the
  record.

## Region box and ERA5

`country=` on the source, and the region config you clone, select which ERA5
subset is used. If your fleet lies outside the boxes already on disk, fetch that
subset first (`scripts/fetch/era5.py`) and point the config's `[era5]`
`path`/`bbox` at it. The correction maths, clustering, curves, and metrics are
region-agnostic; only the ERA5 box and the two CSVs change.

## Handling private data

The input tree is git-ignored, so private turbine or SCADA data stays out of
version control if you keep it there. Nothing on this path writes your raw
generation records into an output artefact: the correction field and the metrics
contain fitted parameters and skill scores only. As with any PyVWF result, treat
the output as screening-level analysis rather than an accredited yield
assessment.
