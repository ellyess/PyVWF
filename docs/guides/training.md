# Training and evaluation

PyVWF trains and evaluates one **region** at a time through the validation
harness. A region is a single TOML file in `configs/regions/`; the harness reads
it, fits its factors on the training years, and scores them on the
test year, which the fit never sees. The design is in
[`design/harness.md`](../design/harness.md).

## The region config

Everything a run needs is in `configs/regions/<code>.toml`:

```toml
[region]
code = "NZ"
name = "New Zealand (EMI-dispatched fleet)"

[observations]
source = "emi-nz"          # adapter in src/pyvwf/sources/
obs_level = "turbine"      # "turbine" or "country"
obs_unit = "farm"          # turbine, farm, plant, complex or country
train_years = [2019, 2023] # inclusive
test_years = [2024]

[era5]
path = "era5/NZ"           # under the input root
bbox = [166.0, 179.0, -48.0, -36.0]
file_tag = "NZ"

[correction]
model = "affine-wind"      # registered correction model
cluster_list = [1, 4, 7]   # spatial cluster counts to fit
time_slices = ["fixed", "season"]

[seasons]                  # EXPLICIT month lists, never the NH default
summer = [12, 1, 2]
autumn = [3, 4, 5]
winter = [6, 7, 8]
spring = [9, 10, 11]
```

Seasons are always explicit so a Southern-Hemisphere region is never scored
against Northern-Hemisphere months. Every key above is required. The optional
ones are `min_cluster_size`, `roughness`, `allow_extrapolation`,
`location_resolution`, `pseudo_replicated_rows`, `station_id_regex` and
`time_convention`; `src/pyvwf/harness/regions.py` gives each one's default.

## Run it

```bash
# Train: fit factors for every (cluster, slice) combination in the config
pyvwf-validate train --region configs/regions/nz.toml

# Evaluate: score a trained run against the test year
pyvwf-validate evaluate --region configs/regions/nz.toml \
    --train-run output/validation/NZ/train-<timestamp>
```

`pyvwf-validate` is installed with the package (`pip install -e .` in a
checkout).

Set `PYVWF_INPUT` on both commands when the run uses another input root; see
[Choose the input root](#choose-the-input-root).

Outputs land under `output/validation/<CODE>/`; see
[`output-structure.md`](output-structure.md).

## Choose the input root

PyVWF reads every input from one directory, the input root. The default is
`input/`. `PYVWF_INPUT` names another one. The input root decides which curve
library a run uses:

| Input root | Curve library | Used for |
|---|---|---|
| `input/` (the default) | the open library | the test suite (the offset-fit pins for DK, FR and ES set `input/combined` themselves), and the scorecard rows CL and AR |
| `input/combined` | the combined library: open and licensed | the turbine-level rows DE, DK, UK, US, BR, AU-NEM and NZ, and every country-level row |

Every country-level row needs `input/combined` because each country grid names
its own Vestas model, which only the licensed library holds, and since
2026-09-25 a country-level run refuses a model key with no curve rather than
falling back.

Follow these rules:

- **Set `PYVWF_INPUT` on every command of a run.** Train and evaluate each
  resolve the curve library again. A command without it uses `input/`.
- **Read the manifest.** Each run records the library it used, by sha256, in
  `run_manifest.json`.
- **Build a separate input root for another library.** Give it the same
  `era5/`, `observations/` and `raw/` as `input/`; symbolic links work. Put
  your `power_curves.csv` and `models.csv` in its own `reference/`.
- **Never copy a licensed library over `input/reference/power_curves.csv`.**
  That file is the open library, which is committed.
  `tests/test_committed_files.py` fails if its content changes.
- **Expect a warning without a library.** An input root with no
  `power_curves.csv` uses the open library shipped inside the package, and
  PyVWF warns.

The fetch and process scripts resolve their default paths under the same input
root.

## Correction models

Selected by `[correction] model`:

| Model | What it fits |
|---|---|
| `affine-wind` | `cor_ws = scalar·ws + offset` per (cluster, slice). The validated baseline. |
| `scalar-only` | The affine model with the offset pinned to 0. A control (see the country-level and South-America findings). |
| `scaled-affine` | Affine plus a per-cluster availability factor. A diagnostic for data problems, not a default. |

## Cluster counts

`cluster_list` fits each cluster count in one run. `k`-means needs `k` at most
the number of units reaching the trainer, and `k` near that ceiling is
one-unit-per-cluster (a fake plateau, `docs/results.md#united-states-and-brazil`). For
country-level regions, `cluster_list` must be `1` or the grid's own cluster
count (see [A country-level region](adding-a-region.md#a-country-level-region)).

A high cluster count also makes a refused factor more likely. A cluster whose
accepted years are not a strict majority of the training years is refused: it
carries no scalar and no offset, and its units get no corrected values. See
[Factors](output-structure.md#factors-factors_slice_kcsv).

## Transfer runs

`transfer` applies one region's factors to another, restricted to the approved
AU↔Europe pair, the only pairing that has been validated:

```bash
pyvwf-validate transfer \
    --region configs/regions/uk.toml \
    --source-region configs/regions/au_nem.toml \
    --source-run output/validation/AU-NEM/train-<timestamp>
```

## Adding a region

No core module changes. [`adding-a-region.md`](adding-a-region.md) lists the
files a new region touches, in order, including a new data source. For where
each region's data comes from and how it is preprocessed, see
[`data-sources.md`](data-sources.md).

## Your own fleet data

PyVWF ships adapters for a number of public national datasets, but you can also
point it at your own fleet without writing any adapter code: supply two CSV files
and use [`ClientCsvTurbineSource`](https://github.com/ellyess/PyVWF/blob/main/src/pyvwf/sources/client_csv.py). Use this
when your observations are private, one-off, or otherwise not worth a dedicated
source module.

### What you provide

**1. Fleet metadata** (one row per unit). Column names are remappable, so you can
keep your own headers and supply a `column_map`. Required after mapping:

| Standard column | Meaning |
| --- | --- |
| `ID` | Unit identifier (string) |
| `lon`, `lat` | Location, degrees |
| `capacity` | Rated capacity (kW, or MW with `capacity_unit="mw"`) |
| `height` | Hub height, metres (> 1) |

Plus **one of**: a `model` key per unit (used directly), or a rotor
`diameter`, in which case a curve is matched from the curve library by specific
power through `pyvwf.curves.add_models`, the same routine the Danish and German
fleets use (a `manufacturer` column sharpens the match). Optional `type`
(`onshore`/`offshore`) defaults to onshore.

**2. Monthly generation** (long form, one row per unit-month), with columns
`ID`, `year`, `month`, and a value column that is either generated energy
(default; converted to capacity factor using the unit's capacity and the hours
in the month) or a capacity factor directly (`generation_is_cf=True`).

### Worked example

```python
import dataclasses
from pathlib import Path
from pyvwf.harness.regions import load_region
from pyvwf.harness.driver import run_train, run_evaluate
from pyvwf.sources.client_csv import ClientCsvTurbineSource

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

### Region box and ERA5

`country=` on the source, and the region config you clone, select which ERA5
subset is used. If your fleet lies outside the boxes already on disk, fetch that
subset first (`scripts/fetch/era5.py`) and point the config's `[era5]`
`path`/`bbox` at it. The correction maths, clustering, curves, and metrics are
region-agnostic; only the ERA5 box and the two CSVs change.

### Handling private data

The input root is git-ignored, so private turbine or SCADA data stays out of
version control if you keep it there. Nothing on this path writes your raw
generation records into an output artefact: the correction field and the metrics
contain fitted parameters and skill scores only. As with any PyVWF result, treat
the output as screening-level analysis rather than an accredited yield
assessment.

## The legacy path, removed

Until 2026-09-24 PyVWF also carried the `PyVWF` class, its `pyvwf-train`
console script and two batch scripts, which produced the thesis-era runs. The
harness is now the only path; it runs the same correction functions, and
`tests/test_harness_corrections.py` pins that equivalence bit for bit. The last
commit with the class is recorded in [`publications.md`](../publications.md).
