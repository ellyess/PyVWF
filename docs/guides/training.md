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
source = "emi-nz"          # adapter in src/vwf/sources/
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
`time_convention`; `src/vwf/harness/regions.py` gives each one's default.

## Run it

```bash
# Train: fit factors for every (cluster, slice) combination in the config
pyvwf-validate train --region configs/regions/nz.toml

# Evaluate: score a trained run against the test year
pyvwf-validate evaluate --region configs/regions/nz.toml \
    --train-run output/validation/NZ/train-<timestamp>
```

`pyvwf-validate` is installed with the package. In a checkout without an
install, `python scripts/analysis/validate_region.py` takes the same
arguments.

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
| `input/` (the default) | the open library | the test suite, the country-level rows, CL and AR |
| `input/combined` | the combined library: open and licensed | the turbine-level rows DE, DK, UK, US, BR, AU-NEM and NZ |

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
one-unit-per-cluster (a fake plateau, `docs/findings/region-us-br.md`). For
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
files a new region touches, in order, and
[`adding-an-adapter.md`](adding-an-adapter.md) covers a new data source. For where
each region's data comes from and how it is preprocessed, see
[`data-sources.md`](data-sources.md).

## The legacy path, removed

Until 2026-09-24 PyVWF also carried a second path: the `PyVWF` class in
`src/vwf/vwf.py`, its `pyvwf-train` console script and the batch scripts
`train_all_bias_corrections.py` and `evaluate_all_pyvwf_runs.py`, which
produced the thesis-era runs under `output/runs/`. It duplicated the harness's
orchestration with its own defaults and output layout, and diverged from it on
the country level, so it was removed and the harness is the only path. The
harness still runs the same correction functions (`vwf.correction`,
`vwf.wind`, `vwf.data`), and `tests/test_harness_corrections.py` pins that
equivalence bit for bit. To reproduce a legacy run, check out the last commit
that has the class, recorded in [`publications.md`](../publications.md).
