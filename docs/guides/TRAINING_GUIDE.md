# Training and evaluation

PyVWF trains and evaluates one **region** at a time through the validation
harness. A region is a single TOML file in `configs/regions/`; the harness reads
it, fits correction factors on the training window, and scores them on a
held-out test year. The design is in
[`design/HARNESS_DESIGN.md`](../design/HARNESS_DESIGN.md).

## The region config

Everything a run needs is in `configs/regions/<code>.toml`:

```toml
[region]
code = "NZ"
name = "New Zealand (EMI-dispatched fleet)"

[observations]
source = "emi-nz"          # adapter in src/vwf/sources/
obs_level = "turbine"      # "turbine" or "country"
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
against Northern-Hemisphere months.

## Run it

```bash
# Train: fit factors for every (cluster, slice) combination in the config
python scripts/analysis/validate_region.py train \
    --region configs/regions/nz.toml

# Evaluate: score a trained run against the test year
python scripts/analysis/validate_region.py evaluate \
    --region configs/regions/nz.toml \
    --train-run output/validation/NZ/train-<timestamp>
```

For a production run, point `PYVWF_INPUT` at an input root carrying the licensed
curve library (`input/combined`); otherwise the open bundled library is used
(with a warning).

Outputs land under `output/validation/<CODE>/`; see
[`OUTPUT_STRUCTURE.md`](OUTPUT_STRUCTURE.md).

## Correction models

Selected by `[correction] model`:

| Model | What it fits |
|---|---|
| `affine-wind` | `cor_ws = scalar·ws + offset` per (cluster, slice). The validated baseline. |
| `scalar-only` | The affine model with the offset pinned to 0. A control (see the country-level and South-America findings). |
| `scaled-affine` | Affine plus a per-cluster availability factor. A diagnostic for data problems, not a default. |

## Cluster counts

`cluster_list` fits each spatial resolution in one run. `k`-means needs
`k ≤ n_farms` reaching the trainer, and `k` near that ceiling is
one-farm-per-cluster (a fake plateau, `docs/findings/us_br_first_run.md`). For
country-level regions, `cluster_list` must be `1` or the grid's own cluster
count (see [`DATA_SOURCES.md`](DATA_SOURCES.md#4-country-level-entso-e-workflow)).

## Transfer runs

`transfer` applies one region's factors to another, restricted to the approved
AU↔Europe pair on this branch:

```bash
python scripts/analysis/validate_region.py transfer \
    --region configs/regions/uk.toml \
    --source-region configs/regions/au_nem.toml \
    --source-run output/validation/AU-NEM/train-<timestamp>
```

## Adding a region

Write one `ObservationSource` adapter and one TOML config; nothing in the
pipeline changes. See
[`ADDING_AN_OBSERVATION_SOURCE.md`](ADDING_AN_OBSERVATION_SOURCE.md). For where
each region's data comes from and how it is preprocessed, see
[`DATA_SOURCES.md`](DATA_SOURCES.md).
