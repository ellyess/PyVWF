# The Python Virtual Wind Farm (PyVWF) model

[![CI](https://github.com/ellyess/PyVWF/actions/workflows/ci.yml/badge.svg)](https://github.com/ellyess/PyVWF/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-green)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21236619-blue)](https://doi.org/10.5281/zenodo.21236619)

PyVWF is an open-source Python framework for processing, bias-correcting, and simulating wind resources and wind power generation using reanalysis data (e.g. ERA5), turbine metadata, and observed generation data. The novelty of this model comes from the bias correction process used to improve the simulations from ERA-5. The simulated wind time-series can be both corrected and uncorrected.

`PyVWF` is a Python rewrite of the [VWF model](https://github.com/renewables-ninja/vwf/tree/master) developed by Iain Staffell. The wind energy simulations on [Renewables.ninja](https://www.renewables.ninja/) are based on the VWF model.

## What makes it different

Raw reanalysis winds carry systematic, location-dependent biases, so capacity factors simulated straight from ERA5 drift away from what wind fleets actually generate. PyVWF learns a per-cluster, per-time-slice linear correction of the wind speed (`w_corrected = α·w + β`) from observed generation, then converts the corrected wind to power. Unlike API-only or general-purpose reanalysis-to-power tools, it exposes the full **training** workflow for the correction factors and lets you compute them at finer spatial and temporal resolution than the conventional national-scale factors, through configurable spatial clustering and temporal grouping. The factors are yours to inspect and retrain, and simulated capacity factors track observations far more closely than uncorrected reanalysis.

## Status and provenance

PyVWF implements the granular bias-correction method introduced in Benmoufok et al. (2024, *Energy*, lead author), [doi:10.1016/j.energy.2024.133759](https://doi.org/10.1016/j.energy.2024.133759). The method is also applied in co-authored follow-on work that extends the framework to high-resolution UK bias correction (Wang et al., 2026, *Energy Conversion and Management*, [doi:10.1016/j.enconman.2026.121066](https://doi.org/10.1016/j.enconman.2026.121066)). It ships a synthetic-data `pytest` suite with continuous integration on Python 3.10 to 3.12, and is pip-installable.

New here? See [How it works](#how-it-works) for the five-step pipeline, then [Quickstart](#quickstart) to run it.

## Contents

- [How it works](#how-it-works)
- [Key features](#key-features)
- [Installation](#installation)
- [Input data](#input-data)
- [Quickstart](#quickstart)
- [Visualising a run](#visualising-a-run)
- [Going further](#going-further)
- [Detailed usage and reference](#detailed-usage-and-reference)
- [Testing](#testing)
- [Assumptions and limitations](#assumptions-and-limitations)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [Contributing](#contributing)
- [Credits and contact](#credits-and-contact)

## How it works

PyVWF supports the following workflow:

1. **Ingest wind reanalysis data** (e.g. ERA5 wind fields)
2. **Interpolate wind speeds to turbine hub height**
3. **Apply bias correction** using observed generation data
4. **Convert wind speeds to power / capacity factor** via turbine power curves
5. **Validate simulated output** against observations

```mermaid
flowchart TD
    A[ERA5 reanalysis winds] --> B[Hub-height extrapolation via log wind profile]
    B --> C[Interpolation to turbine locations]
    C --> D["Bias correction on wind speed: w_corrected = α·w + β"]
    O[Observed generation] -. learns α, β per cluster and time slice .-> D
    D --> E[Power curve conversion]
    E --> F[Capacity factors]
```

The correction is applied to the **wind speed**, before the power-curve
conversion, so the non-linear speed-to-power step operates on corrected winds.

The framework is intended for **daily to monthly** analysis at **turbine, regional, or national scale**.

## Key features

- ERA5-based wind speed processing
- Hub-height extrapolation with configurable methods
- Statistical bias correction of wind speeds
- Power curve-based generation modelling
- Modular, research-friendly Python codebase
- Version-pinned environment for reproducibility
- **Automated test suite** (`pytest`) and continuous integration

## Installation

PyVWF provides a Conda environment (`environment.yaml`) with the core
scientific dependencies pinned to exact versions for reproducibility across
systems.

#### 1. Clone the repository

```bash
git clone https://github.com/ellyess/PyVWF.git
cd PyVWF
```

#### 2. Create the environment

```bash
conda env create -f environment.yaml
conda activate pyvwf
```

#### 3. Verify installation

```bash
python -c "import pandas, xarray, scipy; print('Environment OK')"
```

#### Optional: install via pip

Instead of Conda, you can install PyVWF into any Python ≥3.10 environment
straight from the cloned repository (run from the repo root after step 1):

```bash
pip install -e .            # installs pyvwf-train, the linear bias-correction pipeline
```

The base install carries everything needed to simulate, bias-correct, evaluate
and plot. Fetching *new* input data needs extra libraries, which are kept out of
the base install because they are irrelevant to most users:

```bash
pip install -e ".[data]"    # ENTSO-E API client + Excel/Parquet readers
pip install -e ".[dev]"     # pytest, ruff, mypy
```

| Extra   | Pulls in                        | Needed for                                                    |
|---------|---------------------------------|---------------------------------------------------------------|
| (none)  | core scientific + geospatial     | simulation, bias correction, evaluation, `vwf.viz`            |
| `data`  | `entsoe-py`, `openpyxl`, `pyarrow` | `vwf.datasets` ENTSO-E download and raw fleet-registry parsing |
| `dev`   | `pytest`, `ruff`, `mypy`         | running the test suite, linting, type checking                |
| `docs`  | `sphinx`, `myst-parser`          | building the API reference in `docs/`                          |

#### Pointing PyVWF at your input data

PyVWF looks for input data under `input/` in the working directory, which is the
layout of a repository checkout. If you installed PyVWF and work elsewhere, set
`PYVWF_INPUT` to wherever your data lives:

```bash
export PYVWF_INPUT=/data/pyvwf-inputs   # holds power_curves.csv, models.csv, era5/, ...
```

The package bundles an **open turbine curve library** (69 real machines plus 7
composites from NREL/turbine-models, BSD-3-Clause, VWF-smoothed) so
that it imports and runs on real curve physics out of the box. Fleets are
matched to these curves by specific power rather than machine identity, and
PyVWF warns whenever it falls back to the bundled files; for
manufacturer-specific production runs, supply your own curve library (see
[`input/README.md`](input/README.md)).

After install, the `pyvwf-train` console command is available on your PATH:

| Command       | Wraps                                              |
|---------------|----------------------------------------------------|
| `pyvwf-train` | `PyVWF.train` + `simulate_cf` for one country/year |

Run it with `--help` to see arguments.

## Input data

PyVWF reads all inputs from an `input/` directory at the repository root. The
paths are defined centrally in [`src/vwf/config.py`](src/vwf/config.py)
(`PyVWFPaths`), so that file is the source of truth if anything here is unclear.

There are two workflows, and you only need the data for the one you run:

- **Turbine-level** (DK, DE, UK): individual turbine metadata plus observed
  generation.
- **Country-level** (NL, FR, BE, NO, SE, ES, IT, PT, IE): aggregated national
  generation from ENTSO-E, with sampling grid points generated by PyVWF.

Both workflows need ERA5 reanalysis winds and the shared power-curve and
turbine-model reference files.

### What you need

| Data | Format | Used by | Location |
|---|---|---|---|
| ERA5 reanalysis winds | NetCDF | both | `input/era5/EU/*.nc` |
| Power curves | CSV | both | `input/power_curves.csv` |
| Turbine model specs | CSV | both | `input/models.csv` |
| Turbine metadata | CSV | turbine-level | `input/turbine_level_data/<CC>/` |
| Turbine observed generation | CSV | turbine-level | `input/turbine_level_data/<CC>/` |
| Country grid points | CSV | country-level | `input/country_level_data/grid_points/<cc>/` |
| Country observations | CSV | country-level | `input/country_level_data/observations/<cc>/` |

`<CC>` is the upper-case country code (`DK`), `<cc>` the lower-case form (`nl`).

### Directory layout

```
input/
├── era5/
│   └── EU/                       # all ERA5 NetCDF files go here (any filenames)
│       ├── era5_combined_2015_EU.nc
│       └── ...
├── power_curves.csv              # wind speed -> power, one column per model
├── models.csv                    # manufacturer, model, capacity, diameter, ...
├── turbine_level_data/           # turbine-level workflow (DK, DE, UK)
│   ├── DK/
│   │   ├── dk_md.csv             # turbine metadata
│   │   └── dk_obs_2002_2020.csv  # observed monthly generation
│   ├── DE/
│   │   ├── DE_md.csv
│   │   ├── DE_data.csv
│   │   └── geolocate.germany.csv
│   └── UK/
│       ├── uk_md.csv
│       └── ukobs.csv
├── country_level_data/           # country-level workflow (generated, see below)
│   ├── grid_points/<cc>/<cc>_grid_points_<YYYY>.csv
│   └── observations/<cc>/<cc>_train_<Y1>_<Y2>.csv
└── regions/                      # region shapes for onshore/offshore clustering
    ├── country_shapes.geojson
    └── offshore_shapes.geojson
```

### 1. ERA5 reanalysis winds

Download from the Copernicus Climate Data Store:
[ERA5 hourly single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download).
You need the 100m u- and v-components of wind, plus **either**:

- the 10m u- and v-components of wind (PyVWF then derives surface roughness,
  which is more accurate; use `--calc-z0`), **or**
- forecast surface roughness (`fsr`).

Place every NetCDF file in `input/era5/EU/`. PyVWF loads them with a single
`*.nc` glob and combines them by coordinates, so the individual filenames do
not matter. Training and test years are separated in code by time selection,
not by directory, so all years live in the same folder.

### 2. Turbine metadata and observed generation (turbine-level)

Each supported country has a folder under `input/turbine_level_data/<CC>/` with
the metadata and observation files named exactly as listed in the layout above
(the loaders in [`src/vwf/loaders/turbine_loaders.py`](src/vwf/loaders/turbine_loaders.py)
read these fixed filenames). Metadata provides turbine ID, location, capacity,
rotor diameter, and hub height; observations provide monthly generation.

The repository does not ship turbine metadata or observed generation: such
datasets are typically proprietary, so you supply your own in the layout above
(`input/turbine_level_data/` is gitignored for this reason). To see the
workflow run without any of these files, use the bundled synthetic example
described in the [Quickstart](#quickstart). Do not assume redistribution
rights for data you obtain elsewhere.

### 3. Power curves and turbine models

- `input/power_curves.csv`: wind speed in the first column, then one column per
  turbine model giving capacity factor. The shipped file is the **open turbine
  curve library** (BSD-3-Clause, per-column provenance in
  `input/power_curves_provenance.csv`); manufacturer-specific libraries are
  typically proprietary and are not redistributed here. See
  [`input/README.md`](input/README.md) for provenance details and pointers.
- `input/models.csv`: turbine model reference (manufacturer, model, offshore
  flag, capacity, rotor diameter, power density).

### 4. Country-level data (optional)

The country-level workflow uses generated inputs rather than files you download
by hand. Run:

```bash
python src/vwf/datasets/generate_country_level_training_data.py
```

This fetches national generation from the ENTSO-E Transparency Platform (an API
key is required; see [docs/ENTSOE_API_GUIDE.md](docs/ENTSOE_API_GUIDE.md)) and
writes grid points and train/test observation splits under
`input/country_level_data/`.

### Paths flagged for confirmation

A few input paths are inferred from the code or ship as examples, and are worth
confirming for your own setup:

- **ERA5 variable set.** PyVWF expects `u100`/`v100` and either `u10`/`v10` or
  `fsr`. If your files use different variable names, adjust
  [`src/vwf/datasets/era5.py`](src/vwf/datasets/era5.py).
- **`input/era5/invariant/`.** An invariant file exists in some setups but is
  not read by the default `prep_era5` loader; treat it as optional.
- **Turbine observation filenames** are hard-coded per country in the loaders
  (for example `dk_obs_2002_2020.csv`). New countries need a matching loader
  branch, not just a file drop.

## Quickstart

### Runnable example (no data download)

To see the full workflow end-to-end **without downloading ERA5 or any turbine
data**, run the bundled synthetic example (about a minute):

```bash
python examples/run_minimal.py
```

It preps a small bundled ERA5-shaped NetCDF, trains a per-cluster linear bias
correction against synthetic observed capacity factors, and reports the error
reduction. The weather and observations are synthetic; the power curves come
from the bundled open library. See
[`examples/data/README.md`](examples/data/README.md).

### PyVWF Training Example (Denmark)

A simple, step-by-step quickstart demonstrating the complete PyVWF workflow:

```bash
# After installing (conda env, or `pip install -e .`) the console script is on your PATH:
pyvwf-train --outdir outputs/demo_DK_2020 --country DK --year-test 2020 --calc-z0

# Equivalent (from a repo checkout):
python examples/quick_run.py \
    --outdir outputs/demo_DK_2020 \
    --country DK \
    --year-test 2020 \
    --calc-z0
```

This runs the full research pipeline:
1. Train bias-correction factors using available training data
2. Simulate capacity factor (CF) time series for test year
3. Generate diagnostic plots and error metrics
4. Write all results to output directory

Key options:
- `--outdir`: Output directory (folders and files are created here)
- `--country`: Country code (e.g. DK, DE)
- `--year-test`: Year to simulate
- `--cluster-mode`: all | onshore | offshore
- `--cluster-list`: List of cluster counts to evaluate
- `--time-res-list`: fixed | season | bimonth | month

## Visualising a run

`vwf.viz` turns a run's outputs into diagnostic figures: how well the
corrected simulation reproduces the observed capacity-factor distribution,
what the correction learned spatially, and how error responds to the two
hyperparameters (cluster count and temporal resolution). The
`load_results()` helper reads any PyVWF run directory back into a `Results`
object, so the plot functions are self-contained:

```python
from vwf.viz import load_results, plot_cf_distribution, plot_qq

res = load_results("outputs/DK", country="DK", year=2020)

sims = {
    "uncorrected": res.uncorrected,
    "linear":      res.corrected[(1000, "bimonth")],
}

fig = plot_cf_distribution(res.obs, sims)   # histograms + ECDFs + tail inset
fig.savefig("cf_distribution.png", dpi=150)

fig = plot_qq(res.obs, sims)                # QQ vs the y=x diagonal
fig.savefig("cf_qq.png", dpi=150)
```

![CF distribution diagnostic](docs/img/viz_distribution.png)

The legend annotates each series with its mean and KS distance to observed.
The tail inset (top right) zooms into `CF ≥ 0.7` so distributional differences
between sims in the upper tail remain visible.

### What did the correction learn?

`plot_correction_factor_map()` colours each cluster's Voronoi cell by its
learned scalar and offset, on a diverging scale centred at the neutral value
(scalar = 1, offset = 0), so over- and under-correction read at a glance.
Pass the *training* fleet so the deterministic clustering reproduces the
cluster IDs the factors were fitted on:

```python
from shapely.geometry import box

from vwf.viz import plot_correction_factor_map

fig = plot_correction_factor_map(
    res.factors[(1000, "bimonth")],   # one (n_clu, time_res) configuration
    res.train_turb_info,              # the fleet the factors were fitted on
    boundary=box(8.0, 54.5, 13.0, 57.8),  # optional: clip cells to a region
)                                         # (any shapely geometry, GeoDataFrame,
fig.savefig("factor_map.png", dpi=150)    #  or path to a GeoJSON outline)
```

![Correction factor map](docs/img/viz_factor_map.png)

`plot_factor_joint()` shows the same factors in factor space: the joint
distribution of scalar vs offset with marginal histograms, with guides at the
neutral values. Tight clustering around (1, 0) means the reanalysis needed
little correction:

```python
from vwf.viz import plot_factor_joint

fig = plot_factor_joint(res.factors[(1000, "bimonth")])
fig.savefig("factor_joint.png", dpi=150)
```

![Factor joint distribution](docs/img/viz_factor_joint.png)

### Per-turbine bias

`plot_sim_vs_obs()` scatters each turbine's mean simulated CF against its
mean observed CF; distance from the y=x diagonal is that turbine's bias, and
the panel is annotated with fleet-level MBE/RMSE. It reads the wide CF files
a run writes to disk:

```python
import pandas as pd
from vwf.viz import plot_sim_vs_obs

cf_dir = "outputs/DK/results/capacity-factor"
fig = plot_sim_vs_obs(
    pd.read_csv(f"{cf_dir}/DK_2020_unc_cf.csv"),
    pd.read_csv(f"{cf_dir}/DK_2020_obs_cf.csv"),
    turb_info=res.turb_info,          # optional: colour onshore/offshore
)
fig.savefig("sim_vs_obs.png", dpi=150)
```

![Per-turbine sim vs obs](docs/img/viz_sim_vs_obs.png)

### Choosing `n_clu` and `time_res`

`plot_error_vs_clusters()` takes the tidy metrics table written by
`scripts/evaluate_all_pyvwf_runs.py` (`pyvwf_evaluation_metrics.csv`) and
plots error against cluster count, one line per temporal resolution, with the
uncorrected error as a reference:

```python
import pandas as pd
from vwf.viz import plot_error_vs_clusters

metrics = pd.read_csv("outputs/DK/pyvwf_evaluation_metrics.csv")
fig = plot_error_vs_clusters(metrics[metrics["country"] == "DK"])
fig.savefig("error_vs_clusters.png", dpi=150)
```

![Error vs clusters](docs/img/viz_error_vs_clusters.png)

A data-free reproduction of all six figures is in
[`examples/viz_demo.py`](examples/viz_demo.py).

## Going further

### Full multi-country training

For comprehensive analysis across all supported countries:

```bash
# List available configuration sets
python scripts/train_all_bias_corrections.py --list

# Run turbine + country workflows
python scripts/train_all_bias_corrections.py --sets turbine_grid country_grid_2015_2021_2023
```

See [PIPELINE.md](PIPELINE.md) for the full script execution order.

## Detailed usage and reference

The full documentation, including the guides below and an API reference
generated from the docstrings, is hosted at
[pyvwf.readthedocs.io](https://pyvwf.readthedocs.io/). The same content lives
in `docs/` as plain Markdown, so everything is also readable directly on
GitHub. Start with the [documentation index](docs/README.md) for a suggested
reading order, or jump to a specific reference:

- [Data requirements](docs/DATA_REQUIREMENTS.md): input data formats and how to download ERA5 winds.
- [Output structure](docs/OUTPUT_STRUCTURE.md): the layout of a run directory and the files it produces.
- [PIPELINE.md](PIPELINE.md): the script execution order.

## Testing

PyVWF ships with a `pytest` suite that runs on **synthetic data** (no ERA5 or
ENTSO-E access required), covering the wind log-law/interpolation, power-curve
conversion, the linear bias-correction optimiser, temporal-resolution utilities,
the visualisation layer, and the packaging invariants.

```bash
pip install -e ".[dev]"
pytest                     # run the suite
ruff check src/vwf tests   # lint
mypy                       # type check
```

Continuous integration (`.github/workflows/ci.yml`) runs, for every pull
request and push to `main`:

- **Lint and type check**: `ruff` and `mypy` (the package ships a `py.typed`
  marker, so type information is exported to downstream users).
- **Test**: the suite plus the end-to-end example on Python 3.10–3.12,
  installed from `pyproject.toml` so the declared dependencies are exercised
  exactly as a fresh `pip install` of the package would get them. Coverage is gated, so
  it cannot silently regress.
- **Docs**: builds the API reference and guides with `-W`, so a broken
  docstring or an orphaned page fails rather than quietly degrading the site.
- **Package**: builds the sdist and wheel, validates the distribution
  metadata with `twine`, then installs the wheel into a clean environment and
  imports it with no repository on `sys.path`.

The documentation is generated from the docstrings with Sphinx:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html -W
open docs/_build/html/index.html
```

The version lives in exactly one place, `vwf.__version__`, from which
`pyproject.toml` reads it dynamically; `tests/test_packaging.py` asserts it
follows semantic versioning and stays in step with `CITATION.cff`.

## Assumptions and limitations

- Bias correction is statistical, not physical
- Accuracy depends on the quality and representativeness of observed data
- ERA5 spatial resolution may limit turbine-level accuracy
- Wake effects are not explicitly modelled
- Power curve selection strongly influences results

These assumptions should be considered when interpreting results.

## Reproducibility

- Core scientific dependencies are pinned to exact versions in `environment.yaml`
- Deterministic methods are used where possible
- Results should be reproducible across systems using `environment.yaml`

For published work, we recommend citing the repository and documenting:

- ERA5 data version
- Bias correction training period
- Power curve sources

## Citation

If you use PyVWF in academic work, please cite **both** the software and the
method paper.

**The software**: archived on Zenodo. This is the *concept* DOI: it always
resolves to the latest release, so it stays correct as PyVWF is updated. To pin
the exact version you ran, use the version-specific DOI from the
[Zenodo record](https://doi.org/10.5281/zenodo.21236619).

> Benmoufok, E. F., Warder, S. C., and Piggott, M. D. *PyVWF: An open Python
> framework for bias-corrected wind power simulation from reanalysis data.*
> Zenodo. [doi:10.5281/zenodo.21236619](https://doi.org/10.5281/zenodo.21236619)

**The method**, the bias-correction approach PyVWF implements:

> Benmoufok, E. F., Warder, S. C., Zhu, E., Bhaskaran, B., Staffell, I., and
> Piggott, M. D. (2024).
> *Improving wind power modelling through granular spatial and temporal bias
> correction of reanalysis data.* Energy.
> [doi:10.1016/j.energy.2024.133759](https://doi.org/10.1016/j.energy.2024.133759)

Machine-readable metadata for both is in [`CITATION.cff`](CITATION.cff).

## Contributing

Contributions are welcome, especially:

- Documentation improvements
- Additional bias correction methods
- Validation case studies
- Performance optimisations

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, how to run the
tests and linter, how to report bugs or request features, and pull-request
guidelines. Please open an issue to discuss larger changes before submitting a
pull request.

## Credits and contact

The PyVWF code is developed by Ellyess F. Benmoufok. You can email me via benmoufok.ellyess@gmail.com.

The original VWF code this is based on is developed by Iain Staffell.  You can try emailing them at i.staffell@imperial.ac.uk

PyVWF is part of the [Renewables.ninja](https://renewables.ninja) project, developed by Stefan Pfenninger and Iain Staffell.  Use the [contacts page](https://www.renewables.ninja/about) there.
