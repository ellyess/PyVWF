# The Python Virtual Wind Farm (PyVWF) model

PyVWF is a research-oriented Python framework for processing, bias-correcting, and simulating wind resources and wind power generation using reanalysis data (e.g. ERA5), turbine metadata, and observed generation data. The novelty of this model comes from the bias correction process used to improve the simulations from ERA-5. The simulated wind time-series can be both corrected and uncorrected.

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
- [Visualising distributional fit](#visualising-distributional-fit)
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

The framework is intended for **daily to monthly** analysis at **turbine, regional, or national scale**.

## Key features

- ERA5-based wind speed processing
- Hub-height extrapolation with configurable methods
- Statistical bias correction of wind speeds
- Power curve–based generation modelling
- Modular, research-friendly Python codebase
- Version-pinned environment for reproducibility
- **Automated test suite** (`pytest`) and continuous integration

## Installation

PyVWF uses a fully pinned Conda environment to ensure reproducibility across systems.

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

PyVWF is also a regular pip package covering the bias-correction pipeline:

```bash
pip install pyvwf            # pyvwf-train, the linear bias-correction pipeline
```

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

The repository ships example turbine data for DK, DE, and UK. The origin and
licensing of these datasets, and whether they may be redistributed, are being
confirmed and will be documented here. Do not assume redistribution rights for
data you obtain elsewhere.

### 3. Power curves and turbine models

- `input/power_curves.csv`: wind speed in the first column, then one column per
  turbine model giving power output. The shipped file demonstrates the expected
  format; production curves may be proprietary.
- `input/models.csv`: turbine model reference (manufacturer, model, offshore
  flag, capacity, rotor diameter, power density).

### 4. Country-level data (optional)

The country-level workflow uses generated inputs rather than files you download
by hand. Run:

```bash
python vwf/datasets/generate_country_level_training_data.py
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

### PyVWF Training Example (Denmark)

A simple, step-by-step quickstart demonstrating the complete PyVWF workflow:

```bash
# After `pip install pyvwf` the console script is on your PATH:
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

## Visualising distributional fit

`vwf.viz` turns a run's outputs into two diagnostic figures showing how well a
corrected simulation reproduces the observed capacity-factor distribution. The
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
between sims in the upper tail remain visible. A data-free reproduction is in
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

Reference material that used to live inline has moved into `docs/` to keep this
page focused. Start with the [documentation index](docs/README.md) for a
suggested reading order, or jump to a specific reference:

- [Data requirements](docs/DATA_REQUIREMENTS.md): input data formats and how to download ERA5 winds.
- [Output structure](docs/OUTPUT_STRUCTURE.md): the layout of a run directory and the files it produces.
- [PIPELINE.md](PIPELINE.md): the script execution order.

## Testing

PyVWF ships with a `pytest` suite that runs on **synthetic data** (no ERA5 or
ENTSO-E access required), covering the wind log-law/interpolation, power-curve
conversion, the linear bias-correction optimiser, temporal-resolution utilities,
and the distributional visualisation layer.

```bash
pip install -e ".[dev]"
pytest                     # run the suite
ruff check src/vwf tests   # lint
```

Continuous integration (`.github/workflows/ci.yml`) runs the suite and linter on
Python 3.10–3.12 for every push and pull request.

## Assumptions and limitations

- Bias correction is statistical, not physical
- Accuracy depends on the quality and representativeness of observed data
- ERA5 spatial resolution may limit turbine-level accuracy
- Wake effects are not explicitly modelled
- Power curve selection strongly influences results

These assumptions should be considered when interpreting results.

## Reproducibility

- All dependencies are version-pinned
- Deterministic methods are used where possible
- Results should be reproducible across systems using `environment.yaml`

For published work, we recommend citing the repository and documenting:

- ERA5 data version
- Bias correction training period
- Power curve sources

## Citation

If you use PyVWF in academic work, please cite the repository using the
metadata in [`CITATION.cff`](CITATION.cff).

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
