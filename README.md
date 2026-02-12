# The Python Virtual Wind Farm (PyVWF) model

PyVWF is a research-oriented Python framework for processing, bias-correcting, and simulating wind resources and wind power generation using reanalysis data (e.g. ERA5), turbine metadata, and observed generation data. The novelty of this model comes from the bias correction process used to improve the simulations from ERA-5. The simulated wind time-series can be both corrected and uncorrected.

`PyVWF` is a Python rewrite of the [VWF model](https://github.com/renewables-ninja/vwf/tree/master) developed by Iain Staffell. The wind energy simulations on [Renewables.ninja](https://www.renewables.ninja/) are based on the VWF model. 

## Overview

PyVWF supports the following workflow:

1. **Ingest wind reanalysis data** (e.g. ERA5 wind fields)
2. **Interpolate wind speeds to turbine hub height**
3. **Apply bias correction** using observed generation data
4. **Convert wind speeds to power / capacity factor** via turbine power curves
5. **Validate simulated output** against observations

The framework is intended for **daily to monthly** analysis at **turbine, regional, or national scale**.

## Key Features

- ERA5-based wind speed processing
- Hub-height extrapolation with configurable methods
- Statistical bias correction of wind speeds
- Power curve–based generation modelling
- Modular, research-friendly Python codebase
- Version-pinned environment for reproducibility
- **ML correction prediction experiments** (see `ml/` directory - experimental and for research use only)

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

## Quickstart

> 💡 **New users:** See [DATA_SOURCES.md](DATA_SOURCES.md) for information on where to obtain turbine metadata and generation observations for different countries.

### PyVWF Training Example (Denmark)

A simple, step-by-step quickstart demonstrating the complete PyVWF workflow:

```bash
# Basic usage with Denmark data
python examples/pyvwf_quickstart_denmark.py

# With custom options
python examples/pyvwf_quickstart_denmark.py \
    --year-test 2020 \
    --clusters 5 \
    --time-res month \
    --calc-z0
```

This script demonstrates:
1. Loading turbine metadata and observations
2. Processing ERA5 reanalysis data
3. Spatial clustering of turbines
4. Training bias correction factors (scalar + offset)
5. Simulating test year with corrections applied
6. Validation against observations
7. Exporting results and metrics

Output includes:
- Turbine metadata and cluster assignments
- Bias correction factors (scalar/offset)
- Simulated capacity factors (corrected & uncorrected)
- Validation metrics comparing with observations

### Full Research Run

For comprehensive analysis with multiple configurations:

```bash
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

## Geospatial Utilities - Categorizing Turbines

PyVWF now includes utilities to automatically categorize turbines as onshore or offshore based on their geographic location relative to region definitions (GeoJSON files).

### Basic Usage

```python
import pandas as pd
from vwf import add_domain_column, filter_by_domain

# Load turbine data with lat/lon coordinates
turbines = pd.read_csv('input/turbines.csv')

# Automatically categorize based on GeoJSON regions
turbines = add_domain_column(
    turbines,
    onshore_geojson='input/regions/country_shapes.geojson',
    offshore_geojson='input/regions/north_sea_shape.geojson',
)

# The 'domain' column now contains 'onshore', 'offshore', or 'unknown'
print(turbines['domain'].value_counts())

# Filter by domain
onshore_turbines = filter_by_domain(turbines, "onshore")
offshore_turbines = filter_by_domain(turbines, "offshore")
```

### Advanced Options

```python
# Use faster spatial join method (requires rtree or pygeos)
turbines = add_domain_column(
    turbines,
    onshore_geojson='regions/onshore.geojson',
    offshore_geojson='regions/offshore.geojson',
    method='spatial_join',  # Default, fast
    prefer_onshore=True,     # If point in both regions, use onshore
)

# Or use point-in-polygon (slower but more reliable)
turbines = add_domain_column(
    turbines,
    onshore_geojson='regions/onshore.geojson',
    offshore_geojson='regions/offshore.geojson',
    method='point_in_polygon',
)

# Handle custom column names
turbines = add_domain_column(
    turbines,
    onshore_geojson='regions/onshore.geojson',
    offshore_geojson='regions/offshore.geojson',
    lon_col='longitude',
    lat_col='latitude',
)
```

### Atlite Integration - Export Corrections to Grid

> ⚠️ **Experimental (research use only):** The ML workflows in `ml/` and the atlite export utilities are experimental and intended for research purposes, not production use.

Export PyVWF bias corrections onto an atlite cutout grid for wind simulations:

```python
from vwf import export_pyvwf_grid

# Export correction factors to atlite cutout grid
export_pyvwf_grid(
    cutout_nc='cutouts/europe-2023.nc',
    points_csv='out/correction_points.csv',  # Your bias correction results
    out_nc='output/bias_grid.nc',
    onshore_geojson='input/regions/country_shapes.geojson',
    offshore_geojson='input/regions/north_sea_shape.geojson',
    variogram_model='spherical',  # Spatial interpolation method
    n_closest_onshore=50,         # Local kriging for performance
    n_closest_offshore=80,
)
```

The output NetCDF contains gridded `scalar` and `offset` correction fields that can be applied to atlite wind power calculations.

**Quick demo (creates synthetic data):**
```bash
python examples/atlite_quickstart.py
```

For more examples, see:
- `examples/atlite_quickstart.py` - Ready-to-run demo with synthetic data
- `examples/atlite_export_examples.py` - 8 comprehensive examples
- `examples/categorize_turbines.py` - Geospatial classification examples

## Machine Learning Experiments

The `ml/` directory contains experimental scripts for predicting correction factors using terrain/climate features. **Note: These experiments showed that ML transfer learning is ineffective for correction factors.**

Key findings:
- ❌ Cross-country prediction: All R² < 0 (worse than baseline)
- ❌ Within-country: Only Denmark shows weak positive R² (0.029), likely spurious
- ✅ Conclusion: Region-specific calibration with local observations required

See [ml/README.md](ml/README.md) for full documentation and [ml/WITHIN_COUNTRY_RESULTS.md](ml/WITHIN_COUNTRY_RESULTS.md) for detailed results.

**Not recommended for production use** - included for research transparency.

## Machine Learning-Based Bias Correction

PyVWF now supports machine learning models to learn the relationship between terrain features and bias correction factors. This enables:

- **Physical understanding**: Identify which terrain features drive model bias
- **Spatial transfer**: Apply corrections to regions without observations
- **Improved interpolation**: Use terrain predictors instead of pure spatial interpolation

### Quick Start

```python
from vwf import export_ml_correction_grid

# Train ML model and export gridded corrections in one step
export_ml_correction_grid(
    corrections_csv='out/correction_points.csv',
    grid_nc='cutouts/europe-2023.nc',
    out_nc='out/ml_bias_grid.nc',
    terrain_nc='input/terrain/europe_terrain.nc',  # Optional: elevation, slope, etc.
    coastline_geojson='input/regions/coastline.geojson',  # Optional: for distance-to-coast
    model_type='random_forest',  # Or: gradient_boosting, xgboost, lightgbm, ridge
    n_estimators=200,
    max_depth=15,
)
```

### Training Individual Models

```python
from vwf.ml_correction import create_feature_matrix, train_correction_model

# Load correction points and add terrain features
corrections = pd.read_csv('out/correction_points.csv')

features = create_feature_matrix(
    corrections,
    terrain_nc='input/terrain/europe_terrain.nc',
    coastline_geojson='input/regions/coastline.geojson',
)

# Train model for scalar correction
scalar_model = train_correction_model(
    features,
    target_col='scalar',
    model_type='random_forest',
    cv_folds=5,
)

# View feature importance
print(scalar_model['feature_importance'])
```

### Comparing Methods

```python
from vwf.ml_correction import compare_interpolation_methods

# Compare different ML models
comparison = compare_interpolation_methods(
    features,
    feature_cols=['elevation', 'slope', 'roughness', 'distance_to_coast_km'],
    target_col='scalar',
    models=['random_forest', 'gradient_boosting', 'xgboost', 'ridge'],
    cv_folds=5,
)

# Results show R², MAE, RMSE for each model
print(comparison)
```

### Transfer Learning

Train on one region and apply to another:

```python
from vwf.ml_correction import train_correction_model, predict_correction_grid

# Train on UK data
uk_features = create_feature_matrix(uk_corrections, terrain_nc='terrain.nc')
model = train_correction_model(uk_features, target_col='scalar')

# Apply to Germany (no observations needed)
de_corrections = predict_correction_grid(
    model,
    grid_nc='cutouts/germany-2023.nc',
    terrain_nc='terrain.nc',
)
```

### Required Dependencies

Install ML dependencies:

```bash
# Basic ML (required)
pip install scikit-learn

# Optional: Advanced models
pip install xgboost lightgbm
```

For comprehensive examples, see `examples/ml_terrain_correction.py`.

## Data Requirements

PyVWF expects the following input data types.

### Required Inputs

|Data|Format|Description|
|---|---|---|
|Reanalysis wind data|NetCDF|ERA5 wind components (e.g. u100, v100)|
|Turbine metadata|CSV|Location, capacity, hub height, turbine model|
|Observed generation|CSV|Time series of wind generation or capacity factor|
|Power curves|CSV|Wind speed to power conversion|

The files you should provide are:

- Observation data for all training years placed in `input/country-data/observation/`. Example files are in the repository.
- Reanalysis data for all training years and test years in `data/era5/<country>/<test/train>/`
- Turbine metadata which contains information such as the height, latitude, longitude, turbine ID, turbine model and capacity placed in `data/turb_info/`. An example is provided, plan to make this file easier to create.
- Wind turbine power curves in a .csv file with model names in each column providing the power output with respect to wind speed. Due to proprietary data used in our curve file an example of the format is shown in `input/power_curves.csv`

**For detailed information about data sources, including where to obtain turbine metadata and generation observations for different countries, see [DATA_SOURCES.md](DATA_SOURCES.md).**

#### Download reanalysis wind speed data

Download the necessary input ERA-5 data (Years in a period can be downloaded separately or together as they will be joined. Ensure training data is separate to validation):

- ECMWF's [ERA-5 reanalysis](https://cds-beta.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download), the required variables are either:
  - 100m u-component of wind, 100m v-component of wind, 10m u-component of wind and 10m v-component of wind (surface roughness is calculated instead and is more accurate).
  - 100m u-component of wind, 100m v-component of wind and Forecast surface roughness. 

## Output structure

All PyVWF outputs are written to the user-specified output directory (`outdir`)
and organised by run configuration. Each run is fully self-contained.

### Directory layout

```text
outdir/
└── run/
    └── <run_name>/
        ├── plots/
        ├── results/
        │   ├── capacity-factor/
        │   └── wind-speed/
        └── training/
            ├── correction-factors/
            └── simulated-turbines/
```

The `<run_name>` encodes the scenario configuration (e.g. country, correction
mode, surface roughness treatment).

#### Plots (`plots/`)

Diagnostic figures summarising model performance:

- `*_full_error_appendix.png`
Overall error metrics across all clusters and time resolutions.

- `*_spatial_focus_error_appendix.png`
Error metrics emphasising spatial structure.

- `*_temporal_focus_error_appendix.png`
Error metrics emphasising temporal variability.

- These figures are intended for appendix or supplementary material.

#### Capacity factor results (`results/capacity-factor/`)

CSV time series of simulated and observed capacity factor:

- `<COUNTRY>_<YEAR>_<time_res>_<k>_cor_cf.csv`
Bias-corrected capacity factor for a given time-resolution and cluster count.

- `<COUNTRY>_<YEAR>_unc_cf.csv`
Uncorrected (raw reanalysis-based) capacity factor.

- `<COUNTRY>_<YEAR>_obs_cf.csv`
Observed capacity factor used for validation.

All files share a common time index.

## Assumptions & Limitations

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

If you use PyVWF in academic work, please cite the repository.  
A `CITATION.cff` file may be added in future releases.

## Contributing

Contributions are welcome, especially:

- Documentation improvements
- Additional bias correction methods
- Validation case studies
- Performance optimisations

Please open an issue to discuss changes before submitting a pull request.

## Credits & Contact

The PyVWF code is developed by Ellyess F. Benmoufok. You can email me via benmoufok.ellyess@gmail.com.

The original VWF code this is based on is developed by Iain Staffell.  You can try emailing them at i.staffell@imperial.ac.uk

PyVWF is part of the [Renewables.ninja](https://renewables.ninja) project, developed by Stefan Pfenninger and Iain Staffell.  Use the [contacts page](https://www.renewables.ninja/about) there.
