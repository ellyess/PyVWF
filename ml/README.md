# Machine Learning Correction Factor Prediction

This directory contains all scripts, documentation, and results for ML-based prediction of wind power correction factors across European regions.

## Overview

Attempted to predict PyVWF correction factors in new regions using terrain and climate features. **Key finding**: ML transfer learning is ineffective for correction factors - region-specific calibration required.

## Scripts

### Core Training
- **`train_europe_ml_corrections.py`** - Main ML training script
  - Supports spatial cross-validation (--validation-countries)
  - Multiple models: Ridge, Lasso, Elastic Net, SVR, Random Forest, Gradient Boosting
  - Feature extraction: terrain, spatial context, ERA5 mismatch, climate, geographic
  - Caching system for fast re-runs (--force-extract to regenerate)
  - Fast mode (--fast) skips slow ERA5 climate extraction
  - Usage: `python train_europe_ml_corrections.py --countries DK,UK --validation-countries DE`

### Analysis & Comparison
- **`compare_ml_models.py`** - Compare 8 ML algorithms
  - Tests Ridge, Lasso, Elastic Net, SVR, Random Forest, Gradient Boosting, XGBoost, LightGBM
  - Outputs comparison table and plots
  - Usage: `python compare_ml_models.py`

- **`review_ml_results.py`** - Review diagnostic plots and generate reports
  - Analyzes training results from output directories
  - Usage: `python review_ml_results.py`

- **`compare_feature_sets_denmark.py`** - Feature ablation study for Denmark
  - Tests terrain only, terrain+spatial, terrain+ERA5, etc.
  - Usage: `python compare_feature_sets_denmark.py`

### Data Preparation
- **`download_terrain_data.py`** - Download ETOPO, Natural Earth coastlines
  - Downloads and processes terrain data for European domain
  - Usage: `python download_terrain_data.py`

- **`quick_terrain_setup.py`** - Generate synthetic terrain for testing
  - Fast alternative (~10 seconds vs hours for real data)
  - Usage: `python quick_terrain_setup.py`

- **`download_era5_features.py`** - Download ERA5 variables from CDS
  - Downloads invariant fields (geopotential, roughness, land-sea mask)
  - Downloads atmospheric stability variables (BLH, heat flux, temperature)
  - Usage: `python download_era5_features.py --invariant-only`

## Documentation

- **`ML_CORRECTION_GUIDE.md`** - Comprehensive ML methodology guide
- **`TERRAIN_DATA_GUIDE.md`** - Terrain feature requirements and sources
- **`WITHIN_COUNTRY_RESULTS.md`** - Detailed results and analysis

## Results

### Cross-Country Spatial Validation (Ridge Regression)
| Training | Validation | R² | RMSE | Finding |
|----------|------------|-----|------|---------|
| DK+UK | DE | -0.034 | 0.252 | Poor transfer |
| DK+DE | UK | -0.193 | 0.407 | Very poor |
| UK+DE | DK | -1.011 | 0.479 | Catastrophic |

### Within-Country Random Split
| Country | Points | R² | CV R² | RMSE |
|---------|--------|-----|-------|------|
| Denmark | 700 | +0.029 | -0.044 | 0.125 |
| Germany | 500 | -0.022 | -0.085 | 0.210 |
| UK | 320 | -0.071 | -0.064 | 0.322 |

**Conclusion**: Only Denmark shows weak positive R² (0.029), and even that's questionable given negative CV scores. Correction factors are not predictable from terrain/climate features.

## Features Tested

1. **Terrain (6)**: elevation, slope, aspect, roughness, curvature, distance_to_coast
2. **Spatial Context (3)**: nearby_mean_correction, nearby_correction_variance, turbine_density
3. **ERA5 Mismatch (3)**: era5_elevation_error, era5_roughness, subgrid_terrain_variance
4. **Geographic (4)**: latitude, longitude, abs_latitude, coastal_latitude
5. **Climate (4)**: mean_wind_speed, wind_speed_std, wind_speed_p90, wind_speed_p10
6. **Turbine (2)**: hub_height_m, rotor_diameter_m (DK only)

## Output Directories

- **`ml_europe/`** - Main training outputs
  - `europe_correction_model.pkl` - Trained Ridge model
  - `europe_corrections_ml.nc` - Predicted corrections on European grid
  - `scalar_predictions.png`, `offset_predictions.png` - Diagnostic plots
  - `cache/` - Cached extracted features (CSV files)

- **`ml_comparison/`** - Model comparison outputs
  - `scalar/` - Scalar correction results
  - `offset/` - Offset correction results
  - Model comparison tables and plots

## Key Findings

### Why ML Transfer Learning Failed

1. **Methodological Differences**: Each country uses different calibration procedures
2. **Model Configuration**: Turbine characteristics and measurement uncertainty vary
3. **Local Practices**: Site-specific adjustments not captured by terrain features
4. **Observation Quality**: Different instruments, temporal coverage, QC procedures
5. **Physical Complexity**: Corrections depend on atmospheric stability, mesoscale patterns

### Feature Engineering Attempts

All feature additions made performance **worse**:
- Terrain baseline: R² = -0.034
- + Climate features: R² = -0.076
- + Spatial context: R² = -0.377
- + ERA5 mismatch: R² = -0.432

Even with ERA5 elevation errors up to 2.2km in German Alps, features didn't explain correction factors.

### Research Implications

**Region-specific calibration required**. Cannot reliably predict corrections in new areas from terrain/climate features alone. Each region needs:
- Local observational data (wind farm production)
- Country-specific calibration methodology
- Validation against actual measurements
- Understanding of local wind resource assessment practices

## Requirements

```bash
# Python packages
pip install pandas numpy xarray netCDF4 geopandas shapely
pip install scikit-learn matplotlib seaborn
pip install xgboost lightgbm  # For model comparison
pip install cdsapi  # For ERA5 download

# Data sources
# - Correction factors: run/{COUNTRY}-{TYPE}-obs_turbine-corrected-calc_z0/training/correction-factors/
# - Terrain: input/terrain/terrain_north_sea_full.nc
# - ERA5: input/era5/invariant/era5_invariant_europe.nc
# - Coastlines: input/terrain/coastlines.geojson
```

## Future Work (Not Recommended)

Given the poor results, ML-based correction prediction is **not recommended**. However, if pursuing:

1. **More sophisticated features**: Atmospheric stability, mesoscale wind patterns, seasonal variations
2. **Deep learning**: CNN/RNN architectures to capture spatial-temporal patterns
3. **Physics-informed ML**: Incorporate wind profile theory, boundary layer physics
4. **Ensemble methods**: Combine multiple models with uncertainty quantification
5. **Transfer learning**: Pre-train on global reanalysis, fine-tune on local observations

**Reality check**: Even these approaches unlikely to overcome fundamental limitation that correction factors reflect calibration methodology, not just physical terrain/climate.

## Contact

For questions about PyVWF ML correction prediction:
- See main README.md in repository root
- Check SUPPLEMENTARY_METHODS.md for PyVWF methodology
- Review WITHIN_COUNTRY_RESULTS.md for detailed analysis
