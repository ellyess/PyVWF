# Experimental Extensions

> **Experimental (research use only).** The grid-export and machine-learning
> workflows below build on the core, validated bias-correction pipeline. They
> are intended for research, not production. The selling point of PyVWF is the
> granular linear bias correction; these extensions are exploratory.

Both extensions live under `vwf.extensions` and opt in via the `[ml]` extra
where needed (`pip install pyvwf[ml]`). Their command-line drivers are
`pyvwf-grid` and `pyvwf-ml`, and the full multi-method research pipeline lives
under `scripts/pyvwf_to_grid/` and `scripts/pyvwf_ml/` (see
[PIPELINE.md](../PIPELINE.md)).

## Grid export (atlite integration)

Export PyVWF bias corrections onto an atlite cutout grid for wind simulations:

```python
from vwf import export_pyvwf_grid

# Export correction factors to atlite cutout grid
export_pyvwf_grid(
    cutout_nc='cutouts/europe-2023.nc',
    points_csv='output/correction_points.csv',  # Your bias correction results
    out_nc='output/bias_grid.nc',
    onshore_geojson='input/regions/country_shapes.geojson',
    offshore_geojson='input/regions/north_sea_shape.geojson',
    variogram_model='spherical',  # Spatial interpolation method
    n_closest_onshore=50,         # Local kriging for performance
    n_closest_offshore=80,
)
```

The output NetCDF contains gridded `scalar` and `offset` correction fields that can be applied to atlite wind power calculations.

## Machine learning-based bias correction

PyVWF includes ML models to learn the relationship between terrain/environmental features and bias correction factors. This enables:

- **Physical understanding**: Identify which terrain features drive model bias
- **Spatial transfer**: Apply corrections to regions without observations
- **Improved interpolation**: Use terrain predictors instead of pure spatial interpolation

### Quick Start

```python
from vwf import export_ml_correction_grid

# Train ML model and export gridded corrections in one step
export_ml_correction_grid(
    corrections_csv='output/correction_points.csv',
    grid_nc='cutouts/europe-2023.nc',
    out_nc='output/ml_bias_grid.nc',
    terrain_nc='input/terrain/europe_terrain.nc',  # Optional: elevation, slope, etc.
    coastline_geojson='input/regions/coastline.geojson',  # Optional: for distance-to-coast
    model_type='random_forest',  # Or: gradient_boosting, xgboost, lightgbm, ridge
    n_estimators=200,
    max_depth=15,
)
```

### Training Individual Models

```python
from vwf.extensions.ml import create_feature_matrix, train_correction_model

# Load correction points and add terrain features
corrections = pd.read_csv('output/correction_points.csv')

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
from vwf.extensions.ml import compare_interpolation_methods

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

### Pipeline Scripts

For the full ML pipeline (terrain data download, feature engineering, model training, and figure generation), see [PIPELINE.md](../PIPELINE.md) Stage 3.

### Required Dependencies

Install ML dependencies:

```bash
# Basic ML (required)
pip install scikit-learn

# Optional: Advanced models
pip install xgboost lightgbm
```
