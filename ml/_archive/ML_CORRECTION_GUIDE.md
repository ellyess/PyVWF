# Machine Learning-Based Bias Correction with Terrain Features

## Overview

The PyVWF package now includes a comprehensive ML module (`vwf.ml_correction`) that enables you to:

1. **Learn relationships** between terrain features and bias correction factors
2. **Transfer corrections** to regions without observations
3. **Improve spatial interpolation** using physical predictors
4. **Understand drivers** of model bias through feature importance

## Key Capabilities

### 1. Terrain Feature Extraction

Extract and compute terrain features from various data sources:

- **From NetCDF files**: Elevation, slope, roughness, curvature
- **Computed derivatives**: Gradient, aspect, terrain roughness
- **Distance metrics**: Distance to coastline, distance to specific features
- **Coordinate-based**: Normalized lat/lon, spatial patterns

### 2. ML Model Training

Support for multiple model types:

- **Tree-based**: Random Forest, Gradient Boosting
- **Advanced**: XGBoost, LightGBM (optional, high performance)
- **Linear**: Ridge, Lasso, Elastic Net
- **Custom**: Any scikit-learn compatible estimator

### 3. Model Evaluation

- **Cross-validation**: K-fold CV with multiple metrics (R², MAE, RMSE)
- **Feature importance**: Understand which terrain features matter most
- **Model comparison**: Automatically compare multiple model types
- **Spatial CV**: Option for spatial blocking (future enhancement)

### 4. Spatial Prediction

- **Grid prediction**: Apply trained models to regular grids
- **Masked regions**: Separate onshore/offshore predictions
- **NetCDF export**: Compatible with atlite workflow

## Workflow

### Basic Workflow

```python
from vwf.ml_correction import (
    create_feature_matrix,
    train_correction_model,
    predict_correction_grid,
)

# 1. Load correction points from your bias correction workflow
corrections = pd.read_csv('output/correction_points.csv')

# 2. Add terrain features
features = create_feature_matrix(
    corrections,
    terrain_nc='input/terrain/europe_terrain.nc',
    coastline_geojson='input/regions/coastline.geojson',
)

# 3. Train model
model = train_correction_model(
    features,
    target_col='scalar',
    model_type='random_forest',
    cv_folds=5,
)

# 4. Apply to grid
grid_corrections = predict_correction_grid(
    model,
    grid_nc='cutouts/europe-2023.nc',
    terrain_nc='input/terrain/europe_terrain.nc',
)
```

### One-Step Pipeline

```python
from vwf import export_ml_correction_grid

# Complete pipeline in one function call
export_ml_correction_grid(
    corrections_csv='output/correction_points.csv',
    grid_nc='cutouts/europe-2023.nc',
    out_nc='output/ml_bias_grid.nc',
    terrain_nc='input/terrain/europe_terrain.nc',
    coastline_geojson='input/regions/coastline.geojson',
    model_type='random_forest',
)
```

## Use Cases

### 1. Understanding Bias Drivers

Train models and examine feature importance to understand what causes bias:

```python
model = train_correction_model(features, target_col='scalar', model_type='random_forest')

# View feature importance
print(model['feature_importance'])
# Example output:
#       feature  importance
#    elevation      0.342
#        slope      0.218
#   roughness      0.156
#  distance_km     0.128
```

### 2. Transfer Learning

Train on data-rich regions, apply to data-sparse regions:

```python
# Train on UK (has observations)
uk_model = train_correction_model(uk_features, target_col='scalar')

# Apply to regions without observations
de_corrections = predict_correction_grid(uk_model, grid_nc='germany.nc')
fr_corrections = predict_correction_grid(uk_model, grid_nc='france.nc')
```

### 3. Improved Interpolation

Instead of pure spatial interpolation (kriging, IDW), use terrain features:

```python
# Compare spatial vs. ML methods
from vwf.ml_correction import compare_interpolation_methods

comparison = compare_interpolation_methods(
    features,
    feature_cols=['elevation', 'slope', 'roughness'],
    target_col='scalar',
    models=['random_forest', 'gradient_boosting', 'ridge'],
)
# Shows which approach has better CV performance
```

### 4. Separate Onshore/Offshore Models

Different terrain characteristics affect onshore vs. offshore differently:

```python
from vwf import filter_by_domain

# Train separate models
onshore_features = filter_by_domain(features, 'onshore')
offshore_features = filter_by_domain(features, 'offshore')

onshore_model = train_correction_model(onshore_features, target_col='scalar')
offshore_model = train_correction_model(offshore_features, target_col='scalar')
```

## Installation

### Required Dependencies

```bash
# Basic ML functionality
pip install scikit-learn

# For terrain feature computation
pip install xarray netcdf4

# Already part of PyVWF
pip install pandas numpy geopandas
```

### Optional Dependencies

```bash
# Advanced gradient boosting (recommended for best performance)
pip install xgboost lightgbm

# For terrain processing
pip install rasterio  # If working with GeoTIFF elevation data
```

## Data Requirements

### Correction Points CSV

Must contain at minimum:
- `lon`, `lat`: Coordinates
- `scalar`, `offset`: Correction factors (from bias correction workflow)

Optional columns:
- Terrain features (if pre-computed)
- `domain` or `type`: Onshore/offshore classification

### Terrain Data

Recommended terrain features:

1. **Elevation** (meters above sea level)
2. **Slope** (degrees)
3. **Terrain roughness** (std of elevation in neighborhood)
4. **Curvature** (concavity/convexity)
5. **Distance to coast** (km)
6. **Land cover** (optional: forest, urban, water, etc.)

Sources:
- **SRTM**: 30m or 90m resolution DEM
- **ETOPO**: Global bathymetry and elevation
- **EU-DEM**: 25m resolution for Europe
- **GEBCO**: Bathymetry for offshore

### Region Masks

GeoJSON files for:
- Onshore regions (land boundaries)
- Offshore regions (sea areas)
- Coastline (for distance calculation)

## Performance Considerations

### Model Selection

| Model | Speed | Accuracy | Interpretability | Best For |
|-------|-------|----------|------------------|----------|
| Random Forest | Fast | High | Good | General use, stable |
| Gradient Boosting | Medium | High | Good | When accuracy matters |
| XGBoost | Fast | Very High | Good | Large datasets |
| LightGBM | Very Fast | Very High | Good | Very large datasets |
| Ridge/Lasso | Very Fast | Medium | Excellent | Linear relationships |

### Tips for Large Datasets

1. **Use XGBoost or LightGBM** for datasets > 100k points
2. **Subsample for training** if needed (use stratified sampling)
3. **Feature selection** to reduce dimensionality
4. **Parallelization**: Set `n_jobs=-1` for tree models

### Memory Optimization

- Load terrain data lazily with `xarray`
- Process grids in chunks for very large domains
- Use `dask` for distributed computing (future enhancement)

## Validation

The module uses cross-validation by default:

```python
model = train_correction_model(
    features,
    target_col='scalar',
    cv_folds=5,  # 5-fold cross-validation
)

# Access CV results
cv_scores = model['cv_scores']
print(f"Mean R²: {cv_scores['test_r2'].mean():.3f}")
print(f"Std R²: {cv_scores['test_r2'].std():.3f}")
```

### Spatial Cross-Validation

For spatially-correlated data, consider spatial blocking (custom implementation):

```python
from sklearn.model_selection import GroupKFold

# Group by spatial tiles
features['tile'] = (features['lon'] // 2).astype(int) * 1000 + (features['lat'] // 2).astype(int)

cv = GroupKFold(n_splits=5)
# Use in cross_validate with groups=features['tile']
```

## Examples

All examples are in the `examples/` directory:

- **`ml_quickstart.py`**: Basic demo with synthetic data
- **`ml_terrain_correction.py`**: Comprehensive examples covering all use cases

Run quickstart:
```bash
python examples/ml_quickstart.py
```

## Research Applications

### 1. Bias Attribution

Quantify how much of the bias is explained by different terrain features:

- Elevation effects (orographic forcing)
- Slope effects (flow acceleration)
- Roughness effects (drag, turbulence)
- Coastal effects (land-sea transitions)

### 2. Model Improvement

Use insights to improve the underlying wind model:

- Identify systematic biases
- Guide parameterization improvements
- Suggest new correction schemes

### 3. Uncertainty Quantification

- Use ensemble models for prediction intervals
- Bootstrap for confidence intervals
- Cross-validation for out-of-sample error estimates

### 4. Publication-Ready Plots

```python
import matplotlib.pyplot as plt

# Feature importance plot
importance = model['feature_importance']
plt.barh(importance['feature'], importance['importance'])
plt.xlabel('Importance')
plt.title('Terrain Feature Importance for Bias Correction')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)

# Actual vs predicted
plt.scatter(y_true, y_pred, alpha=0.5)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
plt.xlabel('Actual Correction Factor')
plt.ylabel('Predicted Correction Factor')
plt.title(f'Model Performance (R² = {r2:.3f})')
plt.savefig('predicted_vs_actual.png', dpi=300)
```

## Troubleshooting

### "Insufficient samples after filtering"

- Check for NaN values in features or target
- Ensure at least 10-20 samples per feature
- Consider feature selection to reduce dimensionality

### "Could not find lon/lat coordinates"

- Ensure NetCDF has 'lon' and 'lat' coordinates
- Or 'x' and 'y' coordinates (will be renamed automatically)
- Check coordinate names with `ds.coords`

### "ModuleNotFoundError: No module named 'xgboost'"

- XGBoost/LightGBM are optional dependencies
- Either install them: `pip install xgboost lightgbm`
- Or use built-in models: `model_type='random_forest'`

### Poor model performance (R² < 0.3)

- Check if terrain features are informative for your region
- Try different model types
- Add more features (interactions, polynomials)
- Ensure sufficient training data
- Check for data quality issues

## Future Enhancements

Planned features:

- [ ] Spatial cross-validation
- [ ] Neural network models (deep learning)
- [ ] Automatic hyperparameter tuning
- [ ] Ensemble methods
- [ ] Temporal features (season, month)
- [ ] Uncertainty quantification
- [ ] Online learning (incremental updates)
- [ ] GPU acceleration

## Citation

If you use this ML module in your research, please cite:

```bibtex
@software{pyvwf_ml,
  title={PyVWF: Machine Learning-Based Bias Correction for Wind Power Modeling},
  author={[Your Name]},
  year={2026},
  url={https://github.com/ellyess/PyVWF}
}
```

## Contact & Support

- GitHub Issues: [https://github.com/ellyess/PyVWF/issues](https://github.com/ellyess/PyVWF/issues)
- Documentation: See `README.md` and example scripts
- Examples: `examples/ml_terrain_correction.py`
