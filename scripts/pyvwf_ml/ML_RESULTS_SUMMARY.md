# ML Prediction of ERA5 Bias Correction Factors -- Results Summary

**Dataset**: 1,474 correction centroids from 14 configurations across Europe
**Terrain**: Real ETOPO 2022 elevation data (`terrain_europe_full.nc`, lon -12 to 35, lat 35 to 72)
**Features**: 8 (5 terrain + 3 spatial)

---

## 1. Dataset

### Unified Corrections

All correction factors are produced by the same PyVWF calibration pipeline applied consistently across all regions, eliminating methodology-driven biases that affected earlier ML attempts.

| Source | Country/Region | Clusters | Obs Level |
|--------|---------------|----------|-----------|
| `all_corrections_centroids.csv` | NL | 5 | country |
| | FR | 10 | country |
| | BE | 3 | country |
| | NO | 5 | country |
| | ES | 4 | country |
| | SE | 4 | country |
| | IT | 3 | country |
| | PT | 3 | country |
| | IE | 3 | country |
| | DE-onshore | 500 | turbine |
| | DK-onshore | 629 | turbine |
| | DK-offshore | 2 | turbine |
| | UK-onshore | 293 | turbine |
| | UK-offshore | 10 | turbine |
| **Total** | | **1,474** | |

The dataset is dominated by turbine-level configurations: DE-onshore (34%), DK-onshore (43%), and UK-onshore (20%) account for 96% of samples. Country-level entries contribute only ~40 points total.

### Correction Value Ranges

- **Scalar**: 0.216 to 4.644 (multiplicative; 1.0 = no correction)
- **Offset**: -8.45 to +2.78 m/s (additive; 0.0 = no correction)

### Feature Set

| Feature | Source | Description |
|---------|--------|-------------|
| `terrain_elevation` | ETOPO 2022 | Elevation above sea level (m) |
| `terrain_slope` | ETOPO-derived | Terrain slope (degrees) |
| `terrain_aspect` | ETOPO-derived | Terrain aspect/orientation (degrees) |
| `terrain_roughness` | ETOPO-derived | Local roughness (std of elevation) |
| `terrain_curvature` | ETOPO-derived | Terrain curvature |
| `abs_lat` | Coordinates | Absolute latitude (climate proxy) |
| `lon_normalized` | Coordinates | Normalised longitude |
| `lat_normalized` | Coordinates | Normalised latitude |

94 NaN values in terrain features (points outside ETOPO coverage, e.g. offshore) were filled with feature medians.

---

## 2. Model Comparison (5-Fold Random CV)

Five model types were compared using 5-fold random cross-validation on all 1,474 samples.

| Model | Scalar R² | Scalar MAE | Offset R² | Offset MAE |
|-------|-----------|------------|-----------|------------|
| **Random Forest** | **0.351** | **0.137** | **0.414** | **0.541** |
| **Gradient Boosting** | **0.356** | 0.144 | 0.374 | 0.564 |
| Ridge | 0.098 | 0.173 | 0.158 | 0.684 |
| Lasso | -0.011 | 0.181 | 0.123 | 0.677 |
| Elastic Net | 0.050 | 0.175 | 0.141 | 0.674 |

**Observations**:

1. Tree-based models (Random Forest, Gradient Boosting) substantially outperform linear models. Correction patterns are non-linear -- physically expected given the complex interactions between terrain, atmospheric flow, and ERA5 grid-cell averaging.

2. Random Forest and Gradient Boosting achieve similar scalar R² (~0.35), but Random Forest has better offset performance (R² = 0.41 vs 0.37) and lower MAE for both targets.

3. Linear models explain less than 10% of scalar variance (Ridge) or have negative R² (Lasso), confirming that simple linear relationships between terrain and corrections do not exist.

---

## 3. Random Forest Detailed Results

### Cross-Validation (5-fold)

| Target | R² | MAE | RMSE |
|--------|-----|-----|------|
| Scalar | 0.351 +/- 0.145 | 0.137 +/- 0.006 | 0.228 +/- 0.033 |
| Offset | 0.414 +/- 0.098 | 0.541 +/- 0.018 | 0.822 +/- 0.055 |

### Training Set (full data, for reference)

| Target | R² | MAE | RMSE | Bias |
|--------|-----|-----|------|------|
| Scalar | 0.653 | 0.093 | 0.173 | 0.001 |
| Offset | 0.714 | 0.363 | 0.584 | -0.002 |

The gap between training R² (0.65-0.71) and CV R² (0.35-0.41) indicates moderate overfitting, which is typical for tree-based models on moderately-sized datasets. The CV scores are the honest assessment of prediction accuracy.

### Feature Importance

Feature importance plots are in `output/grid_run/turbine_grid/ml_results/plots/`:
- `scalar_feature_importance.png`
- `offset_feature_importance.png`
- `scalar_predictions.png` / `offset_predictions.png` (actual vs predicted scatter + residuals)
- `scalar_spatial.png` / `offset_spatial.png` (spatial maps)

---

## 4. Comparison to Spatial Interpolation

### ML vs IDW at Control Points

| Method | Scalar R² | Scalar MAE | Offset R² | Offset MAE |
|--------|-----------|------------|-----------|------------|
| IDW | 0.779 | 0.095 | 0.754 | 0.383 |
| ML (RF) | 0.653 | 0.093 | 0.714 | 0.363 |

IDW achieves higher R² at the control points because it directly interpolates known values -- nearest-neighbour effects give it an inherent advantage at training locations. ML has slightly better MAE, suggesting it smooths outliers better.

### Spatial Interpolation Cross-Validation (from grid comparison script)

The spatial interpolation methods were also cross-validated (5-fold spatial CV by longitude):

| Method | Scalar MAE | Scalar RMSE | Offset MAE | Offset RMSE |
|--------|------------|-------------|------------|-------------|
| **IDW** | **0.166** | **0.261** | **0.657** | **0.988** |
| **Kriging** | **0.182** | **0.260** | **0.669** | **0.910** |
| RBF | 0.234 | 0.333 | 0.864 | 1.147 |

IDW and Kriging perform similarly in spatial CV, with IDW having slightly lower MAE and Kriging slightly lower RMSE. RBF (thin plate spline) overshoots in extrapolation regions.

### Comparison Context

The ML random CV MAE (0.137 scalar, 0.541 offset) is lower than the spatial interpolation CV MAE (0.166 scalar, 0.657 offset), but these use different CV strategies (random vs spatial), making direct comparison imprecise. The key takeaway is that both approaches achieve useful prediction accuracy, with IDW/Kriging being simpler and ML providing potentially richer extrapolation through terrain features.

---

## 5. Transfer Learning Analysis

### The Core Problem

Previous experiments tested whether ML models trained on some countries can predict corrections for held-out countries (spatial cross-validation). All such experiments produced negative R², meaning the model predictions are worse than simply predicting the overall mean.

### Summary of Previous Spatial CV Results

| Holdout Country | With Spatial Features | Terrain Only | Notes |
|----------------|----------------------|-------------|-------|
| Germany (scalar R²) | -0.21 to -1.05 | +0.005 to -0.46 | Near break-even terrain-only |
| UK (scalar R²) | -0.10 to -1.91 | -0.21 to -0.57 | Consistently fails |
| Denmark (scalar R²) | -1.88 | -0.12 | Large test set (51% of data) |

### Why Transfer Fails

1. **Spatial features encode country identity**: Longitude and latitude correlate strongly with which country a point belongs to. When an entire country is held out, the model has never seen that lon/lat range and extrapolation fails.

2. **Removing spatial features helps but is insufficient**: Terrain-only models reduce overfitting dramatically (e.g. DK scalar R² from -1.88 to -0.12) but still do not achieve positive R² in most holdout tests.

3. **Country-specific effects persist**: Even with unified methodology, corrections reflect country-specific factors beyond terrain:
   - Different turbine fleets (hub heights 80-140m, varied rotor diameters)
   - Different observation network densities
   - Regional climate patterns (Atlantic maritime vs continental)
   - Different ERA5 performance by weather regime

4. **Real terrain did not solve the problem**: Testing with real ETOPO data (vs synthetic terrain) improved Germany holdout from R² = -0.46 to -0.10 (78% improvement) but UK worsened from -0.45 to -0.57. Terrain quality was a contributing factor but not the fundamental limitation.

### Implications

- ML is suitable for **spatial interpolation** (filling gaps between known correction points)
- ML is **not suitable for transfer** to entirely new geographic regions without local calibration data
- For new regions, use the physics-based PyVWF pipeline with local observations, then interpolate the resulting corrections spatially

---

## 6. Spatial Interpolation Grids

The grid comparison script (`compare_unified_corrections_to_grid.py`) produced interpolated correction surfaces on an ERA5 0.25-degree European grid:

### Output NetCDF Files

Located in `output/grid_run/turbine_grid/grid_comparison/`:

| File | Method | Description |
|------|--------|-------------|
| `europe_corrections_nearest.nc` | Nearest Neighbour | Sharp boundaries between regions |
| `europe_corrections_idw.nc` | IDW (power=2) | Smooth spatial interpolation |
| `europe_corrections_rbf.nc` | RBF (thin plate spline) | Smooth but overshoots at edges |
| `europe_corrections_kriging.nc` | Ordinary Kriging | Geostatistical interpolation |

All grids have a distance mask applied: points beyond 5 degrees from any control point are set to neutral values (scalar=1.0, offset=0.0) to prevent unrealistic extrapolation.

Unmasked versions are also available (`*_unmasked.nc`).

### Comparison Plots

- `interpolation_comparison_scalar.png` -- side-by-side scalar fields for all methods
- `interpolation_comparison_offset.png` -- side-by-side offset fields for all methods
- `cv_scores_comparison.png` -- bar chart comparing spatial CV scores

---

## 7. Thesis Presentation

### Key Messages

1. **Unified methodology enables meaningful ML experiments**: By applying the same PyVWF calibration procedure across all regions, corrections reflect actual terrain-driven bias patterns rather than methodology artefacts.

2. **Terrain + spatial features explain 35-41% of correction variance**: Random Forest cross-validation R² = 0.35 (scalar), 0.41 (offset). Non-linear models are essential.

3. **Transfer learning to new countries does not work**: Spatial CV yields negative R². Corrections have country-specific components (turbine fleet, climate regime, observation quality) not captured by terrain alone.

4. **Spatial interpolation is the practical approach**: IDW and Kriging provide reliable correction surfaces for ERA5 grid cells between known correction points. A 5-degree distance mask prevents unrealistic extrapolation.

5. **ML and IDW achieve comparable performance**: Both predict corrections with similar accuracy. IDW is simpler and more interpretable; ML provides some smoothing benefit.

### Recommended Figures

| Figure | File Location | Shows |
|--------|--------------|-------|
| Model comparison | `ml_results/model_comparison.csv` | RF/GB >> linear models |
| Scalar predictions | `ml_results/plots/scalar_predictions.png` | Actual vs predicted scatter |
| Offset predictions | `ml_results/plots/offset_predictions.png` | Actual vs predicted scatter |
| Feature importance (scalar) | `ml_results/plots/scalar_feature_importance.png` | Which features matter |
| Feature importance (offset) | `ml_results/plots/offset_feature_importance.png` | Which features matter |
| Spatial predictions | `ml_results/plots/scalar_spatial.png` | Map of true vs predicted |
| Interpolation comparison | `grid_comparison/interpolation_comparison_scalar.png` | IDW vs RBF vs Kriging maps |
| CV scores comparison | `grid_comparison/cv_scores_comparison.png` | Method performance bars |

### Suggested Narrative

The ML experiments serve two purposes in the thesis:

**Positive result**: The unified correction dataset enables meaningful ML modelling. Random Forest explains 35-41% of correction variance from terrain and spatial features, confirming that corrections have systematic spatial structure linked to physical terrain characteristics.

**Negative result (equally important)**: Transfer learning fails. Models trained on European countries cannot predict corrections for held-out countries. This demonstrates that wind speed bias corrections are fundamentally location-specific, driven by factors beyond terrain (turbine technology, local climate, observation network quality). This motivates the use of spatial interpolation methods (IDW, Kriging) rather than ML extrapolation for extending corrections to new regions.

---

## 8. File Reference

### Scripts

| Script | Purpose |
|--------|---------|
| `ml/scripts/train_unified_ml_corrections.py` | ML model training and evaluation |
| `scripts/pyvwf_to_grid/create_unified_correction_dataframe.py` | Create unified corrections CSV/GeoJSON |
| `scripts/pyvwf_to_grid/compare_unified_corrections_to_grid.py` | Spatial interpolation grid comparison |
| `scripts/evaluate_all_pyvwf_runs.py` | Evaluate all PyVWF runs (MAE, RMSE, MBE) |

### Data

| File | Description |
|------|-------------|
| `output/grid_run/turbine_grid/all_corrections_centroids.csv` | 1,474 correction centroids |
| `output/grid_run/turbine_grid/all_corrections_polygons.geojson` | Cluster polygon geometries |
| `output/grid_run/turbine_grid/all_corrections_centroids.geojson` | Centroid point geometries |
| `input/terrain/terrain_europe_full.nc` | Real ETOPO terrain features |
| `output/runs/turbine_grid/pyvwf_evaluation_metrics.csv` | Per-run evaluation metrics |

### Results

| Directory | Contents |
|-----------|----------|
| `output/grid_run/turbine_grid/ml_results/` | ML model comparison, training summary, diagnostic plots |
| `output/grid_run/turbine_grid/grid_comparison/` | Interpolated NetCDF grids, CV scores, comparison plots |
