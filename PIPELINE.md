# PyVWF Pipeline: Script Execution Order

Three-stage pipeline: **PyVWF** (bias correction training) → **PyVWF Grid** (spatial interpolation) → **PyVWF ML** (machine learning corrections).

---

## Stage 1: PyVWF Core — Bias Correction Training

Train bias correction factors from observed vs reanalysis capacity factors.

### 1.0 Prerequisites: Generate Country-Level Training Data

```bash
python vwf/datasets/generate_country_level_training_data.py
```

- Fetches ENTSO-E capacity factor observations via API
- Creates grid sample points and Voronoi regions per country
- Splits into training/test years
- **Generates** `input/country_level_data/pyvwf_config.py` (used by all subsequent scripts)
- **Output:** `input/country_level_data/observations/`, `input/country_level_data/grid_points/`
- Required for country-level workflows (NL, FR, BE, NO, SE, ES, IT, PT, IE)
- Not needed if only running turbine-level (DK, DE, UK)

### 1.1 Train All Corrections (Recommended)

```bash
# List available configuration sets
python train_all_bias_corrections.py --list

# Run turbine + country workflows
python train_all_bias_corrections.py --sets turbine_grid country_grid_2015_2021_2023
```

Master orchestrator supporting all training configurations:

| Config Set | Level | Countries | Training Years |
|---|---|---|---|
| `turbine_fixed_2015_2019` | Turbine | DK, DE, UK | 2015-2019 |
| `turbine_dk_research` | Turbine | DK (onshore+offshore) | 2015-2019 |
| `turbine_grid` | Turbine | DK, DE, UK | 2015-2019 |
| `country_grid_2015_2021_2023` | Country | NL, FR, BE, NO, SE, ES, IT, PT, IE | 2015-2021 |

**Output:** `output/runs/<prefix>/<country-run>/training/correction-factors/`

### 1.2 Evaluate Corrections

```bash
python scripts/evaluate_all_pyvwf_runs.py --prefix turbine_grid
```

- Calculates MAE, RMSE, R² for corrected vs uncorrected capacity factors
- **Output:** `output/runs/turbine_grid/pyvwf_evaluation_metrics.csv`

---

## Stage 2: PyVWF Grid — Spatial Interpolation to ERA5 Grid

Interpolate point-based corrections onto a continuous European grid for atlas applications.

### 2.1 Create Unified Correction Dataset

```bash
python scripts/pyvwf_to_grid/create_unified_correction_dataframe.py
```

- Combines all Stage 1 corrections (turbine + country) into one dataset
- Extracts cluster centroid coordinates
- **Requires:** Stage 1 outputs in `output/runs/turbine_grid/`
- **Output:**
  - `output/pyvwf_to_grid/all_corrections_centroids.csv` (used by everything downstream)
  - `output/pyvwf_to_grid/all_corrections_polygons.geojson`
  - `output/pyvwf_to_grid/all_corrections_centroids.geojson`

### 2.2 Generate Cluster Geometries (if needed)

```bash
python scripts/pyvwf_to_grid/generate_turbine_cluster_geometries.py
```

- Creates Voronoi cluster geometries clipped to country boundaries
- Only needed if geometry files are missing from Stage 1
- **Output:** `output/pyvwf_to_grid/cluster_geometries/`

### 2.3 Interpolate to Grid

```bash
python scripts/pyvwf_to_grid/compare_unified_corrections_to_grid.py
```

- Interpolates corrections onto 0.25 deg ERA5 grid using 4 methods:
  Nearest Neighbour, IDW, RBF, Kriging
- Runs spatial cross-validation to compare methods
- Distance masking (>500 km from control points → neutral correction)
- **Requires:** `output/pyvwf_to_grid/all_corrections_centroids.csv`
- **Output:**
  - `output/pyvwf_to_grid/grid_comparison/europe_corrections_{method}.nc`
  - `output/pyvwf_to_grid/grid_comparison/cv_scores.csv`

### 2.4 Visualize Results

```bash
python scripts/pyvwf_to_grid/visualize_corrections_and_interpolations.py
```

- Maps of raw clusters, interpolated surfaces, method comparisons, difference maps
- **Requires:** Steps 2.1 + 2.3 outputs
- **Output:** `output/pyvwf_to_grid/grid_comparison/maps/` (17 maps)

### 2.5 Evaluate Grid Corrections

```bash
python scripts/pyvwf_to_grid/evaluate_grid_corrections.py
```

- Validates gridded corrections by comparing capacity factors at observation locations
- Tests against observed, uncorrected, and cluster-based values
- Handles turbine-level (DK, DE, UK) and country-level (NL, FR, BE) with appropriate aggregation
- **Requires:** Steps 2.1 + 2.3 outputs
- **Output:** `output/pyvwf_to_grid/grid_evaluation/`

### 2.6 Generate Best Correction Grids

```bash
python scripts/pyvwf_to_grid/generate_best_correction_grids.py
```

- Generates optimized correction NetCDF files ready for Atlite/PyPSA-Eur application
- Produces Hybrid Kriging grid (with variance-based masking) and IDW grid
- Uses best variogram configurations per target
- **Requires:** Steps 2.1 + 2.3 outputs
- **Output:** `output/pyvwf_to_grid/` (production-ready `.nc` files)

### 2.7 Generate Chapter 4 Thesis Figures

```bash
python scripts/pyvwf_to_grid/generate_ch4_grid_plots.py
```

- 10 publication-quality figures for thesis Chapter 4 (grid interpolation)
- Control point maps, correction distributions, interpolation surfaces, CV scores, grid vs cluster comparisons
- **Requires:** Steps 2.1 + 2.3 + 1.2 outputs
- **Output:** `output/pyvwf_to_grid/analysis_plots/ch4_grid_interpolation/`

### 2.8 Test Kriging Improvements (Optional)

```bash
python scripts/pyvwf_to_grid/test_kriging_improvements.py
```

- Compares 13 Kriging configurations via 5-fold spatial CV
- Tests variogram models, coordinate systems, Universal vs Ordinary Kriging
- **Output:** `output/pyvwf_to_grid/grid_comparison/kriging_improvement_cv_scores.csv`

---

## Stage 3: PyVWF ML — Machine Learning Corrections

Train ML models to predict correction factors from terrain and environmental features.

### 3.0 Prerequisites: Download Input Data

```bash
# Download and process terrain data (elevation, slope, aspect, roughness)
python scripts/pyvwf_ml/download_terrain_data.py

# Download CORINE land cover (optional, requires Copernicus account)
python scripts/pyvwf_ml/download_corine_data.py
```

- **Output:**
  - `input/terrain/terrain_europe_full.nc`
  - `input/terrain/coastlines.geojson`
  - `input/terrain/corine_europe.nc` (optional)

### 3.1 Enhance Terrain Features

```bash
python scripts/pyvwf_ml/enhance_terrain_features.py \
    --input-nc input/terrain/terrain_europe_full.nc
```

- Adds derived features: distance to coast, sub-grid variance, coastal indicator, terrain complexity, aspect categories
- Uses cKDTree for fast coastline distance computation
- **Requires:** Step 3.0 outputs
- **Output:** `input/terrain/terrain_europe_enhanced.nc`

### 3.2 Prepare Turbine Fleet Features

```bash
python scripts/pyvwf_ml/prepare_turbine_fleet_features.py
```

- Links turbine metadata (hub height, rotor diameter, capacity) to correction clusters
- Spatially assigns turbines to nearest centroid via KDTree
- **Requires:** `output/pyvwf_to_grid/all_corrections_centroids.csv` (from Stage 2.1)
- **Output:** `output/pyvwf_to_grid/fleet_features.csv`

### 3.3 Train ML Models

```bash
# Basic training with random CV
python scripts/pyvwf_ml/train_unified_ml_corrections.py

# With spatial cross-validation (recommended for honest evaluation)
python scripts/pyvwf_ml/train_unified_ml_corrections.py --cv-strategy spatial_lon

# Compare all models
python scripts/pyvwf_ml/train_unified_ml_corrections.py --compare-models

# Feature ablation study
python scripts/pyvwf_ml/train_unified_ml_corrections.py --ablation
```

Supported models: Random Forest, Gradient Boosting, XGBoost, LightGBM, Ridge, Lasso, ElasticNet

CV strategies: `random` (baseline), `spatial_lon` (longitude folds), `leave_country_out`

Feature groups: terrain, ERA5 invariants, turbine fleet, CORINE land cover, spatial

- **Requires:** Steps 2.1 + 3.1 (+ optionally 3.0 CORINE, 3.2 fleet features)
- **Output:** `output/pyvwf_ml/unified_ml/` or `output/pyvwf_ml/ml_results/`

### 3.4 Generate Figures

```bash
# Chapter 2 ML figures (standalone)
python scripts/pyvwf_ml/generate_ch2_ml_plots.py

# Combined grid + ML figures
python scripts/analyse_grid_and_ml_results.py
```

- **Ch2 figures:** ML model comparison, feature importance, predictions scatter, random vs spatial CV, ML vs interpolation
- **Combined figures:** Correction maps, distributions, interpolation surfaces, CV scores + ML results
- **Output:**
  - `output/pyvwf_ml/analysis_plots/ch2_ml_models/`
  - `output/pyvwf_to_grid/analysis_plots/`

---

## Quick Reference: Minimal Pipeline

```bash
# 1. Train corrections
python train_all_bias_corrections.py --sets turbine_grid

# 2. Unify and interpolate to grid
python scripts/pyvwf_to_grid/create_unified_correction_dataframe.py
python scripts/pyvwf_to_grid/compare_unified_corrections_to_grid.py

# 3. (Optional) ML models
python scripts/pyvwf_ml/download_terrain_data.py
python scripts/pyvwf_ml/enhance_terrain_features.py
python scripts/pyvwf_ml/train_unified_ml_corrections.py --cv-strategy spatial_lon
```

## Dependency Graph

```
generate_country_level_training_data.py
          │
          v
train_all_bias_corrections.py ──────────────────────────────────────┐
          │                                                         │
          v                                                         v
create_unified_correction_dataframe.py          evaluate_all_pyvwf_runs.py
          │                                              │
          ├──────────────────────┐                       │
          v                      v                       v
compare_unified_corrections   download_terrain_data   generate_ch4_grid_plots.py
_to_grid.py                        │                       │
          │                        v                       │
          ├──────────   enhance_terrain_features.py        │
          │          │           │                         │
          v          │  prepare_turbine_fleet_features.py  │
evaluate_grid_       │           │                         │
corrections.py       │           v                         v
          │          │  train_unified_ml_corrections   analyse_grid_and
          v          │           │                     _ml_results.py
generate_best_       │           v
correction_grids.py  │  generate_ch2_ml_plots.py
                     │
                     v
          visualize_corrections
          _and_interpolations.py
```
