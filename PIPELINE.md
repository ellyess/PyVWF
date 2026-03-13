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

### 2.4 Evaluate Grid Corrections

```bash
python scripts/pyvwf_to_grid/evaluate_grid_corrections.py
```

- Validates gridded corrections by comparing capacity factors at observation locations
- Tests against observed, uncorrected, and cluster-based values
- Handles turbine-level (DK, DE, UK) and country-level (NL, FR, BE) with appropriate aggregation
- **Requires:** Steps 2.1 + 2.3 outputs
- **Output:** `output/pyvwf_to_grid/grid_evaluation/`

### 2.5 Generate Best Correction Grids

```bash
python scripts/pyvwf_to_grid/generate_best_correction_grids.py
```

- Generates optimized correction NetCDF files ready for Atlite/PyPSA-Eur application
- Produces Hybrid Kriging grid (with variance-based masking) and IDW grid
- Uses best variogram configurations per target
- **Requires:** Steps 2.1 + 2.3 outputs
- **Output:** `output/pyvwf_to_grid/` (production-ready `.nc` files)

### 2.6 Generate Chapter 4 Thesis Figures

```bash
python scripts/pyvwf_to_grid/generate_ch4_grid_plots.py
```

- 10 publication-quality figures for thesis Chapter 4 (grid interpolation)
- Control point maps, correction distributions, interpolation surfaces, CV scores, grid vs cluster comparisons
- **Requires:** Steps 2.1 + 2.3 + 1.2 outputs
- **Output:** `output/pyvwf_to_grid/analysis_plots/ch4_grid_interpolation/`

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

### 3.3 Train Centroid-Level ML Models

```bash
# Compare all models with random CV
python scripts/pyvwf_ml/train_unified_ml_corrections.py --compare-models

# With spatial cross-validation
python scripts/pyvwf_ml/train_unified_ml_corrections.py --cv-strategy spatial_lon
```

Supported models: Random Forest, Gradient Boosting, XGBoost, LightGBM, Ridge, Lasso, ElasticNet, SVR

- **Requires:** Steps 2.1 + 3.1 (+ optionally 3.0 CORINE, 3.2 fleet features)
- **Output:** `output/pyvwf_ml/unified_ml/`

### 3.4 Run Turbine-Level Model Comparisons

```bash
python scripts/pyvwf_ml/run_turbine_model_comparisons.py
```

- Runs 3 experiments: 35-feature default, 35-feature tuned, 7-feature Lasso-selected tuned
- 8 models compared per experiment (RF, GBM, XGBoost, LightGBM, Ridge, Lasso, EN, SVR)
- **Requires:** Turbine-level correction data + enhanced terrain features
- **Output:** `output/pyvwf_ml/turbine_35feat_default/`, `turbine_35feat_tuned/`, `turbine_7feat_tuned/`

### 3.5 Generate Chapter 5 Figures

```bash
python scripts/pyvwf_ml/generate_ch5_ml_plots.py
```

- ML model comparison, feature importance, predictions scatter, random vs spatial CV, ML vs interpolation
- **Output:** `output/pyvwf_ml/analysis_plots/ch5_ml_models/`

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

# 3b. (Optional) Turbine-level model comparisons
python scripts/pyvwf_ml/run_turbine_model_comparisons.py
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
          │          │  train_unified_ml_corrections   generate_ch3_plots.py
          v          │           │
generate_best_       │           ├── run_turbine_model_comparisons.py
correction_grids.py  │           │
                     │           v
                     │  generate_ch5_ml_plots.py
```
