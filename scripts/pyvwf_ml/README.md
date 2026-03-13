# PyVWF ML Corrections Module

This directory contains machine learning experiments for predicting ERA5 wind speed bias correction factors from terrain and environmental features, using the unified European corrections dataset.

## Overview

The core question: **can terrain features predict where and how much ERA5 wind speeds need correcting?**

Two levels of analysis are explored:

1. **Centroid-level**: 1,729 Voronoi correction centroids across 12 European countries (14 country–mode combinations), using 37 spatial/terrain/ERA5 features.
2. **Turbine-level**: 23,009 individual turbine samples from DE, DK, and UK, using 35 terrain/ERA5/turbine features. This enables direct comparison of model types and feature selection strategies.

### Key Results (Turbine-Level, 7 Lasso-Selected Features)

| Model | Scalar R² (CV) | Scalar MAE | Offset R² (CV) | Offset MAE |
|-------|----------------|------------|----------------|------------|
| **Elastic Net** | **0.295** | **0.151** | **0.295** | **0.620** |
| Ridge | 0.293 | 0.151 | 0.293 | 0.621 |
| Lasso | 0.279 | 0.153 | 0.279 | 0.631 |
| Random Forest | 0.273 | 0.151 | 0.254 | 0.631 |
| Gradient Boosting | 0.275 | 0.153 | 0.252 | 0.637 |
| XGBoost | 0.256 | 0.155 | 0.229 | 0.644 |
| LightGBM | 0.258 | 0.155 | 0.233 | 0.644 |
| SVR | 0.205 | 0.162 | 0.202 | 0.659 |

With Lasso feature selection (7 features: `era5_wind_night_mean`, `subgrid_variance`, `era5_weibull_k`, `era5_wind_seasonal_range`, `is_forest`, `aspect_category`, `curvature`), linear models slightly outperform tree-based models, achieving R² ≈ 0.295 for both targets.

### ML vs Spatial Interpolation (at control points)

| Method | Scalar MAE | Offset MAE |
|--------|------------|------------|
| **IDW** | **0.161** | **0.641** |
| ML (Elastic Net, turbine-level) | 0.151 | 0.620 |

ML achieves slightly better MAE than IDW, though these use different evaluation strategies (random CV vs spatial CV), making direct comparison approximate.

---

## Directory Structure

```
scripts/pyvwf_ml/
├── README.md                             # This file
├── ML_RESULTS_SUMMARY.md                 # Consolidated results and analysis
├── download_terrain_data.py              # Download ETOPO elevation data
├── download_corine_data.py               # Download CORINE Land Cover 2018
├── enhance_terrain_features.py           # Add derived terrain features
├── prepare_turbine_fleet_features.py     # Link turbine metadata to clusters
├── train_unified_ml_corrections.py       # Train centroid-level ML corrections
├── run_turbine_model_comparisons.py      # Run turbine-level model comparison experiments
└── generate_ch5_ml_plots.py              # Thesis Chapter 5 figures
```

### Key Outputs

- `output/pyvwf_ml/unified_ml/` — Centroid-level ML model results
- `output/pyvwf_ml/turbine_35feat_default/` — Turbine-level, 35 features, default hyperparameters
- `output/pyvwf_ml/turbine_35feat_tuned/` — Turbine-level, 35 features, tuned hyperparameters
- `output/pyvwf_ml/turbine_7feat_tuned/` — Turbine-level, 7 Lasso-selected features, tuned hyperparameters

---

## Usage

### 1. Train Centroid-Level Models

```bash
python scripts/pyvwf_ml/train_unified_ml_corrections.py \
    --compare-models \
    --output-dir output/pyvwf_ml/unified_ml
```

### 2. Run Turbine-Level Model Comparisons

```bash
python scripts/pyvwf_ml/run_turbine_model_comparisons.py
```

Runs three experiments: 35-feature default, 35-feature tuned, and 7-feature tuned (Lasso selection).

### 3. Generate Chapter 5 Figures

```bash
python scripts/pyvwf_ml/generate_ch5_ml_plots.py
```

---

## Data Pipeline

The ML module sits downstream of the correction pipeline:

1. **PyVWF runs** (`output/runs/turbine_grid/`) — generate per-cluster correction factors
2. **Unified dataframe** (`scripts/pyvwf_to_grid/create_unified_correction_dataframe.py`) — combines all corrections with cluster geometry centroids
3. **Spatial interpolation** (`scripts/pyvwf_to_grid/compare_unified_corrections_to_grid.py`) — IDW/RBF/Kriging grid comparison
4. **Centroid-level ML** (`scripts/pyvwf_ml/train_unified_ml_corrections.py`) — trains models on centroid features
5. **Turbine-level ML** (`scripts/pyvwf_ml/run_turbine_model_comparisons.py`) — trains models on individual turbine features

---

## Interpretation and Limitations

### What ML captures

At the turbine level with 7 Lasso-selected features, Elastic Net explains ~30% of variance in both scalar and offset targets. The selected features are primarily ERA5-derived wind climatology variables and terrain characteristics, suggesting corrections correlate with systematic local climate patterns.

### Transfer learning caveat

Spatial cross-validation (holding out entire countries) yields negative R², meaning ML **does not transfer** to entirely new geographic regions. Corrections have country-specific components (turbine fleet, climate regime, observation quality) not captured by terrain features alone.

For new regions, use the physics-based PyVWF pipeline with local observations, then interpolate the resulting corrections spatially using IDW or Kriging.
