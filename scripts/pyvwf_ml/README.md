# PyVWF ML Corrections Module

This directory contains machine learning experiments for predicting ERA5 wind speed bias correction factors from terrain features, using the unified European corrections dataset.

## Overview

The core question: **can terrain features predict where and how much ERA5 wind speeds need correcting?**

We train ML models on 1,474 Voronoi correction regions across 14 country/region configurations (9 countries with country-level observations + 5 turbine-level configurations for DE, DK, UK). The corrections were generated using a unified PyVWF calibration methodology, ensuring consistency across all regions.

### Key Results

| Model | Scalar R² (CV) | Scalar MAE | Offset R² (CV) | Offset MAE |
|-------|----------------|------------|----------------|------------|
| **Random Forest** | **0.351** | **0.137** | **0.414** | **0.541** |
| Gradient Boosting | 0.356 | 0.144 | 0.374 | 0.564 |
| Ridge | 0.098 | 0.173 | 0.158 | 0.684 |
| Lasso | -0.011 | 0.181 | 0.123 | 0.677 |
| Elastic Net | 0.050 | 0.175 | 0.141 | 0.674 |

Random Forest and Gradient Boosting perform similarly and clearly outperform linear models, confirming that correction patterns are non-linear.

### ML vs Spatial Interpolation (at control points)

| Method | Scalar R² | Scalar MAE | Offset R² | Offset MAE |
|--------|-----------|------------|-----------|------------|
| **IDW** | **0.779** | 0.095 | **0.754** | 0.383 |
| ML (Random Forest) | 0.653 | **0.093** | 0.714 | **0.363** |

IDW outperforms ML on R² at the control point locations (where IDW naturally excels), while ML achieves slightly better MAE. The more informative comparison is cross-validation performance, where ML achieves R² = 0.35-0.41 on truly held-out data.

---

## Directory Structure

```
ml/
├── README.md                     # This file
├── scripts/
│   ├── train_unified_ml_corrections.py   # Main ML training script
│   ├── enhance_terrain_features.py       # Add derived terrain features
│   ├── download_terrain_data.py          # Download real ETOPO elevation
│   └── quick_terrain_setup.py            # Generate synthetic terrain
├── docs/
│   └── ML_RESULTS_SUMMARY.md    # Consolidated results and analysis
├── _archive/                     # Previous experiments and old docs
└── results/                      # Experiment outputs
```

### Key Outputs

Results from the current `turbine_grid` run are in:
- `output/grid_run/turbine_grid/ml_results/` -- ML model comparison, plots, training summary
- `output/grid_run/turbine_grid/grid_comparison/` -- Spatial interpolation (IDW, RBF, Kriging) grids and CV scores

---

## Usage

### 1. Train with Random CV (default)

```bash
python ml/scripts/train_unified_ml_corrections.py \
    --output-dir output/grid_run/turbine_grid/ml_results
```

### 2. Compare All Models

```bash
python ml/scripts/train_unified_ml_corrections.py \
    --compare-models \
    --output-dir output/grid_run/turbine_grid/ml_results
```

### 3. Spatial Cross-Validation (Country Holdout)

```bash
# Hold out Germany
python ml/scripts/train_unified_ml_corrections.py \
    --validation-countries DE-onshore \
    --output-dir ml/results/spatial_cv_de

# Hold out UK
python ml/scripts/train_unified_ml_corrections.py \
    --validation-countries UK-onshore,UK-offshore \
    --output-dir ml/results/spatial_cv_uk
```

### 4. Terrain-Only Test (No Spatial Features)

```bash
python ml/scripts/train_unified_ml_corrections.py \
    --exclude-spatial-features \
    --output-dir ml/results/terrain_only
```

---

## Data Pipeline

The ML module sits downstream of the correction pipeline:

1. **PyVWF runs** (`output/runs/turbine_grid/`) -- generate per-cluster correction factors
2. **Unified dataframe** (`scripts/pyvwf_to_grid/create_unified_correction_dataframe.py`) -- combines all corrections with cluster geometry centroids
3. **Spatial interpolation** (`scripts/pyvwf_to_grid/compare_unified_corrections_to_grid.py`) -- IDW/RBF/Kriging grid comparison
4. **ML training** (`ml/scripts/train_unified_ml_corrections.py`) -- trains models on terrain features

### Input Files

| File | Description |
|------|-------------|
| `output/grid_run/turbine_grid/all_corrections_centroids.csv` | 1,474 correction centroids with scalar/offset |
| `input/terrain/terrain_europe_full.nc` | Real ETOPO terrain (elevation, slope, aspect, roughness, curvature) |
| `output/grid_run/turbine_grid/grid_comparison/europe_corrections_idw.nc` | IDW-interpolated correction grid |

---

## Interpretation and Limitations

### What ML captures

With 8 features (5 terrain + 3 spatial), Random Forest explains ~35% of scalar variance and ~41% of offset variance in cross-validation. The spatial features (lon/lat) contribute substantially to this. Tree-based models capture non-linear interactions between terrain characteristics and correction magnitude.

### Transfer learning caveat

Previous experiments (documented in `docs/ML_RESULTS_SUMMARY.md`) showed that spatial cross-validation (holding out entire countries) yields negative R². This means:

- ML works well for **interpolation** within regions where training data exists
- ML **does not transfer** reliably to completely new countries/regions
- Spatial features (lon/lat) are essential for interpolation but prevent generalisation
- Terrain-only models achieve modest R² (0.15-0.24) but still fail in spatial CV

For prediction at new locations without nearby training data, use IDW or Kriging interpolation of existing correction factors instead.

### Practical recommendation

For extending corrections across Europe:
- **Within trained regions**: ML or IDW both work well
- **Between trained regions**: IDW/Kriging spatial interpolation is more reliable
- **Untrained countries**: Use PyVWF physics-based corrections with local observations

---

## Requirements

```bash
pip install scikit-learn>=1.0 xarray pandas numpy matplotlib seaborn
```
