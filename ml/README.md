# PyVWF ML Corrections Module

**Status**: ⚠️ **Real Terrain Tested - Transfer Learning Still Fails**
**Last Updated**: February 10, 2025

This directory contains machine learning experiments for predicting ERA5 bias correction factors from terrain features using the unified European corrections dataset.

---

## Quick Summary: What We Learned

### The Good News ✅
- **Real terrain significantly improves predictions**: Germany spatial CV improved 78% (R² from -0.46 to -0.10)
- **Random CV gains 45%**: Within-region predictions improved from R² = 0.17 to 0.24 with real terrain
- **Germany nearly viable**: Scalar R² = -0.10 is close to break-even for transfer learning
- **Distance to coastline matters**: 2nd most important feature (23.5%), revealing ERA5 struggles with coastal transitions

### The Critical Issue ⚠️
- **Transfer learning STILL FAILS**: Even with real ETOPO terrain, spatial CV remains negative (Germany: -0.10, UK: -0.57)
- **UK got worse with real terrain**: R² deteriorated from -0.45 to -0.57, suggesting regional idiosyncrasies
- **Country-specific effects dominate**: Corrections have components not explained by terrain alone
- **Synthetic terrain was A limiting factor, not THE limiting factor**: Real terrain helped but didn't solve transfer problem

### Current Recommendation
**Not ready for publication** - Spatial CV still negative despite real terrain. Next steps: Add climate features, turbine metadata, or use hierarchical modeling to account for country-specific effects.

---

## Directory Structure

### Core Scripts
- **`train_unified_ml_corrections.py`** ⭐ - Main ML training script with spatial/random CV options
- **`enhance_terrain_features.py`** - Add derived features (distance to coast, complexity, etc.)
- **`download_terrain_data.py`** - Download real ETOPO elevation and coastline data
- **`quick_terrain_setup.py`** - Generate synthetic terrain for testing

### Documentation
- **`REAL_TERRAIN_RESULTS.md`** ⭐ - **READ THIS FIRST** - Real ETOPO terrain results and comparison to synthetic
- **`COMPLETE_SPATIAL_CV_COMPARISON.md`** - Comprehensive comparison of all 6 synthetic terrain configurations
- **`SPATIAL_CV_FAILURE_ANALYSIS.md`** - Detailed analysis of why spatial CV failed
- **`ML_COMPLETE_SUMMARY.md`** - Original summary (before spatial CV testing)
- **`UNIFIED_ML_RESULTS.md`** - Initial results with random CV only
- **`ENHANCED_FEATURES_COMPARISON.md`** - Baseline vs enhanced features

### Results Directories

**Spatial CV Results with REAL ETOPO Terrain** ⭐:
- `spatial_cv_de_real_terrain/` - Germany holdout, terrain only, REAL terrain (R² = **-0.10**) ⚠️ Much better!
- `spatial_cv_uk_real_terrain/` - UK holdout, terrain only, REAL terrain (R² = **-0.57**) ✗ Got worse
- `random_cv_real_terrain/` - Random CV, terrain only, REAL terrain (R² = **+0.24**) ✅ Improved

**Spatial CV Results with Synthetic Terrain** (for comparison):
- `spatial_cv_de/` - Germany holdout with spatial features (R² = -1.05) ✗
- `spatial_cv_uk/` - UK holdout with spatial features (R² = -1.91) ✗
- `spatial_cv_de_terrain_only/` - Germany holdout, terrain only (R² = -0.46) ⚠️
- `spatial_cv_uk_terrain_only/` - UK holdout, terrain only (R² = -0.45) ⚠️

**Random CV Results**:
- `unified_ml/` - Baseline: 8 features (terrain + spatial)
- `unified_ml_enhanced/` - Enhanced: 13 features (terrain + spatial + derived)
- `terrain_only_random_cv/` - Terrain-only: 10 features, no spatial (R² = 0.17-0.24)

### Archive
- `_archive_old_attempts/` - Previous failed ML attempts with country-specific corrections

---

## Key Results Comparison

| Configuration | Features | Validation | Scalar R² | Offset R² | Conclusion |
|--------------|----------|------------|-----------|-----------|------------|
| **A. Original** | Terrain + Spatial | Random CV | **+0.37** | **+0.47** | Misleading success |
| **B. Germany holdout** | Terrain + Spatial | Spatial CV | **-1.05** | **-0.90** | Catastrophic |
| **C. UK holdout** | Terrain + Spatial | Spatial CV | **-1.91** | **-0.59** | Even worse |
| **D. Germany (terrain-only)** | Terrain only | Spatial CV | **-0.46** | **-0.57** | 56% better! |
| **E. UK (terrain-only)** | Terrain only | Spatial CV | **-0.45** | **-0.50** | 77% better! |
| **F. Terrain-only CV** | Terrain only | Random CV | **+0.17** | **+0.24** | True terrain power |

### Key Insights
1. **Spatial features drove apparent success** (contributed 55% of R² in config A)
2. **Spatial features BLOCK transfer** (configs B→D and C→E show massive improvement when removed)
3. **True terrain predictive power is modest** (R² = 0.17-0.24 in config F)
4. **Transfer still fails** even without spatial features (configs D & E still negative)

---

## Usage

### 1. Train with Random CV (Baseline)
```bash
python train_unified_ml_corrections.py \
    --model-type random_forest \
    --terrain-nc input/terrain/terrain_north_sea_enhanced.nc \
    --output-dir ml/test_run
```

### 2. Test Spatial CV (Country Holdout)
```bash
# Hold out Germany
python train_unified_ml_corrections.py \
    --model-type random_forest \
    --terrain-nc input/terrain/terrain_north_sea_enhanced.nc \
    --validation-countries DE-onshore \
    --output-dir ml/spatial_cv_germany

# Hold out UK
python train_unified_ml_corrections.py \
    --validation-countries UK-onshore,UK-offshore \
    --output-dir ml/spatial_cv_uk
```

### 3. Terrain-Only Test (No Spatial Features)
```bash
python train_unified_ml_corrections.py \
    --model-type random_forest \
    --terrain-nc input/terrain/terrain_north_sea_enhanced.nc \
    --exclude-spatial-features \
    --validation-countries DE-onshore \
    --output-dir ml/spatial_cv_de_terrain_only
```

### 4. Download Real ETOPO Terrain
```bash
# Download and process real elevation data
python download_terrain_data.py --region north_sea --output-dir ../input/terrain

# Add derived features
python enhance_terrain_features.py

# Retrain with real terrain
python train_unified_ml_corrections.py \
    --terrain-nc input/terrain/terrain_north_sea_full.nc \
    --validation-countries DE-onshore \
    --exclude-spatial-features \
    --output-dir ml/spatial_cv_real_terrain
```

---

## Why Did Previous Attempts Fail?

See `_archive_old_attempts/` for previous experiments with country-specific corrections:

**Previous Results** (Country-Specific Calibrations):
- Transfer learning R²: **-0.03 to -1.01** (all negative!)
- Within-country R²: **+0.03 at best** (barely positive)
- Reason: Different calibration methodologies between countries masked terrain relationships

**Current Results** (Unified Calibrations):
- Random CV R²: **+0.37 to +0.47** (appears successful)
- Spatial CV R²: **-1.05 to -1.91** (catastrophic with spatial features)
- Spatial CV R²: **-0.45 to -0.57** (still negative, but much better without spatial features)
- Terrain-only CV R²: **+0.17 to +0.24** (honest assessment of terrain power)

**Breakthrough**: Unified methodology was necessary but not sufficient - spatial features were hiding the real problem.

---

## Next Steps

### Critical Test (IN PROGRESS)
**Download real ETOPO terrain data** (currently using synthetic terrain):
- Real elevation, coastlines, roughness expected to improve R² by +5-10%
- If spatial CV becomes positive → Transfer learning viable → Publishable
- If spatial CV remains negative → Transfer fundamentally limited → Need hierarchical modeling

### If Real Terrain Succeeds
1. Generate European grid predictions for atlite
2. Write publication documenting breakthrough
3. Implement in PyVWF production workflow

### If Real Terrain Fails
1. Document country-specific effects in corrections
2. Implement hierarchical/mixed-effects model (country baseline + terrain adjustments)
3. Accept that transfer learning has fundamental limitations
4. Use interpolation methods (IDW/Kriging) for practical applications

---

## Feature Importance (from Terrain-Only Model)

When spatial features are excluded, terrain features show:

| Rank | Feature | Importance | Physical Meaning |
|------|---------|------------|------------------|
| 1 | **distance_to_coast** | 35-40% | Coastal roughness transitions |
| 2 | **elevation** | ~15% | Orographic forcing |
| 3 | **roughness** | ~12% | Surface drag |
| 4 | **subgrid_variance** | ~10% | Local complexity |
| 5 | **slope, aspect, curvature** | 5-8% each | Terrain shape |
| 6 | **complexity, is_coastal** | 3-5% | Combined metrics |

**Key Finding**: Distance to coastline becomes dominant when spatial coordinates removed, revealing ERA5's struggle with land-sea transitions.

---

## File Size Warning

**Real terrain download**: ~1.2GB ETOPO global file + processing
**Synthetic terrain**: ~5MB (fast, good for testing)
**Results directories**: ~2-10MB each (diagnostic plots at 300 DPI)

---

## Requirements

```bash
# Core ML packages
pip install scikit-learn>=1.0 xarray pandas numpy matplotlib seaborn

# For real terrain processing
pip install geopandas rasterio

# For coastline features
pip install shapely
```

---

## References

- ETOPO 2022 Global Relief Model: https://www.ncei.noaa.gov/products/etopo-global-relief-model
- Natural Earth Coastlines: https://www.naturalearthdata.com/
- PyVWF: https://github.com/yourusername/PyVWF

---

## Contact

For questions about the ML module or to report issues:
- Open an issue in the PyVWF repository
- See main project documentation

---

**Status Summary**:
- ✅ Unified corrections dataset created (1,712 regions)
- ✅ ML training infrastructure implemented
- ✅ Enhanced terrain features added (distance to coast, etc.)
- ✅ Spatial cross-validation testing completed (synthetic terrain)
- ✅ **Real ETOPO terrain downloaded and tested**
- ⚠️ **Significant improvement but transfer still fails** (Germany R² = -0.10, UK R² = -0.57)
- ⚠️ **Real terrain not sufficient alone** - need climate features or hierarchical modeling
- ❌ **Not ready for publication yet**

**Recommendation**: Add climate features (Köppen zones, ERA5 climatology) and turbine metadata, OR accept that transfer learning is limited and use IDW for spatial interpolation.
