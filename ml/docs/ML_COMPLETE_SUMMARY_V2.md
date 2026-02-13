# ML Model Comparison Results - Updated Unified Corrections (V2)

**Date**: February 11, 2026
**Dataset**: 1,729 correction centroids from 14 country/region configurations
**Experiments**: 10 completed (model comparison, enhanced features, terrain-only, spatial CV)

---

## Executive Summary

This document presents updated ML model comparison results using the expanded unified corrections dataset (1,729 samples, up from previous experiments). Key findings:

1. **Random CV Performance**: Random Forest achieves R² 0.33 (scalar) and 0.42 (offset) when interpolating within the dataset
2. **Spatial Features Dominate**: Spatial coordinates (lon/lat) account for ~55% of explained variance
3. **Transfer Learning Failure**: All spatial cross-validation experiments show negative R² when transferring to new countries
4. **Critical Insight**: Spatial features essential for random CV but cause severe overfitting in spatial transfer learning

**Recommendation**: Use Random Forest for spatial interpolation within known regions, but DO NOT rely on transfer learning to new countries without local correction data.

---

## Dataset Overview

### Unified Corrections Dataset
- **Location**: `output/unified_corrections/all_corrections_centroids.csv`
- **Total Samples**: 1,729 correction centroids
- **Quality**: No missing values, no infinite values
- **Date Generated**: February 11, 2026

### Country Distribution

| Country/Region | Samples | Percentage |
|---------------|---------|------------|
| DK-onshore | 884 | 51.1% |
| DE-onshore | 500 | 28.9% |
| UK-onshore | 293 | 16.9% |
| FR | 10 | 0.6% |
| UK-offshore | 10 | 0.6% |
| NL, NO, ES, SE | 4-5 each | <1% each |
| BE, IT, PT, IE | 3 each | <1% each |
| DK-offshore | 2 | <1% |

**Key Observation**: Denmark onshore dominates the dataset (51%), which has implications for spatial CV experiments.

### Feature Sets

**Baseline (8 features)**:
- Terrain: elevation, slope, aspect, roughness, curvature (5 features)
- Spatial: abs_lat, lon_normalized, lat_normalized (3 features)

**Enhanced (13 features)**:
- Baseline + distance_to_coast, is_coastal, subgrid_variance, complexity, aspect_category (8 terrain + 5 enhanced spatial)

**Terrain-only (5 features)**:
- Only terrain features, no spatial coordinates

---

## Phase 1: Random Cross-Validation Results

### Experiment 1: Model Comparison (5 Models)

**Objective**: Compare 5 different ML models on the same dataset to identify best performer.

**Models Tested**:
1. Random Forest (ensemble, non-linear)
2. Gradient Boosting (ensemble, sequential)
3. Ridge Regression (linear, L2 regularization)
4. Lasso Regression (linear, L1 regularization)
5. Elastic Net (linear, L1+L2 regularization)

**Results**:

| Model | Scalar R² | Scalar MAE | Offset R² | Offset MAE |
|-------|-----------|------------|-----------|------------|
| **Random Forest** | **0.330** | **0.136** | **0.420** | **0.434** |
| Gradient Boosting | 0.270 | 0.141 | 0.391 | 0.453 |
| Ridge | 0.143 | 0.162 | 0.185 | 0.544 |
| Lasso | 0.013 | 0.174 | 0.165 | 0.553 |
| Elastic Net | 0.107 | 0.165 | 0.180 | 0.546 |

**Findings**:
- ✅ Random Forest is the clear winner (18% better R² than Gradient Boosting for scalar)
- Random Forest performs better for both scalar (multiplicative) and offset (additive) corrections
- Linear models (Ridge, Lasso, Elastic Net) perform poorly (R² < 0.15 for scalar)
- This confirms correction patterns are highly non-linear

**Output Directory**: `ml/model_comparison_random_cv_v2/`

---

### Experiment 2: Enhanced Features (13 Features)

**Objective**: Test if adding more terrain features (distance_to_coast, etc.) improves performance.

**Features Added**:
- `distance_to_coast` - Distance to nearest coastline
- `is_coastal` - Binary flag for coastal locations
- `subgrid_variance` - Elevation variance within grid cell
- `complexity` - Terrain complexity index
- `aspect_category` - Categorical aspect (N, S, E, W)

**Results**:

| Metric | Enhanced (13 features) | Baseline (8 features) | Difference |
|--------|------------------------|----------------------|------------|
| Scalar R² | 0.327 ± 0.081 | 0.330 ± 0.083 | -0.003 |
| Offset R² | 0.414 ± 0.066 | 0.420 ± 0.063 | -0.006 |
| Scalar MAE | 0.136 ± 0.010 | 0.136 ± 0.010 | 0.000 |
| Offset MAE | 0.435 ± 0.029 | 0.434 ± 0.028 | +0.001 |

**Findings**:
- ❌ Enhanced features provide NO improvement over baseline
- Performance is essentially identical (differences within noise)
- The 5 baseline terrain features + 3 spatial features are sufficient
- Adding 5 more terrain features doesn't capture additional variance

**Interpretation**: Wind bias corrections are primarily driven by:
1. Geographic location (spatial features)
2. Basic terrain characteristics (elevation, slope, roughness)
3. Advanced terrain metrics (coastline distance, complexity) add no value

**Output Directory**: `ml/unified_ml_enhanced_v2/`

---

### Experiment 3: Terrain-Only (No Spatial Features)

**Objective**: Quantify the importance of spatial features (lon/lat) by testing terrain-only model.

**Features Used**: Only 5 terrain features (elevation, slope, aspect, roughness, curvature)

**Results**:

| Metric | Terrain-Only | With Spatial (Baseline) | Difference |
|--------|-------------|-------------------------|------------|
| Scalar R² | 0.157 ± 0.098 | 0.330 ± 0.083 | **-0.173** |
| Offset R² | 0.186 ± 0.061 | 0.420 ± 0.063 | **-0.234** |
| Scalar MAE | 0.167 ± 0.010 | 0.136 ± 0.010 | +0.031 |
| Offset MAE | 0.561 ± 0.028 | 0.434 ± 0.028 | +0.127 |

**Findings**:
- ✅ **Spatial features account for ~55% of explained variance**
  - Scalar: 0.330 - 0.157 = 0.173 R² (52% of baseline)
  - Offset: 0.420 - 0.186 = 0.234 R² (56% of baseline)
- Terrain-only R² still positive (0.16-0.19), so terrain does provide some signal
- MAE increases by 23% (scalar) and 29% (offset) without spatial features

**Critical Insight**: Geographic location (lon/lat) is the single most important predictor of wind bias corrections. This makes sense because:
- Different regions have different wind regimes
- Coastal vs inland locations have distinct bias patterns
- Latitude affects solar heating and atmospheric stability

**Output Directory**: `ml/terrain_only_random_cv_v3/`

---

## Phase 2: Spatial Cross-Validation Results (Transfer Learning)

### Overview

Spatial cross-validation tests whether models trained on data from some countries can accurately predict corrections for completely held-out countries. This simulates the real-world scenario of applying ML corrections to a country with no local validation data.

**Method**: Leave-one-country-out cross-validation
- **Training**: All countries except the held-out country
- **Testing**: Only the held-out country
- **Goal**: Positive R² indicates successful transfer learning

---

### Experiment 5: Germany Holdout (With Spatial Features)

**Setup**:
- Training: 1,229 samples from 13 regions (all except DE-onshore)
- Testing: 500 samples from DE-onshore
- Features: 8 (5 terrain + 3 spatial)

**Results**:

| Metric | Germany Holdout | Expected (Random CV) |
|--------|----------------|----------------------|
| Scalar R² | **-0.209** | 0.330 |
| Offset R² | **-0.132** | 0.420 |
| Scalar MAE | 0.201 | 0.136 |
| Offset MAE | 0.491 | 0.434 |

**Interpretation**:
- ❌ **Negative R² = Worse than predicting the mean**
- Model predictions are actively harmful compared to naive mean prediction
- Transfer learning from other countries to Germany FAILS completely
- Spatial features learned from other countries don't generalize

**Output Directory**: `ml/spatial_cv_de_v3/`

---

### Experiment 6: Germany Holdout (Terrain-Only)

**Setup**:
- Training: 1,229 samples from 13 regions
- Testing: 500 samples from DE-onshore
- Features: 5 (terrain only, NO spatial)

**Results**:

| Metric | Terrain-Only | With Spatial | Improvement |
|--------|-------------|-------------|------------|
| Scalar R² | **+0.005** | -0.209 | **+0.214** |
| Offset R² | -0.253 | -0.132 | -0.121 |
| Scalar MAE | 0.176 | 0.201 | -0.025 |

**Key Finding**:
- ✅ **Terrain-only achieves slightly POSITIVE R² for scalar (0.005)**
- This is a massive improvement over spatial features (-0.209)
- Removing spatial features reduces overfitting
- For offset, terrain-only is worse (-0.253 vs -0.132)

**Interpretation**:
- Spatial features (lon/lat) cause severe overfitting in transfer learning
- The model memorizes geographic patterns that don't generalize
- Terrain features provide weak but generalizable signal

**Output Directory**: `ml/spatial_cv_de_terrain_only_v3/`

---

### Experiment 7: UK Holdout (With Spatial Features)

**Setup**:
- Training: 1,436 samples from 13 regions (all except UK-onshore)
- Testing: 293 samples from UK-onshore
- Features: 8 (5 terrain + 3 spatial)

**Results**:

| Metric | UK Holdout | Expected (Random CV) |
|--------|-----------|----------------------|
| Scalar R² | **-0.101** | 0.330 |
| Offset R² | **-0.415** | 0.420 |
| Scalar MAE | 0.256 | 0.136 |
| Offset MAE | 0.942 | 0.434 |

**Findings**:
- ❌ Negative R² for both scalar and offset
- Offset R² particularly bad (-0.415)
- MAE nearly doubles for scalar, more than doubles for offset
- Transfer learning to UK also FAILS

**Output Directory**: `ml/spatial_cv_uk_v3/`

---

### Experiment 8: UK Holdout (Terrain-Only)

**Setup**:
- Training: 1,436 samples from 13 regions
- Testing: 293 samples from UK-onshore
- Features: 5 (terrain only)

**Results**:

| Metric | Terrain-Only | With Spatial | Difference |
|--------|-------------|-------------|------------|
| Scalar R² | -0.209 | **-0.101** | -0.108 (worse) |
| Offset R² | **-0.031** | -0.415 | **+0.384** (better) |
| Scalar MAE | 0.264 | 0.256 | +0.008 |
| Offset MAE | 0.831 | 0.942 | -0.111 |

**Key Finding**:
- ⚠️ **Mixed results** - different from Germany:
  - Scalar: Spatial features better (-0.101 vs -0.209)
  - Offset: Terrain-only MUCH better (-0.031 vs -0.415)
- UK behaves differently than Germany in transfer learning

**Interpretation**: The relative benefit of spatial vs terrain-only features varies by target country, suggesting country-specific bias patterns.

**Output Directory**: `ml/spatial_cv_uk_terrain_only_v2/`

---

### Experiment 9: Denmark Holdout (With Spatial Features) **[NEW]**

**Setup**:
- Training: 845 samples from 13 regions (all except DK-onshore)
- Testing: 884 samples from DK-onshore (51% of total dataset!)
- Features: 8 (5 terrain + 3 spatial)

**Special Note**: This is a challenging test because:
1. Denmark is the largest single region (884 samples)
2. Training set is smaller than test set (845 vs 884)
3. Tests if small diverse training set can predict large homogeneous test set

**Results**:

| Metric | Denmark Holdout | Expected (Random CV) |
|--------|----------------|----------------------|
| Scalar R² | **-1.884** | 0.330 |
| Offset R² | **-2.386** | 0.420 |
| Scalar MAE | 0.207 | 0.136 |
| Offset MAE | 0.932 | 0.434 |

**Findings**:
- ❌ **CATASTROPHIC FAILURE** - R² worse than -2!
- Much worse than Germany (-0.21) or UK (-0.10)
- Model predictions are extremely harmful
- Small training set + spatial features = severe overfitting

**Why So Bad?**:
1. Denmark represents 51% of dataset, leaving only 845 training samples
2. Model has less data to learn generalizable patterns
3. Spatial features memorize small training set
4. Denmark turbines have unique characteristics not captured elsewhere

**Output Directory**: `ml/spatial_cv_dk_v3/`

---

### Experiment 10: Denmark Holdout (Terrain-Only) **[NEW]**

**Setup**:
- Training: 845 samples from 13 regions
- Testing: 884 samples from DK-onshore
- Features: 5 (terrain only)

**Results**:

| Metric | Terrain-Only | With Spatial | Improvement |
|--------|-------------|-------------|------------|
| Scalar R² | **-0.121** | -1.884 | **+1.763** |
| Offset R² | -0.430 | -2.386 | **+1.956** |
| Scalar MAE | 0.125 | 0.207 | -0.082 |
| Offset MAE | 0.573 | 0.932 | -0.359 |

**Key Finding**:
- ✅ **MASSIVE IMPROVEMENT** from removing spatial features:
  - Scalar R² improves by +1.76 (from -1.88 to -0.12)
  - Offset R² improves by +1.96 (from -2.39 to -0.43)
  - This is a 93% reduction in overfitting!
- Terrain-only R² still negative, but now comparable to Germany/UK
- MAE dramatically reduced (40% for scalar, 38% for offset)

**Critical Insight**: When training set is small relative to test set, spatial features cause EXTREME overfitting. Removing them is essential for any hope of generalization.

**Output Directory**: `ml/spatial_cv_dk_terrain_only/`

---

## Summary of Transfer Learning Results

### All Spatial CV Experiments

| Country Holdout | With Spatial (Scalar R²) | Terrain-Only (Scalar R²) | Improvement |
|----------------|-------------------------|-------------------------|-------------|
| Germany | -0.209 | **+0.005** | +0.214 |
| UK | **-0.101** | -0.209 | -0.108 |
| Denmark | -1.884 | **-0.121** | +1.763 |

| Country Holdout | With Spatial (Offset R²) | Terrain-Only (Offset R²) | Improvement |
|----------------|-------------------------|-------------------------|-------------|
| Germany | **-0.132** | -0.253 | -0.121 |
| UK | -0.415 | **-0.031** | +0.384 |
| Denmark | -2.386 | **-0.430** | +1.956 |

**Overall Patterns**:

1. **All spatial CV experiments show negative R²** - transfer learning fundamentally fails
2. **Terrain-only generally better** - especially for Denmark (dramatic) and Germany/UK offset
3. **Country-specific patterns** - UK scalar is exception where spatial helps
4. **Training set size matters** - Denmark (smallest training set) shows worst overfitting

---

## Critical Insights and Recommendations

### 1. Spatial Features: Double-Edged Sword

**Random CV (Interpolation)**:
- ✅ Spatial features essential - account for 55% of variance
- ✅ Achieve R² 0.33 scalar, 0.42 offset
- ✅ Use Random Forest with spatial features

**Spatial CV (Extrapolation)**:
- ❌ Spatial features cause severe overfitting
- ❌ Negative R² in all countries
- ❌ Improvement of 0.21-1.76 R² when removed

### 2. Why Transfer Learning Fails

**Geographic Specificity**:
- Wind bias patterns are location-specific
- Coastal effects, latitude, local climate all matter
- Models memorize training country locations

**Non-Transferable Patterns**:
- Netherlands turbines in flat low-lying terrain
- Norway turbines in mountainous high-elevation terrain
- Germany turbines in mixed terrain
- Model learns "Netherlands = low scalar", but this doesn't help predict Germany

**Spatial Feature Trap**:
- Model learns "lon=5, lat=52 → scalar=0.2"
- This pattern doesn't generalize to lon=10, lat=54 (Germany)
- Removing spatial forces model to use terrain, which is more transferable

### 3. Terrain Features Alone Are Insufficient

Even terrain-only models show:
- Germany scalar: R² +0.005 (barely positive)
- UK offset: R² -0.031 (slightly negative)
- Denmark: R² -0.12 to -0.43 (clearly negative)

**Conclusion**: Terrain features provide weak generalizable signal, but not enough for practical use.

### 4. Recommendations for Thesis

**For Spatial Interpolation (Within Known Regions)**:
- ✅ **Use Random Forest with spatial features**
- ✅ R² 0.33-0.42 is reasonable for this problem
- ✅ Works well when interpolating between known turbines in same geographic region

**For Transfer Learning (New Countries)**:
- ❌ **DO NOT use ML models trained on other countries**
- ❌ Even terrain-only models show negative/near-zero R²
- ❌ Predictions will be worse than naive mean

**For New Countries Without Local Data**:
1. Collect local validation data (even small sample)
2. Use PyVWF physics-based corrections
3. Consider ensemble approaches
4. Accept that transferability is limited

### 5. Model Selection

| Model | Random CV (Scalar) | Random CV (Offset) | Recommendation |
|-------|-------------------|-------------------|----------------|
| Random Forest | **0.330** | **0.420** | ✅ Best overall |
| Gradient Boosting | 0.270 | 0.391 | Good alternative |
| Ridge | 0.143 | 0.185 | ❌ Too simple |
| Lasso | 0.013 | 0.165 | ❌ Too simple |
| Elastic Net | 0.107 | 0.180 | ❌ Too simple |

**Conclusion**: Random Forest is definitively the best model for this problem.

---

## Comparison to Previous Results

### Dataset Growth

| Metric | Previous | Current (V2) | Change |
|--------|----------|-------------|---------|
| Total samples | ~900-1200 | 1,729 | +44-92% |
| Denmark samples | ~500-600 | 884 | +47-77% |
| Germany samples | ~450 | 500 | +11% |
| UK samples | ~250 | 293 | +17% |

### Random CV Performance

| Model | Previous Scalar R² | Current Scalar R² | Change |
|-------|--------------------|-------------------|---------|
| Random Forest | 0.35-0.37 | 0.330 | -0.02 to -0.04 |
| Terrain-only | 0.17-0.24 | 0.157 | -0.00 to -0.08 |

**Observations**:
- Performance slightly decreased with more data (likely due to increased diversity)
- More countries → more geographic variation → harder to predict
- Still within expected range

### Spatial CV Performance

Previous spatial CV experiments showed similar negative R² patterns:
- Germany: R² -1.05 (previous) vs -0.21 (current) - **IMPROVED**
- UK: R² -1.91 (previous) vs -0.10 (current) - **IMPROVED**

**Note**: Improvement may be due to:
1. More training data from diverse countries
2. Better data quality in unified corrections
3. Expanded geographic coverage reduces extreme extrapolation

---

## Artifacts and Output Directories

### Phase 1: Random CV
- `ml/model_comparison_random_cv_v2/` - 5-model comparison
- `ml/unified_ml_enhanced_v2/` - Enhanced 13-feature model
- `ml/terrain_only_random_cv_v3/` - Terrain-only (no spatial)

### Phase 2: Spatial CV
- `ml/spatial_cv_de_v3/` - Germany holdout + spatial
- `ml/spatial_cv_de_terrain_only_v3/` - Germany holdout terrain-only
- `ml/spatial_cv_uk_v3/` - UK holdout + spatial
- `ml/spatial_cv_uk_terrain_only_v2/` - UK holdout terrain-only
- `ml/spatial_cv_dk_v3/` - **NEW** Denmark holdout + spatial
- `ml/spatial_cv_dk_terrain_only/` - **NEW** Denmark holdout terrain-only

### Summary Files
- `ml/ml_comparison_results_v2.csv` - All results in tabular format
- `ml/ML_COMPLETE_SUMMARY_V2.md` - This document

---

## Thesis Presentation Guidance

### Key Messages

1. **ML works well for spatial interpolation (R² 0.33-0.42)** when predicting corrections at new turbine locations within the same geographic region.

2. **Transfer learning to new countries fails completely** - all spatial CV experiments show negative R² due to geographic specificity of wind bias patterns.

3. **Spatial features are a double-edged sword** - essential for interpolation but cause severe overfitting in extrapolation.

4. **Recommendation**: Use ML for dense turbine regions to interpolate between known points, but collect local validation data for new countries.

### Figures to Include

1. **Model Comparison Bar Chart**: Show Random Forest beats all other models
2. **Feature Importance**: Show spatial features dominate (55% of variance)
3. **Spatial CV Results**: Show all negative R² for transfer learning
4. **Terrain-Only vs Spatial**: Scatter plot showing overfitting reduction

### Talking Points

- ML is not a silver bullet - works for interpolation, fails for extrapolation
- Geographic location is the strongest predictor, but this doesn't transfer
- Terrain features alone are insufficient (R² < 0.2)
- Physics-based PyVWF corrections remain essential for new countries

---

## Future Work

### Potential Improvements

1. **Climate-Normalized Features**:
   - Add long-term climate variables (mean wind speed, seasonality)
   - These may be more transferable than raw coordinates

2. **Hierarchical Models**:
   - Separate country-level and within-country effects
   - May improve transfer learning

3. **Quantile Mapping**:
   - Instead of predicting scalar/offset, predict full distribution
   - More robust to outliers

4. **Ensemble Approaches**:
   - Combine ML (for interpolation) with physics-based PyVWF (for extrapolation)
   - Adaptive weighting based on confidence

### Data Collection Priorities

1. **More countries**: Expand beyond Europe (US, Asia, etc.)
2. **Longer time series**: Test temporal stability of corrections
3. **Real ETOPO terrain**: Fix current implementation issues
4. **Offshore wind**: Currently underrepresented (12 samples total)

---

## Conclusions

This comprehensive ML model comparison on the expanded unified corrections dataset (1,729 samples) confirms and extends previous findings:

✅ **Spatial interpolation works**: Random Forest achieves R² 0.33-0.42 for predicting corrections within known regions

✅ **Geographic location dominates**: Spatial features account for 55% of explained variance

❌ **Transfer learning fails**: All country holdout experiments show negative R² due to geographic specificity

❌ **Terrain alone insufficient**: Terrain-only models achieve R² < 0.2, not practical for deployment

**Practical Recommendation**: Use Random Forest ML corrections for dense turbine regions where spatial interpolation is valid, but rely on physics-based PyVWF corrections for new countries without local validation data. The key insight is that wind bias corrections are fundamentally location-specific and do not transfer well across geographic boundaries.

---

**End of Document**
