# Spatial Cross-Validation Results - Critical Findings

**Date**: February 10, 2025
**Status**: ⚠️ **MAJOR ISSUE DISCOVERED**
**Conclusion**: ML model does NOT generalize to new countries

---

## Executive Summary

Spatial cross-validation (holding out entire countries) reveals that the ML model **fails catastrophically** when predicting corrections for geographic regions not in the training set:

| Holdout Country | Scalar R² | Offset R² | Finding |
|----------------|-----------|-----------|---------|
| **Germany** | **-1.05** | **-0.90** | Catastrophic failure |
| **UK** | **-1.91** | **-0.59** | Even worse |

**vs Random CV** (previous results):
- Scalar R²: +0.37 (appeared successful)
- Offset R²: +0.47 (appeared successful)

**Critical Insight**: The positive R² in random CV was **misleading**. The model learned spatial patterns (lon/lat coordinates), not terrain-correction relationships.

---

## Spatial Cross-Validation Results

### Germany Holdout (DE-onshore)

**Training Set**:
- Countries: NL, FR, BE, NO, DK-onshore, DK-offshore, UK-onshore, UK-offshore
- Samples: 1,180 regions

**Validation Set** (held out):
- Country: DE-onshore
- Samples: 498 regions

**Results**:
- **Scalar R²**: -1.05 (predictions worse than predicting mean!)
- **Scalar MAE**: 0.247 (vs 0.121 in random CV - doubled!)
- **Offset R²**: -0.90
- **Offset MAE**: 0.612 (vs 0.402 in random CV - 52% worse!)

**Interpretation**: Model completely fails to predict German corrections when trained on other countries.

### UK Holdout (UK-onshore + UK-offshore)

**Training Set**:
- Countries: NL, FR, BE, NO, DE-onshore, DK-onshore, DK-offshore
- Samples: 1,398 regions

**Validation Set** (held out):
- Countries: UK-onshore, UK-offshore
- Samples: 280 regions

**Results**:
- **Scalar R²**: -1.91 (even worse than Germany!)
- **Scalar MAE**: 0.455
- **Offset R²**: -0.59
- **Offset MAE**: 0.975

**Interpretation**: Model catastrophically fails on UK. Predictions are worse than simply guessing the training set mean.

---

## Why Does This Happen?

### Root Cause: Spatial Features Dominate

From previous feature importance analysis (random CV):

| Feature | Importance | Type | Generalizes? |
|---------|------------|------|--------------|
| **lon_normalized** | 30.5% | Spatial | ❌ NO - country-specific |
| **distance_to_coast** | 23.5% | Terrain | ✅ Maybe |
| **lat_normalized** | 13.4% | Spatial | ❌ NO - country-specific |
| **abs_lat** | 10.2% | Climate | ⚠️ Partially |
| **subgrid_variance** | 4.3% | Terrain | ✅ Yes |
| **aspect** | 4.0% | Terrain | ✅ Yes |
| **elevation** | 3.9% | Terrain | ✅ Yes |
| **roughness** | 3.7% | Terrain | ✅ Yes |

**Problem**: **~54% of importance from spatial features** (lon_normalized + lat_normalized + abs_lat)

These spatial features effectively encode **"which country is this?"** rather than physical terrain-correction relationships.

### What the Model Actually Learned

**Random CV (appeared to work)**:
- Model learns: "Points at lon=7°, lat=51° tend to have scalar ~1.2"
- This works in random CV because German points are split between train/test
- Test set contains German points at similar lon/lat → predictions work

**Spatial CV (reveals truth)**:
- Model learns: "Points at lon=7°, lat=51° tend to have scalar ~1.2"
- But: ALL German points are in validation set
- Model has never seen lon=7-15°, lat=47-55° (Germany's range)
- Extrapolation fails catastrophically

**Analogy**: The model memorized "France is at lon=2°, UK at lon=-2°, Germany at lon=10°" instead of learning "mountainous terrain increases corrections by X%".

---

## Why Random CV Was Misleading

### Random 5-Fold CV Approach
```
All 1,678 points randomly shuffled and split:
Fold 1: [DE-point-1, UK-point-7, NL-point-3, FR-point-12, ...]  ← Train
        [DE-point-5, UK-point-2, DK-point-8, ...]              ← Test
```

**Result**: Every fold contains points from ALL countries at ALL lon/lat values.

**Why it worked**: Spatial interpolation, not terrain learning:
- Test point in Germany at (lon=10.5, lat=50.2)
- Training set has nearby German points at (lon=10.3, lat=50.0), (lon=10.7, lat=50.5)
- Model uses lon/lat to interpolate → appears successful

### Spatial CV Approach (True Test)
```
Countries split:
Train: [ALL NL points, ALL FR points, ALL UK points, ...]  ← 8 countries
Test:  [ALL DE points]                                      ← 1 country (completely held out)
```

**Result**: Test country's lon/lat range never seen in training.

**Why it failed**: No spatial memorization possible:
- Test point in Germany at (lon=10.5, lat=50.2)
- Training set has NO points in lon=7-15° range (Germany's extent)
- Model tries to extrapolate using lon/lat → catastrophic failure

---

## Comparison to Previous Country-Level Attempts

Recall that previous ML attempts with country-specific corrections also failed with **negative R²** scores:

| Training | Testing | R² | Status |
|----------|---------|-----|--------|
| DK+UK | DE | -0.03 | Failed |
| DK+DE | UK | -0.19 | Failed |
| UK+DE | DK | -1.01 | Catastrophic |

**Then**: We attributed failure to "inconsistent calibration methodology"

**Now**: With unified corrections, spatial CV STILL fails (R² = -1.05, -1.91)

**Revised Understanding**: The problem wasn't just methodology - it's that **spatial features block terrain learning**:
1. Spatial features (lon/lat) always correlate with country identity
2. Model learns country-specific patterns (masked by spatial coordinates)
3. When predicting new country, model fails even with unified methodology

---

## What Does This Mean?

### 1. Random CV Results Are NOT Evidence of Terrain Learning ❌

Previous conclusion: "ML works because unified methodology reveals terrain-correction relationships"

**Reality**: ML appears to work because spatial features enable **memorization of country locations**, not learning of terrain physics.

**Evidence**:
- Random CV R² = +0.37 ✓ (but misleading!)
- Spatial CV R² = -1.05 to -1.91 ✗ (reveals truth)

### 2. ML Cannot Currently Transfer to New Regions ❌

**Implication**: Cannot use trained model to predict corrections for countries/regions without training data.

**Use cases that FAIL**:
- ✗ Train on (DK, DE, UK) → predict Spain
- ✗ Train on European data → predict North America
- ✗ Train on onshore → predict new offshore region

### 3. ML Is NOT Better Than IDW for Interpolation ⚠️

Previous claim: "ML beats IDW by 20% on MAE"

**Reconsideration**: When comparing ML to IDW:
- **Random CV**: Both methods see all geographic regions → both use spatial smoothing
- **ML advantage**: Mostly from spatial features (lon/lat), not terrain
- **True test**: Spatial CV shows ML fails where IDW would also struggle (extrapolation)

**Revised conclusion**: ML and IDW are both spatial interpolation methods. ML doesn't learn terrain physics better than IDW.

### 4. Unified Methodology Alone Is Insufficient ⚠️

**Previous belief**: Consistent calibration methodology enables ML to learn terrain relationships

**Reality**: Even with unified methodology, spatial features dominate (54% importance) and block terrain learning.

**Missing ingredient**: Need to remove or control for spatial patterns to force model to learn terrain.

---

## How to Fix This

### Option 1: Remove Spatial Features ⭐ (Recommended First Try)

**Remove**:
- `lon_normalized` (30.5% importance)
- `lat_normalized` (13.4% importance)
- `abs_lat` (10.2% importance) - maybe keep as climate proxy?

**Keep only physical terrain features**:
- elevation, slope, aspect, roughness, curvature
- distance_to_coast, is_coastal, subgrid_variance, complexity

**Expected outcome**:
- Random CV R² will DROP (maybe to 0.15-0.25)
- But spatial CV R² might become POSITIVE (model forced to learn terrain)
- True test of whether terrain is predictive

**Command**:
```bash
# Modify train_unified_ml_corrections.py to exclude spatial features
python ml/train_unified_ml_corrections.py \
    --terrain-nc input/terrain/terrain_north_sea_enhanced.nc \
    --validation-countries DE-onshore \
    --output-dir ml/spatial_cv_de_no_spatial \
    --exclude-spatial-features
```

### Option 2: Add Country Fixed Effects

**Approach**: Add country as categorical feature:
```python
# One-hot encode country
features_df = pd.get_dummies(features_df, columns=['country_code'], prefix='country')
```

**Effect**:
- Model learns country-specific baseline + terrain adjustments
- Within each country, terrain features explain deviations from country mean
- But: still can't predict NEW countries (country_XX feature unseen)

**Use case**: Interpolate within countries, not transfer to new countries.

### Option 3: Hierarchical/Mixed-Effects Model

**Approach**: Two-stage model:
1. **Stage 1**: Predict country-level mean corrections (from climate, geography, observation network)
2. **Stage 2**: Predict within-country deviations from terrain (elevation, roughness, etc.)

**Advantage**: Separates country-level effects from terrain effects.

**Challenge**: Requires careful modeling of variance components.

### Option 4: Use Real Physical Terrain Data

**Current limitation**: Synthetic terrain may not capture true terrain-correction relationships.

**Action**: Download real ETOPO/SRTM elevation, real land cover:
```bash
python ml/download_terrain_data.py  # Get real terrain
```

**Expected impact**:
- If terrain is truly predictive: R² improves with real data
- If spatial memorization: No improvement (confirms diagnosis)

**Recommendation**: Try this AFTER removing spatial features.

---

## Revised Recommendations

### Immediate Actions (High Priority)

1. **Retrain WITHOUT spatial features** ⭐
   - Remove lon_normalized, lat_normalized, abs_lat
   - Test both random CV and spatial CV (DE, UK holdouts)
   - Compare results to understand terrain-only predictive power

2. **Document revised findings**
   - Update ML_COMPLETE_SUMMARY.md with spatial CV failure
   - Clarify that previous "success" was spatial memorization
   - Explain limitations for transfer learning

3. **Download real terrain data**
   - Get actual ETOPO elevation, real coastlines
   - Retrain with real terrain + no spatial features
   - Final test of terrain predictiveness

### Research Implications

**Previous claim**: "ML successfully predicts corrections from terrain - unified methodology was the breakthrough"

**Revised claim**: "ML with spatial features memorizes country locations - true terrain predictiveness remains unproven"

**Key question**: Can ML predict corrections from PURE terrain features (no spatial coordinates)?
- If YES → terrain truly drives corrections, transfer learning viable
- If NO → corrections are country-specific, interpolation-only approach needed

### Publication Strategy

**Not publishable yet**:
- Spatial CV failure is critical flaw
- Cannot claim "terrain learning" when using spatial coordinates
- Need to demonstrate positive spatial CV results

**Path to publication**:
1. Remove spatial features
2. Show positive spatial CV R² with terrain-only features
3. Use real terrain data for final results
4. Then: "ML predicts corrections from terrain physics across Europe"

---

## Summary Statistics

| Validation Approach | Scalar R² | Offset R² | Interpretation |
|-------------------|-----------|-----------|----------------|
| **Random 5-Fold CV** | +0.37 ± 0.03 | +0.47 ± 0.03 | Misleading - spatial memorization |
| **Spatial CV (Germany)** | **-1.05** | **-0.90** | True test - catastrophic failure |
| **Spatial CV (UK)** | **-1.91** | **-0.59** | Even worse failure |

**Conclusion**: Current ML approach does NOT learn generalizable terrain-correction relationships.

**Action Required**: Remove spatial features and retrain to test true terrain predictiveness.

---

## Files Generated

- `ml/spatial_cv_de/training_summary.txt` - Germany holdout results
- `ml/spatial_cv_uk/training_summary.txt` - UK holdout results
- `ml/spatial_cv_de/plots/` - Diagnostic plots (6 files)
- `ml/spatial_cv_uk/plots/` - Diagnostic plots (6 files)

---

**Status**: ⚠️ Spatial cross-validation reveals ML failure
**Action**: Remove spatial features and retrain
**Next**: Test whether pure terrain features are predictive
**Date**: February 10, 2025
