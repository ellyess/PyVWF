# Within-Country ML Correction Prediction Results

## Summary

Testing whether terrain/climate features can predict correction factors **within a single country** (random train/test split, no spatial holdout).

## Results by Country

| Country | Points | Test R² | CV R² (5-fold) | RMSE | MAE | Scalar Range |
|---------|--------|---------|----------------|------|-----|--------------|
| **Denmark** | 700 | **+0.029** | -0.044 | 0.125 | 0.100 | [0.307, 1.174] |
| Germany | 500 | **-0.022** | -0.085 | 0.210 | 0.165 | [0.346, 3.330] |
| UK | 320 | **-0.071** | -0.064 | 0.322 | 0.261 | [0.251, 2.340] |

## Denmark Feature Ablation (700 points, Ridge, 80/20 split)

| Feature Set | Num Features | Test R² | CV R² | RMSE | MAE |
|-------------|--------------|---------|-------|------|-----|
| **All features** | 18 | **0.029** | -0.044 | 0.125 | 0.100 |
| Terrain + Spatial | 9 | 0.022 | -0.028 | 0.125 | 0.101 |
| Terrain + Geographic | 8 | 0.012 | -0.041 | 0.126 | 0.103 |
| Terrain + ERA5 | 9 | 0.011 | -0.043 | 0.126 | 0.103 |
| **Terrain only** | 6 | 0.009 | -0.032 | 0.126 | 0.103 |

### Features by Category:
- **Terrain (6)**: elevation, slope, aspect, roughness, curvature, distance_to_coast_km
- **Spatial Context (3)**: nearby_mean_correction, nearby_correction_variance, turbine_density
- **ERA5 Mismatch (3)**: era5_elevation_error, era5_roughness, subgrid_terrain_variance
- **Geographic (4)**: latitude, longitude, abs_latitude, coastal_latitude
- **Turbine (2)**: hub_height_m, rotor_diameter_m (only DK has data)

## Key Findings

### 1. Country-Specific Differences
- **Denmark only** shows weak positive test R² (0.029)
- UK and Germany: negative R² even within-country
- CV scores negative for all countries → overfitting, no robust patterns

### 2. Feature Impact (Denmark)
- **Spatial context** helps most (+0.012 R² over terrain only)
- All feature sets improve test R² incrementally
- **BUT**: All CV scores remain negative, suggesting test R² is unstable

### 3. Variance Heterogeneity
- **Denmark**: narrow range (0.87 scalar spread) → easier to predict
- **Germany**: wide range (2.98 scalar spread) → harder to predict
- **UK**: very wide range (2.09 scalar spread) + small sample (320) → hardest

## Cross-Country vs Within-Country Comparison

### Cross-Country Spatial Validation (from previous tests)
| Training | Validation | Test R² | Finding |
|----------|------------|---------|---------|
| DK+UK | DE | -0.034 | Poor transfer |
| DK+DE | UK | -0.193 | Very poor transfer |
| UK+DE | DK | -1.011 | Catastrophic failure |

### Within-Country Random Split
| Country | Test R² | Finding |
|---------|---------|---------|
| DK | +0.029 | Weak positive signal |
| DE | -0.022 | Still fails |
| UK | -0.071 | Still fails |

**Conclusion**: Even removing spatial validation (allowing train/test from same geographic area), only Denmark shows weak positive R². This suggests:
1. Denmark's corrections may have internal structure related to terrain
2. Germany and UK corrections are not predictable from terrain/climate at all
3. CV negativity indicates no robust generalizable patterns

## Statistical Context

### Baseline Comparison
For reference, predicting the mean correction factor:
- R² = 0.0 (by definition)
- Denmark achieves R² = 0.029 → **2.9% better than mean**
- Germany R² = -0.022 → **2.2% worse than mean**
- UK R² = -0.071 → **7.1% worse than mean**

### Cross-Validation Reality Check
All countries show negative CV R²:
- **Denmark**: CV = -0.044 (test = +0.029) → 7.3 percentage points gap
- **Germany**: CV = -0.085 (test = -0.022) → 6.3 percentage points gap  
- **UK**: CV = -0.064 (test = -0.071) → minimal gap

This indicates Denmark's positive test R² may be **lucky split** rather than true predictive power.

## Research Paper Implications

### Main Finding
**ML-based prediction of wind power correction factors is ineffective**, whether:
- Cross-country transfer (all R² negative)
- Within-country prediction (only DK weakly positive with questionable CV)

### Hypotheses for Failure
1. **Methodological differences**: Each country uses different calibration procedures
2. **Model configuration**: Turbine characteristics, measurement uncertainty vary
3. **Local practices**: Site-specific adjustments not captured by terrain
4. **Observation quality**: Different instruments, temporal coverage, quality control
5. **Physical complexity**: Correction factors may depend on atmospheric stability, mesoscale patterns not captured by static terrain

### Recommendation
**Region-specific calibration required**. Cannot reliably predict corrections in new areas from terrain/climate features alone. Each region needs:
- Local observational data
- Country-specific calibration methodology
- Validation against actual wind farm production

## Technical Notes

- Model: Ridge Regression (α=1.0)
- Scaling: StandardScaler on features
- Split: 80/20 random (within-country) or spatial holdout (cross-country)
- Random seed: 42
- Cross-validation: 5-fold stratified on training set
- Missing turbine data: Filled with training set mean (only affects DK)
- ERA5 elevation errors: Up to 2.2km in German Alps, but doesn't help prediction
