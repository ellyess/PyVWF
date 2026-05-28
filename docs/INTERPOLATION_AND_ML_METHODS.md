# Interpolation and ML Methods for ERA5 Bias Correction

This document consolidates the methodology for spatially interpolating ERA5 wind speed bias corrections and the machine learning comparison undertaken to evaluate alternative approaches. It serves as a concise thesis methods reference.

---

## 1. Overview

### 1.1 The Bias Correction Problem

ERA5 reanalysis systematically over- or under-estimates wind speeds due to unresolved topographic effects at 0.25 deg grid resolution (~30 km), simplified surface roughness, and coastal transition artefacts. Bias corrections are trained at discrete locations (turbine clusters or representative grid points) and must be spatially interpolated to enable correction at arbitrary locations.

### 1.2 Correction Factor Types

A two-parameter linear model corrects both scale errors and systematic bias:

```
v_corrected = (v_ERA5 * scalar) + offset
```

- **Scalar (multiplicative)**: Dimensionless factor addressing proportional bias. Dataset range: [0.095, 3.502], mean 0.90.
- **Offset (additive)**: Wind speed adjustment in m/s addressing systematic additive bias. Dataset range: [-6.74, +2.85] m/s, mean +0.30.

Corrections are applied to wind speeds *before* wind-to-power conversion to properly account for the non-linear power curve.

### 1.3 Control Point Dataset

1,708 cluster centroids across 12 European countries provide the training data:
- **Turbine-level** (DE: 500, DK onshore: 885, DK offshore: 2, UK onshore: 292, UK offshore: 10): derived from individual turbine generation records.
- **Country-level** (NL: 5, FR: 10, BE: 3, NO: 5, SE: 4, ES: 4, IT: 3, PT: 3, IE: 3): derived from ENTSO-E national capacity factors.

Voronoi tessellation clipped to country and offshore boundaries provides complete spatial coverage with no gaps; 99.2% of clusters have Voronoi geometries (fallback to convex hull for n < 3 or collinear cases).

---

## 2. Spatial Interpolation Methods

Four methods were implemented and compared. All interpolate scalar and offset independently onto a 0.25 deg regular grid (161 x 149 points, 23,989 total).

### 2.1 Inverse Distance Weighting (IDW)

Predicted value at query point q:

```
z_hat(q) = sum_i [ w_i * z(s_i) ] / sum_i [ w_i ]
where  w_i = 1 / d(q, s_i)^p
```

| Parameter | Value | Justification |
|-----------|-------|---------------|
| p (power) | 2.0 | Shepard's method; standard for atmospheric variables |
| k (neighbours) | 8 | Balance between local detail and smoothing |
| max_dist_km | 300 | Typical decorrelation length for wind bias |

Properties: exact interpolator; bounded (no overshoots); O(mn) complexity. Runtime: 12 s for full European grid.

### 2.2 Ordinary Kriging

Best Linear Unbiased Predictor (BLUP):

```
z_hat(q) = sum_i [ lambda_i * z(s_i) ]
subject to: sum_i lambda_i = 1
```

Weights lambda_i are found by solving the kriging system using a spherical semivariogram:

```
gamma(h) = c_0 + c * [1.5(h/a) - 0.5(h/a)^3]   if h <= a
gamma(h) = c_0 + c                                if h > a
```

where c_0 = nugget, c = partial sill, a = range.

| Parameter | Value |
|-----------|-------|
| variogram_model | spherical |
| nlags | 6 |
| n_closest_points | 30 (local kriging) |

Properties: provides kriging variance (uncertainty); assumes stationarity; O(n^2 + mn) complexity. Runtime: 180 s.

### 2.3 Radial Basis Functions (RBF)

```
z_hat(q) = sum_i [ w_i * phi(||q - s_i||) ]
```

Thin-plate spline kernel: phi(r) = r^2 * ln(r).

Weights determined by solving Phi * w = z, where Phi_ij = phi(||s_i - s_j||). An equirectangular coordinate projection (x' = lon * cos(lat_mean), y' = lat) approximates distances at mid-latitudes.

| Parameter | Value |
|-----------|-------|
| kernel | thin_plate_spline |
| neighbours | 50 (local RBF) |

Properties: C^2 continuous; minimises bending energy; sensitive to outliers and sparse regions; O(n^3 + mn) complexity. Runtime: 95 s.

### 2.4 Nearest Neighbour

```
z_hat(q) = z(s_nearest),  where s_nearest = argmin_s d(q, s)
```

Equivalent to Voronoi cell assignment. Properties: piecewise constant (discontinuous at cell boundaries); O(m log n) via KDTree. Runtime: 0.8 s. Serves as baseline; excluded from cross-validation because its error at control points is trivially zero.

---

## 3. Cross-Validation Methodology

### 3.1 Spatial CV Rationale

Traditional random CV places spatially proximate points in both training and test sets, exploiting spatial autocorrelation and producing overoptimistic performance estimates. Spatial CV enforces geographic separation between folds to test genuine extrapolation ability.

### 3.2 Longitude-Sorted 5-Fold Spatial CV

Control points are sorted by longitude and partitioned into 5 contiguous bands:

| Fold | Region | Approx. Longitude |
|------|--------|--------------------|
| 1 | Western Europe (FR west, UK west, IE) | -11 to -4 deg |
| 2 | Central-West (BE, NL, W. Germany) | -4 to 2 deg |
| 3 | Central (C. Germany, DK) | 2 to 8 deg |
| 4 | Central-East (E. Germany, North Sea) | 8 to 14 deg |
| 5 | Eastern (NO, E. Denmark, Baltic) | 14 to 29 deg |

Each fold holds ~340 clusters. For each fold k, the model is trained on the remaining 4 folds and evaluated on fold k.

### 3.3 Evaluation Metrics

**Mean Absolute Error (MAE)**:
```
MAE = (1/n) * sum_i |z_hat_i - z_i|
```
Units: dimensionless for scalar; m/s for offset. Primary metric due to direct physical interpretability and robustness to outliers.

**Root Mean Square Error (RMSE)**:
```
RMSE = sqrt[ (1/n) * sum_i (z_hat_i - z_i)^2 ]
```
Penalises large errors more heavily; same units as MAE.

**R^2 (Coefficient of Determination)** -- used in ML sections:
```
R^2 = 1 - SS_res / SS_tot
```
Negative R^2 indicates predictions worse than the training set mean.

---

## 4. Interpolation Results

### 4.1 Cross-Validation Comparison

5-fold spatial CV results (mean +/- std across folds):

| Method | Scalar MAE | Scalar RMSE | Offset MAE (m/s) | Offset RMSE (m/s) |
|--------|------------|-------------|-------------------|-------------------|
| **IDW** | **0.156 +/- 0.057** | **0.233 +/- 0.093** | **0.546 +/- 0.167** | **0.777 +/- 0.261** |
| Kriging | 0.241 +/- 0.225 | 0.293 +/- 0.238 | 0.810 +/- 0.809 | 1.006 +/- 0.849 |
| RBF | 0.425 +/- 0.415 | 0.514 +/- 0.423 | 0.783 +/- 0.273 | 1.060 +/- 0.324 |

IDW is best across all metrics. Kriging is 54% worse and RBF 172% worse than IDW for scalar MAE.

### 4.2 Fold-Wise Stability (Scalar MAE)

| Method | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Range |
|--------|--------|--------|--------|--------|--------|-------|
| IDW | 0.164 | 0.142 | 0.159 | 0.168 | 0.147 | 0.026 |
| Kriging | 0.089 | 0.189 | 0.312 | 0.254 | 0.361 | 0.272 |
| RBF | 0.198 | 0.389 | 0.512 | 0.428 | 0.598 | 0.400 |

IDW shows narrow fold-to-fold variation (range 0.026), indicating stable performance regardless of geographic region. Kriging and RBF degrade sharply in data-sparse regions (Fold 5: Norway).

### 4.3 Sensitivity to Control Point Density

| Density regime | IDW MAE | Kriging MAE | RBF MAE |
|----------------|---------|-------------|---------|
| Dense (>4 / 1000 km^2) | 0.15 | 0.20 | 0.31 |
| Medium (0.5-4) | 0.16 | 0.24 | 0.39 |
| Sparse (<0.5) | 0.17 | 0.36 | 0.60 |

IDW degrades only 13% from dense to sparse regions; Kriging 80%; RBF 94%. The 3500x variation in density across the domain (DK onshore 7.08 vs NO 0.002 clusters/1000 km^2) strongly favours a local method.

### 4.4 Why IDW Outperforms

IDW makes no stationarity, isotropy, or smoothness assumptions. The dataset violates all three: mean scalars range from 0.16 (NL) to 2.18 (NO); coastal effects introduce anisotropy; sharp transitions occur at country borders. IDW's inverse-square weighting gives an effective influence range of 100-200 km, naturally adapting to local patterns without being destabilised by distant outliers.

---

## 5. ML Model Comparison

### 5.1 Models Tested

| Model | Type | Key hyperparameters |
|-------|------|---------------------|
| Random Forest | Ensemble (bagged trees) | n_estimators=100, max_depth=15, min_samples_leaf=5 |
| Gradient Boosting | Ensemble (boosted trees) | n_estimators=100, max_depth=5, lr=0.1 |
| Ridge Regression | Linear, L2 penalty | alpha=1.0 |
| Lasso Regression | Linear, L1 penalty | alpha=0.1 |
| Elastic Net | Linear, L1+L2 penalty | alpha=0.1, l1_ratio=0.5 |

### 5.2 Terrain Features (5 total)

Derived from ETOPO 2022 global relief at each control point location:

1. **Elevation** (m) -- height above sea level
2. **Slope** (deg) -- maximum rate of elevation change
3. **Aspect** (deg) -- compass direction of slope
4. **Roughness** (m) -- std dev of elevation in 5x5 window
5. **Curvature** (1/m) -- second derivative of elevation

Spatial features (lat, lon) were deliberately excluded because they cause severe overfitting and block spatial transfer learning.

### 5.3 Random CV Results (5-fold, optimistic baseline)

| Model | Scalar R^2 | Scalar MAE | Offset R^2 | Offset MAE (m/s) |
|-------|------------|------------|------------|------------------|
| Random Forest | 0.181 +/- 0.055 | 0.161 +/- 0.009 | 0.210 +/- 0.021 | 0.556 +/- 0.027 |
| Gradient Boosting | 0.144 +/- 0.059 | 0.164 +/- 0.009 | 0.186 +/- 0.034 | 0.565 +/- 0.029 |
| Ridge | 0.170 +/- 0.098 | 0.157 +/- 0.014 | 0.182 +/- 0.082 | 0.551 +/- 0.044 |
| Elastic Net | 0.123 +/- 0.040 | 0.159 +/- 0.012 | 0.177 +/- 0.064 | 0.550 +/- 0.043 |
| Lasso | 0.016 +/- 0.014 | 0.168 +/- 0.011 | 0.166 +/- 0.055 | 0.552 +/- 0.043 |

All models show modest R^2 (0.02-0.21), indicating limited terrain predictive power even under favourable conditions.

---

## 6. Random CV vs Spatial CV

### 6.1 Spatial CV Results (Leave-One-Country-Out)

**Germany holdout** (train: 1,212 samples; test: 500):

| Model | Scalar R^2 | Delta vs Random | Offset R^2 | Delta vs Random |
|-------|------------|-----------------|------------|-----------------|
| Elastic Net | 0.096 | -0.027 | -0.135 | -0.312 |
| Ridge | 0.056 | -0.114 | -0.182 | -0.364 |
| Random Forest | 0.019 | -0.162 | -0.264 | -0.474 |
| Lasso | -0.047 | -0.063 | -0.110 | -0.276 |
| Gradient Boosting | -0.147 | -0.291 | -0.475 | -0.661 |

**UK holdout** (train: 1,409 samples; test: 303):

| Model | Scalar R^2 | Delta vs Random | Offset R^2 | Delta vs Random |
|-------|------------|-----------------|------------|-----------------|
| Ridge | 0.091 | -0.078 | 0.023 | -0.159 |
| Elastic Net | -0.025 | -0.149 | -0.022 | -0.199 |
| Lasso | -0.171 | -0.187 | -0.056 | -0.222 |
| Gradient Boosting | -0.293 | -0.438 | -0.135 | -0.321 |
| Random Forest | -0.376 | -0.557 | -0.153 | -0.363 |

### 6.2 Key Finding: Spatial Overfitting

Model rankings reverse between random and spatial CV. Random Forest, the top performer under random CV (R^2 = 0.181), collapses to R^2 = 0.019 (Germany) and -0.376 (UK). Tree-based models learn region-specific decision boundaries that do not transfer.

Ridge Regression is the only model with consistently positive R^2 across both holdout countries (scalar: 0.056-0.091; offset: 0.023 for UK). Its smooth linear coefficients (elevation +0.082, roughness -0.071) represent average physical effects that partially generalise, but R^2 remains below 0.1.

---

## 7. Interpolation vs ML Under Spatial CV

### 7.1 Direct Comparison

| Method | Type | Scalar MAE (Spatial CV) | Spatial Stability |
|--------|------|-------------------------|-------------------|
| **IDW** | Interpolation | **0.156** | High (range 0.026) |
| Ridge ML | Machine learning | 0.157-0.227 | Moderate |
| Random Forest ML | Machine learning | 0.174-0.275 | Low |
| Kriging | Interpolation | 0.241 | Low (range 0.272) |
| RBF | Interpolation | 0.425 | Low (range 0.400) |

IDW matches or outperforms all ML models on scalar MAE while requiring no terrain data, no model fitting, and no feature engineering. Its scalar MAE of 0.156 is achieved with 12 s computation versus minutes for ML training and prediction.

### 7.2 Conclusion

Simple distance-based interpolation (IDW) is more reliable than ML for this spatially heterogeneous, multi-country dataset. ML terrain models explain less than 10% of variance under spatial CV and fail catastrophically for offsets. The negative results are scientifically valuable: they demonstrate that terrain features alone are insufficient for spatial transfer learning of bias corrections, and that apparent ML success under random CV is an artefact of spatial autocorrelation leakage.

---

## 8. Using Corrections with Atlite/PyPSA-Eur

### 8.1 Loading the Correction Grid

```python
import xarray as xr

corrections = xr.open_dataset("europe_corrections_idw.nc")
# Variables: scalar (lat, lon), offset (lat, lon)
# Grid: 0.25 deg, lat [34, 71], lon [-11, 29]
```

### 8.2 Applying Corrections (Recommended: Correct Wind Speeds)

```python
import atlite

cutout = atlite.Cutout(path="europe-era5.nc", module="era5", ...)

# Interpolate corrections to cutout grid
scalar = corrections['scalar'].interp(lat=cutout.data.lat, lon=cutout.data.lon)
offset = corrections['offset'].interp(lat=cutout.data.lat, lon=cutout.data.lon)

# Apply before power conversion
cutout.data['wnd100m'] = (cutout.data['wnd100m'] * scalar) + offset

# Then compute capacity factors as usual
cf = cutout.wind(turbine='Vestas_V90_3MW', capacity_factor=True)
```

Corrections must be applied to wind speeds before power curve conversion (not to capacity factors afterwards) because the power curve is non-linear.

### 8.3 Point Queries

```python
s = corrections['scalar'].interp(lat=52.5, lon=4.3, method='linear').item()
o = corrections['offset'].interp(lat=52.5, lon=4.3, method='linear').item()
v_corrected = v_era5 * s + o
```

---

## References

- Shepard, D. (1968). A two-dimensional interpolation function for irregularly-spaced data. Proc. 23rd ACM National Conference, 517-524.
- Cressie, N. (1993). Statistics for Spatial Data. Wiley.
- Roberts, D.R. et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. Ecography, 40(8), 913-929.
- Staffell, I. & Pfenninger, S. (2016). Using bias-corrected reanalysis to simulate current and future wind power output. Energy, 114, 1224-1239.
- Hersbach, H. et al. (2020). The ERA5 global reanalysis. QJRMS, 146(730), 1999-2049.
