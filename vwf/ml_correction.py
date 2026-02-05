"""
Machine learning-based bias correction using terrain features.

This module implements ML models to learn the relationship between
terrain features and bias correction factors (scalar/offset).

Useful for:
- Transferring corrections to regions without observations
- Understanding physical drivers of model bias
- Improving spatial interpolation of corrections
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Any
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Optional ML models
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    _HAS_SKLEARN_MODELS = True
except ImportError:
    _HAS_SKLEARN_MODELS = False

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False


# =============================================================================
# TERRAIN FEATURE EXTRACTION
# =============================================================================

def extract_terrain_features_from_netcdf(
    nc_path: Path | str,
    *,
    lon: np.ndarray | None = None,
    lat: np.ndarray | None = None,
    variables: list[str] | None = None,
) -> pd.DataFrame:
    """Extract terrain features from a NetCDF file.

    Args:
        nc_path: Path to NetCDF file containing terrain data.
        lon: Longitude points to sample. If None, extract full grid.
        lat: Latitude points to sample. If None, extract full grid.
        variables: Variables to extract. If None, extract all.

    Returns:
        DataFrame with terrain features at each location.
    """
    ds = xr.open_dataset(nc_path)
    
    # Standardize coordinate names
    if 'longitude' in ds.coords:
        ds = ds.rename({'longitude': 'lon', 'latitude': 'lat'})
    if 'x' in ds.coords and 'y' in ds.coords:
        ds = ds.rename({'x': 'lon', 'y': 'lat'})
    
    if variables is not None:
        ds = ds[variables]
    
    if lon is not None and lat is not None:
        # Sample at specific points
        if len(lon) != len(lat):
            raise ValueError("lon and lat must have the same length")
        
        # Create point dataset
        points = []
        for i, (ln, lt) in enumerate(zip(lon, lat)):
            try:
                point = ds.sel(lon=ln, lat=lt, method='nearest')
                feat = {var: float(point[var].values) for var in ds.data_vars}
                feat['lon'] = ln
                feat['lat'] = lt
                points.append(feat)
            except Exception as e:
                warnings.warn(f"Failed to extract features at ({ln}, {lt}): {e}")
                continue
        
        return pd.DataFrame(points)
    else:
        # Return full grid
        data = {}
        for var in ds.data_vars:
            data[var] = ds[var].values.flatten()
        
        LON, LAT = np.meshgrid(ds.lon.values, ds.lat.values)
        data['lon'] = LON.flatten()
        data['lat'] = LAT.flatten()
        
        return pd.DataFrame(data)


def compute_terrain_derivatives(
    elevation: xr.DataArray,
    *,
    resolution_deg: float = 0.25,
) -> xr.Dataset:
    """Compute terrain derivatives from elevation data.

    Args:
        elevation: Elevation data with lat/lon coordinates.
        resolution_deg: Grid resolution in degrees for gradient calculation.

    Returns:
        Dataset with slope, aspect, roughness, and curvature fields.
    """
    # Convert degrees to meters (approximate at mid-latitude)
    lat_mid = float(elevation.lat.mean())
    m_per_deg_lat = 111000.0
    m_per_deg_lon = 111000.0 * np.cos(np.radians(lat_mid))
    
    dx = resolution_deg * m_per_deg_lon
    dy = resolution_deg * m_per_deg_lat
    
    # Gradients
    grad_x = elevation.differentiate('lon') / dx  # m/m
    grad_y = elevation.differentiate('lat') / dy
    
    # Slope (degrees)
    slope_rad = np.arctan(np.sqrt(grad_x**2 + grad_y**2))
    slope_deg = np.rad2deg(slope_rad)
    
    # Aspect (degrees from north)
    aspect = np.rad2deg(np.arctan2(grad_x, grad_y)) % 360
    
    # Terrain roughness (std of elevation in neighborhood)
    # Simple approximation: rolling std
    roughness = elevation.rolling(lat=3, lon=3, center=True).std()
    
    # Curvature (2nd derivatives)
    curv_x = grad_x.differentiate('lon') / dx
    curv_y = grad_y.differentiate('lat') / dy
    curvature = curv_x + curv_y
    
    return xr.Dataset({
        'elevation': elevation,
        'slope': slope_deg,
        'aspect': aspect,
        'roughness': roughness,
        'curvature': curvature,
        'grad_x': grad_x,
        'grad_y': grad_y,
    })


def add_distance_to_coast(
    df: pd.DataFrame,
    *,
    coastline_geojson: Path | str,
    lon_col: str = 'lon',
    lat_col: str = 'lat',
) -> pd.DataFrame:
    """Add a distance-to-coast feature to a DataFrame.

    Args:
        df: Points with lon/lat columns.
        coastline_geojson: GeoJSON with coastline geometry.
        lon_col: Longitude column name.
        lat_col: Latitude column name.

    Returns:
        DataFrame with a ``distance_to_coast_km`` column added.
    """
    coastline = gpd.read_file(coastline_geojson)
    if coastline.crs is None:
        coastline = coastline.set_crs('EPSG:4326')
    else:
        coastline = coastline.to_crs('EPSG:4326')
    
    # Union coastline geometries
    coast_geom = coastline.geometry.union_all() if hasattr(coastline.geometry, 'union_all') else coastline.geometry.unary_union
    
    # Create point GeoDataFrame
    points = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs='EPSG:4326'
    )
    
    # Calculate distance in degrees, then convert to km
    # (rough approximation: 1 degree ≈ 111 km at equator)
    points['distance_to_coast_km'] = points.geometry.distance(coast_geom) * 111.0
    
    df['distance_to_coast_km'] = points['distance_to_coast_km'].values
    return df


def create_feature_matrix(
    corrections: pd.DataFrame,
    *,
    terrain_nc: Path | str | None = None,
    coastline_geojson: Path | str | None = None,
    additional_features: list[str] | None = None,
    lon_col: str = 'lon',
    lat_col: str = 'lat',
) -> pd.DataFrame:
    """Create a feature matrix combining corrections with terrain features.

    Args:
        corrections: Correction factors with lon/lat and scalar/offset columns.
        terrain_nc: NetCDF with terrain features.
        coastline_geojson: GeoJSON for distance-to-coast calculation.
        additional_features: Column names in corrections to use as features.
        lon_col: Longitude column name.
        lat_col: Latitude column name.

    Returns:
        Combined feature matrix.
    """
    df = corrections.copy()
    
    # Extract terrain features if provided
    if terrain_nc is not None:
        terrain = extract_terrain_features_from_netcdf(
            terrain_nc,
            lon=df[lon_col].values,
            lat=df[lat_col].values,
        )
        # Merge on coordinates
        df = pd.merge(
            df,
            terrain,
            on=[lon_col, lat_col],
            how='left',
            suffixes=('', '_terrain')
        )
    
    # Add distance to coast
    if coastline_geojson is not None:
        df = add_distance_to_coast(df, coastline_geojson=coastline_geojson)
    
    # Add coordinate-based features
    df['lat_norm'] = (df[lat_col] - df[lat_col].mean()) / df[lat_col].std()
    df['lon_norm'] = (df[lon_col] - df[lon_col].mean()) / df[lon_col].std()
    
    return df


# =============================================================================
# ML MODEL TRAINING
# =============================================================================

def get_model(
    model_type: Literal[
        'random_forest', 'gradient_boosting', 'xgboost', 'lightgbm',
        'ridge', 'lasso', 'elastic_net'
    ],
    **kwargs,
) -> BaseEstimator:
    """Get a scikit-learn compatible ML model.

    Args:
        model_type: Type of model to create.
        **kwargs: Model-specific hyperparameters.

    Returns:
        Scikit-learn compatible model.

    Raises:
        ImportError: If required ML dependencies are missing.
        ValueError: If ``model_type`` is unknown.
    """
    if not _HAS_SKLEARN_MODELS:
        raise ImportError("scikit-learn is required for ML models")
    
    defaults = {
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 10,
            'random_state': 42,
            'n_jobs': -1,
        },
        'gradient_boosting': {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'random_state': 42,
        },
        'xgboost': {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'random_state': 42,
            'n_jobs': -1,
        },
        'lightgbm': {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'random_state': 42,
            'n_jobs': -1,
        },
        'ridge': {'alpha': 1.0},
        'lasso': {'alpha': 1.0},
        'elastic_net': {'alpha': 1.0, 'l1_ratio': 0.5},
    }
    
    # Merge defaults with user kwargs
    params = {**defaults.get(model_type, {}), **kwargs}
    
    if model_type == 'random_forest':
        return RandomForestRegressor(**params)
    elif model_type == 'gradient_boosting':
        return GradientBoostingRegressor(**params)
    elif model_type == 'xgboost':
        if not _HAS_XGB:
            raise ImportError("xgboost is required. Install with: pip install xgboost")
        return xgb.XGBRegressor(**params)
    elif model_type == 'lightgbm':
        if not _HAS_LGB:
            raise ImportError("lightgbm is required. Install with: pip install lightgbm")
        return lgb.LGBMRegressor(**params)
    elif model_type == 'ridge':
        return Ridge(**params)
    elif model_type == 'lasso':
        return Lasso(**params)
    elif model_type == 'elastic_net':
        return ElasticNet(**params)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def train_correction_model(
    features: pd.DataFrame,
    *,
    target_col: Literal['scalar', 'offset'],
    feature_cols: list[str] | None = None,
    model_type: str = 'random_forest',
    scale_features: bool = True,
    cv_folds: int = 5,
    **model_kwargs,
) -> dict[str, Any]:
    """Train an ML model to predict correction factors from features.

    Args:
        features: Feature matrix with target column.
        target_col: Which correction factor to predict (``"scalar"`` or ``"offset"``).
        feature_cols: Column names to use as features. If None, auto-detect.
        model_type: Type of model (see ``get_model``).
        scale_features: Whether to standardize features.
        cv_folds: Number of cross-validation folds.
        **model_kwargs: Model-specific hyperparameters.

    Returns:
        Dictionary containing the trained model, CV scores, feature importance,
        and the list of feature columns used.
    """
    # Auto-detect feature columns if not provided
    if feature_cols is None:
        exclude = {target_col, 'scalar', 'offset', 'lon', 'lat', 'ID', 'turbine_id', 'domain', 'type'}
        feature_cols = [c for c in features.columns if c not in exclude]
    
    print(f"Training {model_type} to predict {target_col}")
    print(f"Using {len(feature_cols)} features: {feature_cols[:5]}{'...' if len(feature_cols) > 5 else ''}")
    
    # Prepare data
    X = features[feature_cols].copy()
    y = features[target_col].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    mask = np.isfinite(y)
    X = X.loc[mask]
    y = y.loc[mask]
    
    if len(X) < 10:
        raise ValueError(f"Insufficient samples after filtering: {len(X)}")
    
    print(f"Training on {len(X)} samples")
    
    # Create model
    model = get_model(model_type, **model_kwargs)
    
    # Optionally wrap in pipeline with scaling
    if scale_features:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model),
        ])
    else:
        pipeline = model
    
    # Cross-validation
    cv_results = cross_validate(
        pipeline,
        X, y,
        cv=KFold(n_splits=cv_folds, shuffle=True, random_state=42),
        scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'],
        return_train_score=True,
    )
    
    # Train final model on all data
    pipeline.fit(X, y)
    
    # Extract feature importance (if available)
    feature_importance = None
    try:
        if scale_features:
            final_model = pipeline.named_steps['model']
        else:
            final_model = pipeline
        
        if hasattr(final_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': final_model.feature_importances_
            }).sort_values('importance', ascending=False)
    except Exception:
        pass
    
    # Print results
    print(f"\nCross-validation results ({cv_folds} folds):")
    print(f"  R² (test):  {cv_results['test_r2'].mean():.3f} ± {cv_results['test_r2'].std():.3f}")
    print(f"  MAE (test): {-cv_results['test_neg_mean_absolute_error'].mean():.3f}")
    print(f"  RMSE (test): {np.sqrt(-cv_results['test_neg_mean_squared_error'].mean()):.3f}")
    
    return {
        'model': pipeline,
        'cv_scores': cv_results,
        'feature_importance': feature_importance,
        'feature_cols': feature_cols,
        'target_col': target_col,
    }


def predict_correction_grid(
    model_result: dict,
    *,
    grid_nc: Path | str,
    terrain_nc: Path | str | None = None,
    coastline_geojson: Path | str | None = None,
    mask: xr.DataArray | None = None,
) -> xr.DataArray:
    """Apply a trained model to predict corrections on a spatial grid.

    Args:
        model_result: Output from ``train_correction_model``.
        grid_nc: NetCDF defining the output grid (lon/lat coords).
        terrain_nc: NetCDF with terrain features.
        coastline_geojson: GeoJSON for distance-to-coast.
        mask: Optional boolean mask for where to predict.

    Returns:
        Predicted corrections on the grid.
    """
    # Load grid
    grid_ds = xr.open_dataset(grid_nc)
    if 'x' in grid_ds.coords and 'y' in grid_ds.coords:
        grid_ds = grid_ds.rename({'x': 'lon', 'y': 'lat'})
    
    lon = grid_ds.lon.values
    lat = grid_ds.lat.values
    
    # Create feature matrix for grid points
    LON, LAT = np.meshgrid(lon, lat)
    grid_df = pd.DataFrame({
        'lon': LON.flatten(),
        'lat': LAT.flatten(),
    })
    
    # Add terrain features
    if terrain_nc is not None or coastline_geojson is not None:
        grid_df = create_feature_matrix(
            grid_df,
            terrain_nc=terrain_nc,
            coastline_geojson=coastline_geojson,
        )
    
    # Prepare features (match training)
    X_grid = grid_df[model_result['feature_cols']].copy()
    X_grid = X_grid.fillna(X_grid.median())
    
    # Predict
    predictions = model_result['model'].predict(X_grid)
    
    # Reshape to grid
    pred_grid = predictions.reshape(len(lat), len(lon))
    
    # Create DataArray
    pred_da = xr.DataArray(
        pred_grid,
        coords={'lat': lat, 'lon': lon},
        dims=('lat', 'lon'),
        name=model_result['target_col'],
    )
    
    # Apply mask if provided
    if mask is not None:
        pred_da = pred_da.where(mask)
    
    return pred_da


# =============================================================================
# COMPARISON & EVALUATION
# =============================================================================

def compare_interpolation_methods(
    corrections: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: Literal['scalar', 'offset'],
    models: list[str] = ['random_forest', 'gradient_boosting', 'ridge'],
    cv_folds: int = 5,
) -> pd.DataFrame:
    """Compare different ML models for correction prediction.

    Args:
        corrections: Training data with features and targets.
        feature_cols: Features to use.
        target_col: Target to predict.
        models: Model types to compare.
        cv_folds: Number of CV folds.

    Returns:
        Comparison of model performance.
    """
    results = []
    
    for model_type in models:
        try:
            print(f"\n{'='*60}")
            print(f"Training: {model_type}")
            print(f"{'='*60}")
            
            result = train_correction_model(
                corrections,
                target_col=target_col,
                feature_cols=feature_cols,
                model_type=model_type,
                cv_folds=cv_folds,
            )
            
            cv = result['cv_scores']
            results.append({
                'model': model_type,
                'r2_mean': cv['test_r2'].mean(),
                'r2_std': cv['test_r2'].std(),
                'mae_mean': -cv['test_neg_mean_absolute_error'].mean(),
                'rmse_mean': np.sqrt(-cv['test_neg_mean_squared_error'].mean()),
            })
        except Exception as e:
            print(f"Failed to train {model_type}: {e}")
            continue
    
    comparison = pd.DataFrame(results).sort_values('r2_mean', ascending=False)
    
    print(f"\n{'='*60}")
    print("Model Comparison Summary")
    print(f"{'='*60}")
    print(comparison.to_string(index=False))
    
    return comparison


def export_ml_correction_grid(
    *,
    corrections_csv: Path | str,
    grid_nc: Path | str,
    out_nc: Path | str,
    terrain_nc: Path | str | None = None,
    coastline_geojson: Path | str | None = None,
    onshore_mask_geojson: Path | str | None = None,
    offshore_mask_geojson: Path | str | None = None,
    model_type: str = 'random_forest',
    **model_kwargs,
) -> Path:
    """Train ML models and export gridded corrections.

    Args:
        corrections_csv: CSV with correction factors and coordinates.
        grid_nc: NetCDF defining output grid.
        out_nc: Output NetCDF path.
        terrain_nc: Terrain features NetCDF.
        coastline_geojson: Coastline GeoJSON for distance calculation.
        onshore_mask_geojson: GeoJSON mask for onshore regions.
        offshore_mask_geojson: GeoJSON mask for offshore regions.
        model_type: ML model type.
        **model_kwargs: Model hyperparameters.

    Returns:
        Path to the output NetCDF file.
    """
    print("="*60)
    print("ML-Based Bias Correction Grid Export")
    print("="*60)
    
    # Load corrections
    corrections = pd.read_csv(corrections_csv)
    print(f"Loaded {len(corrections)} correction points")
    
    # Create feature matrix
    features = create_feature_matrix(
        corrections,
        terrain_nc=terrain_nc,
        coastline_geojson=coastline_geojson,
    )
    
    # Train models for scalar and offset
    scalar_result = train_correction_model(
        features,
        target_col='scalar',
        model_type=model_type,
        **model_kwargs,
    )
    
    offset_result = train_correction_model(
        features,
        target_col='offset',
        model_type=model_type,
        **model_kwargs,
    )
    
    # Load masks if provided
    from vwf.geospatial import union_geometries
    
    onshore_mask = None
    offshore_mask = None
    
    if onshore_mask_geojson:
        grid_ds = xr.open_dataset(grid_nc)
        lon = grid_ds.x.values if 'x' in grid_ds.coords else grid_ds.lon.values
        lat = grid_ds.y.values if 'y' in grid_ds.coords else grid_ds.lat.values
        
        from vwf.atlite_export import mask_from_geojson_fast
        onshore_mask = mask_from_geojson_fast(lon, lat, Path(onshore_mask_geojson), name='onshore')
        
        if offshore_mask_geojson:
            offshore_mask = mask_from_geojson_fast(lon, lat, Path(offshore_mask_geojson), name='offshore')
            offshore_mask = offshore_mask & (~onshore_mask)
    
    # Predict grids
    scalar_grid = predict_correction_grid(
        scalar_result,
        grid_nc=grid_nc,
        terrain_nc=terrain_nc,
        coastline_geojson=coastline_geojson,
        mask=onshore_mask if onshore_mask is not None else None,
    )
    
    offset_grid = predict_correction_grid(
        offset_result,
        grid_nc=grid_nc,
        terrain_nc=terrain_nc,
        coastline_geojson=coastline_geojson,
        mask=onshore_mask if onshore_mask is not None else None,
    )
    
    # Combine into dataset
    ds_out = xr.Dataset({
        'scalar': scalar_grid,
        'offset': offset_grid,
    })
    
    # Add metadata
    ds_out.attrs.update({
        'title': 'ML-predicted bias correction fields',
        'model_type': model_type,
        'source': str(corrections_csv),
        'terrain_data': str(terrain_nc) if terrain_nc else 'none',
    })
    
    # Save
    out_nc = Path(out_nc)
    out_nc.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(out_nc)
    
    print(f"\nSaved ML corrections to: {out_nc}")
    
    # Print feature importance
    if scalar_result['feature_importance'] is not None:
        print("\nTop 10 features for SCALAR:")
        print(scalar_result['feature_importance'].head(10).to_string(index=False))
    
    if offset_result['feature_importance'] is not None:
        print("\nTop 10 features for OFFSET:")
        print(offset_result['feature_importance'].head(10).to_string(index=False))
    
    return out_nc
