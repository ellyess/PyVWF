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

from scipy.spatial import cKDTree
from scipy.special import gamma as _gamma_func

from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, cross_validate, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Optional ML models
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.neural_network import MLPRegressor
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


def extract_era5_invariant_features(
    df: pd.DataFrame,
    *,
    invariant_nc: Path | str,
    lon_col: str = 'lon',
    lat_col: str = 'lat',
) -> pd.DataFrame:
    """Add ERA5 invariant surface fields as features.

    Extracts land-sea mask, surface roughness length, and ERA5 model
    orography from an ERA5 invariant NetCDF file.

    Args:
        df: DataFrame with lon/lat columns.
        invariant_nc: Path to ERA5 invariant NetCDF (containing z, lsm, fsr).
        lon_col: Longitude column name.
        lat_col: Latitude column name.

    Returns:
        DataFrame with added columns: ``era5_elevation``, ``era5_lsm``,
        ``era5_roughness_length``.
    """
    ds = xr.open_dataset(invariant_nc)

    # Standardize coordinate names
    rename = {}
    if 'latitude' in ds.coords:
        rename['latitude'] = 'lat'
    if 'longitude' in ds.coords:
        rename['longitude'] = 'lon'
    if rename:
        ds = ds.rename(rename)

    # Squeeze single time dimension if present
    if 'valid_time' in ds.dims:
        ds = ds.isel(valid_time=0)
    elif 'time' in ds.dims:
        ds = ds.isel(time=0)

    lons = df[lon_col].values
    lats = df[lat_col].values

    for var, out_col, scale in [
        ('z', 'era5_elevation', 1.0 / 9.80665),  # geopotential -> meters
        ('lsm', 'era5_lsm', 1.0),
        ('fsr', 'era5_roughness_length', 1.0),
    ]:
        if var in ds:
            vals = ds[var].interp(
                lon=xr.DataArray(lons, dims='points'),
                lat=xr.DataArray(lats, dims='points'),
                method='linear',
            ).values * scale
            df[out_col] = vals

    ds.close()
    return df


def extract_era5_wind_climatology(
    df: pd.DataFrame,
    *,
    era5_wind_dir: Path | str,
    lon_col: str = 'lon',
    lat_col: str = 'lat',
    years: list[int] | None = None,
) -> pd.DataFrame:
    """Add ERA5 100m wind speed climatology features.

    Computes long-term mean and standard deviation of wind speed at
    each point from hourly u100/v100 fields.

    Args:
        df: DataFrame with lon/lat columns.
        era5_wind_dir: Directory containing ERA5 u100/v100 NetCDF files
            with naming pattern ``era5_u100_v100_YYYY_*.nc``.
        lon_col: Longitude column name.
        lat_col: Latitude column name.
        years: Years to include. If None, uses all available files.

    Returns:
        DataFrame with ``era5_wind_mean`` and ``era5_wind_std`` columns.
    """
    era5_dir = Path(era5_wind_dir)
    files = sorted(era5_dir.glob('era5_u100_v100_*.nc'))
    if not files:
        files = sorted(era5_dir.glob('era5_combined_*.nc'))
    if years is not None:
        files = [f for f in files if any(str(y) in f.name for y in years)]

    if not files:
        warnings.warn(f"No ERA5 wind files found in {era5_dir}")
        return df

    lons = df[lon_col].values
    lats = df[lat_col].values

    # Accumulate windspeed statistics across years
    ws_sum = np.zeros(len(df))
    ws_sq_sum = np.zeros(len(df))
    n_total = 0

    for fpath in files:
        ds = xr.open_dataset(fpath, engine='netcdf4')

        # Standardize coordinate names
        rename = {}
        if 'latitude' in ds.coords:
            rename['latitude'] = 'lat'
        if 'longitude' in ds.coords:
            rename['longitude'] = 'lon'
        if rename:
            ds = ds.rename(rename)

        # Compute wind speed per timestep, then mean/std over time
        ws = np.sqrt(ds['u100']**2 + ds['v100']**2)
        time_dim = 'valid_time' if 'valid_time' in ws.dims else 'time'
        ws_mean_grid = ws.mean(dim=time_dim)
        ws_std_grid = ws.std(dim=time_dim)

        # Interpolate to point locations
        mean_vals = ws_mean_grid.interp(
            lon=xr.DataArray(lons, dims='points'),
            lat=xr.DataArray(lats, dims='points'),
            method='linear',
        ).values
        std_vals = ws_std_grid.interp(
            lon=xr.DataArray(lons, dims='points'),
            lat=xr.DataArray(lats, dims='points'),
            method='linear',
        ).values

        ws_sum += np.nan_to_num(mean_vals)
        ws_sq_sum += np.nan_to_num(mean_vals**2)
        n_total += 1
        ds.close()

    if n_total > 0:
        df['era5_wind_mean'] = ws_sum / n_total
        # Inter-annual variability
        df['era5_wind_std'] = std_vals  # std within last year as proxy
        print(f"  ERA5 wind climatology from {n_total} files: "
              f"mean={df['era5_wind_mean'].mean():.2f} m/s")

    return df


def extract_era5_extended_wind_features(
    df: pd.DataFrame,
    *,
    era5_wind_dir: Path | str,
    era5_wind10_dir: Path | str | None = None,
    lon_col: str = 'lon',
    lat_col: str = 'lat',
    years: list[int] | None = None,
) -> pd.DataFrame:
    """Add extended ERA5 wind features: shear, seasonal, Weibull, diurnal.

    Computes features from hourly u100/v100 and u10/v10 data,
    processing year by year to limit memory usage.

    Args:
        df: DataFrame with lon/lat columns.
        era5_wind_dir: Directory containing ERA5 u100/v100 NetCDF files
            with naming pattern ``era5_u100_v100_YYYY_*.nc``.
        era5_wind10_dir: Directory containing ERA5 u10/v10 NetCDF files
            with naming pattern ``era5_u10_v10_YYYY_*.nc``. If None,
            uses era5_wind_dir (same directory).
        lon_col: Longitude column name.
        lat_col: Latitude column name.
        years: Years to include. If None, uses all available files.

    Returns:
        DataFrame with added columns: ``era5_wind_shear``,
        ``era5_wind_winter_mean``, ``era5_wind_summer_mean``,
        ``era5_wind_seasonal_range``, ``era5_weibull_k``,
        ``era5_weibull_a``, ``era5_diurnal_amplitude``,
        ``era5_wind_night_mean``.
    """
    era5_dir = Path(era5_wind_dir)
    era5_10_dir = Path(era5_wind10_dir) if era5_wind10_dir else era5_dir

    files_100 = sorted(era5_dir.glob('era5_u100_v100_*.nc'))
    if not files_100:
        files_100 = sorted(era5_dir.glob('era5_combined_*.nc'))
    if years is not None:
        files_100 = [f for f in files_100 if any(str(y) in f.name for y in years)]

    if not files_100:
        warnings.warn(f"No ERA5 u100/v100 files found in {era5_dir}")
        return df

    lons = df[lon_col].values
    lats = df[lat_col].values
    n_pts = len(df)

    # Accumulators across years (interpolated to points)
    shear_sum = np.zeros(n_pts)
    winter_sum = np.zeros(n_pts)
    summer_sum = np.zeros(n_pts)
    day_sum = np.zeros(n_pts)
    night_sum = np.zeros(n_pts)
    weibull_mean_sum = np.zeros(n_pts)
    weibull_std_sum = np.zeros(n_pts)
    n_shear_years = 0
    n_years = 0

    for fpath_100 in files_100:
        # Extract year from filename
        parts = fpath_100.stem.split('_')
        year_str = [p for p in parts if p.isdigit() and len(p) == 4]
        if year_str:
            year = int(year_str[0])
        else:
            year = None

        ds100 = xr.open_dataset(fpath_100, engine='netcdf4')

        # Standardize coordinate names
        rename = {}
        if 'latitude' in ds100.coords:
            rename['latitude'] = 'lat'
        if 'longitude' in ds100.coords:
            rename['longitude'] = 'lon'
        if rename:
            ds100 = ds100.rename(rename)

        # Identify time dimension
        time_dim = 'valid_time' if 'valid_time' in ds100.dims else 'time'

        ws100 = np.sqrt(ds100['u100']**2 + ds100['v100']**2)

        # --- WIND SHEAR ---
        fpath_10 = None
        if year is not None:
            candidates = sorted(era5_10_dir.glob(f'era5_u10_v10_{year}_*.nc'))
            if not candidates:
                candidates = sorted(era5_10_dir.glob(f'era5_combined_{year}_*.nc'))
            if candidates:
                fpath_10 = candidates[0]

        if fpath_10 is not None and fpath_10.exists():
            ds10 = xr.open_dataset(fpath_10, engine='netcdf4')
            rename10 = {}
            if 'latitude' in ds10.coords:
                rename10['latitude'] = 'lat'
            if 'longitude' in ds10.coords:
                rename10['longitude'] = 'lon'
            if rename10:
                ds10 = ds10.rename(rename10)

            ws10 = np.sqrt(ds10['u10']**2 + ds10['v10']**2)

            # Clip to avoid log(0) and div-by-zero
            ws100_c = ws100.clip(min=0.5)
            ws10_c = ws10.clip(min=0.5)

            ratio = ws100_c / ws10_c
            # Filter physically unreasonable ratios
            ratio = ratio.where((ratio > 0.1) & (ratio < 10.0))
            alpha_hourly = np.log(ratio) / np.log(10.0)
            alpha_mean_grid = alpha_hourly.mean(dim=time_dim, skipna=True)

            shear_vals = alpha_mean_grid.interp(
                lon=xr.DataArray(lons, dims='points'),
                lat=xr.DataArray(lats, dims='points'),
                method='linear',
            ).values
            shear_sum += np.nan_to_num(shear_vals)
            n_shear_years += 1
            ds10.close()

        # --- SEASONAL MEANS ---
        months = ds100[time_dim].dt.month

        winter_mask = months.isin([12, 1, 2])
        summer_mask = months.isin([6, 7, 8])

        ws_winter_grid = ws100.sel({time_dim: winter_mask}).mean(dim=time_dim)
        ws_summer_grid = ws100.sel({time_dim: summer_mask}).mean(dim=time_dim)

        winter_vals = ws_winter_grid.interp(
            lon=xr.DataArray(lons, dims='points'),
            lat=xr.DataArray(lats, dims='points'),
            method='linear',
        ).values
        summer_vals = ws_summer_grid.interp(
            lon=xr.DataArray(lons, dims='points'),
            lat=xr.DataArray(lats, dims='points'),
            method='linear',
        ).values

        winter_sum += np.nan_to_num(winter_vals)
        summer_sum += np.nan_to_num(summer_vals)

        # --- DIURNAL ---
        hours = ds100[time_dim].dt.hour
        day_mask = (hours >= 6) & (hours < 18)
        night_mask = ~day_mask

        ws_day_grid = ws100.sel({time_dim: day_mask}).mean(dim=time_dim)
        ws_night_grid = ws100.sel({time_dim: night_mask}).mean(dim=time_dim)

        day_vals = ws_day_grid.interp(
            lon=xr.DataArray(lons, dims='points'),
            lat=xr.DataArray(lats, dims='points'),
            method='linear',
        ).values
        night_vals = ws_night_grid.interp(
            lon=xr.DataArray(lons, dims='points'),
            lat=xr.DataArray(lats, dims='points'),
            method='linear',
        ).values

        day_sum += np.nan_to_num(day_vals)
        night_sum += np.nan_to_num(night_vals)

        # --- WEIBULL: accumulate mean and std ---
        ws_mean_grid = ws100.mean(dim=time_dim)
        ws_std_grid = ws100.std(dim=time_dim)

        mean_vals = ws_mean_grid.interp(
            lon=xr.DataArray(lons, dims='points'),
            lat=xr.DataArray(lats, dims='points'),
            method='linear',
        ).values
        std_vals = ws_std_grid.interp(
            lon=xr.DataArray(lons, dims='points'),
            lat=xr.DataArray(lats, dims='points'),
            method='linear',
        ).values

        weibull_mean_sum += np.nan_to_num(mean_vals)
        weibull_std_sum += np.nan_to_num(std_vals)

        n_years += 1
        ds100.close()

    if n_years > 0:
        # Wind shear
        if n_shear_years > 0:
            df['era5_wind_shear'] = shear_sum / n_shear_years
        else:
            df['era5_wind_shear'] = np.nan

        # Seasonal
        df['era5_wind_winter_mean'] = winter_sum / n_years
        df['era5_wind_summer_mean'] = summer_sum / n_years
        df['era5_wind_seasonal_range'] = (
            df['era5_wind_winter_mean'] - df['era5_wind_summer_mean']
        )

        # Diurnal
        df['era5_diurnal_amplitude'] = (day_sum - night_sum) / n_years
        df['era5_wind_night_mean'] = night_sum / n_years

        # Weibull via method of moments (Justus 1978 approximation)
        pooled_mean = weibull_mean_sum / n_years
        pooled_std = weibull_std_sum / n_years
        cv = pooled_std / np.maximum(pooled_mean, 1e-6)

        k_vals = (1.0 / np.maximum(cv, 0.01)) ** 1.086
        df['era5_weibull_k'] = k_vals
        df['era5_weibull_a'] = pooled_mean / _gamma_func(
            1.0 + 1.0 / np.maximum(k_vals, 0.5)
        )

        print(f"  ERA5 extended wind features from {n_years} files:")
        print(f"    wind_shear:       mean={df['era5_wind_shear'].mean():.3f}")
        print(f"    winter_mean:      mean={df['era5_wind_winter_mean'].mean():.2f} m/s")
        print(f"    summer_mean:      mean={df['era5_wind_summer_mean'].mean():.2f} m/s")
        print(f"    seasonal_range:   mean={df['era5_wind_seasonal_range'].mean():.2f} m/s")
        print(f"    weibull_k:        mean={df['era5_weibull_k'].mean():.2f}")
        print(f"    weibull_a:        mean={df['era5_weibull_a'].mean():.2f}")
        print(f"    diurnal_amp:      mean={df['era5_diurnal_amplitude'].mean():.2f} m/s")

    return df


def extract_land_cover_features(
    df: pd.DataFrame,
    *,
    corine_nc: Path | str,
    lon_col: str = 'lon',
    lat_col: str = 'lat',
) -> pd.DataFrame:
    """Add land cover features from CORINE or ESA CCI data.

    Extracts land cover class at each point and derives binary
    indicators and roughness length estimates.  Supports both
    CORINE class codes (111-523) and ESA CCI/C3S LCCS codes
    (10-220), auto-detected from values.

    Args:
        df: DataFrame with lon/lat columns.
        corine_nc: Path to land cover NetCDF with a
            ``land_cover_class`` or ``lccs_class`` variable.
        lon_col: Longitude column name.
        lat_col: Latitude column name.

    Returns:
        DataFrame with land-cover indicator columns and
        ``roughness_from_lc``.
    """
    ds = xr.open_dataset(corine_nc)

    # Standardize coordinate names
    rename = {}
    if 'latitude' in ds.coords:
        rename['latitude'] = 'lat'
    if 'longitude' in ds.coords:
        rename['longitude'] = 'lon'
    if rename:
        ds = ds.rename(rename)

    # Squeeze time dimension if present
    if 'time' in ds.dims:
        ds = ds.isel(time=0)

    lons = df[lon_col].values
    lats = df[lat_col].values

    lc_var = 'land_cover_class'
    if lc_var not in ds:
        for alt in ['lccs_class', 'lc', 'Band1']:
            if alt in ds:
                lc_var = alt
                break

    if lc_var in ds:
        # Nearest-neighbor for categorical data
        lc_vals = ds[lc_var].sel(
            lon=xr.DataArray(lons, dims='points'),
            lat=xr.DataArray(lats, dims='points'),
            method='nearest',
        ).values.astype(float)

        df['land_cover_class'] = lc_vals

        # Detect if CCI (values < 250) or CORINE (values > 100)
        max_val = np.nanmax(lc_vals)
        is_cci = max_val < 250 and np.nanmin(lc_vals[lc_vals > 0]) < 100

        if is_cci:
            # ESA CCI LCCS class mappings
            df['is_urban'] = (lc_vals == 190).astype(float)
            df['is_agricultural'] = np.isin(lc_vals, [10, 11, 12, 20, 30, 40]).astype(float)
            df['is_forest'] = np.isin(lc_vals, [50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 100]).astype(float)
            df['is_bare'] = np.isin(lc_vals, [200, 201, 202, 150, 151, 152, 153]).astype(float)
            df['is_water'] = np.isin(lc_vals, [210, 220]).astype(float)

            z0_map = {'urban': 1.5, 'agricultural': 0.1, 'forest': 1.0,
                      'bare': 0.005, 'water': 0.0002}
            roughness = np.full(len(df), 0.3)  # default (shrubland/grassland)
            roughness[df['is_urban'] == 1] = z0_map['urban']
            roughness[df['is_agricultural'] == 1] = z0_map['agricultural']
            roughness[df['is_forest'] == 1] = z0_map['forest']
            roughness[df['is_bare'] == 1] = z0_map['bare']
            roughness[df['is_water'] == 1] = z0_map['water']
            df['roughness_from_lc'] = roughness
            print(f"  Land cover (CCI): {len(lc_vals)} points, "
                  f"{(df['is_forest']==1).sum()} forest, "
                  f"{(df['is_agricultural']==1).sum()} agricultural, "
                  f"{(df['is_urban']==1).sum()} urban")
        else:
            # CORINE class ranges -> binary indicators
            df['is_urban'] = ((lc_vals >= 111) & (lc_vals <= 142)).astype(float)
            df['is_agricultural'] = ((lc_vals >= 211) & (lc_vals <= 244)).astype(float)
            df['is_forest'] = ((lc_vals >= 311) & (lc_vals <= 313)).astype(float)
            df['is_bare'] = ((lc_vals >= 331) & (lc_vals <= 335)).astype(float)
            df['is_water'] = ((lc_vals >= 511) & (lc_vals <= 523)).astype(float)

            z0_map = {'urban': 1.5, 'agricultural': 0.1, 'forest': 1.0,
                      'bare': 0.005, 'water': 0.0002}
            roughness = np.full(len(df), 0.1)
            roughness[df['is_urban'] == 1] = z0_map['urban']
            roughness[df['is_agricultural'] == 1] = z0_map['agricultural']
            roughness[df['is_forest'] == 1] = z0_map['forest']
            roughness[df['is_bare'] == 1] = z0_map['bare']
            roughness[df['is_water'] == 1] = z0_map['water']
            df['roughness_from_lc'] = roughness

    ds.close()
    return df


def add_turbine_fleet_features(
    corrections: pd.DataFrame,
    *,
    turbine_metadata: dict[str, pd.DataFrame],
    lon_col: str = 'lon',
    lat_col: str = 'lat',
    country_col: str = 'country_code',
) -> pd.DataFrame:
    """Add turbine fleet aggregate features to corrections.

    For countries with spatially referenced turbine metadata (lon/lat),
    turbines are assigned to the nearest correction centroid and
    per-cluster statistics are computed.  For countries without spatial
    data, country-wide medians are used.

    Args:
        corrections: Corrections DataFrame with lon, lat, country_code.
        turbine_metadata: Mapping of country code to turbine metadata
            DataFrame.  Each DataFrame should have columns ``height``
            (hub height in m), ``diameter`` (rotor diameter in m), and
            ``capacity`` (kW).  Optionally ``lon`` and ``lat`` for
            spatial matching.
        lon_col: Longitude column name.
        lat_col: Latitude column name.
        country_col: Country code column name.

    Returns:
        DataFrame with ``mean_hub_height``, ``mean_rotor_diameter``,
        and ``mean_capacity`` columns added.
    """
    df = corrections.copy()
    df['mean_hub_height'] = np.nan
    df['mean_rotor_diameter'] = np.nan
    df['mean_capacity'] = np.nan

    for country, turb_df in turbine_metadata.items():
        country_mask = df[country_col].str.startswith(country)
        if not country_mask.any():
            continue

        has_coords = 'lon' in turb_df.columns and 'lat' in turb_df.columns
        turb = turb_df.dropna(subset=['height', 'diameter', 'capacity'])

        if has_coords and len(turb) > 0:
            turb = turb.dropna(subset=['lon', 'lat'])
            if len(turb) == 0:
                continue

            # Spatial matching: assign each turbine to nearest centroid
            corr_sub = df.loc[country_mask]
            tree = cKDTree(corr_sub[[lon_col, lat_col]].values)
            _, indices = tree.query(turb[['lon', 'lat']].values)

            turb = turb.copy()
            turb['_centroid_idx'] = corr_sub.index[indices]

            agg = turb.groupby('_centroid_idx').agg(
                mean_hub_height=('height', 'mean'),
                mean_rotor_diameter=('diameter', 'mean'),
                mean_capacity=('capacity', 'mean'),
            )
            for col in ['mean_hub_height', 'mean_rotor_diameter', 'mean_capacity']:
                df.loc[agg.index, col] = agg[col]
        else:
            # No coordinates -- use country-wide medians
            medians = {
                'mean_hub_height': turb['height'].median() if 'height' in turb else np.nan,
                'mean_rotor_diameter': turb['diameter'].median() if 'diameter' in turb else np.nan,
                'mean_capacity': turb['capacity'].median() if 'capacity' in turb else np.nan,
            }
            for col, val in medians.items():
                df.loc[country_mask, col] = val

    # Fill remaining NaN with global median
    for col in ['mean_hub_height', 'mean_rotor_diameter', 'mean_capacity']:
        df[col] = df[col].fillna(df[col].median())

    return df


def build_turbine_level_dataset(
    corrections: pd.DataFrame,
    turbine_metadata: dict[str, pd.DataFrame],
    *,
    de_geolocate: Path | str | None = None,
    lon_col: str = 'lon',
    lat_col: str = 'lat',
    country_col: str = 'country_code',
) -> pd.DataFrame:
    """Expand centroid-level corrections to individual turbine rows.

    Each turbine is assigned to its nearest correction centroid (via
    cKDTree) and inherits that centroid's scalar/offset.  The turbine's
    own hub height, rotor diameter and capacity are kept as direct
    features rather than cluster aggregates.

    Args:
        corrections: Centroid-level correction factors with lon, lat,
            scalar, offset, and country_code columns.
        turbine_metadata: Mapping of country prefix (e.g. ``"DK"``) to
            a DataFrame with columns ``lon``, ``lat``, ``height``,
            ``diameter``, ``capacity``.
        de_geolocate: Optional path to German postcode geolocation CSV
            (columns: postcode, lon, lat).  When provided, DE turbines
            are geolocated via their postcode extracted from the ID
            column ``V1``.
        lon_col: Longitude column name in *corrections*.
        lat_col: Latitude column name in *corrections*.
        country_col: Country code column name in *corrections*.

    Returns:
        DataFrame with one row per turbine, containing:
        lon, lat, scalar, offset, country_code, hub_height,
        rotor_diameter, capacity.
    """
    rows: list[pd.DataFrame] = []

    for country_prefix, turb_df in turbine_metadata.items():
        # Find matching corrections (e.g. 'DK' matches 'DK-onshore', 'DK-offshore')
        mask = corrections[country_col].str.startswith(country_prefix)
        corr_sub = corrections.loc[mask]
        if corr_sub.empty:
            continue

        # Prepare turbine data with coordinates
        turb = turb_df.copy()

        # For DE: geolocate via postcode if coordinates missing
        if country_prefix == 'DE' and 'lon' not in turb.columns and de_geolocate is not None:
            geo = pd.read_csv(de_geolocate)
            if 'V1' in turb.columns:
                turb['_postcode'] = turb['V1'].str.extract(r'^(\d{4,5})', expand=False)
                turb['_postcode'] = pd.to_numeric(turb['_postcode'], errors='coerce')
                geo['postcode'] = geo['postcode'].astype(int)
                turb = turb.merge(
                    geo[['postcode', 'lon', 'lat']],
                    left_on='_postcode', right_on='postcode', how='inner',
                )
                turb = turb.drop(columns=['_postcode', 'postcode'], errors='ignore')

        # Standardize column names
        col_map = {}
        if 'Tower..m.' in turb.columns:
            col_map['Tower..m.'] = 'height'
        if 'Rotor..m.' in turb.columns:
            col_map['Rotor..m.'] = 'diameter'
        if 'kW' in turb.columns:
            col_map['kW'] = 'capacity'
        if col_map:
            turb = turb.rename(columns=col_map)

        # Need lon, lat, height, diameter, capacity
        required = ['lon', 'lat', 'height', 'diameter', 'capacity']
        if not all(c in turb.columns for c in required):
            continue

        turb = turb.dropna(subset=required)
        for col in required:
            turb[col] = pd.to_numeric(turb[col], errors='coerce')
        turb = turb.dropna(subset=required)

        # Filter obvious outliers
        turb = turb[
            (turb['capacity'] > 0) & (turb['capacity'] < 20000)
            & (turb['diameter'] > 5) & (turb['diameter'] < 300)
            & (turb['height'] > 10) & (turb['height'] < 300)
        ]
        if turb.empty:
            continue

        # Assign each turbine to nearest correction centroid
        tree = cKDTree(corr_sub[[lon_col, lat_col]].values)
        _, indices = tree.query(turb[['lon', 'lat']].values)

        matched_corr = corr_sub.iloc[indices]
        turbine_rows = pd.DataFrame({
            'lon': turb['lon'].values,
            'lat': turb['lat'].values,
            'scalar': matched_corr['scalar'].values,
            'offset': matched_corr['offset'].values,
            'country_code': matched_corr[country_col].values,
            'hub_height': turb['height'].values,
            'rotor_diameter': turb['diameter'].values,
            'capacity': turb['capacity'].values,
        })
        rows.append(turbine_rows)

    if not rows:
        raise ValueError("No turbines could be matched to corrections")

    result = pd.concat(rows, ignore_index=True)
    print(f"  Built turbine-level dataset: {len(result)} turbines")
    for cc in result['country_code'].unique():
        n = (result['country_code'] == cc).sum()
        print(f"    {cc}: {n}")
    return result


def create_feature_matrix(
    corrections: pd.DataFrame,
    *,
    terrain_nc: Path | str | None = None,
    coastline_geojson: Path | str | None = None,
    invariant_nc: Path | str | None = None,
    corine_nc: Path | str | None = None,
    turbine_metadata: dict[str, pd.DataFrame] | None = None,
    era5_wind_dir: Path | str | None = None,
    era5_wind10_dir: Path | str | None = None,
    additional_features: list[str] | None = None,
    lon_col: str = 'lon',
    lat_col: str = 'lat',
) -> pd.DataFrame:
    """Create a feature matrix combining corrections with terrain features.

    Args:
        corrections: Correction factors with lon/lat and scalar/offset columns.
        terrain_nc: NetCDF with terrain features.
        coastline_geojson: GeoJSON for distance-to-coast calculation.
        invariant_nc: ERA5 invariant NetCDF for land-sea mask, roughness, etc.
        corine_nc: CORINE land cover NetCDF.
        turbine_metadata: Dict mapping country code to turbine metadata
            DataFrames for fleet features.
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
        # Assign columns directly (preserves row order, avoids
        # cartesian product when duplicate lon/lat exist)
        for col in terrain.columns:
            if col in (lon_col, lat_col):
                continue
            df[col] = terrain[col].values

    # Add distance to coast
    if coastline_geojson is not None:
        df = add_distance_to_coast(df, coastline_geojson=coastline_geojson)

    # Add ERA5 invariant features
    if invariant_nc is not None:
        df = extract_era5_invariant_features(
            df, invariant_nc=invariant_nc, lon_col=lon_col, lat_col=lat_col,
        )
        # Compute elevation mismatch if both terrain and ERA5 elevation exist
        if 'elevation' in df.columns and 'era5_elevation' in df.columns:
            df['elevation_mismatch'] = df['era5_elevation'] - df['elevation']

    # Add CORINE land cover features
    if corine_nc is not None:
        df = extract_land_cover_features(
            df, corine_nc=corine_nc, lon_col=lon_col, lat_col=lat_col,
        )

    # Add turbine fleet features
    if turbine_metadata is not None:
        df = add_turbine_fleet_features(
            df, turbine_metadata=turbine_metadata,
            lon_col=lon_col, lat_col=lat_col,
        )

    # Add ERA5 wind climatology
    if era5_wind_dir is not None:
        df = extract_era5_wind_climatology(
            df, era5_wind_dir=era5_wind_dir,
            lon_col=lon_col, lat_col=lat_col,
        )

    # Add ERA5 extended wind features (shear, seasonal, Weibull, diurnal)
    if era5_wind_dir is not None:
        df = extract_era5_extended_wind_features(
            df, era5_wind_dir=era5_wind_dir,
            era5_wind10_dir=era5_wind10_dir,
            lon_col=lon_col, lat_col=lat_col,
        )

    # Derived features
    # Specific power: capacity per swept area (W/m²)
    diam_col = 'rotor_diameter' if 'rotor_diameter' in df.columns else 'mean_rotor_diameter'
    cap_col = 'capacity' if 'capacity' in df.columns else 'mean_capacity'
    if diam_col in df.columns and cap_col in df.columns:
        swept_area = np.pi * (df[diam_col] / 2) ** 2
        df['specific_power'] = (df[cap_col] * 1000) / swept_area.replace(0, np.nan)

    # Add coordinate-based features
    df['lat_norm'] = (df[lat_col] - df[lat_col].mean()) / df[lat_col].std()
    df['lon_norm'] = (df[lon_col] - df[lon_col].mean()) / df[lon_col].std()

    return df


# =============================================================================
# CROSS-VALIDATION STRATEGIES
# =============================================================================

def get_cv_splitter(
    features: pd.DataFrame,
    *,
    cv_strategy: Literal['random', 'spatial_lon', 'leave_country_out'] = 'random',
    n_splits: int = 5,
    country_col: str = 'country_code',
    lon_col: str = 'lon',
    random_state: int = 42,
) -> KFold | list[tuple[np.ndarray, np.ndarray]]:
    """Get cross-validation splits for correction model evaluation.

    Args:
        features: Feature matrix (used for spatial/group splitting).
        cv_strategy: One of ``"random"`` (standard KFold), ``"spatial_lon"``
            (contiguous longitude blocks), or ``"leave_country_out"``
            (group by country_code).
        n_splits: Number of folds (ignored for leave_country_out).
        country_col: Column with country/region labels.
        lon_col: Longitude column (for spatial splitting).
        random_state: Random seed.

    Returns:
        A scikit-learn CV splitter or explicit list of (train_idx, test_idx).
    """
    if cv_strategy == 'random':
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    if cv_strategy == 'spatial_lon':
        # Sort by longitude, divide into contiguous blocks
        lon_vals = features[lon_col].values
        sorted_idx = np.argsort(lon_vals)
        fold_size = len(sorted_idx) // n_splits
        splits = []
        for i in range(n_splits):
            start = i * fold_size
            end = (i + 1) * fold_size if i < n_splits - 1 else len(sorted_idx)
            test_idx = sorted_idx[start:end]
            train_idx = np.setdiff1d(np.arange(len(features)), test_idx)
            splits.append((train_idx, test_idx))
        return splits

    if cv_strategy == 'leave_country_out':
        if country_col not in features.columns:
            warnings.warn(
                f"Column '{country_col}' not found, falling back to random CV"
            )
            return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        # Each large country is its own fold; small countries pooled
        countries = features[country_col].values
        unique_countries = np.unique(countries)
        counts = {c: (countries == c).sum() for c in unique_countries}

        # Countries with >= 50 samples get their own fold
        large = [c for c, n in counts.items() if n >= 50]
        small = [c for c, n in counts.items() if n < 50]

        splits = []
        for country in large:
            test_mask = countries == country
            test_idx = np.where(test_mask)[0]
            train_idx = np.where(~test_mask)[0]
            splits.append((train_idx, test_idx))

        # Pool small countries into one fold if any exist
        if small:
            test_mask = np.isin(countries, small)
            test_idx = np.where(test_mask)[0]
            train_idx = np.where(~test_mask)[0]
            if len(test_idx) > 0:
                splits.append((train_idx, test_idx))

        return splits

    raise ValueError(f"Unknown cv_strategy: {cv_strategy}")


# =============================================================================
# ML MODEL TRAINING
# =============================================================================

def get_model(
    model_type: Literal[
        'random_forest', 'gradient_boosting', 'xgboost', 'lightgbm',
        'ridge', 'lasso', 'elastic_net', 'mlp',
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
        'mlp': {
            'hidden_layer_sizes': (64, 32),
            'activation': 'relu',
            'solver': 'adam',
            'max_iter': 500,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'random_state': 42,
        },
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
    elif model_type == 'mlp':
        return MLPRegressor(**params)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def _get_param_distributions(model_type: str) -> dict | None:
    """Return hyperparameter search distributions for a model type."""
    from scipy.stats import uniform, randint, loguniform

    dists = {
        'random_forest': {
            'n_estimators': randint(50, 500),
            'max_depth': randint(3, 30),
            'min_samples_split': randint(2, 50),
            'min_samples_leaf': randint(1, 30),
            'max_features': uniform(0.3, 0.7),
        },
        'gradient_boosting': {
            'n_estimators': randint(50, 500),
            'max_depth': randint(2, 15),
            'learning_rate': loguniform(0.01, 0.5),
            'min_samples_split': randint(2, 50),
            'subsample': uniform(0.5, 0.5),
        },
        'ridge': {
            'alpha': loguniform(0.001, 1000),
        },
        'lasso': {
            'alpha': loguniform(0.0001, 100),
        },
        'elastic_net': {
            'alpha': loguniform(0.0001, 100),
            'l1_ratio': uniform(0.1, 0.9),
        },
        'xgboost': {
            'n_estimators': randint(50, 500),
            'max_depth': randint(2, 15),
            'learning_rate': loguniform(0.01, 0.5),
            'subsample': uniform(0.5, 0.5),
            'colsample_bytree': uniform(0.3, 0.7),
            'reg_alpha': loguniform(0.001, 10),
            'reg_lambda': loguniform(0.001, 10),
        },
        'lightgbm': {
            'n_estimators': randint(50, 500),
            'max_depth': randint(2, 15),
            'learning_rate': loguniform(0.01, 0.5),
            'subsample': uniform(0.5, 0.5),
            'colsample_bytree': uniform(0.3, 0.7),
            'reg_alpha': loguniform(0.001, 10),
            'reg_lambda': loguniform(0.001, 10),
            'min_child_samples': randint(5, 100),
        },
        'mlp': {
            'hidden_layer_sizes': [(32,), (64,), (128,),
                                   (32, 16), (64, 32), (128, 64),
                                   (64, 32, 16), (128, 64, 32)],
            'alpha': loguniform(1e-5, 1e-1),
            'learning_rate_init': loguniform(1e-4, 1e-2),
            'batch_size': [32, 64, 128, 256],
        },
    }
    return dists.get(model_type)


def train_correction_model(
    features: pd.DataFrame,
    *,
    target_col: Literal['scalar', 'offset'],
    feature_cols: list[str] | None = None,
    model_type: str = 'random_forest',
    scale_features: bool = True,
    cv_folds: int = 5,
    cv_strategy: Literal['random', 'spatial_lon', 'leave_country_out'] = 'random',
    log_target: bool = False,
    tune_hyperparams: bool = False,
    sample_weight_col: str | None = None,
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
        cv_strategy: Cross-validation strategy: ``"random"``, ``"spatial_lon"``,
            or ``"leave_country_out"``.
        log_target: If True, apply log transform to target before training
            and inverse-transform predictions.  Only valid for positive targets
            (e.g. scalar).
        tune_hyperparams: If True, run RandomizedSearchCV to find better
            hyperparameters before the final CV evaluation.
        sample_weight_col: Optional column to use for sample weights during
            training (e.g. inverse cluster size for deduplication).
        **model_kwargs: Model-specific hyperparameters.

    Returns:
        Dictionary containing the trained model, CV scores, feature importance,
        and the list of feature columns used.
    """
    # Auto-detect feature columns if not provided
    if feature_cols is None:
        exclude = {
            target_col, 'scalar', 'offset', 'lon', 'lat', 'ID', 'turbine_id',
            'domain', 'type', 'country_code', 'country_name', 'cluster',
            'cluster_mode', 'obs_level', 'area_km2', 'land_cover_class',
        }
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

    # Sample weights
    weights = None
    if sample_weight_col is not None and sample_weight_col in features.columns:
        weights = features.loc[mask, sample_weight_col].values

    # Log transform (only for strictly positive targets like scalar)
    if log_target:
        if (y <= 0).any():
            warnings.warn("log_target requested but target has non-positive values; skipping")
            log_target = False
        else:
            y = np.log(y)
            print(f"  Applied log transform to {target_col}")

    if len(X) < 10:
        raise ValueError(f"Insufficient samples after filtering: {len(X)}")

    print(f"Training on {len(X)} samples")

    # Create model
    model = get_model(model_type, **model_kwargs)

    # Hyperparameter tuning
    if tune_hyperparams:
        param_dists = _get_param_distributions(model_type)
        if param_dists:
            print(f"  Tuning hyperparameters ({len(param_dists)} params)...")
            cv_tune = get_cv_splitter(
                features, cv_strategy=cv_strategy, n_splits=min(cv_folds, 3),
            )
            search = RandomizedSearchCV(
                model,
                param_dists,
                n_iter=30,
                cv=cv_tune,
                scoring='r2',
                random_state=42,
                n_jobs=-1,
                error_score='raise',
            )
            search.fit(
                StandardScaler().fit_transform(X) if scale_features else X,
                y,
            )
            model = search.best_estimator_
            print(f"  Best params: {search.best_params_}")
            print(f"  Tuning R²: {search.best_score_:.3f}")

    # Optionally wrap in pipeline with scaling
    if scale_features:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model),
        ])
    else:
        pipeline = model

    # Cross-validation with sample weights
    cv_splitter = get_cv_splitter(
        features, cv_strategy=cv_strategy, n_splits=cv_folds,
    )
    fit_params = {}
    if weights is not None:
        # sklearn cross_validate accepts fit_params for sample_weight
        step_prefix = 'model__' if scale_features else ''
        fit_params[f'{step_prefix}sample_weight'] = weights

    cv_results = cross_validate(
        pipeline,
        X, y,
        cv=cv_splitter,
        scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'],
        return_train_score=True,
        fit_params=fit_params if fit_params else None,
    )

    # Train final model on all data
    if fit_params:
        pipeline.fit(X, y, **fit_params)
    else:
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
        elif hasattr(final_model, 'coef_'):
            # Linear models: use absolute coefficient magnitude
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': np.abs(final_model.coef_),
                'coefficient': final_model.coef_,
            }).sort_values('importance', ascending=False)
    except Exception:
        pass
    
    # Print results
    n_folds = len(cv_results['test_r2'])
    print(f"\nCross-validation results ({n_folds} folds, strategy={cv_strategy}):")
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
    models: list[str] = ['random_forest', 'gradient_boosting', 'ridge', 'lasso', 'elastic_net', 'xgboost', 'lightgbm', 'mlp'],
    cv_folds: int = 5,
    cv_strategy: Literal['random', 'spatial_lon', 'leave_country_out'] = 'random',
    tune_hyperparams: bool = False,
) -> pd.DataFrame:
    """Compare different ML models for correction prediction.

    Args:
        corrections: Training data with features and targets.
        feature_cols: Features to use.
        target_col: Target to predict.
        models: Model types to compare.
        cv_folds: Number of CV folds.
        cv_strategy: Cross-validation strategy.

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
                cv_strategy=cv_strategy,
                tune_hyperparams=tune_hyperparams,
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


def select_important_features(
    corrections: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: Literal['scalar', 'offset'],
    method: str = 'lasso',
    cv_folds: int = 5,
    cv_strategy: Literal['random', 'spatial_lon', 'leave_country_out'] = 'random',
    tune_hyperparams: bool = False,
    top_n: int | None = None,
    threshold: float = 0.0,
) -> tuple[list[str], pd.DataFrame]:
    """Select important features using model-based importance ranking.

    Trains a model on all features, extracts feature importances, and
    returns only the features deemed important.

    For linear models (lasso, elastic_net, ridge), importance is based on
    the absolute value of scaled coefficients. Lasso and Elastic Net
    naturally drive unimportant coefficients to zero.

    For tree-based models, importance is from ``feature_importances_``.

    Args:
        corrections: Training data with features and targets.
        feature_cols: Full set of candidate features.
        target_col: Target variable (``'scalar'`` or ``'offset'``).
        method: Model type to use for ranking features.
        cv_folds: Number of CV folds for evaluation.
        cv_strategy: Cross-validation strategy.
        tune_hyperparams: Whether to tune hyperparameters before ranking.
        top_n: If set, keep only the top N features.
        threshold: Minimum importance to keep (default 0.0 keeps all
            non-zero features for Lasso).

    Returns:
        Tuple of (selected feature names, full importance DataFrame).
    """
    print(f"\n{'='*60}")
    print(f"FEATURE SELECTION ({target_col.upper()})")
    print(f"{'='*60}")
    print(f"Method: {method}")
    print(f"Starting features: {len(feature_cols)}")

    # Train model to get importances
    result = train_correction_model(
        corrections,
        target_col=target_col,
        feature_cols=feature_cols,
        model_type=method,
        cv_folds=cv_folds,
        cv_strategy=cv_strategy,
        tune_hyperparams=tune_hyperparams,
    )

    fi = result.get('feature_importance')
    if fi is None:
        print("  WARNING: No feature importance available, keeping all features")
        return feature_cols, pd.DataFrame()

    # Print full importance ranking
    print(f"\nFeature importance ranking ({method}):")
    for _, row in fi.iterrows():
        coef_str = ''
        if 'coefficient' in fi.columns:
            coef_str = f'  (coef={row["coefficient"]:+.4f})'
        marker = ' *' if row['importance'] <= threshold else ''
        print(f"  {row['feature']:35s}  importance={row['importance']:.4f}{coef_str}{marker}")

    # Select features above threshold
    selected = fi[fi['importance'] > threshold]['feature'].tolist()

    # Optionally limit to top N
    if top_n is not None and len(selected) > top_n:
        selected = fi.head(top_n)['feature'].tolist()

    dropped = [f for f in feature_cols if f not in selected]
    print(f"\nSelected: {len(selected)} features")
    if dropped:
        print(f"Dropped ({len(dropped)}): {dropped}")

    return selected, fi


# =============================================================================
# REGIONAL MODELS
# =============================================================================

def train_regional_models(
    features: pd.DataFrame,
    *,
    target_col: Literal['scalar', 'offset'],
    region_col: str = 'country_code',
    feature_cols: list[str] | None = None,
    model_type: str = 'random_forest',
    min_samples: int = 50,
    cv_strategy: Literal['random', 'spatial_lon', 'leave_country_out'] = 'spatial_lon',
    cv_folds: int = 5,
    fallback_to_global: bool = True,
    **model_kwargs,
) -> dict[str, dict[str, Any]]:
    """Train separate ML models per region.

    Regions with at least ``min_samples`` data points get their own
    model.  Smaller regions fall back to a global model trained on
    all data.

    Args:
        features: Feature matrix with target and region columns.
        target_col: Target column (``"scalar"`` or ``"offset"``).
        region_col: Column identifying regions.
        feature_cols: Feature columns to use.
        model_type: Model type for all regional models.
        min_samples: Minimum samples required for a regional model.
        cv_strategy: CV strategy for evaluation.
        cv_folds: Number of CV folds.
        fallback_to_global: If True, train a global model as fallback.
        **model_kwargs: Passed to ``get_model``.

    Returns:
        Dict mapping region name (or ``"_global"``) to model result dicts.
    """
    results: dict[str, dict[str, Any]] = {}

    if region_col not in features.columns:
        warnings.warn(f"Column '{region_col}' not found. Training global model only.")
        result = train_correction_model(
            features, target_col=target_col, feature_cols=feature_cols,
            model_type=model_type, cv_strategy=cv_strategy,
            cv_folds=cv_folds, **model_kwargs,
        )
        results['_global'] = result
        return results

    regions = features[region_col].unique()
    for region in regions:
        region_data = features[features[region_col] == region]
        if len(region_data) >= min_samples:
            print(f"\n--- Training regional model: {region} ({len(region_data)} samples) ---")
            try:
                result = train_correction_model(
                    region_data, target_col=target_col, feature_cols=feature_cols,
                    model_type=model_type, cv_strategy='random',
                    cv_folds=min(cv_folds, len(region_data) // 10),
                    **model_kwargs,
                )
                results[region] = result
            except Exception as e:
                print(f"  Failed for {region}: {e}")

    # Global fallback model
    if fallback_to_global:
        print(f"\n--- Training global fallback model ({len(features)} samples) ---")
        result = train_correction_model(
            features, target_col=target_col, feature_cols=feature_cols,
            model_type=model_type, cv_strategy=cv_strategy,
            cv_folds=cv_folds, **model_kwargs,
        )
        results['_global'] = result

    return results


def predict_regional(
    regional_models: dict[str, dict[str, Any]],
    features: pd.DataFrame,
    *,
    region_col: str = 'country_code',
) -> np.ndarray:
    """Predict correction factors using regional models.

    Each point is routed to its regional model if one exists,
    otherwise the global fallback model is used.

    Args:
        regional_models: Output from ``train_regional_models``.
        features: Feature matrix with region column.
        region_col: Column identifying regions.

    Returns:
        Array of predictions aligned with ``features`` index.
    """
    predictions = np.full(len(features), np.nan)
    global_model = regional_models.get('_global')

    if region_col in features.columns:
        for region in features[region_col].unique():
            mask = features[region_col] == region
            model_result = regional_models.get(region, global_model)
            if model_result is None:
                continue
            X = features.loc[mask, model_result['feature_cols']].copy()
            X = X.fillna(X.median())
            predictions[mask.values] = model_result['model'].predict(X)
    elif global_model is not None:
        X = features[global_model['feature_cols']].copy()
        X = X.fillna(X.median())
        predictions = global_model['model'].predict(X)

    return predictions


# =============================================================================
# HYBRID IDW + ML
# =============================================================================

def compute_idw_at_points(
    df: pd.DataFrame,
    *,
    target_col: str = 'scalar',
    power: float = 2.0,
    k: int = 10,
    lon_col: str = 'lon',
    lat_col: str = 'lat',
) -> np.ndarray:
    """Compute leave-one-out IDW predictions at each training point.

    For each point, the IDW prediction is computed using the ``k``
    nearest other points.

    Args:
        df: DataFrame with lon, lat, and target columns.
        target_col: Column to interpolate.
        power: IDW distance exponent.
        k: Number of nearest neighbours.
        lon_col: Longitude column name.
        lat_col: Latitude column name.

    Returns:
        Array of LOO-IDW predictions aligned with ``df`` rows.
    """
    coords = df[[lon_col, lat_col]].values
    values = df[target_col].values
    tree = cKDTree(coords)

    predictions = np.full(len(df), np.nan)
    # Query k+1 because the closest point is the point itself
    dists, indices = tree.query(coords, k=k + 1)

    for i in range(len(df)):
        # Skip the self-match (distance ~0)
        nbr_dists = dists[i, 1:]
        nbr_idx = indices[i, 1:]

        # Avoid division by zero
        nbr_dists = np.maximum(nbr_dists, 1e-10)
        weights = 1.0 / nbr_dists ** power
        predictions[i] = np.average(values[nbr_idx], weights=weights)

    return predictions


def train_hybrid_model(
    features: pd.DataFrame,
    *,
    target_col: Literal['scalar', 'offset'],
    feature_cols: list[str] | None = None,
    model_type: str = 'random_forest',
    idw_power: float = 2.0,
    idw_k: int = 10,
    cv_strategy: Literal['random', 'spatial_lon', 'leave_country_out'] = 'spatial_lon',
    cv_folds: int = 5,
    **model_kwargs,
) -> dict[str, Any]:
    """Train a hybrid IDW + ML model on correction residuals.

    First computes leave-one-out IDW predictions, then trains an ML
    model to predict the residual (actual - IDW).  The final
    prediction is IDW + ML_residual.

    Args:
        features: Feature matrix with target and coordinate columns.
        target_col: Target column.
        feature_cols: Feature columns for the ML residual model.
        model_type: ML model type.
        idw_power: IDW distance exponent.
        idw_k: Number of IDW neighbours.
        cv_strategy: CV strategy for evaluation.
        cv_folds: Number of CV folds.
        **model_kwargs: Passed to ``get_model``.

    Returns:
        Dict with ``'ml_result'`` (residual model), ``'idw_params'``,
        and ``'cv_metrics'``.
    """
    # Compute LOO-IDW predictions
    idw_preds = compute_idw_at_points(
        features, target_col=target_col,
        power=idw_power, k=idw_k,
    )

    # Compute residuals
    residuals = features[target_col].values - idw_preds
    residual_col = f'{target_col}_residual'
    features_with_resid = features.copy()
    features_with_resid[residual_col] = residuals
    features_with_resid['idw_prediction'] = idw_preds

    # Train ML on residuals
    print(f"\nTraining hybrid IDW+ML for {target_col}")
    print(f"  IDW MAE: {np.nanmean(np.abs(residuals)):.4f}")
    print(f"  Training ML on {np.isfinite(residuals).sum()} residual samples")

    ml_result = train_correction_model(
        features_with_resid,
        target_col=residual_col,
        feature_cols=feature_cols,
        model_type=model_type,
        cv_strategy=cv_strategy,
        cv_folds=cv_folds,
        **model_kwargs,
    )

    # Evaluate the full hybrid prediction under CV
    cv_splitter = get_cv_splitter(
        features, cv_strategy=cv_strategy, n_splits=cv_folds,
    )

    if isinstance(cv_splitter, KFold):
        splits = list(cv_splitter.split(features))
    else:
        splits = cv_splitter

    from sklearn.metrics import r2_score, mean_absolute_error

    hybrid_r2, hybrid_mae = [], []
    for train_idx, test_idx in splits:
        train_df = features.iloc[train_idx]
        test_df = features.iloc[test_idx]

        # IDW from train points to test points
        train_coords = train_df[['lon', 'lat']].values
        train_vals = train_df[target_col].values
        test_coords = test_df[['lon', 'lat']].values

        tree = cKDTree(train_coords)
        dists, indices = tree.query(test_coords, k=min(idw_k, len(train_coords)))

        idw_test = np.zeros(len(test_df))
        for i in range(len(test_df)):
            d = np.maximum(dists[i], 1e-10)
            w = 1.0 / d ** idw_power
            idw_test[i] = np.average(train_vals[indices[i]], weights=w)

        # ML residual prediction
        X_test = test_df[ml_result['feature_cols']].copy().fillna(
            train_df[ml_result['feature_cols']].median()
        )
        ml_resid_pred = ml_result['model'].predict(X_test)

        hybrid_pred = idw_test + ml_resid_pred
        y_true = test_df[target_col].values

        mask = np.isfinite(y_true) & np.isfinite(hybrid_pred)
        if mask.sum() > 1:
            hybrid_r2.append(r2_score(y_true[mask], hybrid_pred[mask]))
            hybrid_mae.append(mean_absolute_error(y_true[mask], hybrid_pred[mask]))

    hybrid_r2 = np.array(hybrid_r2)
    hybrid_mae = np.array(hybrid_mae)

    print(f"\nHybrid IDW+ML results ({len(splits)} folds, strategy={cv_strategy}):")
    print(f"  R²:  {hybrid_r2.mean():.3f} ± {hybrid_r2.std():.3f}")
    print(f"  MAE: {hybrid_mae.mean():.4f}")

    return {
        'ml_result': ml_result,
        'idw_params': {'power': idw_power, 'k': idw_k},
        'cv_metrics': {
            'r2': hybrid_r2,
            'mae': hybrid_mae,
        },
        'target_col': target_col,
    }


def predict_hybrid_grid(
    hybrid_result: dict[str, Any],
    corrections: pd.DataFrame,
    *,
    grid_nc: Path | str,
    terrain_nc: Path | str | None = None,
    coastline_geojson: Path | str | None = None,
    invariant_nc: Path | str | None = None,
    mask: xr.DataArray | None = None,
) -> xr.DataArray:
    """Apply a trained hybrid IDW+ML model to a spatial grid.

    Computes IDW from training points to each grid cell, then adds
    the ML-predicted residual.

    Args:
        hybrid_result: Output from ``train_hybrid_model``.
        corrections: Training data used for IDW interpolation.
        grid_nc: NetCDF defining the output grid.
        terrain_nc: Terrain NetCDF for feature extraction.
        coastline_geojson: Coastline GeoJSON for distance-to-coast.
        invariant_nc: ERA5 invariant NetCDF.
        mask: Optional boolean mask.

    Returns:
        Predicted corrections on the grid.
    """
    target_col = hybrid_result['target_col']
    idw_params = hybrid_result['idw_params']
    ml_result = hybrid_result['ml_result']

    # Load grid
    grid_ds = xr.open_dataset(grid_nc)
    if 'x' in grid_ds.coords and 'y' in grid_ds.coords:
        grid_ds = grid_ds.rename({'x': 'lon', 'y': 'lat'})
    lon = grid_ds.lon.values
    lat = grid_ds.lat.values

    LON, LAT = np.meshgrid(lon, lat)
    grid_coords = np.column_stack([LON.flatten(), LAT.flatten()])

    # IDW from training points
    train_coords = corrections[['lon', 'lat']].values
    train_vals = corrections[target_col].values
    tree = cKDTree(train_coords)
    k = min(idw_params['k'], len(train_coords))
    dists, indices = tree.query(grid_coords, k=k)

    idw_grid = np.zeros(len(grid_coords))
    for i in range(len(grid_coords)):
        d = np.maximum(dists[i], 1e-10)
        w = 1.0 / d ** idw_params['power']
        idw_grid[i] = np.average(train_vals[indices[i]], weights=w)

    # ML residual prediction
    grid_df = pd.DataFrame({'lon': LON.flatten(), 'lat': LAT.flatten()})
    if terrain_nc is not None or coastline_geojson is not None or invariant_nc is not None:
        grid_df = create_feature_matrix(
            grid_df,
            terrain_nc=terrain_nc,
            coastline_geojson=coastline_geojson,
            invariant_nc=invariant_nc,
        )

    X_grid = grid_df[ml_result['feature_cols']].copy()
    X_grid = X_grid.fillna(X_grid.median())
    ml_resid = ml_result['model'].predict(X_grid)

    # Combine
    hybrid_pred = (idw_grid + ml_resid).reshape(len(lat), len(lon))

    pred_da = xr.DataArray(
        hybrid_pred,
        coords={'lat': lat, 'lon': lon},
        dims=('lat', 'lon'),
        name=target_col,
    )

    if mask is not None:
        pred_da = pred_da.where(mask)

    return pred_da


# =============================================================================
# EXPORT
# =============================================================================

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
    
    onshore_mask = None
    offshore_mask = None
    
    if onshore_mask_geojson:
        grid_ds = xr.open_dataset(grid_nc)
        lon = grid_ds.x.values if 'x' in grid_ds.coords else grid_ds.lon.values
        lat = grid_ds.y.values if 'y' in grid_ds.coords else grid_ds.lat.values
        
        from vwf.extensions.grid.atlite_export import mask_from_geojson_fast
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
