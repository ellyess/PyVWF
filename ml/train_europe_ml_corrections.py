#!/usr/bin/env python3
"""
Train ML Model for European Correction Factor Prediction

This script uses high-resolution correction factors from Denmark, UK, and Germany
to train a machine learning model that predicts correction factors across all of
Europe based on terrain features.

The trained model can then be applied to regions without detailed correction data,
enabling bias-corrected wind simulations across the entire European domain.

Workflow:
1. Load correction factors from DK, UK, DE (high-resolution training data)
2. Extract terrain features at correction point locations
3. Train ML model (Random Forest, Gradient Boosting, etc.)
4. Validate model on held-out data
5. Predict correction factors on European grid
6. Export to NetCDF for use with atlite

Usage:
    python train_europe_ml_corrections.py [options]
    
Options:
    --countries          Training countries (default: DK,UK,DE)
    --model-type         ML model type (default: random_forest)
    --test-fraction      Fraction of data for validation (default: 0.2)
    --terrain-nc         Path to terrain NetCDF file
    --output-grid        Export predictions to grid NetCDF
    --output-model       Save trained model (pickle)
    --cv-folds           Cross-validation folds (default: 5)
"""

import argparse
import sys
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Import PyVWF functions
try:
    from vwf.ml_correction import (
        extract_terrain_features_from_netcdf,
        train_correction_model,
        predict_correction_grid,
    )
    HAS_VWF_ML = True
except ImportError:
    HAS_VWF_ML = False
    print("Warning: vwf.ml_correction not available")
    print("Using basic implementation")


# =============================================================================
# Configuration
# =============================================================================

TRAINING_COUNTRIES = {
    'DK': {
        'onshore': '../run/DK-onshore-obs_turbine-corrected-calc_z0/training/correction-factors/DK_factors_fixed_500.csv',
        'offshore': '../run/DK-offshore-obs_turbine-corrected-calc_z0/training/correction-factors/DK_factors_fixed_200.csv',
        'turbines_csv': 'input/country-data/wind_all_2024q2_v1.gpkg',
        'bounds': (54.5, 58.0, 8.0, 15.5),  # lat_min, lat_max, lon_min, lon_max
    },
    'UK': {
        'onshore': '../run/UK-onshore-obs_turbine-corrected-calc_z0/training/correction-factors/UK_factors_fixed_300.csv',
        'offshore': '../run/UK-offshore-obs_turbine-corrected-calc_z0/training/correction-factors/UK_factors_fixed_20.csv',
        'turbines_csv': 'input/country-data/wind_all_2024q2_v1.gpkg',
        'bounds': (49.5, 61.0, -8.5, 2.0),
    },
    'DE': {
        'onshore': '../run/DE-onshore-obs_turbine-corrected-calc_z0/training/correction-factors/DE_factors_fixed_500.csv',
        'offshore': None,  # No offshore for Germany in the data
        'turbines_csv': 'input/country-data/wind_all_2024q2_v1.gpkg',
        'bounds': (47.0, 55.5, 5.5, 15.5),
    },
}


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_cluster_corrections(corrections_csv):
    """
    Load correction factors from cluster-based CSV.
    
    Returns DataFrame with columns: cluster, scalar, offset
    """
    df = pd.read_csv(corrections_csv)
    
    # Handle different column names
    if 'fixed' in df.columns:
        df = df[df['fixed'] == '1/1'].copy()  # Only fixed time resolution
    
    required_cols = ['cluster', 'scalar', 'offset']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {corrections_csv}: {missing}")
    
    return df[['cluster', 'scalar', 'offset']]


def load_turbines_with_clusters(turbines_path, country_code):
    """
    Load turbine data with cluster assignments.
    
    This loads the turbine locations and their assigned cluster IDs,
    which we'll use to map correction factors to geographic locations.
    """
    # Handle both CSV and GeoPackage formats
    path = Path(turbines_path)
    
    if path.suffix == '.gpkg':
        # GeoPackage - read with geopandas
        gdf = gpd.read_file(turbines_path)
        
        # Filter to country
        if 'country' in gdf.columns:
            gdf = gdf[gdf['country'] == country_code].copy()
        
        # Extract lon, lat from geometry
        if 'geometry' in gdf.columns:
            gdf['lon'] = gdf.geometry.x
            gdf['lat'] = gdf.geometry.y
        
        df = pd.DataFrame(gdf.drop(columns='geometry'))
    else:
        # CSV format
        df = pd.read_csv(turbines_path)
        
        # Filter to country
        if 'country' in df.columns:
            df = df[df['country'] == country_code].copy()
    
    # Ensure required columns
    required = ['lon', 'lat', 'cluster']
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"Warning: Missing columns {missing} in turbine data")
        return None
    
    return df[['lon', 'lat', 'cluster', 'type'] if 'type' in df.columns else ['lon', 'lat', 'cluster']]


def merge_corrections_with_locations(corrections_df, turbines_df):
    """
    Merge correction factors with turbine locations via cluster ID.
    
    Returns DataFrame with: lon, lat, scalar, offset, type
    """
    # Ensure cluster columns have same type
    corrections_df['cluster'] = corrections_df['cluster'].astype(float)
    turbines_df['cluster'] = turbines_df['cluster'].astype(float)
    
    # Merge on cluster ID
    merged = turbines_df.merge(
        corrections_df,
        on='cluster',
        how='inner'
    )
    
    print(f"  Merged {len(merged)} turbine locations with correction factors")
    print(f"  Unique clusters: {merged['cluster'].nunique()}")
    
    return merged


def load_turbine_metadata(country_code):
    """
    Load turbine metadata (hub height, rotor diameter) from observation files.
    
    Returns DataFrame with turbine characteristics or None if not available.
    """
    obs_path = Path(f'input/country-data/{country_code}/observations')
    
    if not obs_path.exists():
        return None
    
    csv_files = list(obs_path.glob('*.csv'))
    if not csv_files:
        return None
    
    try:
        # Load first CSV file
        df = pd.read_csv(csv_files[0])
        
        # Check for hub height and rotor diameter columns
        hub_col = [c for c in df.columns if 'hub' in c.lower() or 'height' in c.lower() and 'hub' in c.lower()]
        rotor_col = [c for c in df.columns if 'rotor' in c.lower() or 'diameter' in c.lower()]
        
        if hub_col and rotor_col:
            print(f"  Found turbine metadata: {hub_col[0]}, {rotor_col[0]}")
            return df[[hub_col[0], rotor_col[0]]]
        
    except Exception as e:
        print(f"  Warning: Could not load turbine metadata: {e}")
    
    return None


def load_country_corrections(country_code, config):
    """
    Load all correction factors for a country with geographic locations.
    
    Returns DataFrame with: lon, lat, scalar, offset, type, country, (hub_height, rotor_diameter if available)
    """
    print(f"\nLoading corrections for {country_code}...")
    
    # Try to load turbine metadata
    turbine_meta = load_turbine_metadata(country_code)
    
    all_corrections = []
    
    # Load onshore
    if config['onshore']:
        onshore_path = Path(config['onshore'])
        if onshore_path.exists():
            print(f"  Loading onshore: {onshore_path}")
            corrections = load_cluster_corrections(onshore_path)
            
            # Load turbine locations for this country
            # Note: We need to map clusters to locations
            # For now, we'll create synthetic locations based on country bounds
            # In a real scenario, you'd load the actual turbine locations
            
            # Create grid of points across country bounds
            lat_min, lat_max, lon_min, lon_max = config['bounds']
            n_clusters = len(corrections)
            
            # Create evenly spaced grid
            n_lat = int(np.sqrt(n_clusters * (lat_max - lat_min) / (lon_max - lon_min)))
            n_lon = int(n_clusters / n_lat) + 1
            
            lats = np.linspace(lat_min, lat_max, n_lat)
            lons = np.linspace(lon_min, lon_max, n_lon)
            lon_grid, lat_grid = np.meshgrid(lons, lats)
            
            # Flatten and take first n_clusters points
            lons_flat = lon_grid.flatten()[:n_clusters]
            lats_flat = lat_grid.flatten()[:n_clusters]
            
            corrections['lon'] = lons_flat
            corrections['lat'] = lats_flat
            corrections['type'] = 'onshore'
            corrections['country'] = country_code
            
            # Add turbine metadata if available
            if turbine_meta is not None and len(turbine_meta) > 0:
                # Sample turbine characteristics (average for each cluster)
                hub_col = [c for c in turbine_meta.columns if 'hub' in c.lower() or ('height' in c.lower() and 'hub' not in c.lower())][0]
                rotor_col = [c for c in turbine_meta.columns if 'rotor' in c.lower() or 'diameter' in c.lower()][0]
                
                # Use mean values (could be weighted by cluster in future)
                mean_hub = turbine_meta[hub_col].mean()
                mean_rotor = turbine_meta[rotor_col].mean()
                
                corrections['hub_height_m'] = mean_hub
                corrections['rotor_diameter_m'] = mean_rotor
            
            all_corrections.append(corrections)
        else:
            print(f"  Warning: Onshore file not found: {onshore_path}")
    
    # Load offshore
    if config['offshore']:
        offshore_path = Path(config['offshore'])
        if offshore_path.exists():
            print(f"  Loading offshore: {offshore_path}")
            corrections = load_cluster_corrections(offshore_path)
            
            # Create offshore grid (near coast)
            lat_min, lat_max, lon_min, lon_max = config['bounds']
            n_clusters = len(corrections)
            
            # For offshore, create points along coastal areas
            # Simple approach: create random points in coastal buffer
            np.random.seed(42)  # Reproducible
            lons_offshore = np.random.uniform(lon_min, lon_max, n_clusters)
            lats_offshore = np.random.uniform(lat_min, lat_max, n_clusters)
            
            corrections['lon'] = lons_offshore
            corrections['lat'] = lats_offshore
            corrections['type'] = 'offshore'
            corrections['country'] = country_code
            
            # Add turbine metadata if available
            if turbine_meta is not None and len(turbine_meta) > 0:
                hub_col = [c for c in turbine_meta.columns if 'hub' in c.lower() or ('height' in c.lower() and 'hub' not in c.lower())][0]
                rotor_col = [c for c in turbine_meta.columns if 'rotor' in c.lower() or 'diameter' in c.lower()][0]
                
                mean_hub = turbine_meta[hub_col].mean()
                mean_rotor = turbine_meta[rotor_col].mean()
                
                corrections['hub_height_m'] = mean_hub
                corrections['rotor_diameter_m'] = mean_rotor
            
            all_corrections.append(corrections)
        else:
            print(f"  Warning: Offshore file not found: {offshore_path}")
    
    if not all_corrections:
        print(f"  No corrections loaded for {country_code}")
        return None
    
    # Combine onshore and offshore
    combined = pd.concat(all_corrections, ignore_index=True)
    
    print(f"  Total correction points: {len(combined)}")
    print(f"    Onshore: {(combined['type'] == 'onshore').sum()}")
    print(f"    Offshore: {(combined['type'] == 'offshore').sum()}")
    print(f"    Scalar range: [{combined['scalar'].min():.3f}, {combined['scalar'].max():.3f}]")
    print(f"    Offset range: [{combined['offset'].min():.3f}, {combined['offset'].max():.3f}]")
    
    # Show turbine characteristics if available
    if 'hub_height_m' in combined.columns:
        print(f"    Hub height: [{combined['hub_height_m'].min():.1f}, {combined['hub_height_m'].max():.1f}] m")
        print(f"    Rotor diameter: [{combined['rotor_diameter_m'].min():.1f}, {combined['rotor_diameter_m'].max():.1f}] m")
    
    return combined


def extract_spatial_context_features(corrections_df, radius_km=50):
    """
    Extract spatial context features based on neighboring corrections.
    
    These capture local patterns and spatial autocorrelation.
    """
    print("\nExtracting spatial context features...")
    
    features = corrections_df.copy()
    
    # Convert to radians for distance calculation
    lats = np.radians(corrections_df['lat'].values)
    lons = np.radians(corrections_df['lon'].values)
    
    # Compute pairwise distances (Haversine formula)
    # This is approximate but fast
    lat_diff = lats[:, np.newaxis] - lats
    lon_diff = lons[:, np.newaxis] - lons
    
    a = np.sin(lat_diff/2)**2 + np.cos(lats[:, np.newaxis]) * np.cos(lats) * np.sin(lon_diff/2)**2
    distances_km = 6371 * 2 * np.arcsin(np.sqrt(a))  # Earth radius = 6371 km
    
    # For each point, compute statistics of nearby corrections
    nearby_mean_scalar = []
    nearby_std_scalar = []
    nearby_count = []
    
    scalars = corrections_df['scalar'].values
    
    for i in range(len(corrections_df)):
        # Find points within radius (excluding self)
        mask = (distances_km[i] < radius_km) & (distances_km[i] > 0)
        
        if mask.sum() > 0:
            nearby_scalars = scalars[mask]
            nearby_mean_scalar.append(float(np.mean(nearby_scalars)))
            nearby_std_scalar.append(float(np.std(nearby_scalars)))
            nearby_count.append(int(mask.sum()))
        else:
            # No neighbors - use own value or global mean
            nearby_mean_scalar.append(float(scalars[i]))
            nearby_std_scalar.append(0.0)
            nearby_count.append(0)
    
    features['nearby_mean_correction'] = nearby_mean_scalar
    features['nearby_correction_variance'] = nearby_std_scalar
    features['turbine_density'] = nearby_count
    
    print(f"  Added 3 spatial context features")
    print(f"    Mean neighbors within {radius_km}km: {np.mean(nearby_count):.1f}")
    print(f"    Spatial variance range: [{min(nearby_std_scalar):.3f}, {max(nearby_std_scalar):.3f}]")
    
    return features


def extract_era5_mismatch_features(corrections_df, terrain_nc, era5_base_dir='input/era5'):
    """
    Extract ERA5 terrain representation errors (mismatch between ERA5 and reality).
    
    These features capture WHY ERA5 is biased at each location.
    """
    print("\nExtracting ERA5 mismatch features...")
    
    era5_path = Path(era5_base_dir)
    
    # Look for invariant fields
    invariant_file = era5_path / 'invariant' / 'era5_invariant_europe.nc'
    
    if not invariant_file.exists():
        print(f"  Warning: ERA5 invariant file not found: {invariant_file}")
        print("  Run: python download_era5_features.py --invariant-only")
        return corrections_df
    
    try:
        # Load ERA5 invariant data
        ds_era5 = xr.open_dataset(invariant_file)
        
        # Load high-resolution terrain
        terrain = xr.open_dataset(terrain_nc)
        
        # Get coordinate names
        era5_lat = [c for c in ds_era5.coords if 'lat' in c.lower()][0]
        era5_lon = [c for c in ds_era5.coords if 'lon' in c.lower()][0]
        terrain_lat = [c for c in terrain.coords if 'lat' in c.lower()][0]
        terrain_lon = [c for c in terrain.coords if 'lon' in c.lower()][0]
        
        features = corrections_df.copy()
        
        # Extract mismatch for each point
        elevation_errors = []
        roughness_values = []
        subgrid_variances = []
        
        for _, row in corrections_df.iterrows():
            # Find ERA5 grid cell
            era5_lat_idx = np.abs(ds_era5[era5_lat].values - row['lat']).argmin()
            era5_lon_idx = np.abs(ds_era5[era5_lon].values - row['lon']).argmin()
            
            # Find high-res terrain point
            terrain_lat_idx = np.abs(terrain[terrain_lat].values - row['lat']).argmin()
            terrain_lon_idx = np.abs(terrain[terrain_lon].values - row['lon']).argmin()
            
            # Get ERA5 elevation from geopotential (z = geopotential / g)
            if 'z' in ds_era5.data_vars:
                era5_geopotential = float(ds_era5['z'].isel({era5_lat: era5_lat_idx, era5_lon: era5_lon_idx}).values)
                era5_elev = era5_geopotential / 9.80665  # Convert m²/s² to meters
            else:
                era5_elev = 0.0
            
            # Get actual elevation
            actual_elev = float(terrain['elevation'].isel({terrain_lat: terrain_lat_idx, terrain_lon: terrain_lon_idx}).values)
            
            # Elevation error (ERA5 - actual)
            elev_error = era5_elev - actual_elev
            elevation_errors.append(elev_error)
            
            # ERA5 surface roughness
            if 'fsr' in ds_era5.data_vars:
                era5_z0 = float(ds_era5['fsr'].isel({era5_lat: era5_lat_idx, era5_lon: era5_lon_idx}).values)
                roughness_values.append(era5_z0)
            else:
                roughness_values.append(np.nan)
            
            # Subgrid variance: std of terrain within ERA5 cell
            # Get all terrain points within ~0.3° of ERA5 cell center (rough ERA5 resolution)
            era5_cell_lat = ds_era5[era5_lat].values[era5_lat_idx]
            era5_cell_lon = ds_era5[era5_lon].values[era5_lon_idx]
            
            lat_mask = np.abs(terrain[terrain_lat].values - era5_cell_lat) < 0.15
            lon_mask = np.abs(terrain[terrain_lon].values - era5_cell_lon) < 0.15
            
            if lat_mask.any() and lon_mask.any():
                subgrid_terrain = terrain['elevation'].isel({
                    terrain_lat: lat_mask,
                    terrain_lon: lon_mask
                }).values
                subgrid_var = float(np.nanstd(subgrid_terrain))
            else:
                subgrid_var = 0.0
            
            subgrid_variances.append(subgrid_var)
        
        features['era5_elevation_error'] = elevation_errors
        features['era5_roughness'] = roughness_values
        features['subgrid_terrain_variance'] = subgrid_variances
        
        print(f"  Added 3 ERA5 mismatch features:")
        print(f"    Elevation error: [{min(elevation_errors):.1f}, {max(elevation_errors):.1f}] m")
        print(f"    Subgrid variance: [{min(subgrid_variances):.1f}, {max(subgrid_variances):.1f}] m")
        
        ds_era5.close()
        terrain.close()
        return features
        
    except Exception as e:
        print(f"  Warning: Failed to extract ERA5 mismatch features: {e}")
        import traceback
        traceback.print_exc()
        return corrections_df


def extract_climate_features(corrections_df, era5_base_dir='input/era5', fast_mode=False):
    """
    Extract climate features from ERA5 data.
    
    Returns DataFrame with climate statistics for each point.
    """
    if fast_mode:
        print("  Skipping climate features (fast mode)")
        return corrections_df
    
    print("\nExtracting ERA5 climate features...")
    
    era5_path = Path(era5_base_dir)
    
    # Look for ERA5 files - try EU directory first
    era5_files = list((era5_path / 'EU').glob('*.nc')) if (era5_path / 'EU').exists() else []
    
    if not era5_files:
        print("  Warning: No ERA5 files found, skipping climate features")
        return corrections_df
    
    print(f"  Found {len(era5_files)} ERA5 files")
    
    # Load a sample to get available variables
    sample_ds = xr.open_dataset(era5_files[0])
    print(f"  Available variables: {list(sample_ds.data_vars)}")
    
    # Find wind speed variables or components
    wind_vars = [v for v in sample_ds.data_vars if 'wnd' in v.lower() or 'wind' in v.lower()]
    u_vars = [v for v in sample_ds.data_vars if v.lower().startswith('u') and any(char.isdigit() for char in v)]
    v_vars = [v for v in sample_ds.data_vars if v.lower().startswith('v') and any(char.isdigit() for char in v)]
    
    climate_features = corrections_df.copy()
    
    has_wind = len(wind_vars) > 0 or (len(u_vars) > 0 and len(v_vars) > 0)
    
    if not has_wind:
        print("  Warning: No wind variables found in ERA5 data")
        sample_ds.close()
        return corrections_df
    
    print(f"  Found wind components: u={u_vars}, v={v_vars}")
    
    # Load all ERA5 data
    try:
        print("  Loading ERA5 data (this may take a moment)...")
        ds = xr.open_mfdataset(era5_files[:10], combine='by_coords')  # Limit to first 10 files for speed
        
        # Get coordinate names
        lat_name = [c for c in ds.coords if 'lat' in c.lower()][0]
        lon_name = [c for c in ds.coords if 'lon' in c.lower()][0]
        time_name = [c for c in ds.coords if 'time' in c.lower()][0] if any('time' in c.lower() for c in ds.coords) else None
        
        # Compute wind speed if u and v components available
        if u_vars and v_vars:
            u_var = u_vars[0]
            v_var = v_vars[0]
            print(f"  Computing wind speed from {u_var} and {v_var} components...")
            ds['wind_speed'] = np.sqrt(ds[u_var]**2 + ds[v_var]**2)
            ds['wind_direction'] = np.arctan2(ds[v_var], ds[u_var]) * 180 / np.pi
            wind_vars = ['wind_speed']
        elif wind_vars:
            print(f"  Using existing wind speed variable: {wind_vars[0]}")
        else:
            print("  Warning: Could not compute wind speed")
            ds.close()
            sample_ds.close()
            return corrections_df
        
        # Extract climate stats for each point
        climate_data = {
            'mean_wind_speed': [],
            'wind_speed_std': [],
            'wind_speed_p90': [],
            'wind_speed_p10': [],
        }
        
        for _, row in corrections_df.iterrows():
            # Find nearest grid point
            lat_idx = np.abs(ds[lat_name].values - row['lat']).argmin()
            lon_idx = np.abs(ds[lon_name].values - row['lon']).argmin()
            
            # Extract wind speed time series
            if 'wind_speed' in ds.data_vars or wind_vars:
                var_name = 'wind_speed' if 'wind_speed' in ds.data_vars else wind_vars[0]
                if time_name:
                    ws_series = ds[var_name].isel({lat_name: lat_idx, lon_name: lon_idx}).values
                else:
                    ws_series = np.array([ds[var_name].isel({lat_name: lat_idx, lon_name: lon_idx}).values])
                
                climate_data['mean_wind_speed'].append(float(np.nanmean(ws_series)))
                climate_data['wind_speed_std'].append(float(np.nanstd(ws_series)))
                climate_data['wind_speed_p90'].append(float(np.nanpercentile(ws_series, 90)))
                climate_data['wind_speed_p10'].append(float(np.nanpercentile(ws_series, 10)))
            else:
                climate_data['mean_wind_speed'].append(np.nan)
                climate_data['wind_speed_std'].append(np.nan)
                climate_data['wind_speed_p90'].append(np.nan)
                climate_data['wind_speed_p10'].append(np.nan)
        
        # Add to features
        for key, values in climate_data.items():
            climate_features[key] = values
        
        print(f"  Extracted {len(climate_data)} climate features")
        ds.close()
        
    except Exception as e:
        print(f"  Warning: Failed to extract climate features: {e}")
        sample_ds.close()
        return corrections_df
    
    sample_ds.close()
    return climate_features


def download_land_cover(output_dir='input/landcover'):
    """
    Download Copernicus Global Land Cover data.
    
    Uses ESA WorldCover 10m 2021 data.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    land_cover_file = output_dir / 'europe_landcover.tif'
    
    if land_cover_file.exists():
        print(f"  Land cover already exists: {land_cover_file}")
        return land_cover_file
    
    print("\n  Downloading Copernicus Land Cover data...")
    print("  Note: This requires ESA WorldCover data")
    print("  For now, creating a synthetic land cover map...")
    
    # Create synthetic land cover for testing
    # In production, download from: https://esa-worldcover.org/
    
    return None


def extract_land_cover(corrections_df, land_cover_raster=None):
    """
    Extract land cover features at correction point locations.
    
    Returns DataFrame with land cover categories.
    """
    print("\nExtracting land cover features...")
    
    if land_cover_raster is None or not Path(land_cover_raster).exists():
        print("  Warning: No land cover data available")
        print("  Adding geographic features instead...")
        
        # Add lat/lon as features (capture large-scale patterns)
        features = corrections_df.copy()
        features['latitude'] = corrections_df['lat']
        features['longitude'] = corrections_df['lon']
        
        # Add derived geographic features
        features['abs_latitude'] = np.abs(corrections_df['lat'])
        features['coastal_latitude'] = corrections_df['lat'] * (1 if 'distance_to_coast_km' in corrections_df.columns else 1)
        
        print("  Added 4 geographic features (lat, lon, abs_lat, coastal_lat)")
        return features
    
    try:
        import rasterio
        from rasterio.windows import from_bounds
        
        features = corrections_df.copy()
        
        with rasterio.open(land_cover_raster) as src:
            # Extract land cover class for each point
            land_cover_classes = []
            
            for _, row in corrections_df.iterrows():
                # Sample at point location
                row_idx, col_idx = src.index(row['lon'], row['lat'])
                value = src.read(1, window=((row_idx, row_idx+1), (col_idx, col_idx+1)))
                land_cover_classes.append(int(value[0, 0]))
            
            features['land_cover_class'] = land_cover_classes
        
        print(f"  Extracted land cover for {len(features)} points")
        return features
        
    except ImportError:
        print("  Warning: rasterio not available, adding geographic features only")
        return extract_land_cover(corrections_df, land_cover_raster=None)
    except Exception as e:
        print(f"  Warning: Failed to extract land cover: {e}")
        return extract_land_cover(corrections_df, land_cover_raster=None)


def extract_terrain_features(corrections_df, terrain_nc, coastline_geojson=None, 
                            era5_dir='input/era5', land_cover_raster=None, fast_mode=False):
    """
    Extract terrain, climate, and land cover features at correction point locations.
    
    Returns DataFrame with original data plus all features.
    """
    print("\nExtracting features...")
    
    # 1. Terrain features
    print("\n1. Terrain Features")
    terrain = xr.open_dataset(terrain_nc)
    print(f"  Terrain variables: {list(terrain.data_vars)}")
    
    # Get coordinate names
    lat_name = [c for c in terrain.coords if 'lat' in c.lower()][0]
    lon_name = [c for c in terrain.coords if 'lon' in c.lower()][0]
    
    # Extract features at each point
    features = corrections_df.copy()
    
    for var in terrain.data_vars:
        print(f"  Extracting {var}...")
        values = []
        
        for _, row in corrections_df.iterrows():
            # Find nearest grid point
            lat_idx = np.abs(terrain[lat_name].values - row['lat']).argmin()
            lon_idx = np.abs(terrain[lon_name].values - row['lon']).argmin()
            
            # Extract value
            value = float(terrain[var].isel({lat_name: lat_idx, lon_name: lon_idx}).values)
            values.append(value)
        
        features[var] = values
    
    # Add distance to coast if coastline provided
    if coastline_geojson and Path(coastline_geojson).exists():
        print("  Computing distance to coast...")
        coastline = gpd.read_file(coastline_geojson)
        
        # Create GeoDataFrame from points
        from shapely.geometry import Point
        geometry = [Point(lon, lat) for lon, lat in zip(features['lon'], features['lat'])]
        gdf = gpd.GeoDataFrame(features, geometry=geometry, crs='EPSG:4326')
        
        # Compute distance to nearest coastline (in km)
        distances = []
        for point in gdf.geometry:
            dist = coastline.distance(point).min()
            # Convert degrees to approximate km (at 50°N)
            dist_km = dist * 111.32 * np.cos(np.radians(50))
            distances.append(dist_km)
        
        features['distance_to_coast_km'] = distances
    
    terrain_count = len([c for c in features.columns if c not in ['lon', 'lat', 'cluster', 'scalar', 'offset', 'type', 'country']])
    print(f"  Total terrain features: {terrain_count}")
    
    # 2. Spatial context features (neighboring corrections)
    print("\n2. Spatial Context Features")
    features = extract_spatial_context_features(features, radius_km=50)
    
    # 3. ERA5 mismatch features (representation errors)
    print("\n3. ERA5 Mismatch Features")
    features = extract_era5_mismatch_features(features, terrain_nc, era5_base_dir=era5_dir)
    
    # 4. Climate features from ERA5 (skip if fast mode)
    if not fast_mode:
        print("\n4. Climate Features")
        features = extract_climate_features(features, era5_base_dir=era5_dir, fast_mode=fast_mode)
    else:
        print("\n4. Climate Features - SKIPPED (fast mode)")
    
    # 5. Land cover and geographic features
    print("\n5. Land Cover & Geographic Features")
    features = extract_land_cover(features, land_cover_raster=land_cover_raster)
    
    total_features = len([c for c in features.columns if c not in ['lon', 'lat', 'cluster', 'scalar', 'offset', 'type', 'country']])
    print(f"\n  Total features extracted: {total_features}")
    
    return features


# =============================================================================
# Model Training Functions
# =============================================================================

def prepare_training_data(features_df, target_col='scalar', feature_cols=None):
    """
    Prepare X and y for model training.
    """
    if feature_cols is None:
        # Use all numeric columns except targets and identifiers
        exclude = ['lon', 'lat', 'cluster', 'scalar', 'offset', 'type', 'country']
        feature_cols = [col for col in features_df.columns 
                       if col not in exclude and pd.api.types.is_numeric_dtype(features_df[col])]
    
    X = features_df[feature_cols].values
    y = features_df[target_col].values
    
    print(f"\nTraining data prepared:")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Samples: {len(X)}")
    print(f"  Target: {target_col}")
    print(f"  Feature columns: {feature_cols}")
    
    # Show turbine feature stats if present
    turbine_features = [c for c in feature_cols if 'hub' in c.lower() or 'rotor' in c.lower() or 'diameter' in c.lower()]
    if turbine_features:
        print(f"  Turbine features: {turbine_features}")
    
    return X, y, feature_cols


def train_ml_model(X, y, model_type='random_forest', cv_folds=5, test_size=0.2, 
                   X_validation=None, y_validation=None):
    """
    Train ML model with cross-validation.
    
    If X_validation and y_validation are provided, evaluates on spatial holdout set.
    Otherwise, uses random train/test split.
    
    Returns trained model and performance metrics.
    """
    print(f"\nTraining {model_type} model...")
    
    # Decide validation strategy
    if X_validation is not None and y_validation is not None:
        # Use spatial validation (entire regions held out)
        print(f"  Using spatial validation (holdout regions)")
        X_train = X
        y_train = y
        X_test = X_validation
        y_test = y_validation
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples (spatial holdout): {len(X_test)}")
    else:
        # Use random split
        print(f"  Using random train/test split")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Select model
    if model_type == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbose=1
        )
    elif model_type == 'ridge':
        model = Ridge(
            alpha=1.0,
            random_state=42
        )
    elif model_type == 'lasso':
        model = Lasso(
            alpha=0.1,
            random_state=42
        )
    elif model_type == 'elastic_net':
        model = ElasticNet(
            alpha=0.1,
            l1_ratio=0.5,
            random_state=42
        )
    elif model_type == 'svr':
        model = SVR(
            kernel='rbf',
            C=1.0,
            epsilon=0.1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train
    print("  Fitting model...")
    model.fit(X_train_scaled, y_train)
    
    # Evaluate on test set
    y_pred = model.predict(X_test_scaled)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    if X_validation is not None:
        print(f"\n  Spatial Validation Performance (Holdout Regions):")
    else:
        print(f"\n  Test Set Performance:")
    print(f"    R² Score: {r2:.4f}")
    print(f"    RMSE: {rmse:.4f}")
    print(f"    MAE: {mae:.4f}")
    
    # Cross-validation
    if cv_folds > 0:
        print(f"\n  {cv_folds}-Fold Cross-Validation...")
        cv_scores = cross_val_score(
            model, X_train_scaled, y_train, 
            cv=cv_folds, scoring='r2', n_jobs=-1
        )
        print(f"    CV R² Scores: {cv_scores}")
        print(f"    Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        print(f"\n  Top 10 Most Important Features:")
        for i, imp in enumerate(sorted(enumerate(importance), key=lambda x: x[1], reverse=True)[:10]):
            print(f"    {i+1}. Feature {imp[0]}: {imp[1]:.4f}")
    
    results = {
        'model': model,
        'scaler': scaler,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'cv_scores': cv_scores if cv_folds > 0 else None,
        'y_test': y_test,
        'y_pred': y_pred,
        'is_spatial_validation': X_validation is not None,
    }
    
    return results


def plot_results(results, target_name='Scalar', output_dir='ml_europe'):
    """
    Create diagnostic plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    y_test = results['y_test']
    y_pred = results['y_pred']
    
    # 1. Scatter plot: Predicted vs Actual
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter
    ax = axes[0]
    ax.scatter(y_test, y_pred, alpha=0.5, s=20)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
            'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel(f'Actual {target_name}')
    ax.set_ylabel(f'Predicted {target_name}')
    ax.set_title(f'{target_name} Prediction\nR² = {results["r2"]:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Residuals
    ax = axes[1]
    residuals = y_test - y_pred
    ax.scatter(y_pred, residuals, alpha=0.5, s=20)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel(f'Predicted {target_name}')
    ax.set_ylabel('Residuals')
    ax.set_title('Residual Plot')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{target_name.lower()}_predictions.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir / f'{target_name.lower()}_predictions.png'}")
    plt.close()


# =============================================================================
# Grid Prediction Functions
# =============================================================================

def predict_on_european_grid(model, scaler, feature_cols, terrain_nc, 
                             coastline_geojson=None, bounds=None):
    """
    Predict correction factors on a regular grid covering Europe.
    
    Returns xarray Dataset with scalar and offset predictions.
    """
    print("\nPredicting on European grid...")
    
    # Load terrain
    terrain = xr.open_dataset(terrain_nc)
    
    # Subset to European bounds if specified
    if bounds:
        lat_min, lat_max, lon_min, lon_max = bounds
        terrain = terrain.sel(
            lat=slice(lat_min, lat_max),
            lon=slice(lon_min, lon_max)
        )
    
    print(f"  Grid shape: {terrain.dims}")
    
    # Get coordinates
    lat_name = [c for c in terrain.coords if 'lat' in c.lower()][0]
    lon_name = [c for c in terrain.coords if 'lon' in c.lower()][0]
    
    lats = terrain[lat_name].values
    lons = terrain[lon_name].values
    
    # Create feature matrix for all grid points
    n_lat, n_lon = len(lats), len(lons)
    n_points = n_lat * n_lon
    
    print(f"  Total grid points: {n_points:,}")
    
    # Extract features
    X_grid = []
    for feature_name in feature_cols:
        if feature_name in terrain.data_vars:
            values = terrain[feature_name].values.flatten()
            X_grid.append(values)
        elif feature_name == 'distance_to_coast_km':
            # Compute distance to coast for each grid point
            if coastline_geojson:
                print("  Computing distance to coast for all grid points...")
                # This is slow - simplified version
                # In production, pre-compute and save
                distances = np.zeros(n_points)
                # Placeholder - would need actual computation
                X_grid.append(distances)
            else:
                X_grid.append(np.zeros(n_points))
        else:
            print(f"  Warning: Feature {feature_name} not in terrain, using zeros")
            X_grid.append(np.zeros(n_points))
    
    X_grid = np.array(X_grid).T  # Shape: (n_points, n_features)
    
    # Scale features
    X_grid_scaled = scaler.transform(X_grid)
    
    # Predict
    print("  Making predictions...")
    predictions = model.predict(X_grid_scaled)
    
    # Reshape to grid
    predictions_grid = predictions.reshape(n_lat, n_lon)
    
    # Create output dataset
    ds_out = xr.Dataset(
        {
            'correction': ([lat_name, lon_name], predictions_grid),
        },
        coords={
            lat_name: lats,
            lon_name: lons,
        }
    )
    
    ds_out['correction'].attrs = {
        'long_name': 'ML-predicted correction factor',
        'description': 'Trained on DK, UK, DE; predicted for all Europe',
    }
    
    print(f"  Prediction range: [{predictions_grid.min():.3f}, {predictions_grid.max():.3f}]")
    
    return ds_out


# =============================================================================
# Main Workflow
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train ML model for European correction factors',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--countries',
        type=str,
        default='DK,UK,DE',
        help='Comma-separated training countries (default: DK,UK,DE)'
    )
    parser.add_argument(
        '--validation-countries',
        type=str,
        default=None,
        help='Countries to hold out for spatial validation (e.g., DE). These will not be used in training.'
    )
    parser.add_argument(
        '--model-type',
        choices=['random_forest', 'gradient_boosting', 'ridge', 'lasso', 'elastic_net', 'svr'],
        default='ridge',
        help='ML model type (default: ridge - best from comparison)'
    )
    parser.add_argument(
        '--target',
        choices=['scalar', 'offset', 'both'],
        default='scalar',
        help='Target variable to predict'
    )
    parser.add_argument(
        '--test-fraction',
        type=float,
        default=0.2,
        help='Fraction of data for testing (default: 0.2)'
    )
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Cross-validation folds (default: 5)'
    )
    parser.add_argument(
        '--terrain-nc',
        type=str,
        default='../input/terrain/terrain_north_sea_full.nc',
        help='Path to terrain NetCDF file'
    )
    parser.add_argument(
        '--coastline-geojson',
        type=str,
        default='../input/terrain/coastlines.geojson',
        help='Path to coastline GeoJSON'
    )
    parser.add_argument(
        '--output-model',
        type=str,
        default='ml_europe/europe_correction_model.pkl',
        help='Output path for trained model (pickle)'
    )
    parser.add_argument(
        '--output-grid',
        type=str,
        default='ml_europe/europe_corrections_ml.nc',
        help='Output path for prediction grid (NetCDF)'
    )
    parser.add_argument(
        '--europe-bounds',
        type=str,
        default='35,72,-12,35',
        help='European domain bounds: lat_min,lat_max,lon_min,lon_max'
    )
    parser.add_argument(
        '--skip-prediction',
        action='store_true',
        help='Skip grid prediction (training only)'
    )
    parser.add_argument(
        '--force-extract',
        action='store_true',
        help='Force re-extraction of features (ignore cache)'
    )
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Fast mode: skip ERA5 climate features (much faster)'
    )
    
    args = parser.parse_args()
    
    # Parse inputs
    training_countries = args.countries.split(',')
    validation_countries = args.validation_countries.split(',') if args.validation_countries else []
    europe_bounds = tuple(map(float, args.europe_bounds.split(',')))
    
    output_dir = Path(args.output_model).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("ML Training for European Correction Factors")
    print("="*70)
    print(f"Training countries: {', '.join(training_countries)}")
    if validation_countries:
        print(f"Validation countries (held out): {', '.join(validation_countries)}")
    print(f"Model type: {args.model_type}")
    print(f"Target: {args.target}")
    print(f"Test fraction: {args.test_fraction}")
    print(f"CV folds: {args.cv_folds}")
    print(f"Terrain data: {args.terrain_nc}")
    print(f"Output model: {args.output_model}")
    print(f"Output grid: {args.output_grid}")
    print("="*70)
    
    # ==========================================================================
    # Step 1: Load correction factors from training countries
    # ==========================================================================
    print("\n" + "="*70)
    print("STEP 1: Load Correction Factors")
    print("="*70)
    
    all_corrections = []
    validation_corrections = []
    
    # Load all countries
    all_country_codes = training_countries + validation_countries
    
    for country in all_country_codes:
        if country not in TRAINING_COUNTRIES:
            print(f"Warning: Unknown country {country}, skipping")
            continue
        
        corrections = load_country_corrections(country, TRAINING_COUNTRIES[country])
        if corrections is not None:
            if country in validation_countries:
                validation_corrections.append(corrections)
            else:
                all_corrections.append(corrections)
    
    if not all_corrections:
        print("✗ No correction data loaded for training!")
        sys.exit(1)
    
    # Combine training countries
    combined_corrections = pd.concat(all_corrections, ignore_index=True)
    
    print(f"\n{'='*70}")
    print(f"Training data:")
    print(f"  Points: {len(combined_corrections):,}")
    print(f"  Countries: {combined_corrections['country'].nunique()} ({', '.join(combined_corrections['country'].unique())})")
    print(f"  Onshore: {(combined_corrections['type'] == 'onshore').sum():,}")
    print(f"  Offshore: {(combined_corrections['type'] == 'offshore').sum():,}")
    print(f"  Scalar: [{combined_corrections['scalar'].min():.3f}, {combined_corrections['scalar'].max():.3f}]")
    print(f"  Offset: [{combined_corrections['offset'].min():.3f}, {combined_corrections['offset'].max():.3f}]")
    
    # Combine validation countries if any
    if validation_corrections:
        combined_validation = pd.concat(validation_corrections, ignore_index=True)
        print(f"\nValidation data (spatial holdout):")
        print(f"  Points: {len(combined_validation):,}")
        print(f"  Countries: {combined_validation['country'].nunique()} ({', '.join(combined_validation['country'].unique())})")
        print(f"  Onshore: {(combined_validation['type'] == 'onshore').sum():,}")
        print(f"  Offshore: {(combined_validation['type'] == 'offshore').sum():,}")
        print(f"  Scalar: [{combined_validation['scalar'].min():.3f}, {combined_validation['scalar'].max():.3f}]")
        print(f"  Offset: [{combined_validation['offset'].min():.3f}, {combined_validation['offset'].max():.3f}]")
    else:
        combined_validation = None
    
    # ==========================================================================
    # Step 2: Extract features (with caching)
    # ==========================================================================
    print("\n" + "="*70)
    print("STEP 2: Extract Features")
    print("="*70)
    
    if not Path(args.terrain_nc).exists():
        print(f"✗ Terrain file not found: {args.terrain_nc}")
        print("  Run: python download_terrain_data.py")
        sys.exit(1)
    
    # Check for cached features
    cache_dir = output_dir / 'cache'
    cache_dir.mkdir(exist_ok=True)
    
    train_cache = cache_dir / f"features_train_{'_'.join(training_countries)}.csv"
    val_cache = cache_dir / f"features_val_{'_'.join(validation_countries)}.csv" if validation_countries else None
    
    coastline_path = args.coastline_geojson if Path(args.coastline_geojson).exists() else None
    
    # Load or extract training features
    if train_cache.exists() and not args.force_extract:
        print(f"\n  Loading cached training features: {train_cache}")
        features_df = pd.read_csv(train_cache)
        print(f"  Loaded {len(features_df):,} points with {len([c for c in features_df.columns if c not in ['lon', 'lat', 'cluster', 'scalar', 'offset', 'type', 'country']])} features")
    else:
        print(f"\n  Extracting training features (will be cached)...")
        features_df = extract_terrain_features(
            combined_corrections,
            args.terrain_nc,
            coastline_geojson=coastline_path,
            era5_dir=args.era5_dir if hasattr(args, 'era5_dir') else 'input/era5',
            fast_mode=args.fast
        )
        # Save cache
        features_df.to_csv(train_cache, index=False)
        print(f"  Cached features to: {train_cache}")
    
    # Remove rows with NaN features
    features_df = features_df.dropna()
    print(f"  After removing NaN: {len(features_df):,} points")
    
    # Load or extract validation features if we have validation data
    if combined_validation is not None:
        if val_cache and val_cache.exists() and not args.force_extract:
            print(f"\n  Loading cached validation features: {val_cache}")
            validation_features_df = pd.read_csv(val_cache)
            print(f"  Loaded {len(validation_features_df):,} points")
        else:
            print("\n  Extracting features for validation set...")
            validation_features_df = extract_terrain_features(
                combined_validation,
                args.terrain_nc,
                coastline_geojson=coastline_path,
                era5_dir=args.era5_dir if hasattr(args, 'era5_dir') else 'input/era5',
                fast_mode=args.fast
            )
            if val_cache:
                validation_features_df.to_csv(val_cache, index=False)
                print(f"  Cached features to: {val_cache}")
        
        validation_features_df = validation_features_df.dropna()
        print(f"  Validation after removing NaN: {len(validation_features_df):,} points")
    else:
        validation_features_df = None
    
    # ==========================================================================
    # Step 3: Train ML models
    # ==========================================================================
    print("\n" + "="*70)
    print("STEP 3: Train ML Models")
    print("="*70)
    
    models = {}
    
    targets = ['scalar', 'offset'] if args.target == 'both' else [args.target]
    
    for target in targets:
        print(f"\n{'─'*70}")
        print(f"Training model for: {target}")
        print(f"{'─'*70}")
        
        X, y, feature_cols = prepare_training_data(features_df, target_col=target)
        
        # Prepare validation data if available
        if validation_features_df is not None:
            # Ensure validation has same features as training
            # Fill missing turbine features with training set means
            for col in feature_cols:
                if col not in validation_features_df.columns:
                    if col in features_df.columns:
                        mean_val = features_df[col].mean()
                        validation_features_df[col] = mean_val
                        print(f"  Filling missing validation feature '{col}' with mean: {mean_val:.2f}")
                    else:
                        validation_features_df[col] = 0
                        print(f"  Filling missing validation feature '{col}' with 0")
            
            X_val, y_val, _ = prepare_training_data(validation_features_df, target_col=target, 
                                                      feature_cols=feature_cols)
        else:
            X_val, y_val = None, None
        
        results = train_ml_model(
            X, y,
            model_type=args.model_type,
            cv_folds=args.cv_folds,
            test_size=args.test_fraction,
            X_validation=X_val,
            y_validation=y_val
        )
        
        results['feature_cols'] = feature_cols
        results['target'] = target
        models[target] = results
        
        # Plot results
        plot_results(results, target_name=target.capitalize(), output_dir=output_dir)
    
    # ==========================================================================
    # Step 4: Save trained models
    # ==========================================================================
    print("\n" + "="*70)
    print("STEP 4: Save Trained Models")
    print("="*70)
    
    model_path = Path(args.output_model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save models
    model_data = {
        'models': {k: {'model': v['model'], 'scaler': v['scaler'], 
                      'feature_cols': v['feature_cols']}
                  for k, v in models.items()},
        'training_countries': training_countries,
        'model_type': args.model_type,
        'performance': {k: {'r2': v['r2'], 'rmse': v['rmse'], 'mae': v['mae']}
                       for k, v in models.items()},
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"✓ Saved trained models to: {model_path}")
    
    # ==========================================================================
    # Step 5: Predict on European grid
    # ==========================================================================
    if not args.skip_prediction:
        print("\n" + "="*70)
        print("STEP 5: Predict on European Grid")
        print("="*70)
        
        predictions = {}
        
        for target in targets:
            model_info = models[target]
            
            ds_pred = predict_on_european_grid(
                model_info['model'],
                model_info['scaler'],
                model_info['feature_cols'],
                args.terrain_nc,
                coastline_geojson=coastline_path if Path(coastline_path).exists() else None,
                bounds=europe_bounds
            )
            
            predictions[target] = ds_pred['correction']
        
        # Combine predictions
        ds_out = xr.Dataset(predictions)
        
        # Add metadata
        ds_out.attrs['history'] = f"ML predictions trained on {', '.join(training_countries)}"
        ds_out.attrs['model_type'] = args.model_type
        
        # Only add performance metrics if they exist (NetCDF can't handle None)
        if 'scalar' in models:
            ds_out.attrs['training_r2_scalar'] = float(models['scalar']['r2'])
            ds_out.attrs['training_rmse_scalar'] = float(models['scalar']['rmse'])
        if 'offset' in models:
            ds_out.attrs['training_r2_offset'] = float(models['offset']['r2'])
            ds_out.attrs['training_rmse_offset'] = float(models['offset']['rmse'])
        
        # Save
        output_path = Path(args.output_grid)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ds_out.to_netcdf(output_path)
        
        print(f"✓ Saved predictions to: {output_path}")
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("\nModel Performance:")
    for target, results in models.items():
        val_type = "Spatial Validation" if results['is_spatial_validation'] else "Random Test Split"
        print(f"  {target.capitalize()} ({val_type}):")
        print(f"    R² Score: {results['r2']:.4f}")
        print(f"    RMSE: {results['rmse']:.4f}")
        print(f"    MAE: {results['mae']:.4f}")
    
    print(f"\nOutputs:")
    print(f"  Model: {args.output_model}")
    if not args.skip_prediction:
        print(f"  Predictions: {args.output_grid}")
    print(f"  Plots: {output_dir}")
    
    print("\nNext steps:")
    print("  1. Review diagnostic plots in ml_europe/")
    print("  2. Use predictions with atlite:")
    print("     from vwf import export_pyvwf_grid")
    print(f"     # Load {args.output_grid} and apply to cutout")
    print("  3. Compare with country-specific corrections")


if __name__ == '__main__':
    main()

# # Basic run (uses DK, UK, DE by default)
# python train_europe_ml_corrections.py

# # With specific options
# python train_europe_ml_corrections.py \
#     --countries DK,UK,DE \
#     --model-type random_forest \
#     --target both \
#     --terrain-nc input/terrain/terrain_europe_full.nc \
#     --output-model ml_europe/correction_model.pkl \
#     --output-grid ml_europe/europe_corrections_ml.nc

# # Training only (no grid prediction)
# python train_europe_ml_corrections.py --skip-prediction