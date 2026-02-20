#!/usr/bin/env python3
"""Enhance terrain dataset with additional features for ML.

This script adds computed features to an existing terrain NetCDF:
- Distance to coastline (using fast cKDTree lookup)
- Sub-grid terrain variance (roughness in larger neighborhood)
- Coastal indicator (within 50km of coast)
- Terrain complexity index
- Aspect categories (cardinal directions)

Usage:
    # Default (North Sea):
    python enhance_terrain_features.py

    # Europe-wide:
    python enhance_terrain_features.py \
        --input-nc input/terrain/terrain_europe_full.nc \
        --output-nc input/terrain/terrain_europe_enhanced.nc \
        --coastline input/terrain/coastlines.geojson
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from scipy.ndimage import generic_filter
from scipy.spatial import cKDTree


def compute_distance_to_coast(lons_2d, lats_2d, coastline_gdf):
    """Compute distance to coast using cKDTree for fast lookup.

    Args:
        lons_2d: 2D array of longitudes (from meshgrid).
        lats_2d: 2D array of latitudes (from meshgrid).
        coastline_gdf: GeoDataFrame with coastline geometries.

    Returns:
        2D array of distances in km.
    """
    # Extract coastline vertices
    coast_points = []
    for geom in coastline_gdf.geometry:
        if geom is None:
            continue
        if geom.geom_type == 'MultiLineString':
            for line in geom.geoms:
                coords = np.array(line.coords)
                coast_points.append(coords[:, :2])
        elif geom.geom_type == 'LineString':
            coords = np.array(geom.coords)
            coast_points.append(coords[:, :2])

    if not coast_points:
        raise ValueError("No coastline vertices found")

    coast_coords = np.vstack(coast_points)  # (N, 2) as (lon, lat)
    print(f"  Using {len(coast_coords)} coastline vertices for cKDTree")

    # Scale coordinates to approximate metres
    lat_mid = np.mean(lats_2d)
    scale_lon = 111.0 * np.cos(np.radians(lat_mid))
    scale_lat = 111.0

    coast_scaled = coast_coords.copy()
    coast_scaled[:, 0] *= scale_lon
    coast_scaled[:, 1] *= scale_lat

    grid_points = np.column_stack([
        lons_2d.flatten() * scale_lon,
        lats_2d.flatten() * scale_lat,
    ])

    tree = cKDTree(coast_scaled)
    distances_km, _ = tree.query(grid_points)

    return distances_km.reshape(lons_2d.shape)


def main():
    parser = argparse.ArgumentParser(
        description='Enhance terrain dataset with additional features for ML',
    )
    parser.add_argument(
        '--input-nc', type=str,
        default='input/terrain/terrain_north_sea_full.nc',
        help='Input terrain NetCDF file',
    )
    parser.add_argument(
        '--output-nc', type=str,
        default=None,
        help='Output enhanced NetCDF file (default: <input>_enhanced.nc)',
    )
    parser.add_argument(
        '--coastline', type=str,
        default='input/terrain/coastlines.geojson',
        help='Coastline GeoJSON file',
    )
    args = parser.parse_args()

    input_nc = Path(args.input_nc)
    coastline_path = Path(args.coastline)

    if args.output_nc is None:
        output_nc = input_nc.parent / (input_nc.stem.replace('_full', '') + '_enhanced.nc')
    else:
        output_nc = Path(args.output_nc)

    print("=" * 70)
    print("Enhance Terrain Features for ML")
    print("=" * 70)

    # Check inputs
    if not input_nc.exists():
        print(f"Error: Terrain NetCDF not found: {input_nc}")
        sys.exit(1)

    if not coastline_path.exists():
        print(f"Error: Coastline not found: {coastline_path}")
        sys.exit(1)

    # Load data
    print("\nLoading data...")
    ds = xr.open_dataset(input_nc)
    coastline = gpd.read_file(coastline_path)

    print(f"  Loaded terrain: {input_nc}")
    print(f"  Shape: {ds.elevation.shape}")
    print(f"  Variables: {list(ds.data_vars)}")
    print(f"  Loaded coastline: {coastline_path}")

    # Create coordinate grids
    lons, lats = np.meshgrid(ds.lon.values, ds.lat.values)

    # ========================================================================
    # Feature 1: Distance to Coastline (cKDTree - fast)
    # ========================================================================

    print("\n" + "-" * 70)
    print("Feature 1: Distance to Coastline (cKDTree)")
    print("-" * 70)

    distances = compute_distance_to_coast(lons, lats, coastline)

    ds['distance_to_coast'] = (('lat', 'lon'), distances)
    ds['distance_to_coast'].attrs['units'] = 'km'
    ds['distance_to_coast'].attrs['description'] = 'Distance to nearest coastline'

    print(f"  Range: [{distances.min():.1f}, {distances.max():.1f}] km")
    print(f"  Mean: {distances.mean():.1f} km")

    # ========================================================================
    # Feature 2: Coastal Indicator
    # ========================================================================

    print("\n" + "-" * 70)
    print("Feature 2: Coastal Indicator")
    print("-" * 70)

    coastal_threshold_km = 50
    is_coastal = (distances < coastal_threshold_km).astype(int)

    ds['is_coastal'] = (('lat', 'lon'), is_coastal)
    ds['is_coastal'].attrs['description'] = (
        f'1 if within {coastal_threshold_km}km of coast, 0 otherwise'
    )

    n_coastal = is_coastal.sum()
    pct_coastal = 100 * n_coastal / is_coastal.size
    print(f"  Coastal points: {n_coastal} ({pct_coastal:.1f}%)")

    # ========================================================================
    # Feature 3: Sub-grid Terrain Variance
    # ========================================================================

    print("\n" + "-" * 70)
    print("Feature 3: Sub-grid Terrain Variance")
    print("-" * 70)

    print("  Computing sub-grid variance (5x5 neighborhood)...")
    elev_vals = ds.elevation.values
    # Handle NaN in elevation for offshore areas
    elev_filled = np.nan_to_num(elev_vals, nan=0.0)
    subgrid_variance = generic_filter(
        elev_filled, np.std, size=5, mode='nearest',
    )

    ds['subgrid_variance'] = (('lat', 'lon'), subgrid_variance)
    ds['subgrid_variance'].attrs['units'] = 'm'
    ds['subgrid_variance'].attrs['description'] = (
        'Std of elevation in 5x5 grid cell neighborhood'
    )

    print(f"  Range: [{subgrid_variance.min():.1f}, {subgrid_variance.max():.1f}] m")
    print(f"  Mean: {subgrid_variance.mean():.1f} m")

    # ========================================================================
    # Feature 4: Terrain Complexity Index
    # ========================================================================

    print("\n" + "-" * 70)
    print("Feature 4: Terrain Complexity Index")
    print("-" * 70)

    def safe_normalize(arr):
        vmin, vmax = np.nanmin(arr), np.nanmax(arr)
        if vmax - vmin == 0:
            return np.zeros_like(arr)
        return (arr - vmin) / (vmax - vmin)

    elevation_norm = safe_normalize(elev_filled)
    slope_norm = safe_normalize(np.nan_to_num(ds.slope.values, nan=0.0))
    roughness_norm = safe_normalize(np.nan_to_num(ds.roughness.values, nan=0.0))
    curvature_norm = safe_normalize(np.abs(np.nan_to_num(ds.curvature.values, nan=0.0)))

    complexity = (
        0.3 * elevation_norm
        + 0.3 * slope_norm
        + 0.25 * roughness_norm
        + 0.15 * curvature_norm
    )

    ds['complexity'] = (('lat', 'lon'), complexity)
    ds['complexity'].attrs['description'] = (
        'Terrain complexity index (0-1, weighted combination of '
        'elevation, slope, roughness, curvature)'
    )

    print(f"  Range: [{complexity.min():.3f}, {complexity.max():.3f}]")
    print(f"  Mean: {complexity.mean():.3f}")

    # ========================================================================
    # Feature 5: Aspect Categories
    # ========================================================================

    print("\n" + "-" * 70)
    print("Feature 5: Aspect Categories")
    print("-" * 70)

    aspect_vals = np.nan_to_num(ds.aspect.values, nan=0.0)
    slope_vals = np.nan_to_num(ds.slope.values, nan=0.0)
    aspect_cats = np.zeros_like(aspect_vals, dtype=int)

    flat_mask = slope_vals < 1.0
    aspect_cats[flat_mask] = 0

    for i in range(8):
        if i == 0:  # North (wrap around)
            mask = ((aspect_vals >= 337.5) | (aspect_vals < 22.5)) & (~flat_mask)
        else:
            angle_min = i * 45 - 22.5
            angle_max = i * 45 + 22.5
            mask = ((aspect_vals >= angle_min) & (aspect_vals < angle_max)) & (~flat_mask)
        aspect_cats[mask] = i + 1

    ds['aspect_category'] = (('lat', 'lon'), aspect_cats)
    ds['aspect_category'].attrs['description'] = (
        'Aspect category: 0=flat, 1=N, 2=NE, 3=E, 4=SE, 5=S, 6=SW, 7=W, 8=NW'
    )

    cat_names = ['Flat', 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    for i in range(9):
        n = (aspect_cats == i).sum()
        pct = 100 * n / aspect_cats.size
        print(f"  {cat_names[i]}: {n} cells ({pct:.1f}%)")

    # ========================================================================
    # Save Enhanced Dataset
    # ========================================================================

    print("\n" + "=" * 70)
    print("Saving Enhanced Dataset")
    print("=" * 70)

    ds.attrs['history'] = (
        ds.attrs.get('history', '')
        + ' | Enhanced with coastline, variance, complexity features'
    )
    ds.attrs['enhanced_date'] = pd.Timestamp.now().strftime('%Y-%m-%d')

    output_nc.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_nc)

    print(f"\n  Saved: {output_nc}")
    print(f"  New variables: distance_to_coast, is_coastal, subgrid_variance, "
          f"complexity, aspect_category")
    print(f"  Total variables: {len(ds.data_vars)}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("Feature Summary Statistics")
    print("=" * 70)

    summary = []
    for var in ds.data_vars:
        vals = ds[var].values
        summary.append({
            'variable': var,
            'min': np.nanmin(vals),
            'max': np.nanmax(vals),
            'mean': np.nanmean(vals),
            'std': np.nanstd(vals),
        })
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("ENHANCEMENT COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
