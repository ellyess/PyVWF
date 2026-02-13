#!/usr/bin/env python3
"""Enhance terrain dataset with additional features for ML.

This script adds computed features to the existing terrain NetCDF:
- Distance to coastline
- Sub-grid terrain variance (roughness in larger neighborhood)
- Coastal indicator (within 50km of coast)
- Terrain complexity index

Much faster than downloading full ETOPO - uses existing synthetic terrain
as baseline and adds derived features.

Usage:
    python ml/enhance_terrain_features.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from scipy.ndimage import generic_filter
from shapely.geometry import Point
from shapely.ops import nearest_points

print("=" * 70)
print("Enhance Terrain Features for ML")
print("=" * 70)

# Paths
TERRAIN_NC = Path("input/terrain/terrain_north_sea_full.nc")
COASTLINE_GEOJSON = Path("input/terrain/coastlines.geojson")
OUTPUT_NC = Path("input/terrain/terrain_north_sea_enhanced.nc")

# Check inputs
if not TERRAIN_NC.exists():
    print(f"✗ Terrain NetCDF not found: {TERRAIN_NC}")
    print("  Run ml/quick_terrain_setup.py first")
    sys.exit(1)

if not COASTLINE_GEOJSON.exists():
    print(f"✗ Coastline not found: {COASTLINE_GEOJSON}")
    print("  Coastlines are needed for distance calculation")
    sys.exit(1)

# Load data
print("\nLoading data...")
ds = xr.open_dataset(TERRAIN_NC)
coastline = gpd.read_file(COASTLINE_GEOJSON)

print(f"✓ Loaded terrain: {TERRAIN_NC}")
print(f"  Shape: {ds.elevation.shape}")
print(f"  Variables: {list(ds.data_vars)}")

print(f"✓ Loaded coastline: {COASTLINE_GEOJSON}")
print(f"  Coastline length: {coastline.geometry.length.sum():.0f} degrees")

# Create coordinate grids
lons, lats = np.meshgrid(ds.lon.values, ds.lat.values)

# ============================================================================
# Feature 1: Distance to Coastline
# ============================================================================

print("\n" + "─" * 70)
print("Feature 1: Distance to Coastline")
print("─" * 70)

# Combine all coastline segments
all_coastlines = coastline.geometry.unary_union

# Calculate distance for each grid point
print("  Computing distances (this may take a minute)...")
distances = np.zeros_like(lons)

for i in range(len(ds.lat)):
    if i % 20 == 0:
        print(f"    Row {i}/{len(ds.lat)}")

    for j in range(len(ds.lon)):
        point = Point(lons[i, j], lats[i, j])
        # Distance in degrees (approximate)
        dist = point.distance(all_coastlines)
        # Convert to km (rough approximation: 1 degree ≈ 111 km at mid-latitudes)
        distances[i, j] = dist * 111.0

ds['distance_to_coast'] = (('lat', 'lon'), distances)
ds['distance_to_coast'].attrs['units'] = 'km'
ds['distance_to_coast'].attrs['description'] = 'Distance to nearest coastline'

print(f"✓ Distance to coast computed")
print(f"    Range: [{distances.min():.1f}, {distances.max():.1f}] km")
print(f"    Mean: {distances.mean():.1f} km")

# ============================================================================
# Feature 2: Coastal Indicator
# ============================================================================

print("\n" + "─" * 70)
print("Feature 2: Coastal Indicator")
print("─" * 70)

coastal_threshold_km = 50
is_coastal = (distances < coastal_threshold_km).astype(int)

ds['is_coastal'] = (('lat', 'lon'), is_coastal)
ds['is_coastal'].attrs['description'] = f'1 if within {coastal_threshold_km}km of coast, 0 otherwise'

n_coastal = is_coastal.sum()
pct_coastal = 100 * n_coastal / is_coastal.size

print(f"✓ Coastal indicator created (threshold = {coastal_threshold_km} km)")
print(f"    Coastal points: {n_coastal} ({pct_coastal:.1f}%)")

# ============================================================================
# Feature 3: Sub-grid Terrain Variance
# ============================================================================

print("\n" + "─" * 70)
print("Feature 3: Sub-grid Terrain Variance")
print("─" * 70)

# Compute standard deviation in larger neighborhood
def compute_std(values):
    return np.std(values)

print("  Computing sub-grid variance (5x5 neighborhood)...")
subgrid_variance = generic_filter(
    ds.elevation.values,
    compute_std,
    size=5,
    mode='nearest'
)

ds['subgrid_variance'] = (('lat', 'lon'), subgrid_variance)
ds['subgrid_variance'].attrs['units'] = 'm'
ds['subgrid_variance'].attrs['description'] = 'Std of elevation in 5x5 grid cell neighborhood'

print(f"✓ Sub-grid variance computed")
print(f"    Range: [{subgrid_variance.min():.1f}, {subgrid_variance.max():.1f}] m")
print(f"    Mean: {subgrid_variance.mean():.1f} m")

# ============================================================================
# Feature 4: Terrain Complexity Index
# ============================================================================

print("\n" + "─" * 70)
print("Feature 4: Terrain Complexity Index")
print("─" * 70)

# Combine multiple terrain features into single complexity metric
# Normalize each component to 0-1 range
elevation_norm = (ds.elevation.values - ds.elevation.values.min()) / (ds.elevation.values.max() - ds.elevation.values.min())
slope_norm = (ds.slope.values - ds.slope.values.min()) / (ds.slope.values.max() - ds.slope.values.min())
roughness_norm = (ds.roughness.values - ds.roughness.values.min()) / (ds.roughness.values.max() - ds.roughness.values.min())
curvature_norm = np.abs(ds.curvature.values)
curvature_norm = (curvature_norm - curvature_norm.min()) / (curvature_norm.max() - curvature_norm.min())

# Weighted combination
complexity = (
    0.3 * elevation_norm +
    0.3 * slope_norm +
    0.25 * roughness_norm +
    0.15 * curvature_norm
)

ds['complexity'] = (('lat', 'lon'), complexity)
ds['complexity'].attrs['description'] = 'Terrain complexity index (0-1, weighted combination of elevation, slope, roughness, curvature)'

print(f"✓ Complexity index computed")
print(f"    Range: [{complexity.min():.3f}, {complexity.max():.3f}]")
print(f"    Mean: {complexity.mean():.3f}")

# ============================================================================
# Feature 5: Elevation Aspect Categories
# ============================================================================

print("\n" + "─" * 70)
print("Feature 5: Aspect Categories")
print("─" * 70)

# Categorize aspect into 8 cardinal directions
# Aspect is in degrees (0-360), where 0=North, 90=East, 180=South, 270=West
aspect_cats = np.zeros_like(ds.aspect.values, dtype=int)

# 0: flat/undefined
flat_mask = ds.slope.values < 1.0
aspect_cats[flat_mask] = 0

# 1-8: N, NE, E, SE, S, SW, W, NW
for i in range(8):
    angle_min = i * 45 - 22.5
    angle_max = i * 45 + 22.5

    if i == 0:  # North (wrap around)
        mask = ((ds.aspect.values >= 337.5) | (ds.aspect.values < 22.5)) & (~flat_mask)
    else:
        mask = ((ds.aspect.values >= angle_min) & (ds.aspect.values < angle_max)) & (~flat_mask)

    aspect_cats[mask] = i + 1

ds['aspect_category'] = (('lat', 'lon'), aspect_cats)
ds['aspect_category'].attrs['description'] = 'Aspect category: 0=flat, 1=N, 2=NE, 3=E, 4=SE, 5=S, 6=SW, 7=W, 8=NW'

print(f"✓ Aspect categories computed")
for i in range(9):
    n = (aspect_cats == i).sum()
    pct = 100 * n / aspect_cats.size
    cat_name = ['Flat', 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'][i]
    print(f"    {cat_name}: {n} cells ({pct:.1f}%)")

# ============================================================================
# Save Enhanced Dataset
# ============================================================================

print("\n" + "=" * 70)
print("Saving Enhanced Dataset")
print("=" * 70)

# Update attributes
ds.attrs['history'] = ds.attrs.get('history', '') + f' | Enhanced with coastline, variance, complexity features'
ds.attrs['enhanced_date'] = pd.Timestamp.now().strftime('%Y-%m-%d')

# Save
OUTPUT_NC.parent.mkdir(parents=True, exist_ok=True)
ds.to_netcdf(OUTPUT_NC)

print(f"\n✓ Saved enhanced terrain: {OUTPUT_NC}")
print(f"\nNew variables added:")
print(f"  - distance_to_coast: Distance to nearest coastline (km)")
print(f"  - is_coastal: Binary indicator (within 50km of coast)")
print(f"  - subgrid_variance: Terrain variability in 5x5 neighborhood (m)")
print(f"  - complexity: Terrain complexity index (0-1)")
print(f"  - aspect_category: Categorical aspect (0=flat, 1-8=cardinal directions)")

print(f"\nTotal variables: {len(ds.data_vars)}")
print(f"Original + 5 new features")

# Summary statistics
print("\n" + "=" * 70)
print("Feature Summary Statistics")
print("=" * 70)

summary = []
for var in ds.data_vars:
    vals = ds[var].values
    summary.append({
        'variable': var,
        'min': vals.min(),
        'max': vals.max(),
        'mean': vals.mean(),
        'std': vals.std(),
    })

summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))

print("\n" + "=" * 70)
print("✓ ENHANCEMENT COMPLETE")
print("=" * 70)
print(f"\nUse enhanced terrain with:")
print(f"  python ml/train_unified_ml_corrections.py \\")
print(f"    --terrain-nc {OUTPUT_NC} \\")
print(f"    --output-dir ml/unified_ml_enhanced")
