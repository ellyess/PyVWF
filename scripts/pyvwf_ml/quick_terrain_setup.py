#!/usr/bin/env python3
"""Create a lightweight terrain dataset for ML workflows.

Creates a basic terrain file from existing ERA5 data or a lightweight
alternative (synthetic/GEBCO-style subset). This is faster than downloading
full ETOPO (1GB+).

Usage:
    python quick_terrain_setup.py [--region north_sea|europe]
"""

import argparse
from pathlib import Path
import numpy as np
import xarray as xr
import geopandas as gpd
from scipy.ndimage import uniform_filter
import urllib.request
import sys

# Regions
REGIONS = {
    'north_sea': {
        'bounds': (48, 62, -5, 15),
        'name': 'North Sea',
    },
    'europe': {
        'bounds': (35, 72, -12, 35),
        'name': 'Europe',
    },
}

COASTLINE_URL = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip"


def create_synthetic_terrain(bounds, resolution_deg=0.1):
    """Create a synthetic terrain dataset.

    Args:
        bounds: Tuple of (lat_min, lat_max, lon_min, lon_max).
        resolution_deg: Grid resolution in degrees.

    Returns:
        Dataset with elevation and derivative features.

    Notes:
        This is a placeholder; real DEM data should be used for production.
    """
    lat_min, lat_max, lon_min, lon_max = bounds
    
    # Create grid
    lats = np.arange(lat_min, lat_max, resolution_deg)
    lons = np.arange(lon_min, lon_max, resolution_deg)
    
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Simple synthetic elevation model
    # - Decreases towards coast (simplified)
    # - Some random variation
    
    print("Creating synthetic terrain (rough estimates)...")
    
    # Distance from center
    lat_center = (lat_min + lat_max) / 2
    lon_center = (lon_min + lon_max) / 2
    
    dist_from_center = np.sqrt(
        (lat_grid - lat_center)**2 + (lon_grid - lon_center)**2
    )
    
    # Base elevation (higher inland, lower near coasts)
    elevation = 100 - dist_from_center * 20
    
    # Add some random variation
    np.random.seed(42)
    elevation += np.random.randn(*elevation.shape) * 50
    
    # Ensure reasonable values
    elevation = np.clip(elevation, -100, 1000)
    
    # Compute derivatives
    ds = compute_terrain_derivatives(lats, lons, elevation, resolution_deg)
    
    return ds


def compute_terrain_derivatives(lats, lons, elevation, resolution_deg=0.1):
    """Compute slope, aspect, roughness, and curvature from elevation.

    Args:
        lats: 1D latitude array.
        lons: 1D longitude array.
        elevation: 2D elevation array.
        resolution_deg: Grid resolution in degrees.

    Returns:
        Dataset with elevation and derivative fields.
    """
    print("Computing terrain derivatives...")
    
    # Average latitude for distance calculation
    lat_avg = np.mean(lats)
    
    # Convert resolution from degrees to meters
    m_per_deg_lat = 111132.954
    m_per_deg_lon = 111132.954 * np.cos(np.deg2rad(lat_avg))
    
    dlat_m = resolution_deg * m_per_deg_lat
    dlon_m = resolution_deg * m_per_deg_lon
    
    # Compute gradients
    grad_lat = np.gradient(elevation, dlat_m, axis=0)
    grad_lon = np.gradient(elevation, dlon_m, axis=1)
    
    # Slope (in degrees)
    slope_rad = np.arctan(np.sqrt(grad_lat**2 + grad_lon**2))
    slope = np.rad2deg(slope_rad)
    
    # Aspect (direction of slope)
    aspect = np.rad2deg(np.arctan2(grad_lon, grad_lat))
    aspect = (90 - aspect) % 360
    
    # Roughness (std dev in 3x3 window)
    mean_elev = uniform_filter(elevation, size=3, mode='nearest')
    sq_diff = (elevation - mean_elev) ** 2
    roughness = np.sqrt(uniform_filter(sq_diff, size=3, mode='nearest'))
    
    # Curvature (second derivative)
    grad2_lat = np.gradient(grad_lat, dlat_m, axis=0)
    grad2_lon = np.gradient(grad_lon, dlon_m, axis=1)
    curvature = grad2_lat + grad2_lon
    
    # Create dataset
    ds = xr.Dataset(
        {
            'elevation': (['lat', 'lon'], elevation),
            'slope': (['lat', 'lon'], slope),
            'aspect': (['lat', 'lon'], aspect),
            'roughness': (['lat', 'lon'], roughness),
            'curvature': (['lat', 'lon'], curvature),
        },
        coords={
            'lat': lats,
            'lon': lons,
        }
    )
    
    # Add attributes
    ds['elevation'].attrs = {
        'units': 'meters',
        'long_name': 'Elevation',
        'description': 'Synthetic terrain (for initial ML training)',
    }
    ds['slope'].attrs = {'units': 'degrees', 'long_name': 'Terrain slope'}
    ds['aspect'].attrs = {'units': 'degrees', 'long_name': 'Terrain aspect'}
    ds['roughness'].attrs = {'units': 'meters', 'long_name': 'Terrain roughness'}
    ds['curvature'].attrs = {'units': '1/meters', 'long_name': 'Terrain curvature'}
    
    print(f"  Slope range: {float(slope.min()):.1f}° to {float(slope.max()):.1f}°")
    print(f"  Elevation range: {float(elevation.min()):.1f}m to {float(elevation.max()):.1f}m")
    
    return ds


def download_coastlines(output_dir):
    """Download Natural Earth coastlines as GeoJSON.

    Args:
        output_dir: Directory to store the coastlines data.

    Returns:
        Path to the saved GeoJSON file, or None on failure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    coastline_geojson = output_dir / "coastlines.geojson"
    
    if coastline_geojson.exists():
        print(f"✓ Coastlines already exist: {coastline_geojson}")
        return coastline_geojson
    
    import zipfile
    import tempfile
    
    print("Downloading Natural Earth coastlines...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "coastlines.zip"
        
        # Download
        try:
            urllib.request.urlretrieve(COASTLINE_URL, zip_path)
        except Exception as e:
            print(f"Warning: Could not download coastlines: {e}")
            return None
        
        # Extract
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        
        # Find shapefile
        shp_file = list(Path(tmpdir).glob("*.shp"))[0]
        
        # Convert to GeoJSON
        gdf = gpd.read_file(shp_file)
        gdf.to_file(coastline_geojson, driver='GeoJSON')
    
    print(f"✓ Saved coastlines: {coastline_geojson}")
    return coastline_geojson


def main():
    """Run the quick terrain setup workflow."""
    parser = argparse.ArgumentParser(description='Quick terrain setup')
    parser.add_argument(
        '--region',
        choices=['north_sea', 'europe'],
        default='north_sea',
        help='Region to create terrain for'
    )
    parser.add_argument(
        '--resolution',
        type=float,
        default=0.1,
        help='Grid resolution in degrees (default: 0.1 = ~10km)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../input/terrain',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    region_info = REGIONS[args.region]
    bounds = region_info['bounds']
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Quick Terrain Setup")
    print("="*70)
    print(f"Region: {region_info['name']}")
    print(f"Bounds: lat=[{bounds[0]}, {bounds[1]}], lon=[{bounds[2]}, {bounds[3]}]")
    print(f"Resolution: {args.resolution}° (~{args.resolution * 111:.0f}km)")
    print(f"Output: {output_dir}")
    print("="*70)
    print()
    
    # Create synthetic terrain
    ds = create_synthetic_terrain(bounds, resolution_deg=args.resolution)
    
    # Save
    output_nc = output_dir / f"terrain_{args.region}_full.nc"
    ds.attrs['history'] = "Created by quick_terrain_setup.py"
    ds.attrs['region'] = region_info['name']
    ds.attrs['note'] = "Synthetic terrain for initial ML training"
    
    print(f"\nSaving terrain dataset: {output_nc}")
    ds.to_netcdf(output_nc)
    print("✓ Terrain dataset saved!")
    
    # Download coastlines
    print("\nDownloading coastlines...")
    coastline_file = download_coastlines(output_dir)
    
    # Summary
    print("\n" + "="*70)
    print("SETUP COMPLETE!")
    print("="*70)
    print(f"\nFiles created:")
    print(f"  ✓ Terrain: {output_nc}")
    if coastline_file:
        print(f"  ✓ Coastlines: {coastline_file}")
    
    print(f"\nYou can now run:")
    print(f"  python train_europe_ml_corrections.py \\")
    print(f"      --terrain-nc {output_nc} \\")
    if coastline_file:
        print(f"      --coastline-geojson {coastline_file}")
    
    print("\n" + "="*70)
    print("NOTE: This uses synthetic terrain data")
    print("="*70)
    print("For better results:")
    print("  1. Run: python download_terrain_data.py (downloads real ETOPO)")
    print("     (This takes time as ETOPO is ~1GB)")
    print("  2. Or manually download EU-DEM from Copernicus")
    print("\nThe synthetic terrain is OK for initial testing and")
    print("to see if terrain features help explain your corrections.")
    print("="*70)


if __name__ == '__main__':
    main()
