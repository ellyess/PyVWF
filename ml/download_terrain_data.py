#!/usr/bin/env python3
"""
Download and Process Terrain Data for PyVWF Atlite Export Case Study

This script downloads terrain features needed for ML-based bias correction
covering the European study area (North Sea region + countries: BE, DE, DK, FR, NL, NO, UK).

Steps:
1. Download ETOPO elevation data (global, includes bathymetry)
2. Download EU-DEM for higher resolution over land (optional)
3. Download coastline data from Natural Earth
4. Process to NetCDF format for PyVWF
5. Compute terrain derivatives (slope, roughness, etc.)
6. Create combined terrain dataset

Usage:
    python download_terrain_data.py [--high-res] [--region europe|north_sea|custom]
    
Options:
    --high-res       Download EU-DEM 25m (large files, ~10GB for Europe)
    --region         Study area: 'europe' (default), 'north_sea', or 'custom'
    --custom-bounds  Custom bounds as "lat_min,lat_max,lon_min,lon_max"
"""

import argparse
import subprocess
import sys
from pathlib import Path
import urllib.request
import zipfile
import gzip
import shutil

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from scipy.ndimage import uniform_filter

try:
    import rasterio
    from rasterio.warp import reproject, Resampling, calculate_default_transform
    from rasterio.crs import CRS
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("Warning: rasterio not installed. High-res EU-DEM processing will not be available.")
    print("Install with: pip install rasterio")


# =============================================================================
# Configuration
# =============================================================================

# Study area definitions
REGIONS = {
    'europe': {
        'name': 'Europe',
        'bounds': (35, 72, -12, 35),  # lat_min, lat_max, lon_min, lon_max
        'description': 'Full European domain (Iceland to Greece, Atlantic to Russia)',
    },
    'north_sea': {
        'name': 'North Sea',
        'bounds': (48, 62, -5, 15),  # Covers UK, Scandinavia, Benelux, Germany
        'description': 'North Sea + surrounding countries (UK, NO, DK, DE, NL, BE)',
    },
    'custom': {
        'name': 'Custom',
        'bounds': None,  # Set via --custom-bounds
        'description': 'User-defined region',
    }
}

# Data URLs
# ETOPO 2022 - use the full global dataset (ice surface)
ETOPO_URL = "https://www.ngdc.noaa.gov/thredds/fileServer/global/ETOPO2022/30s/30s_surface_elev_netcdf/ETOPO_2022_v1_30s_N90W180_surface.nc"
ETOPO_BEDROCK_URL = "https://www.ngdc.noaa.gov/thredds/fileServer/global/ETOPO2022/30s/30s_bed_elev_netcdf/ETOPO_2022_v1_30s_N90W180_bed.nc"
COASTLINE_URL = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip"
LAND_URL = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip"

# Alternative smaller ETOPO (15 arc-second = ~450m, smaller file)
ETOPO_15S_URL = "https://www.ngdc.noaa.gov/thredds/fileServer/global/ETOPO2022/15s/15s_surface_elev_netcdf/ETOPO_2022_v1_15s_N90W180_surface.nc"

# EU-DEM info (requires manual download)
EUDEM_INFO = """
EU-DEM High-Resolution Data (25m):
----------------------------------
For high-resolution terrain data over Europe, download EU-DEM from:
https://land.copernicus.eu/imagery-in-situ/eu-dem/eu-dem-v1.1

Steps:
1. Go to the above link
2. Select your region of interest (or download full mosaic)
3. Register/login (free)
4. Download GeoTIFF tiles
5. Place in input/terrain/eudem/
6. Re-run this script with --high-res flag

EU-DEM provides:
- 25m resolution
- Coverage: EEA39 countries (Europe + Turkey + Iceland)
- More accurate than ETOPO over land
"""


# =============================================================================
# Helper Functions
# =============================================================================

def download_file(url, output_path, description="file"):
    """Download file with progress bar."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists():
        print(f"✓ {description} already exists: {output_path}")
        return output_path
    
    print(f"Downloading {description}...")
    print(f"  URL: {url}")
    print(f"  Output: {output_path}")
    
    def reporthook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\r  Progress: {percent}% ")
        sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, output_path, reporthook)
        print("\n✓ Download complete!")
        return output_path
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        if output_path.exists():
            output_path.unlink()
        return None


def extract_archive(archive_path, extract_to=None):
    """Extract zip or gzip archive."""
    archive_path = Path(archive_path)
    
    if extract_to is None:
        extract_to = archive_path.parent
    else:
        extract_to = Path(extract_to)
        extract_to.mkdir(parents=True, exist_ok=True)
    
    if archive_path.suffix == '.zip':
        print(f"Extracting {archive_path.name}...")
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("✓ Extraction complete!")
        return extract_to
    
    elif archive_path.suffix == '.gz':
        print(f"Extracting {archive_path.name}...")
        output_path = extract_to / archive_path.stem
        with gzip.open(archive_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print("✓ Extraction complete!")
        return output_path
    
    else:
        print(f"Unknown archive format: {archive_path.suffix}")
        return archive_path


def subset_netcdf(nc_path, output_path, bounds):
    """Subset NetCDF to study area bounds."""
    lat_min, lat_max, lon_min, lon_max = bounds
    
    print(f"Subsetting to bounds: lat=[{lat_min}, {lat_max}], lon=[{lon_min}, {lon_max}]")
    
    ds = xr.open_dataset(nc_path)
    
    # Find coordinate names
    lat_name = None
    lon_name = None
    for name in ['lat', 'latitude', 'y']:
        if name in ds.coords:
            lat_name = name
            break
    for name in ['lon', 'longitude', 'x']:
        if name in ds.coords:
            lon_name = name
            break
    
    if lat_name is None or lon_name is None:
        print(f"✗ Could not find lat/lon coordinates in {nc_path}")
        print(f"  Available coordinates: {list(ds.coords)}")
        return None
    
    print(f"  Using coordinates: {lat_name}, {lon_name}")
    
    # Subset
    ds_subset = ds.sel({
        lat_name: slice(lat_min, lat_max),
        lon_name: slice(lon_min, lon_max)
    })
    
    # Save
    print(f"  Saving subset to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds_subset.to_netcdf(output_path)
    ds.close()
    ds_subset.close()
    
    print(f"✓ Subset complete! Shape: {dict(ds_subset.dims)}")
    return output_path


def compute_terrain_derivatives(elevation, resolution_deg=0.00833):
    """
    Compute terrain derivatives from elevation.
    
    Parameters
    ----------
    elevation : xarray.DataArray
        Elevation data with lat/lon coordinates
    resolution_deg : float
        Grid resolution in degrees (default 0.00833 ≈ 1km at 45°N)
    
    Returns
    -------
    xarray.Dataset
        Dataset with elevation, slope, aspect, roughness, curvature
    """
    print("Computing terrain derivatives...")
    
    # Get coordinate names
    dims = elevation.dims
    lat_name = [d for d in dims if 'lat' in d.lower() or d == 'y'][0]
    lon_name = [d for d in dims if 'lon' in d.lower() or d == 'x'][0]
    
    # Get coordinates
    lat = elevation[lat_name].values
    lon = elevation[lon_name].values
    elev = elevation.values
    
    # Average latitude for distance calculation
    lat_avg = np.mean(lat)
    
    # Convert resolution from degrees to meters
    m_per_deg_lat = 111132.954  # meters per degree latitude
    m_per_deg_lon = 111132.954 * np.cos(np.deg2rad(lat_avg))  # varies with latitude
    
    dlat_m = resolution_deg * m_per_deg_lat
    dlon_m = resolution_deg * m_per_deg_lon
    
    # Compute gradients (rate of elevation change)
    grad_lat = np.gradient(elev, dlat_m, axis=0)
    grad_lon = np.gradient(elev, dlon_m, axis=1)
    
    # Slope (in degrees)
    slope_rad = np.arctan(np.sqrt(grad_lat**2 + grad_lon**2))
    slope = np.rad2deg(slope_rad)
    
    # Aspect (direction of slope, 0=North, 90=East, 180=South, 270=West)
    aspect = np.rad2deg(np.arctan2(grad_lon, grad_lat))
    aspect = (90 - aspect) % 360  # Convert to compass bearing
    
    # Roughness (std dev of elevation in 3x3 window)
    kernel_size = 3
    mean_elev = uniform_filter(elev, size=kernel_size, mode='nearest')
    sq_diff = (elev - mean_elev) ** 2
    roughness = np.sqrt(uniform_filter(sq_diff, size=kernel_size, mode='nearest'))
    
    # Curvature (second derivative)
    grad2_lat = np.gradient(grad_lat, dlat_m, axis=0)
    grad2_lon = np.gradient(grad_lon, dlon_m, axis=1)
    curvature = grad2_lat + grad2_lon
    
    # Create dataset
    ds = xr.Dataset(
        {
            'elevation': ([lat_name, lon_name], elev),
            'slope': ([lat_name, lon_name], slope),
            'aspect': ([lat_name, lon_name], aspect),
            'roughness': ([lat_name, lon_name], roughness),
            'curvature': ([lat_name, lon_name], curvature),
        },
        coords={
            lat_name: lat,
            lon_name: lon,
        }
    )
    
    # Add attributes
    ds['elevation'].attrs = {
        'units': 'meters',
        'long_name': 'Elevation above sea level',
        'description': 'Terrain elevation (includes bathymetry for offshore)',
    }
    ds['slope'].attrs = {
        'units': 'degrees',
        'long_name': 'Terrain slope',
        'description': 'Steepness of terrain',
    }
    ds['aspect'].attrs = {
        'units': 'degrees',
        'long_name': 'Terrain aspect',
        'description': 'Direction terrain faces (0=N, 90=E, 180=S, 270=W)',
    }
    ds['roughness'].attrs = {
        'units': 'meters',
        'long_name': 'Terrain roughness',
        'description': 'Standard deviation of elevation in local neighborhood',
    }
    ds['curvature'].attrs = {
        'units': '1/meters',
        'long_name': 'Terrain curvature',
        'description': 'Concavity/convexity of terrain',
    }
    
    print("✓ Terrain derivatives computed!")
    print(f"  Slope range: {float(slope.min()):.1f}° to {float(slope.max()):.1f}°")
    print(f"  Roughness range: {float(roughness.min()):.1f}m to {float(roughness.max()):.1f}m")
    
    return ds


def process_eudem_tiles(eudem_dir, output_path, bounds):
    """
    Process EU-DEM GeoTIFF tiles to NetCDF.
    
    This assumes you've manually downloaded EU-DEM tiles from Copernicus.
    """
    if not HAS_RASTERIO:
        print("✗ rasterio required for EU-DEM processing")
        print("  Install with: pip install rasterio")
        return None
    
    eudem_dir = Path(eudem_dir)
    if not eudem_dir.exists():
        print(f"✗ EU-DEM directory not found: {eudem_dir}")
        print(EUDEM_INFO)
        return None
    
    # Find GeoTIFF files
    tif_files = list(eudem_dir.glob("*.tif")) + list(eudem_dir.glob("*.TIF"))
    if not tif_files:
        print(f"✗ No GeoTIFF files found in {eudem_dir}")
        return None
    
    print(f"Found {len(tif_files)} EU-DEM tiles")
    
    # TODO: Implement mosaic and reproject
    # For now, just notify user
    print("EU-DEM processing not yet implemented in this script.")
    print("Please use GDAL to merge and reproject tiles:")
    print(f"  gdal_merge.py -o merged.tif {eudem_dir}/*.tif")
    print(f"  gdalwarp -t_srs EPSG:4326 -te {bounds[2]} {bounds[0]} {bounds[3]} {bounds[1]} \\")
    print(f"           merged.tif {output_path}")
    
    return None


# =============================================================================
# Main Processing
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Download and process terrain data for PyVWF',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--region',
        choices=['europe', 'north_sea', 'custom'],
        default='north_sea',
        help='Study area region'
    )
    parser.add_argument(
        '--custom-bounds',
        type=str,
        help='Custom bounds as "lat_min,lat_max,lon_min,lon_max"'
    )
    parser.add_argument(
        '--high-res',
        action='store_true',
        help='Process EU-DEM high-resolution data (requires manual download)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../input/terrain',
        help='Output directory for terrain data'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip download steps (use existing data)'
    )
    
    args = parser.parse_args()
    
    # Get region bounds
    if args.region == 'custom':
        if not args.custom_bounds:
            print("✗ Error: --custom-bounds required for custom region")
            print("  Example: --custom-bounds 48,62,-5,15")
            sys.exit(1)
        bounds = tuple(map(float, args.custom_bounds.split(',')))
        region_name = 'Custom'
    else:
        region_info = REGIONS[args.region]
        bounds = region_info['bounds']
        region_name = region_info['name']
    
    lat_min, lat_max, lon_min, lon_max = bounds
    
    print("="*70)
    print("PyVWF Terrain Data Download")
    print("="*70)
    print(f"Region: {region_name}")
    print(f"Bounds: lat=[{lat_min}, {lat_max}], lon=[{lon_min}, {lon_max}]")
    print(f"Output: {args.output_dir}")
    print("="*70)
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =============================================================================
    # Step 1: Download ETOPO elevation data
    # =============================================================================
    print("\n" + "="*70)
    print("STEP 1: Download ETOPO Elevation Data")
    print("="*70)
    print("ETOPO provides:")
    print("  - Global coverage")
    print("  - ~1km resolution (30 arc-second)")
    print("  - Includes bathymetry (offshore elevation)")
    print("  - Ice surface elevation (good for wind studies)")
    print()
    
    etopo_raw = output_dir / "etopo_global.nc"
    etopo_subset = output_dir / f"etopo_{args.region}.nc"
    
    if not args.skip_download:
        # Note: ETOPO is large (~1GB), may take time
        downloaded = download_file(
            ETOPO_URL,
            etopo_raw,
            description="ETOPO 2022 global elevation"
        )
        
        if downloaded is None:
            print("✗ ETOPO download failed")
            print("  You can manually download from:")
            print("  https://www.ncei.noaa.gov/products/etopo-global-relief-model")
            print("  and place in:", etopo_raw)
    
    # Subset to study area
    if etopo_raw.exists() and not etopo_subset.exists():
        subset_netcdf(etopo_raw, etopo_subset, bounds)
    elif etopo_subset.exists():
        print(f"✓ ETOPO subset already exists: {etopo_subset}")
    
    # =============================================================================
    # Step 2: Download coastline data
    # =============================================================================
    print("\n" + "="*70)
    print("STEP 2: Download Coastline Data")
    print("="*70)
    print("Natural Earth coastlines for distance calculations")
    print()
    
    coastline_zip = output_dir / "ne_10m_coastline.zip"
    coastline_dir = output_dir / "coastlines"
    coastline_shp = coastline_dir / "ne_10m_coastline.shp"
    
    if not args.skip_download:
        downloaded = download_file(
            COASTLINE_URL,
            coastline_zip,
            description="Natural Earth coastlines (10m)"
        )
        
        if downloaded:
            extract_archive(coastline_zip, coastline_dir)
    
    if coastline_shp.exists():
        print(f"✓ Coastline data ready: {coastline_shp}")
        
        # Convert to GeoJSON for easier use with PyVWF
        coastline_geojson = output_dir / "coastlines.geojson"
        if not coastline_geojson.exists():
            print("Converting to GeoJSON...")
            gdf = gpd.read_file(coastline_shp)
            # Subset to study area (with buffer)
            gdf_subset = gdf.cx[lon_min-5:lon_max+5, lat_min-5:lat_max+5]
            gdf_subset.to_file(coastline_geojson, driver='GeoJSON')
            print(f"✓ Saved: {coastline_geojson}")
    
    # =============================================================================
    # Step 3: Process EU-DEM (optional)
    # =============================================================================
    if args.high_res:
        print("\n" + "="*70)
        print("STEP 3: Process EU-DEM High-Resolution Data")
        print("="*70)
        
        eudem_dir = output_dir / "eudem"
        eudem_output = output_dir / f"eudem_{args.region}.nc"
        
        process_eudem_tiles(eudem_dir, eudem_output, bounds)
    
    # =============================================================================
    # Step 4: Compute terrain derivatives
    # =============================================================================
    print("\n" + "="*70)
    print("STEP 4: Compute Terrain Derivatives")
    print("="*70)
    print("Computing slope, aspect, roughness, curvature...")
    print()
    
    terrain_output = output_dir / f"terrain_{args.region}_full.nc"
    
    if etopo_subset.exists():
        # Load elevation
        ds = xr.open_dataset(etopo_subset)
        
        # Find elevation variable
        elev_var = None
        for var in ['elevation', 'z', 'Band1']:
            if var in ds:
                elev_var = var
                break
        
        if elev_var is None:
            print(f"✗ Could not find elevation variable in {etopo_subset}")
            print(f"  Available variables: {list(ds.data_vars)}")
        else:
            elevation = ds[elev_var]
            
            # Compute derivatives
            # Resolution: ETOPO is 30 arc-second = 0.00833 degrees
            terrain = compute_terrain_derivatives(elevation, resolution_deg=0.00833)
            
            # Add original attributes
            terrain.attrs = ds.attrs
            terrain.attrs['history'] = f"Terrain derivatives computed by download_terrain_data.py"
            terrain.attrs['region'] = region_name
            terrain.attrs['bounds'] = f"lat=[{lat_min},{lat_max}], lon=[{lon_min},{lon_max}]"
            
            # Save
            print(f"Saving terrain dataset to: {terrain_output}")
            terrain.to_netcdf(terrain_output)
            print("✓ Terrain dataset complete!")
            
            ds.close()
    
    # =============================================================================
    # Step 5: Summary and next steps
    # =============================================================================
    print("\n" + "="*70)
    print("DOWNLOAD COMPLETE!")
    print("="*70)
    print("\nFiles created:")
    
    if etopo_subset.exists():
        print(f"  ✓ Elevation (ETOPO):    {etopo_subset}")
    if terrain_output.exists():
        print(f"  ✓ Terrain (full):       {terrain_output}")
    if coastline_shp.exists():
        print(f"  ✓ Coastlines (shp):     {coastline_shp}")
    if (output_dir / "coastlines.geojson").exists():
        print(f"  ✓ Coastlines (geojson): {output_dir / 'coastlines.geojson'}")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. Use terrain data in PyVWF ML training:")
    print()
    print("   from vwf.ml_correction import create_feature_matrix, train_correction_model")
    print()
    print("   # Create feature matrix from terrain")
    print("   features = create_feature_matrix(")
    print("       corrections_df,")
    print(f"       terrain_nc='{terrain_output}',")
    print(f"       coastline_geojson='{output_dir / 'coastlines.geojson'}',")
    print("   )")
    print()
    print("   # Train model")
    print("   model = train_correction_model(")
    print("       features,")
    print("       target_col='scalar',")
    print("       model_type='random_forest',")
    print("   )")
    print()
    print("2. View data:")
    print()
    print(f"   import xarray as xr")
    print(f"   ds = xr.open_dataset('{terrain_output}')")
    print(f"   print(ds)")
    print()
    print("3. For higher resolution over land:")
    print("   - Download EU-DEM from Copernicus")
    print("   - Place in input/terrain/eudem/")
    print("   - Re-run with --high-res flag")
    print()
    print("4. See examples:")
    print("   - examples/ml_terrain_correction.py")
    print("   - examples/pyvwf_quickstart_denmark.py")
    print()
    print("For more information: see TERRAIN_DATA_GUIDE.md")
    print("="*70)


if __name__ == '__main__':
    main()


# # Quick start - North Sea region (your case study area)
# python download_terrain_data.py

# # Full Europe
# python download_terrain_data.py --region europe

# # Custom area
# python download_terrain_data.py --region custom --custom-bounds "54,58,8,13"

# # Skip re-downloading if you have data
# python download_terrain_data.py --skip-download