#!/usr/bin/env python3
"""Download and prepare CORINE Land Cover data for ML features.

Downloads CORINE Land Cover 2018 (CLC2018) and converts to a
manageable NetCDF for use with the ML correction pipeline.

The CORINE dataset classifies European land cover into 44 classes
at 100m resolution.  This script coarsens to ~0.01 degrees (~1km)
and exports as NetCDF.

Usage:
    python download_corine_data.py --output-dir input/terrain

Notes:
    CORINE data requires manual download from Copernicus due to
    authentication requirements.  This script provides instructions
    and processes the downloaded file.

    Alternative: Use the ERA5 invariant surface roughness (fsr) and
    land-sea mask (lsm) which are already available in the project
    without any download.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

try:
    import rasterio
    from rasterio.warp import reproject, Resampling
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


# CORINE class groupings for binary indicators
CORINE_GROUPS = {
    'urban': list(range(111, 143)),          # 111-142: Artificial surfaces
    'agricultural': list(range(211, 245)),    # 211-244: Agricultural areas
    'forest': list(range(311, 314)),          # 311-313: Forest
    'scrub': list(range(321, 325)),           # 321-324: Scrub/herbaceous
    'bare': list(range(331, 336)),            # 331-335: Open spaces
    'wetland': list(range(411, 424)),         # 411-423: Wetlands
    'water': list(range(511, 524)),           # 511-523: Water bodies
}

# Aerodynamic roughness length (m) by CORINE class group
Z0_LOOKUP = {
    'urban': 1.5,
    'agricultural': 0.1,
    'forest': 1.0,
    'scrub': 0.3,
    'bare': 0.005,
    'wetland': 0.05,
    'water': 0.0002,
}

DOWNLOAD_INSTRUCTIONS = """
CORINE Land Cover 2018 - Download Instructions
================================================

The CORINE dataset requires manual download from Copernicus:

1. Go to: https://land.copernicus.eu/en/products/corine-land-cover/clc2018
2. Register/login (free account required)
3. Download the 100m raster (GeoTIFF format)
   File: U2018_CLC2018_V2020_20u1.tif (~4GB)
4. Place the file in: {output_dir}/
5. Re-run this script with:
   python download_corine_data.py --input-tif {output_dir}/U2018_CLC2018_V2020_20u1.tif

Alternative (smaller): Download from the Copernicus Climate Data Store:
   https://cds.climate.copernicus.eu/cdsapp#!/dataset/satellite-land-cover
   (Global land cover at 300m, available as NetCDF)
"""


def process_corine_tif(
    input_tif: Path,
    output_nc: Path,
    bounds: tuple = (35, 72, -12, 35),
    target_res: float = 0.01,
):
    """Process CORINE GeoTIFF to coarsened NetCDF.

    Args:
        input_tif: Path to CORINE GeoTIFF.
        output_nc: Output NetCDF path.
        bounds: (lat_min, lat_max, lon_min, lon_max).
        target_res: Target resolution in degrees.
    """
    if not HAS_RASTERIO:
        print("Error: rasterio is required to process GeoTIFF files.")
        print("Install with: pip install rasterio")
        sys.exit(1)

    lat_min, lat_max, lon_min, lon_max = bounds

    print(f"Processing CORINE GeoTIFF: {input_tif}")
    print(f"  Bounds: lat=[{lat_min},{lat_max}], lon=[{lon_min},{lon_max}]")
    print(f"  Target resolution: {target_res} degrees")

    with rasterio.open(input_tif) as src:
        # Create target grid
        lons = np.arange(lon_min, lon_max, target_res)
        lats = np.arange(lat_max, lat_min, -target_res)  # top to bottom

        from rasterio.transform import from_bounds
        from rasterio.windows import from_bounds as window_from_bounds

        # Read and reproject to target grid
        dst_transform = from_bounds(lon_min, lat_min, lon_max, lat_max, len(lons), len(lats))

        dst_data = np.empty((len(lats), len(lons)), dtype=np.uint8)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_data,
            dst_transform=dst_transform,
            dst_crs='EPSG:4326',
            resampling=Resampling.nearest,  # nearest for categorical
        )

    # Create xarray Dataset
    ds = xr.Dataset(
        {'land_cover_class': (['lat', 'lon'], dst_data.astype(np.int16))},
        coords={'lat': lats, 'lon': lons},
    )
    ds['land_cover_class'].attrs = {
        'long_name': 'CORINE Land Cover 2018 class',
        'source': 'Copernicus CORINE CLC2018',
    }

    # Add binary indicators and roughness
    for group_name, classes in CORINE_GROUPS.items():
        indicator = np.isin(dst_data, classes).astype(np.int8)
        ds[f'is_{group_name}'] = (['lat', 'lon'], indicator)

    # Roughness from land cover
    roughness = np.full_like(dst_data, 0.1, dtype=np.float32)
    for group_name, classes in CORINE_GROUPS.items():
        mask = np.isin(dst_data, classes)
        roughness[mask] = Z0_LOOKUP[group_name]
    ds['roughness_from_lc'] = (['lat', 'lon'], roughness)
    ds['roughness_from_lc'].attrs = {'units': 'm', 'long_name': 'Roughness length from land cover'}

    ds.attrs = {
        'title': 'CORINE Land Cover 2018 (coarsened)',
        'source': str(input_tif),
        'resolution_deg': target_res,
    }

    output_nc.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_nc)
    print(f"  Saved: {output_nc}")
    print(f"  Grid: {len(lats)} x {len(lons)}")


def process_cci_netcdf(
    input_nc: Path,
    output_nc: Path,
    bounds: tuple = (35, 72, -12, 35),
):
    """Process ESA CCI/C3S land cover NetCDF to subset.

    Works with the C3S/CCI land cover dataset available from the
    Copernicus Climate Data Store as NetCDF.

    Args:
        input_nc: Path to CCI land cover NetCDF.
        output_nc: Output NetCDF path.
        bounds: (lat_min, lat_max, lon_min, lon_max).
    """
    lat_min, lat_max, lon_min, lon_max = bounds

    print(f"Processing CCI land cover: {input_nc}")
    ds = xr.open_dataset(input_nc)

    # Standardize coordinate names
    rename = {}
    if 'latitude' in ds.coords:
        rename['latitude'] = 'lat'
    if 'longitude' in ds.coords:
        rename['longitude'] = 'lon'
    if rename:
        ds = ds.rename(rename)

    # Subset to bounds
    ds = ds.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))

    # Find the land cover variable
    lc_var = None
    for v in ['lccs_class', 'lc', 'land_cover_class', 'Band1']:
        if v in ds:
            lc_var = v
            break

    if lc_var is None:
        print(f"  Error: No land cover variable found. Available: {list(ds.data_vars)}")
        return

    # Rename to standard name
    if lc_var != 'land_cover_class':
        ds = ds.rename({lc_var: 'land_cover_class'})

    # Squeeze time if present
    if 'time' in ds.dims:
        ds = ds.isel(time=0)

    output_nc.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_nc)
    print(f"  Saved: {output_nc}")


def main():
    parser = argparse.ArgumentParser(
        description='Download and prepare CORINE land cover data',
    )
    parser.add_argument(
        '--input-tif', type=str, default=None,
        help='Path to downloaded CORINE GeoTIFF',
    )
    parser.add_argument(
        '--input-nc', type=str, default=None,
        help='Path to CCI/C3S land cover NetCDF (alternative)',
    )
    parser.add_argument(
        '--output-dir', type=str, default='input/terrain',
        help='Output directory',
    )
    parser.add_argument(
        '--resolution', type=float, default=0.01,
        help='Target resolution in degrees (default: 0.01)',
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_nc = output_dir / 'corine_europe.nc'

    print("=" * 70)
    print("CORINE Land Cover Data Preparation")
    print("=" * 70)

    if args.input_tif:
        input_path = Path(args.input_tif)
        if input_path.exists():
            process_corine_tif(input_path, output_nc, target_res=args.resolution)
        else:
            print(f"Error: File not found: {input_path}")
            sys.exit(1)
    elif args.input_nc:
        input_path = Path(args.input_nc)
        if input_path.exists():
            process_cci_netcdf(input_path, output_nc)
        else:
            print(f"Error: File not found: {input_path}")
            sys.exit(1)
    else:
        print(DOWNLOAD_INSTRUCTIONS.format(output_dir=output_dir))
        print("\nNo input file provided. See instructions above.")
        print("After downloading, run:")
        print(f"  python {__file__} --input-tif <path_to_corine.tif>")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
