#!/usr/bin/env python3
"""Combine ERA5 wind files with roughness into single optimized files per year.

This script combines:
1. u10, v10 (10-meter winds)
2. u100, v100 (100-meter winds)
3. z0 (surface roughness length) - optional

Into single files per year with optimized encoding for faster I/O.

Usage:
    python combine_era5_files.py --years 2019 2020 2021
    python combine_era5_files.py --all-years  # Process all available years
    python combine_era5_files.py --years 2019 --add-roughness --roughness-source terrain
"""

import argparse
import xarray as xr
import numpy as np
from pathlib import Path
from datetime import datetime


def find_available_years(era5_dir):
    """Find all available years in ERA5 directory.

    Args:
        era5_dir: Path to ERA5 directory.

    Returns:
        List of available years.
    """
    u10_files = list(era5_dir.glob("era5_u10_v10_*_EU.nc"))
    years = []
    for f in u10_files:
        # Extract year from filename
        parts = f.stem.split('_')
        year = int(parts[3])
        years.append(year)
    return sorted(years)


def load_era5_winds(era5_dir, year):
    """Load u10/v10 and u100/v100 for a given year.

    Args:
        era5_dir: Path to ERA5 directory.
        year: Year to load.

    Returns:
        Tuple of (ds_10m, ds_100m) datasets.
    """
    u10_file = era5_dir / f"era5_u10_v10_{year}_months01-12_EU.nc"
    u100_file = era5_dir / f"era5_u100_v100_{year}_months01-12_EU.nc"

    if not u10_file.exists():
        raise FileNotFoundError(f"10m wind file not found: {u10_file}")
    if not u100_file.exists():
        raise FileNotFoundError(f"100m wind file not found: {u100_file}")

    print(f"  Loading 10m winds: {u10_file.name}")
    ds_10m = xr.open_dataset(u10_file)

    print(f"  Loading 100m winds: {u100_file.name}")
    ds_100m = xr.open_dataset(u100_file)

    return ds_10m, ds_100m


def calculate_roughness_from_winds(combined_ds):
    """Calculate surface roughness length (z0) from 10m and 100m winds.

    Uses PyVWF method: derives z0 from logarithmic wind profile shear.

    Args:
        combined_ds: Combined ERA5 dataset with u10, v10, u100, v100.

    Returns:
        Dataset with z0 variable added.

    Method:
        From logarithmic wind profile:
            u(z) = (u*/k) * ln(z/z0)

        Solving for z0 from winds at two heights (10m and 100m):
            z0 = exp[(wnd100m * ln(10) - wnd10m * ln(100)) / (wnd100m - wnd10m)]
    """
    print("  Calculating roughness length from wind shear (PyVWF method)...")

    # Calculate wind speeds
    wnd10m = np.sqrt(combined_ds['u10']**2 + combined_ds['v10']**2)
    wnd100m = np.sqrt(combined_ds['u100']**2 + combined_ds['v100']**2)

    # Clip to avoid division by zero
    wnd10m = wnd10m.clip(min=1e-4)
    wnd100m = wnd100m.clip(min=1e-4)

    # Calculate z0 from logarithmic profile
    num = wnd100m * np.log(10) - wnd10m * np.log(100)
    denom = wnd100m - wnd10m

    # Mask near-zero shear (avoid divide-by-zero)
    denom = denom.where(np.abs(denom) > 1e-4)

    z0_log = num / denom

    # Physical constraint: log(z0) < 0  →  z0 < 1 m
    z0_log = z0_log.where(z0_log < 0)

    # Backward fill missing values in time
    z0_log = z0_log.bfill("time")

    # Clip to realistic roughness range [1e-6 m, 2.0 m]
    z0_log = z0_log.clip(min=np.log(1e-6), max=np.log(2.0))

    # Convert from log space
    z0 = np.exp(z0_log)
    z0 = z0.clip(min=1e-6, max=2.0)

    # Time-average for representative roughness
    z0_mean = z0.mean(dim='time')

    # Add to dataset
    combined_ds['z0'] = z0_mean
    combined_ds['z0'].attrs = {
        'long_name': 'Surface roughness length',
        'units': 'm',
        'method': 'PyVWF: derived from 10m-100m wind shear',
        'description': 'Roughness length (z0) from logarithmic wind profile',
        'valid_range': '1e-6 to 2.0 m',
        'temporal_aggregation': 'time-averaged'
    }

    print(f"    z0 range: [{float(z0_mean.min()):.6f}, {float(z0_mean.max()):.6f}] m")

    return combined_ds


def add_constant_roughness(combined_ds, z0_value=0.03):
    """Add constant surface roughness.

    Args:
        combined_ds: Combined ERA5 dataset.
        z0_value: Constant roughness length (m).

    Returns:
        Dataset with z0 variable added.
    """
    print(f"  Adding constant roughness z0={z0_value}m")
    z0_const = np.full(
        (len(combined_ds.latitude), len(combined_ds.longitude)),
        z0_value
    )
    combined_ds['z0'] = (
        ('latitude', 'longitude'),
        z0_const,
        {
            'long_name': 'Surface roughness length',
            'units': 'm',
            'description': f'Constant roughness length = {z0_value}m'
        }
    )
    return combined_ds


def combine_era5_year(era5_dir, year, add_roughness=False, roughness_source='pyvwf',
                      z0_constant=0.03, output_dir=None):
    """Combine ERA5 files for a single year.

    Args:
        era5_dir: Path to ERA5 directory.
        year: Year to process.
        add_roughness: Whether to add roughness length.
        roughness_source: 'pyvwf' (from wind shear) or 'constant'.
        z0_constant: Constant roughness value (m) if using 'constant' source.
        output_dir: Output directory (default: same as era5_dir).

    Returns:
        Path to output file.
    """
    print(f"\nProcessing year {year}...")
    print("="*60)

    # Load wind data
    ds_10m, ds_100m = load_era5_winds(era5_dir, year)

    # Combine datasets
    print("  Combining 10m and 100m winds...")
    combined_ds = xr.Dataset({
        'u10': ds_10m['u10'],
        'v10': ds_10m['v10'],
        'u100': ds_100m['u100'],
        'v100': ds_100m['v100'],
    })

    # Rename time coordinate if needed
    if 'valid_time' in combined_ds.coords:
        combined_ds = combined_ds.rename({'valid_time': 'time'})

    # Add roughness if requested
    if add_roughness:
        if roughness_source == 'pyvwf':
            combined_ds = calculate_roughness_from_winds(combined_ds)
        else:
            combined_ds = add_constant_roughness(combined_ds, z0_constant)

    # Add metadata
    combined_ds.attrs.update({
        'title': f'Combined ERA5 wind data for Europe - {year}',
        'source': 'ERA5 reanalysis',
        'created': datetime.now().isoformat(),
        'variables': 'u10, v10, u100, v100' + (', z0' if add_roughness else ''),
        'domain': 'Europe (42-72°N, -12-22°E)',
        'temporal_resolution': 'hourly',
    })

    # Set output directory
    if output_dir is None:
        output_dir = era5_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Save with compression
    output_file = output_dir / f"era5_combined_{year}_EU.nc"
    print(f"  Saving combined file: {output_file.name}")

    # Optimize encoding for faster access
    encoding = {}
    for var in combined_ds.data_vars:
        encoding[var] = {
            'zlib': True,
            'complevel': 4,  # Moderate compression (balance speed vs size)
            'shuffle': True,
            'dtype': 'float32',  # Use float32 to save space
        }

    combined_ds.to_netcdf(output_file, encoding=encoding)

    # Close datasets
    ds_10m.close()
    ds_100m.close()
    combined_ds.close()

    # Print file info
    output_size_mb = output_file.stat().st_size / 1e6
    print(f"  ✓ Saved: {output_file.name} ({output_size_mb:.1f} MB)")

    return output_file


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description='Combine ERA5 wind files with optional roughness'
    )
    parser.add_argument(
        '--years',
        type=int,
        nargs='+',
        help='Years to process (e.g., 2019 2020 2021)'
    )
    parser.add_argument(
        '--all-years',
        action='store_true',
        help='Process all available years'
    )
    parser.add_argument(
        '--add-roughness',
        action='store_true',
        help='Add surface roughness length (z0)'
    )
    parser.add_argument(
        '--roughness-source',
        choices=['pyvwf', 'constant'],
        default='pyvwf',
        help='Roughness source: pyvwf (from wind shear) or constant value (default: pyvwf)'
    )
    parser.add_argument(
        '--roughness-value',
        type=float,
        default=0.03,
        help='Constant roughness value (m) if using constant source (default: 0.03)'
    )
    parser.add_argument(
        '--era5-dir',
        type=str,
        default='input/era5/EU',
        help='Directory containing ERA5 files (default: input/era5/EU)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory (default: same as ERA5 directory)'
    )

    args = parser.parse_args()

    # Setup paths
    era5_dir = Path(args.era5_dir)
    if not era5_dir.exists():
        print(f"Error: ERA5 directory not found: {era5_dir}")
        return 1

    # Determine years to process
    if args.all_years:
        years = find_available_years(era5_dir)
        print(f"Found {len(years)} years: {years}")
    elif args.years:
        years = args.years
    else:
        print("Error: Specify --years or --all-years")
        return 1

    # Print configuration
    print("="*80)
    print("ERA5 FILE COMBINATION SCRIPT")
    print("="*80)
    print(f"ERA5 directory: {era5_dir}")
    print(f"Years to process: {years}")
    print(f"Add roughness: {args.add_roughness}")
    if args.add_roughness:
        print(f"Roughness source: {args.roughness_source}")
        if args.roughness_source == 'constant':
            print(f"Roughness value: {args.roughness_value}m")
        else:
            print("Roughness method: PyVWF (from 10m-100m wind shear)")
    print(f"Output directory: {args.output_dir or era5_dir}")
    print("="*80)

    # Process each year
    output_files = []
    for year in years:
        try:
            output_file = combine_era5_year(
                era5_dir=era5_dir,
                year=year,
                add_roughness=args.add_roughness,
                roughness_source=args.roughness_source,
                z0_constant=args.roughness_value,
                output_dir=args.output_dir
            )
            output_files.append(output_file)
        except Exception as e:
            print(f"  ✗ Error processing year {year}: {e}")
            continue

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Successfully processed: {len(output_files)}/{len(years)} years")

    if output_files:
        print("\nOutput files:")
        total_size = 0
        for f in output_files:
            size_mb = f.stat().st_size / 1e6
            total_size += size_mb
            print(f"  ✓ {f.name} ({size_mb:.1f} MB)")
        print(f"\nTotal size: {total_size:.1f} MB")

        # Size comparison
        original_size = sum([
            (era5_dir / f"era5_u10_v10_{year}_months01-12_EU.nc").stat().st_size +
            (era5_dir / f"era5_u100_v100_{year}_months01-12_EU.nc").stat().st_size
            for year in years if (era5_dir / f"era5_u10_v10_{year}_months01-12_EU.nc").exists()
        ]) / 1e6

        savings_pct = (original_size - total_size) / original_size * 100
        print(f"Original total: {original_size:.1f} MB")
        print(f"Space savings: {savings_pct:.1f}%")

        print("\n" + "="*80)
        print("✓ COMPLETE - Your code will run faster with combined files!")
        print("="*80)
        print("\nUsage in your code:")
        print("  ds = xr.open_dataset('input/era5/EU/era5_combined_2019_EU.nc')")
        print("  u10 = ds['u10']")
        print("  v10 = ds['v10']")
        print("  u100 = ds['u100']")
        print("  v100 = ds['v100']")
        if args.add_roughness:
            print("  z0 = ds['z0']  # Derived from wind shear (PyVWF method)")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
