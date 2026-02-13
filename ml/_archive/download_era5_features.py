#!/usr/bin/env python3
"""Download ERA5 variables for ML feature enhancement.

Downloads additional ERA5 variables that can improve correction factor
prediction:
    1. Invariant fields: orography, land-sea mask, surface roughness
    2. Atmospheric stability: boundary layer height, heat flux, temperature
    3. Surface characteristics: skin temperature, friction velocity

Requires:
    cdsapi (pip install cdsapi)

Setup:
    https://cds.climate.copernicus.eu/api-how-to
"""

import cdsapi
from pathlib import Path
import argparse


def download_era5_invariant(output_dir='input/era5/invariant', bounds=None):
    """Download ERA5 invariant (time-independent) fields.

    Args:
        output_dir: Directory for output NetCDF files.
        bounds: Geographic bounds as [north, west, south, east].

    Returns:
        Path to the downloaded NetCDF file, or None on failure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if bounds is None:
        # European domain
        bounds = [72, -12, 35, 35]  # North, West, South, East
    
    print("="*70)
    print("Downloading ERA5 Invariant Fields")
    print("="*70)
    
    c = cdsapi.Client()
    
    output_file = output_dir / 'era5_invariant_europe.nc'
    
    if output_file.exists():
        print(f"✓ Already exists: {output_file}")
        return output_file
    
    print(f"\nDownloading to: {output_file}")
    print("Variables: orography, land-sea mask, surface roughness")
    print("This may take a few minutes...")
    
    try:
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': [
                    'geopotential',  # Surface geopotential (orography * g)
                    'land_sea_mask',
                    'forecast_surface_roughness',  # z0
                ],
                'year': '2020',  # Just need one timestamp for invariant fields
                'month': '01',
                'day': '01',
                'time': '00:00',
                'area': bounds,  # North, West, South, East
            },
            str(output_file)
        )
        print(f"✓ Downloaded: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"✗ Download failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Install CDS API: pip install cdsapi")
        print("  2. Setup credentials: https://cds.climate.copernicus.eu/api-how-to")
        print("  3. Accept ERA5 license terms on the CDS website")
        return None


def download_era5_atmospheric_stability(output_dir='input/era5/stability', 
                                        year=2020, months=None, bounds=None):
    """Download ERA5 atmospheric stability indicators.

    Args:
        output_dir: Directory for output NetCDF files.
        year: Year to download.
        months: List of month strings (e.g., ["01", "07"]).
        bounds: Geographic bounds as [north, west, south, east].

    Returns:
        Output directory path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if bounds is None:
        # European domain
        bounds = [72, -12, 35, 35]  # North, West, South, East
    
    if months is None:
        months = ['01', '07']  # Winter and summer for seasonal patterns
    
    print("="*70)
    print("Downloading ERA5 Atmospheric Stability Variables")
    print("="*70)
    
    c = cdsapi.Client()
    
    for month in months:
        output_file = output_dir / f'era5_stability_{year}_{month}.nc'
        
        if output_file.exists():
            print(f"✓ Already exists: {output_file}")
            continue
        
        print(f"\nDownloading {year}-{month} to: {output_file}")
        print("Variables: BLH, heat flux, temperature, friction velocity")
        print("This will take several minutes...")
        
        try:
            c.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'format': 'netcdf',
                    'variable': [
                        'boundary_layer_height',
                        'surface_sensible_heat_flux',
                        '2m_temperature',
                        'skin_temperature',
                        'friction_velocity',
                    ],
                    'year': str(year),
                    'month': month,
                    'day': [f'{d:02d}' for d in range(1, 32)],
                    'time': [f'{h:02d}:00' for h in range(0, 24, 3)],  # Every 3 hours
                    'area': bounds,
                },
                str(output_file)
            )
            print(f"✓ Downloaded: {output_file}")
            
        except Exception as e:
            print(f"✗ Download failed: {e}")
            continue
    
    return output_dir


def download_era5_surface_roughness_time_varying(output_dir='input/era5/roughness',
                                                 year=2020, bounds=None):
    """Download time-varying surface roughness.

    Args:
        output_dir: Directory for output NetCDF files.
        year: Year to download.
        bounds: Geographic bounds as [north, west, south, east].

    Returns:
        Path to the downloaded NetCDF file, or None on failure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if bounds is None:
        bounds = [72, -12, 35, 35]
    
    print("="*70)
    print("Downloading ERA5 Time-Varying Surface Roughness")
    print("="*70)
    
    c = cdsapi.Client()
    
    output_file = output_dir / f'era5_roughness_{year}.nc'
    
    if output_file.exists():
        print(f"✓ Already exists: {output_file}")
        return output_file
    
    print(f"\nDownloading to: {output_file}")
    print("This includes seasonal roughness variations...")
    
    try:
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': [
                    'forecast_surface_roughness',
                ],
                'year': str(year),
                'month': ['01', '04', '07', '10'],  # Seasonal
                'day': '15',  # Mid-month
                'time': '12:00',
                'area': bounds,
            },
            str(output_file)
        )
        print(f"✓ Downloaded: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return None


def main():
    """Run the ERA5 feature download workflow."""
    parser = argparse.ArgumentParser(
        description='Download ERA5 variables for ML feature enhancement',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--bounds',
        type=str,
        default='72,-12,35,35',
        help='Geographic bounds: North,West,South,East (default: Europe)'
    )
    parser.add_argument(
        '--year',
        type=int,
        default=2020,
        help='Year for time-varying data (default: 2020)'
    )
    parser.add_argument(
        '--months',
        type=str,
        default='01,07',
        help='Months for stability data (default: 01,07 for winter/summer)'
    )
    parser.add_argument(
        '--invariant-only',
        action='store_true',
        help='Download only invariant fields (fastest)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../input/era5',
        help='Output directory (default: input/era5)'
    )
    
    args = parser.parse_args()
    
    # Parse bounds
    bounds = [float(x) for x in args.bounds.split(',')]
    months = args.months.split(',')
    
    print("\n" + "="*70)
    print("ERA5 Feature Download for ML Training")
    print("="*70)
    print(f"Bounds: {bounds} (N,W,S,E)")
    print(f"Year: {args.year}")
    print(f"Months: {', '.join(months)}")
    print(f"Output: {args.output_dir}")
    print("="*70)
    
    # 1. Invariant fields (always download - essential for ERA5 mismatch features)
    print("\n" + "="*70)
    print("STEP 1: Invariant Fields (Orography, Roughness)")
    print("="*70)
    invariant_file = download_era5_invariant(
        output_dir=f'{args.output_dir}/invariant',
        bounds=bounds
    )
    
    if args.invariant_only:
        print("\n" + "="*70)
        print("DONE (invariant only)")
        print("="*70)
        return
    
    # 2. Atmospheric stability indicators
    print("\n" + "="*70)
    print("STEP 2: Atmospheric Stability Variables")
    print("="*70)
    stability_dir = download_era5_atmospheric_stability(
        output_dir=f'{args.output_dir}/stability',
        year=args.year,
        months=months,
        bounds=bounds
    )
    
    # 3. Surface roughness (time-varying)
    print("\n" + "="*70)
    print("STEP 3: Time-Varying Surface Roughness")
    print("="*70)
    roughness_file = download_era5_surface_roughness_time_varying(
        output_dir=f'{args.output_dir}/roughness',
        year=args.year,
        bounds=bounds
    )
    
    # Summary
    print("\n" + "="*70)
    print("DOWNLOAD COMPLETE!")
    print("="*70)
    print("\nDownloaded files:")
    if invariant_file:
        print(f"  Invariant: {invariant_file}")
    if stability_dir:
        print(f"  Stability: {stability_dir}/")
    if roughness_file:
        print(f"  Roughness: {roughness_file}")
    
    print("\nNext steps:")
    print("  1. Run training with ERA5 mismatch features:")
    print("     python train_europe_ml_corrections.py --countries DK,UK --validation-countries DE --force-extract")
    print("  2. The script will automatically use these new ERA5 files")
    print("  3. Review improved model performance")


if __name__ == '__main__':
    main()
