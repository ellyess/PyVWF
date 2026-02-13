#!/usr/bin/env python3
"""Export bimonthly correction factors for atlite.

Trains monthly or bimonthly correction factors and exports them in a format
that atlite can use for bias correction of wind resource data.

Usage:
    python export_bimonthly_corrections_for_atlite.py --country DK --bimonthly
"""

import argparse
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime

def train_monthly_corrections(obs_data, reanalysis_data, grid_points, method='bimonthly'):
    """Train monthly or bimonthly correction factors.

    Args:
        obs_data: Observed capacity factors (time series)
        reanalysis_data: Reanalysis capacity factors (time x turbine)
        grid_points: Grid point metadata with lat/lon/cluster
        method: 'monthly' (12 periods) or 'bimonthly' (6 periods)

    Returns:
        DataFrame with corrections per cluster per time period
    """
    print(f"\nTraining {method} corrections...")

    # Add month to data
    obs_data['month'] = obs_data.index.month

    # Define time periods
    if method == 'bimonthly':
        # 6 bimonthly periods: Jan-Feb, Mar-Apr, May-Jun, Jul-Aug, Sep-Oct, Nov-Dec
        period_map = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3,
                     7: 4, 8: 4, 9: 5, 10: 5, 11: 6, 12: 6}
        n_periods = 6
        period_names = ['Jan-Feb', 'Mar-Apr', 'May-Jun', 'Jul-Aug', 'Sep-Oct', 'Nov-Dec']
    else:  # monthly
        period_map = {i: i for i in range(1, 13)}
        n_periods = 12
        period_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    obs_data['period'] = obs_data['month'].map(period_map)

    # Country-aggregate reanalysis (mean across turbines)
    reanalysis_agg = pd.DataFrame({
        'cf': reanalysis_data.mean(axis=1),
        'month': reanalysis_data.index.month
    })
    reanalysis_agg['period'] = reanalysis_agg['month'].map(period_map)

    # Compute corrections for each period
    corrections = []

    for period in range(1, n_periods + 1):
        # Get data for this period
        obs_period = obs_data[obs_data['period'] == period]['capacity_factor']
        rea_period = reanalysis_agg[reanalysis_agg['period'] == period]['cf']

        if len(obs_period) == 0 or len(rea_period) == 0:
            print(f"  ⚠ Warning: No data for period {period}")
            continue

        # Compute period-specific bias
        obs_mean = obs_period.mean()
        rea_mean = rea_period.mean()

        # Multiple correction approaches
        # 1. Additive: observed = reanalysis + offset
        offset = obs_mean - rea_mean

        # 2. Multiplicative: observed = scalar * reanalysis
        scalar = obs_mean / rea_mean if rea_mean > 0 else 1.0

        # 3. Affine: observed = scalar * reanalysis + offset (fit both)
        # Using simple ratio + offset for robustness
        affine_scalar = scalar
        affine_offset = 0.0  # Could fit properly with least squares

        corrections.append({
            'period': period,
            'period_name': period_names[period - 1],
            'obs_mean': obs_mean,
            'rea_mean': rea_mean,
            'offset': offset,
            'scalar': scalar,
            'affine_scalar': affine_scalar,
            'affine_offset': affine_offset,
            'n_samples': len(obs_period),
        })

        print(f"  Period {period} ({period_names[period-1]}): "
              f"obs={obs_mean:.3f}, rea={rea_mean:.3f}, "
              f"scalar={scalar:.3f}, offset={offset:+.3f}")

    return pd.DataFrame(corrections)

def create_atlite_correction_field(corrections, grid_points, region_shape, method='bimonthly'):
    """Create spatially-distributed correction field for atlite.

    Args:
        corrections: DataFrame with period-specific corrections
        grid_points: Grid points with lat/lon
        region_shape: GeoDataFrame with region boundary
        method: 'monthly' or 'bimonthly'

    Returns:
        xarray Dataset with lat/lon/period dimensions
    """
    print(f"\nCreating {method} correction field...")

    # Create lat/lon grid
    lat_min, lat_max = grid_points['lat'].min(), grid_points['lat'].max()
    lon_min, lon_max = grid_points['lon'].min(), grid_points['lon'].max()

    # Create grid at 0.25° resolution (typical for wind resource assessments)
    lats = np.arange(lat_min, lat_max + 0.25, 0.25)
    lons = np.arange(lon_min, lon_max + 0.25, 0.25)

    n_periods = 6 if method == 'bimonthly' else 12

    # Initialize correction arrays
    scalar_field = np.ones((n_periods, len(lats), len(lons)))
    offset_field = np.zeros((n_periods, len(lats), len(lons)))

    # For simplicity, apply uniform corrections across space
    # (Could interpolate cluster-specific corrections if clusters are defined spatially)
    for i, row in corrections.iterrows():
        period_idx = int(row['period']) - 1
        scalar_field[period_idx, :, :] = row['scalar']
        offset_field[period_idx, :, :] = row['offset']

    # Create xarray Dataset
    ds = xr.Dataset(
        {
            'scalar': (['period', 'lat', 'lon'], scalar_field),
            'offset': (['period', 'lat', 'lon'], offset_field),
        },
        coords={
            'period': np.arange(1, n_periods + 1),
            'lat': lats,
            'lon': lons,
        },
        attrs={
            'description': f'{method.capitalize()} wind bias correction factors',
            'method': method,
            'n_periods': n_periods,
            'created': datetime.now().isoformat(),
        }
    )

    return ds

def export_for_atlite(corrections_ds, output_path, method='bimonthly'):
    """Export corrections in atlite-compatible NetCDF format.

    Args:
        corrections_ds: xarray Dataset with corrections
        output_path: Output file path (.nc)
        method: 'monthly' or 'bimonthly'
    """
    print(f"\nExporting {method} corrections for atlite...")

    # Add metadata for atlite
    corrections_ds.attrs.update({
        'title': f'{method.capitalize()} Wind Bias Corrections',
        'institution': 'Imperial College London',
        'source': 'PyVWF bias correction framework',
        'conventions': 'CF-1.8',
    })

    # Save as NetCDF
    corrections_ds.to_netcdf(
        output_path,
        encoding={
            'scalar': {'dtype': 'float32', 'zlib': True, 'complevel': 4},
            'offset': {'dtype': 'float32', 'zlib': True, 'complevel': 4},
        }
    )

    print(f"✓ Saved: {output_path}")
    print(f"  Size: {Path(output_path).stat().st_size / 1024:.1f} KB")
    print(f"\nUsage in atlite:")
    print(f"  import xarray as xr")
    print(f"  corrections = xr.open_dataset('{output_path}')")
    print(f"  # Apply to wind speeds or capacity factors")

def generate_atlite_application_script(corrections_path, output_script):
    """Generate Python script showing how to use corrections in atlite."""

    script_content = f'''#!/usr/bin/env python3
"""Apply bimonthly corrections to atlite wind resource data.

This script shows how to use the exported correction factors with atlite.

Usage:
    python apply_corrections_to_atlite.py
"""

import atlite
import xarray as xr
import pandas as pd
import numpy as np

# Load correction factors
corrections = xr.open_dataset('{corrections_path}')

print("Correction factors loaded:")
print(f"  Shape: {{corrections.scalar.shape}}")
print(f"  Method: {{corrections.attrs.get('method', 'unknown')}}")

# Load atlite cutout (example for Denmark)
cutout = atlite.Cutout(
    path="denmark-2019-2023.nc",
    module="era5",
    x=slice(8.0, 13.0),
    y=slice(54.5, 58.0),
    time="2019-01-01",
    time_end="2023-12-31",
)

# Option 1: Apply corrections to wind speeds before capacity factor calculation
def apply_wind_speed_corrections(cutout, corrections):
    """Apply corrections to wind speeds."""
    ds = cutout.data

    # Get month from time coordinate
    months = ds['time'].dt.month

    # Map months to periods
    if corrections.attrs.get('method') == 'bimonthly':
        period_map = {{1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3,
                     7: 4, 8: 4, 9: 5, 10: 5, 11: 6, 12: 6}}
    else:
        period_map = {{i: i for i in range(1, 13)}}

    periods = xr.DataArray([period_map[m] for m in months.values],
                          dims='time', coords={{'time': ds.time}})

    # Apply period-specific corrections to wind speeds
    wnd100m = ds['wnd100m']

    # Interpolate corrections to cutout grid
    corrections_interp = corrections.interp(
        lat=ds.lat,
        lon=ds.lon,
        method='nearest'
    )

    # Apply corrections based on period
    wnd100m_corrected = wnd100m.copy()

    for period in range(1, corrections.period.size + 1):
        # Mask for this period
        period_mask = periods == period

        # Get correction for this period
        scalar = corrections_interp.scalar.sel(period=period)
        offset = corrections_interp.offset.sel(period=period)

        # Apply: corrected = scalar * original + offset
        # (offset typically 0 for wind speeds, mainly scalar)
        wnd100m_corrected = xr.where(
            period_mask,
            wnd100m * scalar,
            wnd100m_corrected
        )

    # Update dataset
    ds_corrected = ds.copy()
    ds_corrected['wnd100m'] = wnd100m_corrected

    return ds_corrected

# Option 2: Apply corrections to capacity factors after calculation
def apply_cf_corrections(capacity_factors, corrections, time_index):
    """Apply corrections to pre-calculated capacity factors."""

    # Get months and periods
    months = time_index.month

    if corrections.attrs.get('method') == 'bimonthly':
        period_map = {{1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3,
                     7: 4, 8: 4, 9: 5, 10: 5, 11: 6, 12: 6}}
    else:
        period_map = {{i: i for i in range(1, 13)}}

    periods = np.array([period_map[m] for m in months])

    # Apply period-specific corrections
    cf_corrected = capacity_factors.copy()

    for period in range(1, corrections.period.size + 1):
        period_mask = periods == period

        # Spatially-averaged correction for this period
        scalar = float(corrections.scalar.sel(period=period).mean())
        offset = float(corrections.offset.sel(period=period).mean())

        # Apply: corrected = scalar * original + offset
        cf_corrected[period_mask] = (
            capacity_factors[period_mask] * scalar + offset
        )

    # Clip to valid range [0, 1]
    cf_corrected = np.clip(cf_corrected, 0.0, 1.0)

    return cf_corrected

# Example usage
if __name__ == '__main__':
    print("="*80)
    print("EXAMPLE: Applying Bimonthly Corrections to Atlite")
    print("="*80)

    # Method 1: Correct wind speeds (if you have full cutout)
    # ds_corrected = apply_wind_speed_corrections(cutout, corrections)

    # Method 2: Correct capacity factors (simpler)
    # Assume you have calculated CFs
    example_cf = np.random.rand(8760)  # Example CF time series
    example_time = pd.date_range('2023-01-01', periods=8760, freq='h')

    cf_corrected = apply_cf_corrections(example_cf, corrections, example_time)

    print(f"\\nOriginal CF mean: {{example_cf.mean():.3f}}")
    print(f"Corrected CF mean: {{cf_corrected.mean():.3f}}")
    print(f"Change: {{(cf_corrected.mean() - example_cf.mean()):.3f}}")

    print("\\n✓ Correction applied successfully!")
'''

    with open(output_script, 'w') as f:
        f.write(script_content)

    print(f"\n✓ Generated example script: {output_script}")

def main():
    parser = argparse.ArgumentParser(
        description='Export bimonthly corrections for atlite'
    )
    parser.add_argument('--country', required=True, help='Country code (e.g., DK)')
    parser.add_argument('--bimonthly', action='store_true',
                       help='Use bimonthly (6 periods) instead of monthly (12)')
    parser.add_argument('--output-dir', default='output/atlite_corrections',
                       help='Output directory')

    args = parser.parse_args()

    method = 'bimonthly' if args.bimonthly else 'monthly'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print(f"EXPORTING {method.upper()} CORRECTIONS FOR ATLITE")
    print("="*80)
    print(f"Country: {args.country}")
    print(f"Method: {method}")
    print("="*80)

    # Load training data
    country = args.country.upper()
    obs_dir = f"input/country_level_data/observations/{country.lower()}"

    if country in ['NO', 'SE']:
        train_file = f"{obs_dir}/{country.lower()}_train_2015_2019_aggregated.csv"
    else:
        train_file = f"{obs_dir}/{country.lower()}_train_2015_2019.csv"

    obs_data = pd.read_csv(train_file)
    obs_data['time'] = pd.to_datetime(obs_data.iloc[:, 0], utc=True)
    obs_data = obs_data.set_index('time')

    # Load reanalysis (uncorrected)
    results_dir = f"output/temporal_2015_2019_to_2023/output/run/{country}-all-obs_country-corrected-calc_z0/results/capacity-factor"
    reanalysis_file = f"{results_dir}/{country}_2023_unc_cf.csv"

    if not Path(reanalysis_file).exists():
        print(f"✗ Error: Uncorrected reanalysis file not found: {reanalysis_file}")
        print(f"  Please run temporal training first")
        return 1

    reanalysis_data = pd.read_csv(reanalysis_file, index_col=0, parse_dates=True)

    # Load grid points
    grid_file = f"input/country_level_data/grid_points/{country.lower()}/{country.lower()}_grid_points.csv"
    if not Path(grid_file).exists():
        grid_file = f"input/country_level_data/grid_points/{country.lower()}/{country.lower()}_grid_points_zones.csv"

    grid_points = pd.read_csv(grid_file)

    print(f"\\nLoaded data:")
    print(f"  Observations: {len(obs_data)} timesteps")
    print(f"  Reanalysis: {reanalysis_data.shape}")
    print(f"  Grid points: {len(grid_points)}")

    # Train corrections
    corrections_df = train_monthly_corrections(
        obs_data, reanalysis_data, grid_points, method=method
    )

    # Save corrections table
    corrections_csv = output_dir / f"{country}_{method}_corrections.csv"
    corrections_df.to_csv(corrections_csv, index=False)
    print(f"\\n✓ Saved corrections table: {corrections_csv}")

    # Create spatial correction field
    corrections_ds = create_atlite_correction_field(
        corrections_df, grid_points, None, method=method
    )

    # Export NetCDF
    corrections_nc = output_dir / f"{country}_{method}_corrections.nc"
    export_for_atlite(corrections_ds, corrections_nc, method=method)

    # Generate example script
    script_path = output_dir / f"apply_{method}_corrections_to_atlite.py"
    generate_atlite_application_script(str(corrections_nc), str(script_path))

    print("\\n" + "="*80)
    print("✓ EXPORT COMPLETE")
    print("="*80)
    print(f"\\nGenerated files:")
    print(f"  1. {corrections_csv} - Correction factors table")
    print(f"  2. {corrections_nc} - NetCDF for atlite")
    print(f"  3. {script_path} - Example usage script")
    print("="*80)

    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
