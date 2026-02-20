#!/usr/bin/env python3
"""Generate optimised correction grids for Atlite application.

Produces two correction NetCDF files:
1. Hybrid Kriging: uses the best variogram config per target
   - Scalar: OK, spherical, Euclidean (lowest scalar CV MAE)
   - Offset: OK, linear, geographic (lowest offset CV MAE)
   - Variance-based masking: neutral values where Kriging confidence is low
2. IDW: standard inverse distance weighting with distance masking

Both files are ready for direct use with Atlite/PyPSA-Eur.

Usage:
    PYTHONPATH=. python scripts/pyvwf_to_grid/generate_best_correction_grids.py
    PYTHONPATH=. python scripts/pyvwf_to_grid/generate_best_correction_grids.py --variance-threshold 0.90
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from pykrige.ok import OrdinaryKriging
except ImportError:
    print("ERROR: pykrige is required. Install with: pip install pykrige")
    sys.exit(1)


# =============================================================================
# Configuration
# =============================================================================

UNIFIED_CORRECTIONS_CSV = "output/pyvwf_to_grid/all_corrections_centroids.csv"
OUTPUT_DIR = Path("output/pyvwf_to_grid")

GRID_RESOLUTION = 0.25
EUROPE_EXTENT = {
    'lon_min': -10.0, 'lon_max': 30.0,
    'lat_min': 35.0,  'lat_max': 72.0,
}

# Best Kriging configs from test_kriging_improvements.py CV analysis
SCALAR_CONFIG = {'variogram_model': 'spherical', 'coordinates_type': 'euclidean'}
OFFSET_CONFIG = {'variogram_model': 'linear',    'coordinates_type': 'geographic'}

# IDW parameters
IDW_POWER = 2.0
MAX_DISTANCE_DEG = 5.0  # Distance mask for IDW

# Default variance threshold for Kriging masking
DEFAULT_VARIANCE_THRESHOLD = 0.90


# =============================================================================
# Grid and data loading (adapted from compare_unified_corrections_to_grid.py)
# =============================================================================

def load_control_points(csv_path):
    """Load and prepare control points."""
    print(f"Loading control points from: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['lon', 'lat', 'scalar', 'offset'])
    print(f"  {len(df)} valid control points")
    print(f"  Scalar range: [{df['scalar'].min():.3f}, {df['scalar'].max():.3f}]")
    print(f"  Offset range: [{df['offset'].min():.3f}, {df['offset'].max():.3f}]")
    return df


def create_europe_grid():
    """Create regular 0.25-degree grid covering Europe."""
    lons = np.arange(EUROPE_EXTENT['lon_min'],
                     EUROPE_EXTENT['lon_max'] + GRID_RESOLUTION,
                     GRID_RESOLUTION)
    lats = np.arange(EUROPE_EXTENT['lat_min'],
                     EUROPE_EXTENT['lat_max'] + GRID_RESOLUTION,
                     GRID_RESOLUTION)
    print(f"Grid: {len(lats)} lat x {len(lons)} lon = {len(lats)*len(lons)} points")
    return lons, lats


# =============================================================================
# Kriging with variance
# =============================================================================

def kriging_with_variance(lons, lats, values, grid_lons, grid_lats,
                          variogram_model, coordinates_type, label=""):
    """Run Ordinary Kriging and return both prediction and variance grids.

    Args:
        lons, lats: Control point coordinates.
        values: Control point values to interpolate.
        grid_lons, grid_lats: 1D arrays of grid coordinates.
        variogram_model: PyKrige variogram model name.
        coordinates_type: 'euclidean' or 'geographic'.
        label: Label for logging.

    Returns:
        prediction: 2D array (lat x lon) of interpolated values.
        variance: 2D array (lat x lon) of kriging variance.
        sill: Fitted sill value (nugget + partial sill).
    """
    print(f"  Kriging {label} ({variogram_model}, {coordinates_type})...")

    ok = OrdinaryKriging(
        lons, lats, values,
        variogram_model=variogram_model,
        coordinates_type=coordinates_type,
        verbose=False,
        enable_plotting=False,
    )

    prediction, variance = ok.execute('grid', grid_lons, grid_lats)

    # Extract sill from fitted variogram parameters
    # Parameters depend on model: [psill, range, nugget] for most models
    params = ok.variogram_model_parameters
    if variogram_model == 'linear':
        # Linear: [slope, nugget]
        sill = params[0] * max(
            np.ptp(lons) if coordinates_type == 'euclidean'
            else np.ptp(lons) * 111.0,  # rough km conversion
            1.0
        )  # Linear doesn't have a true sill; use data variance instead
        sill = np.var(values)
        print(f"    Linear model: using data variance as sill = {sill:.4f}")
    else:
        # Exponential, spherical, etc: [psill, range, nugget]
        psill, range_param, nugget = params[0], params[1], params[2]
        sill = psill + nugget
        print(f"    Variogram: psill={psill:.4f}, range={range_param:.2f}, "
              f"nugget={nugget:.4f}, sill={sill:.4f}")

    prediction = np.asarray(prediction)
    variance = np.asarray(variance)

    print(f"    Prediction range: [{prediction.min():.3f}, {prediction.max():.3f}]")
    print(f"    Variance range: [{variance.min():.4f}, {variance.max():.4f}]")

    return prediction, variance, sill


# =============================================================================
# IDW interpolation
# =============================================================================

def interpolate_idw(control_points, grid_lons, grid_lats, power=2.0):
    """Inverse Distance Weighting interpolation."""
    print("  Running IDW interpolation...")

    coords = control_points[['lon', 'lat']].values
    scalars = control_points['scalar'].values
    offsets = control_points['offset'].values

    lon_grid, lat_grid = np.meshgrid(grid_lons, grid_lats)
    target_coords = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])

    batch_size = 10000
    n_targets = len(target_coords)
    scalar_interp = np.zeros(n_targets)
    offset_interp = np.zeros(n_targets)

    for i in range(0, n_targets, batch_size):
        batch = target_coords[i:i+batch_size]
        dists = distance.cdist(batch, coords, metric='euclidean')

        weights = 1.0 / (dists ** power + 1e-10)
        weights = weights / weights.sum(axis=1, keepdims=True)

        scalar_interp[i:i+batch_size] = (weights * scalars).sum(axis=1)
        offset_interp[i:i+batch_size] = (weights * offsets).sum(axis=1)

        # Fix exact matches
        exact_match = dists == 0
        for j, row_match in enumerate(exact_match):
            if row_match.any():
                idx = np.where(row_match)[0][0]
                scalar_interp[i+j] = scalars[idx]
                offset_interp[i+j] = offsets[idx]

    scalar_grid = scalar_interp.reshape(lon_grid.shape)
    offset_grid = offset_interp.reshape(lon_grid.shape)

    print(f"    Scalar range: [{scalar_grid.min():.3f}, {scalar_grid.max():.3f}]")
    print(f"    Offset range: [{offset_grid.min():.3f}, {offset_grid.max():.3f}]")

    return scalar_grid, offset_grid


# =============================================================================
# Masking
# =============================================================================

def apply_variance_mask(scalar_grid, offset_grid, scalar_var, scalar_sill,
                        threshold=0.95):
    """Mask grid points where normalised scalar Kriging variance exceeds threshold.

    Uses only the scalar variance (spherical variogram with bounded sill) for
    masking both fields. The linear variogram used for offset has unbounded
    variance, making its normalised variance meaningless. Since spatial
    reliability depends on control point density (same for both targets),
    the scalar variance is a valid proxy for overall prediction confidence.

    Points above the threshold are set to neutral corrections (scalar=1, offset=0).

    Args:
        scalar_grid, offset_grid: Prediction grids.
        scalar_var: Scalar variance grid (from spherical variogram).
        scalar_sill: Scalar sill value for normalisation.
        threshold: Normalised variance threshold (0-1). Default 0.95.

    Returns:
        Masked scalar_grid, offset_grid, and the mask.
    """
    print(f"\nApplying variance mask (threshold={threshold}, scalar variance only)...")

    # Normalise scalar variance by its sill
    scalar_norm_var = scalar_var / scalar_sill if scalar_sill > 0 else scalar_var

    # Mask where scalar normalised variance exceeds threshold
    unreliable = scalar_norm_var > threshold

    n_total = unreliable.size
    n_masked = unreliable.sum()
    pct = 100 * n_masked / n_total

    print(f"  Scalar norm variance range: [{scalar_norm_var.min():.4f}, "
          f"{scalar_norm_var.max():.4f}]")
    print(f"  Masked: {n_masked}/{n_total} points ({pct:.1f}%)")

    # Apply mask
    scalar_masked = np.where(unreliable, 1.0, scalar_grid)
    offset_masked = np.where(unreliable, 0.0, offset_grid)

    return scalar_masked, offset_masked, unreliable


def apply_distance_mask(scalar_grid, offset_grid, grid_lons, grid_lats,
                        control_points, max_distance_deg=5.0):
    """Mask grid points far from any control point."""
    print(f"\nApplying distance mask (max {max_distance_deg} deg)...")

    lon_grid, lat_grid = np.meshgrid(grid_lons, grid_lats)
    grid_coords = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    control_coords = control_points[['lon', 'lat']].values

    nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn.fit(control_coords)
    distances, _ = nn.kneighbors(grid_coords)
    distance_grid = distances.reshape(lon_grid.shape)

    far_mask = distance_grid > max_distance_deg
    n_masked = far_mask.sum()
    pct = 100 * n_masked / far_mask.size
    print(f"  Distance-masked: {n_masked} points ({pct:.1f}%)")

    scalar_masked = np.where(far_mask, 1.0, scalar_grid)
    offset_masked = np.where(far_mask, 0.0, offset_grid)

    return scalar_masked, offset_masked, distance_grid


# =============================================================================
# NetCDF export
# =============================================================================

def save_correction_nc(path, grid_lons, grid_lats, scalar, offset,
                       method, attrs=None, extra_vars=None):
    """Save correction grid to NetCDF."""
    ds = xr.Dataset(
        {
            'scalar': (['y', 'x'], scalar),
            'offset': (['y', 'x'], offset),
        },
        coords={'y': grid_lats, 'x': grid_lons},
        attrs={
            'method': method,
            'description': 'Wind speed bias correction factors for Atlite',
            'usage': 'v_corrected = v_ERA5 * scalar + offset',
            'neutral_scalar': 1.0,
            'neutral_offset': 0.0,
        },
    )

    if attrs:
        ds.attrs.update(attrs)

    if extra_vars:
        for name, (dims, data) in extra_vars.items():
            ds[name] = (dims, data)

    ds.to_netcdf(path)
    print(f"  Saved: {path}")
    return ds


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate optimised correction grids for Atlite"
    )
    parser.add_argument(
        '--variance-threshold', type=float, default=DEFAULT_VARIANCE_THRESHOLD,
        help=f'Normalised variance threshold for Kriging masking (default: {DEFAULT_VARIANCE_THRESHOLD})'
    )
    parser.add_argument(
        '--output-dir', type=str, default=str(OUTPUT_DIR),
        help=f'Output directory (default: {OUTPUT_DIR})'
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GENERATE BEST CORRECTION GRIDS FOR ATLITE")
    print("=" * 70)

    # Load data
    df = load_control_points(UNIFIED_CORRECTIONS_CSV)
    grid_lons, grid_lats = create_europe_grid()

    cp_lons = df['lon'].values
    cp_lats = df['lat'].values
    cp_scalars = df['scalar'].values
    cp_offsets = df['offset'].values

    # =========================================================================
    # 1. Hybrid Kriging (best config per target)
    # =========================================================================
    print(f"\n{'='*70}")
    print("1. HYBRID KRIGING")
    print(f"{'='*70}")

    # Scalar: OK, spherical, euclidean (best scalar MAE)
    scalar_pred, scalar_var, scalar_sill = kriging_with_variance(
        cp_lons, cp_lats, cp_scalars, grid_lons, grid_lats,
        label="scalar", **SCALAR_CONFIG,
    )

    # Offset: OK, linear, geographic (best offset MAE)
    offset_pred, offset_var, offset_sill = kriging_with_variance(
        cp_lons, cp_lats, cp_offsets, grid_lons, grid_lats,
        label="offset", **OFFSET_CONFIG,
    )

    # Apply variance-based mask (scalar variance only - see docstring)
    scalar_kriging, offset_kriging, var_mask = apply_variance_mask(
        scalar_pred, offset_pred, scalar_var, scalar_sill,
        threshold=args.variance_threshold,
    )

    # Save Kriging grid
    kriging_path = output_dir / 'europe_corrections_kriging_best.nc'
    save_correction_nc(
        kriging_path, grid_lons, grid_lats, scalar_kriging, offset_kriging,
        method='kriging_hybrid',
        attrs={
            'scalar_variogram': SCALAR_CONFIG['variogram_model'],
            'scalar_coordinates': SCALAR_CONFIG['coordinates_type'],
            'offset_variogram': OFFSET_CONFIG['variogram_model'],
            'offset_coordinates': OFFSET_CONFIG['coordinates_type'],
            'variance_threshold': args.variance_threshold,
            'scalar_sill': float(scalar_sill),
            'offset_sill': float(offset_sill),
            'n_control_points': len(df),
        },
        extra_vars={
            'scalar_variance': (['y', 'x'], scalar_var),
            'offset_variance': (['y', 'x'], offset_var),
            'variance_masked': (['y', 'x'], var_mask.astype(np.int8)),
        },
    )

    # =========================================================================
    # 2. IDW with distance masking
    # =========================================================================
    print(f"\n{'='*70}")
    print("2. IDW")
    print(f"{'='*70}")

    scalar_idw, offset_idw = interpolate_idw(df, grid_lons, grid_lats, power=IDW_POWER)

    # Apply distance mask
    scalar_idw_m, offset_idw_m, dist_grid = apply_distance_mask(
        scalar_idw, offset_idw, grid_lons, grid_lats, df,
        max_distance_deg=MAX_DISTANCE_DEG,
    )

    # Save IDW grid
    idw_path = output_dir / 'europe_corrections_idw_best.nc'
    save_correction_nc(
        idw_path, grid_lons, grid_lats, scalar_idw_m, offset_idw_m,
        method='idw',
        attrs={
            'idw_power': IDW_POWER,
            'max_distance_deg': MAX_DISTANCE_DEG,
            'n_control_points': len(df),
        },
        extra_vars={
            'distance_to_nearest_control': (['y', 'x'], dist_grid),
        },
    )

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    print(f"\nControl points: {len(df)}")

    print(f"\nKriging (hybrid, variance-masked):")
    print(f"  Scalar config: {SCALAR_CONFIG}")
    print(f"  Offset config: {OFFSET_CONFIG}")
    print(f"  Variance threshold: {args.variance_threshold}")
    print(f"  Points masked: {var_mask.sum()} / {var_mask.size} "
          f"({100*var_mask.sum()/var_mask.size:.1f}%)")
    valid_scalar = scalar_kriging[~var_mask]
    valid_offset = offset_kriging[~var_mask]
    if len(valid_scalar) > 0:
        print(f"  Valid scalar range: [{valid_scalar.min():.3f}, {valid_scalar.max():.3f}]")
        print(f"  Valid offset range: [{valid_offset.min():.3f}, {valid_offset.max():.3f}]")

    print(f"\nIDW (distance-masked):")
    print(f"  Power: {IDW_POWER}")
    print(f"  Max distance: {MAX_DISTANCE_DEG} deg")
    dist_mask = dist_grid > MAX_DISTANCE_DEG
    print(f"  Points masked: {dist_mask.sum()} / {dist_mask.size} "
          f"({100*dist_mask.sum()/dist_mask.size:.1f}%)")

    print(f"\nOutput files:")
    print(f"  {kriging_path}")
    print(f"  {idw_path}")


if __name__ == '__main__':
    main()
