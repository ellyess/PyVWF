"""Compare spatial interpolation methods for unified correction dataset.

Adapts bias_to_grid_comparison.py to work with the new unified correction structure
containing 1,835 clusters from 7 countries (NL, FR, BE, NO, DE, DK, UK).

This script:
1. Loads unified corrections from CSV
2. Separates onshore/offshore control points
3. Interpolates corrections onto ERA5 0.25° grid using multiple methods
4. Exports NetCDF surfaces and comparison maps
5. Optionally runs cross-validation to compare methods
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.spatial import distance
from scipy.interpolate import RBFInterpolator
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold
import warnings

# Import thesis plotting style
from plotting_style import thesis_plot_style

# Try importing pykrige for kriging
try:
    from pykrige.ok import OrdinaryKriging
    KRIGING_AVAILABLE = True
except ImportError:
    KRIGING_AVAILABLE = False
    warnings.warn("pykrige not available - Ordinary Kriging will be skipped")


# ============================================================================
# Configuration
# ============================================================================

# Input data
UNIFIED_CORRECTIONS_CSV = "output/unified_corrections_temporal/all_corrections_centroids_temporal.csv"

# Output directory
OUTPUT_DIR = Path("output/unified_corrections_grid_comparison_temporal")

# Grid resolution (degrees)
GRID_RESOLUTION = 0.25

# Europe extent
EUROPE_EXTENT = {
    'lon_min': -10.0,
    'lon_max': 30.0,
    'lat_min': 35.0,
    'lat_max': 72.0
}

# Interpolation methods to compare
METHODS = ['nearest', 'idw', 'rbf']
if KRIGING_AVAILABLE:
    METHODS.append('kriging')

# IDW power parameter
IDW_POWER = 2.0

# RBF kernel
RBF_KERNEL = 'thin_plate_spline'

# Cross-validation folds
CV_FOLDS = 5

# Distance mask for neutral values
# Grid points beyond this distance (degrees) from any control point will be set to neutral (scalar=1, offset=0)
MAX_DISTANCE_DEG = 5.0  # ~500km at mid-latitudes


# ============================================================================
# Load and prepare data
# ============================================================================

def load_unified_corrections(csv_path):
    """Load unified corrections CSV.

    Returns:
        DataFrame with columns: country_code, cluster, cluster_mode, obs_level,
                                lon, lat, scalar, offset, area_km2
    """
    print(f"\nLoading unified corrections from: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"  ✓ Loaded {len(df)} correction clusters")
    print(f"    Countries: {df['country_code'].unique()}")
    print(f"    Cluster modes: {df['cluster_mode'].unique()}")
    print(f"    Observation levels: {df['obs_level'].unique()}")

    return df


def prepare_control_points(df):
    """Prepare control points for interpolation.

    Separates onshore and offshore points based on cluster_mode.
    Country-level points (mode='all') are assigned to onshore by default.

    Args:
        df: Unified corrections dataframe.

    Returns:
        Tuple of (onshore_df, offshore_df)
    """
    print("\nPreparing control points...")

    # Onshore: cluster_mode == 'onshore' or 'all' (country-level)
    onshore = df[df['cluster_mode'].isin(['onshore', 'all'])].copy()

    # Offshore: cluster_mode == 'offshore'
    offshore = df[df['cluster_mode'] == 'offshore'].copy()

    print(f"  ✓ Onshore control points: {len(onshore)}")
    print(f"    - Country-level: {len(onshore[onshore['cluster_mode'] == 'all'])}")
    print(f"    - Turbine-level: {len(onshore[onshore['cluster_mode'] == 'onshore'])}")
    print(f"  ✓ Offshore control points: {len(offshore)}")

    # Check for missing values
    for name, subset in [('onshore', onshore), ('offshore', offshore)]:
        n_missing_scalar = subset['scalar'].isna().sum()
        n_missing_offset = subset['offset'].isna().sum()
        if n_missing_scalar > 0 or n_missing_offset > 0:
            warnings.warn(f"{name}: {n_missing_scalar} missing scalars, {n_missing_offset} missing offsets")

    return onshore, offshore


def create_europe_grid(resolution=0.25, extent=EUROPE_EXTENT):
    """Create regular grid covering Europe.

    Args:
        resolution: Grid spacing in degrees.
        extent: Dict with lon_min, lon_max, lat_min, lat_max.

    Returns:
        xarray Dataset with lon/lat coordinates.
    """
    print(f"\nCreating Europe grid at {resolution}° resolution...")

    lons = np.arange(extent['lon_min'], extent['lon_max'] + resolution, resolution)
    lats = np.arange(extent['lat_min'], extent['lat_max'] + resolution, resolution)

    print(f"  ✓ Grid shape: {len(lats)} lat × {len(lons)} lon")
    print(f"  ✓ Total grid points: {len(lats) * len(lons)}")

    ds = xr.Dataset(
        coords={
            'lat': lats,
            'lon': lons,
        }
    )

    return ds


# ============================================================================
# Interpolation methods
# ============================================================================

def interpolate_nearest(control_points, grid_lons, grid_lats):
    """Nearest neighbor interpolation.

    Args:
        control_points: DataFrame with lon, lat, scalar, offset columns.
        grid_lons: 1D array of grid longitudes.
        grid_lats: 1D array of grid latitudes.

    Returns:
        Tuple of (scalar_grid, offset_grid) as 2D arrays.
    """
    print("  Running nearest neighbor interpolation...")

    # Extract control point coordinates and values
    coords = control_points[['lon', 'lat']].values
    scalars = control_points['scalar'].values
    offsets = control_points['offset'].values

    # Create grid of target coordinates
    lon_grid, lat_grid = np.meshgrid(grid_lons, grid_lats)
    target_coords = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])

    # Find nearest neighbor for each grid point
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn.fit(coords)
    distances, indices = nn.kneighbors(target_coords)

    # Map values
    scalar_grid = scalars[indices.ravel()].reshape(lon_grid.shape)
    offset_grid = offsets[indices.ravel()].reshape(lon_grid.shape)

    print(f"    ✓ Interpolated to {lon_grid.size} grid points")

    return scalar_grid, offset_grid


def interpolate_idw(control_points, grid_lons, grid_lats, power=2.0, k=None):
    """Inverse Distance Weighting interpolation.

    Args:
        control_points: DataFrame with lon, lat, scalar, offset columns.
        grid_lons: 1D array of grid longitudes.
        grid_lats: 1D array of grid latitudes.
        power: IDW power parameter (default 2.0).
        k: Number of nearest neighbors to use (default None = all points).

    Returns:
        Tuple of (scalar_grid, offset_grid) as 2D arrays.
    """
    print(f"  Running IDW interpolation (power={power}, k={k})...")

    # Extract control point coordinates and values
    coords = control_points[['lon', 'lat']].values
    scalars = control_points['scalar'].values
    offsets = control_points['offset'].values

    # Create grid of target coordinates
    lon_grid, lat_grid = np.meshgrid(grid_lons, grid_lats)
    target_coords = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])

    # For large grids, process in batches
    batch_size = 10000
    n_targets = len(target_coords)

    scalar_interp = np.zeros(n_targets)
    offset_interp = np.zeros(n_targets)

    for i in range(0, n_targets, batch_size):
        batch = target_coords[i:i+batch_size]

        # Compute distances
        dists = distance.cdist(batch, coords, metric='euclidean')

        # Handle exact matches (distance = 0)
        exact_match = dists == 0

        if k is not None:
            # Use only k nearest neighbors
            nearest_indices = np.argsort(dists, axis=1)[:, :k]
            dists_k = np.take_along_axis(dists, nearest_indices, axis=1)
            scalars_k = scalars[nearest_indices]
            offsets_k = offsets[nearest_indices]
        else:
            dists_k = dists
            scalars_k = np.tile(scalars, (len(batch), 1))
            offsets_k = np.tile(offsets, (len(batch), 1))

        # Compute weights
        weights = 1.0 / (dists_k ** power + 1e-10)  # Add small value to avoid division by zero
        weights = weights / weights.sum(axis=1, keepdims=True)

        # Interpolate
        scalar_interp[i:i+batch_size] = (weights * scalars_k).sum(axis=1)
        offset_interp[i:i+batch_size] = (weights * offsets_k).sum(axis=1)

        # Fix exact matches
        if k is None:
            for j, row_match in enumerate(exact_match[i:i+batch_size]):
                if row_match.any():
                    idx = np.where(row_match)[0][0]
                    scalar_interp[i+j] = scalars[idx]
                    offset_interp[i+j] = offsets[idx]

    # Reshape to grid
    scalar_grid = scalar_interp.reshape(lon_grid.shape)
    offset_grid = offset_interp.reshape(lon_grid.shape)

    print(f"    ✓ Interpolated to {lon_grid.size} grid points")

    return scalar_grid, offset_grid


def interpolate_kriging(control_points, grid_lons, grid_lats):
    """Ordinary Kriging interpolation.

    Args:
        control_points: DataFrame with lon, lat, scalar, offset columns.
        grid_lons: 1D array of grid longitudes.
        grid_lats: 1D array of grid latitudes.

    Returns:
        Tuple of (scalar_grid, offset_grid) as 2D arrays.
    """
    if not KRIGING_AVAILABLE:
        raise ImportError("pykrige not available - cannot perform kriging")

    print("  Running Ordinary Kriging interpolation...")

    # Extract control point coordinates and values
    lons = control_points['lon'].values
    lats = control_points['lat'].values
    scalars = control_points['scalar'].values
    offsets = control_points['offset'].values

    # Create grid
    lon_grid, lat_grid = np.meshgrid(grid_lons, grid_lats)

    # Interpolate scalar
    print("    Kriging scalar...")
    ok_scalar = OrdinaryKriging(
        lons, lats, scalars,
        variogram_model='spherical',
        verbose=False,
        enable_plotting=False
    )
    scalar_grid, _ = ok_scalar.execute('grid', grid_lons, grid_lats)

    # Interpolate offset
    print("    Kriging offset...")
    ok_offset = OrdinaryKriging(
        lons, lats, offsets,
        variogram_model='spherical',
        verbose=False,
        enable_plotting=False
    )
    offset_grid, _ = ok_offset.execute('grid', grid_lons, grid_lats)

    print(f"    ✓ Interpolated to {lon_grid.size} grid points")

    return scalar_grid, offset_grid


def interpolate_rbf(control_points, grid_lons, grid_lats, kernel='thin_plate_spline'):
    """Radial Basis Function interpolation.

    Args:
        control_points: DataFrame with lon, lat, scalar, offset columns.
        grid_lons: 1D array of grid longitudes.
        grid_lats: 1D array of grid latitudes.
        kernel: RBF kernel type.

    Returns:
        Tuple of (scalar_grid, offset_grid) as 2D arrays.
    """
    print(f"  Running RBF interpolation (kernel={kernel})...")

    # Extract control point coordinates and values
    coords = control_points[['lon', 'lat']].values
    scalars = control_points['scalar'].values
    offsets = control_points['offset'].values

    # Create grid of target coordinates
    lon_grid, lat_grid = np.meshgrid(grid_lons, grid_lats)
    target_coords = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])

    # Interpolate scalar
    rbf_scalar = RBFInterpolator(coords, scalars, kernel=kernel)
    scalar_interp = rbf_scalar(target_coords)
    scalar_grid = scalar_interp.reshape(lon_grid.shape)

    # Interpolate offset
    rbf_offset = RBFInterpolator(coords, offsets, kernel=kernel)
    offset_interp = rbf_offset(target_coords)
    offset_grid = offset_interp.reshape(lon_grid.shape)

    print(f"    ✓ Interpolated to {lon_grid.size} grid points")

    return scalar_grid, offset_grid


def interpolate_to_grid(control_points, grid_ds, method='idw', **kwargs):
    """Interpolate control points to grid using specified method.

    Args:
        control_points: DataFrame with lon, lat, scalar, offset columns.
        grid_ds: xarray Dataset with lat/lon coordinates.
        method: Interpolation method ('nearest', 'idw', 'kriging', 'rbf').
        **kwargs: Additional arguments for interpolation method.

    Returns:
        xarray Dataset with scalar and offset fields.
    """
    print(f"\nInterpolating using method: {method.upper()}")

    grid_lons = grid_ds['lon'].values
    grid_lats = grid_ds['lat'].values

    # Remove any NaN values from control points
    control_points = control_points.dropna(subset=['lon', 'lat', 'scalar', 'offset'])

    if len(control_points) == 0:
        raise ValueError("No valid control points after removing NaN values")

    # Call appropriate interpolation method
    if method == 'nearest':
        scalar_grid, offset_grid = interpolate_nearest(control_points, grid_lons, grid_lats)
    elif method == 'idw':
        power = kwargs.get('power', IDW_POWER)
        k = kwargs.get('k', None)
        scalar_grid, offset_grid = interpolate_idw(control_points, grid_lons, grid_lats, power=power, k=k)
    elif method == 'kriging':
        scalar_grid, offset_grid = interpolate_kriging(control_points, grid_lons, grid_lats)
    elif method == 'rbf':
        kernel = kwargs.get('kernel', RBF_KERNEL)
        scalar_grid, offset_grid = interpolate_rbf(control_points, grid_lons, grid_lats, kernel=kernel)
    else:
        raise ValueError(f"Unknown interpolation method: {method}")

    # Create output dataset
    ds = xr.Dataset(
        {
            'scalar': (['lat', 'lon'], scalar_grid),
            'offset': (['lat', 'lon'], offset_grid),
        },
        coords={
            'lat': grid_lats,
            'lon': grid_lons,
        },
        attrs={
            'method': method,
            'n_control_points': len(control_points),
            'description': 'Bias correction factors interpolated to grid',
        }
    )

    return ds


def apply_distance_mask(ds, control_points, max_distance_deg=5.0):
    """Apply neutral correction values (scalar=1, offset=0) to grid points far from control points.

    Sets grid points beyond max_distance from any control point to neutral values,
    preventing unrealistic extrapolation into untrained regions.

    Args:
        ds: xarray Dataset with scalar and offset fields.
        control_points: DataFrame with lon, lat of control points.
        max_distance_deg: Maximum distance (in degrees) from control points to apply interpolated values.
                          Points farther than this get neutral values (scalar=1, offset=0).

    Returns:
        xarray Dataset with masked corrections.
    """
    print(f"\nApplying distance mask (max distance: {max_distance_deg}°)...")

    # Create grid of coordinates
    lon_grid, lat_grid = np.meshgrid(ds['lon'].values, ds['lat'].values)
    grid_coords = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])

    # Control point coordinates
    control_coords = control_points[['lon', 'lat']].values

    # Compute distance from each grid point to nearest control point
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn.fit(control_coords)
    distances, _ = nn.kneighbors(grid_coords)

    # Reshape distance to grid
    distance_grid = distances.reshape(lon_grid.shape)

    # Create mask: True where we should use interpolated values
    valid_mask = distance_grid <= max_distance_deg

    # Count masked points
    n_total = valid_mask.size
    n_valid = valid_mask.sum()
    n_masked = n_total - n_valid
    pct_masked = 100 * n_masked / n_total

    print(f"  ✓ Valid interpolation region: {n_valid:,} points ({100 - pct_masked:.1f}%)")
    print(f"  ✓ Neutral value region: {n_masked:,} points ({pct_masked:.1f}%)")

    # Apply mask: set distant points to neutral values
    ds_masked = ds.copy(deep=True)
    ds_masked['scalar'] = xr.where(valid_mask, ds['scalar'], 1.0)
    ds_masked['offset'] = xr.where(valid_mask, ds['offset'], 0.0)

    # Add distance field for visualization
    ds_masked['distance_to_nearest_control'] = (['lat', 'lon'], distance_grid)

    # Update attributes (use int instead of bool for NetCDF compatibility)
    ds_masked.attrs['max_distance_deg'] = max_distance_deg
    ds_masked.attrs['neutral_value_applied'] = 1  # 1 = True, 0 = False

    return ds_masked


# ============================================================================
# Cross-validation
# ============================================================================

def spatial_cv_split(control_points, n_splits=5):
    """Create spatial cross-validation splits.

    Splits control points spatially (by longitude) to avoid spatial
    autocorrelation bias in validation.

    Args:
        control_points: DataFrame with lon, lat, scalar, offset.
        n_splits: Number of CV folds.

    Returns:
        List of (train_indices, test_indices) tuples.
    """
    # Sort by longitude
    sorted_idx = control_points['lon'].argsort()
    n = len(control_points)

    # Create roughly equal-sized spatial folds
    fold_size = n // n_splits
    splits = []

    for i in range(n_splits):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < n_splits - 1 else n

        test_indices = sorted_idx[test_start:test_end]
        train_indices = np.concatenate([sorted_idx[:test_start], sorted_idx[test_end:]])

        splits.append((train_indices, test_indices))

    return splits


def evaluate_interpolation_cv(control_points, method='idw', n_splits=5, **kwargs):
    """Evaluate interpolation method using cross-validation.

    Args:
        control_points: DataFrame with lon, lat, scalar, offset.
        method: Interpolation method.
        n_splits: Number of CV folds.
        **kwargs: Additional arguments for interpolation method.

    Returns:
        Dict with CV scores (MAE, RMSE) for scalar and offset.
    """
    print(f"\nCross-validating {method.upper()} ({n_splits} folds)...")

    splits = spatial_cv_split(control_points, n_splits=n_splits)

    scalar_mae_scores = []
    scalar_rmse_scores = []
    offset_mae_scores = []
    offset_rmse_scores = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        print(f"  Fold {fold_idx + 1}/{n_splits}...")

        train_data = control_points.iloc[train_idx]
        test_data = control_points.iloc[test_idx]

        # Create temporary grid containing only test locations
        test_lons = test_data['lon'].values
        test_lats = test_data['lat'].values

        # Interpolate
        if method == 'nearest':
            scalar_pred, offset_pred = interpolate_nearest(
                train_data, test_lons, test_lats
            )
        elif method == 'idw':
            power = kwargs.get('power', IDW_POWER)
            k = kwargs.get('k', None)
            # For CV, use point interpolation
            scalar_pred = []
            offset_pred = []
            for lon, lat in zip(test_lons, test_lats):
                s, o = interpolate_idw_point(train_data, lon, lat, power=power, k=k)
                scalar_pred.append(s)
                offset_pred.append(o)
            scalar_pred = np.array(scalar_pred)
            offset_pred = np.array(offset_pred)
        elif method == 'kriging':
            scalar_pred, offset_pred = interpolate_kriging_points(
                train_data, test_lons, test_lats
            )
        elif method == 'rbf':
            kernel = kwargs.get('kernel', RBF_KERNEL)
            scalar_pred, offset_pred = interpolate_rbf_points(
                train_data, test_lons, test_lats, kernel=kernel
            )

        # Compute errors
        scalar_true = test_data['scalar'].values
        offset_true = test_data['offset'].values

        scalar_mae = np.abs(scalar_pred - scalar_true).mean()
        scalar_rmse = np.sqrt(((scalar_pred - scalar_true) ** 2).mean())
        offset_mae = np.abs(offset_pred - offset_true).mean()
        offset_rmse = np.sqrt(((offset_pred - offset_true) ** 2).mean())

        scalar_mae_scores.append(scalar_mae)
        scalar_rmse_scores.append(scalar_rmse)
        offset_mae_scores.append(offset_mae)
        offset_rmse_scores.append(offset_rmse)

        print(f"    Scalar - MAE: {scalar_mae:.4f}, RMSE: {scalar_rmse:.4f}")
        print(f"    Offset - MAE: {offset_mae:.4f}, RMSE: {offset_rmse:.4f}")

    results = {
        'scalar_mae_mean': np.mean(scalar_mae_scores),
        'scalar_mae_std': np.std(scalar_mae_scores),
        'scalar_rmse_mean': np.mean(scalar_rmse_scores),
        'scalar_rmse_std': np.std(scalar_rmse_scores),
        'offset_mae_mean': np.mean(offset_mae_scores),
        'offset_mae_std': np.std(offset_mae_scores),
        'offset_rmse_mean': np.mean(offset_rmse_scores),
        'offset_rmse_std': np.std(offset_rmse_scores),
    }

    print(f"\n  {method.upper()} CV Results:")
    print(f"    Scalar  - MAE: {results['scalar_mae_mean']:.4f} ± {results['scalar_mae_std']:.4f}")
    print(f"            - RMSE: {results['scalar_rmse_mean']:.4f} ± {results['scalar_rmse_std']:.4f}")
    print(f"    Offset  - MAE: {results['offset_mae_mean']:.4f} ± {results['offset_mae_std']:.4f}")
    print(f"            - RMSE: {results['offset_rmse_mean']:.4f} ± {results['offset_rmse_std']:.4f}")

    return results


def interpolate_idw_point(control_points, lon, lat, power=2.0, k=None):
    """IDW interpolation to single point."""
    coords = control_points[['lon', 'lat']].values
    scalars = control_points['scalar'].values
    offsets = control_points['offset'].values

    target = np.array([[lon, lat]])
    dists = distance.cdist(target, coords, metric='euclidean')[0]

    # Handle exact matches
    if (dists == 0).any():
        idx = np.where(dists == 0)[0][0]
        return scalars[idx], offsets[idx]

    # Use k nearest if specified
    if k is not None:
        nearest_idx = np.argsort(dists)[:k]
        dists = dists[nearest_idx]
        scalars = scalars[nearest_idx]
        offsets = offsets[nearest_idx]

    weights = 1.0 / (dists ** power)
    weights /= weights.sum()

    scalar = (weights * scalars).sum()
    offset = (weights * offsets).sum()

    return scalar, offset


def interpolate_kriging_points(control_points, test_lons, test_lats):
    """Kriging interpolation to multiple points."""
    if not KRIGING_AVAILABLE:
        raise ImportError("pykrige not available")

    lons = control_points['lon'].values
    lats = control_points['lat'].values
    scalars = control_points['scalar'].values
    offsets = control_points['offset'].values

    # Scalar
    ok_scalar = OrdinaryKriging(
        lons, lats, scalars,
        variogram_model='spherical',
        verbose=False,
        enable_plotting=False
    )
    scalar_pred = np.array([ok_scalar.execute('points', lon, lat)[0][0]
                            for lon, lat in zip(test_lons, test_lats)])

    # Offset
    ok_offset = OrdinaryKriging(
        lons, lats, offsets,
        variogram_model='spherical',
        verbose=False,
        enable_plotting=False
    )
    offset_pred = np.array([ok_offset.execute('points', lon, lat)[0][0]
                            for lon, lat in zip(test_lons, test_lats)])

    return scalar_pred, offset_pred


def interpolate_rbf_points(control_points, test_lons, test_lats, kernel='thin_plate_spline'):
    """RBF interpolation to multiple points."""
    coords = control_points[['lon', 'lat']].values
    scalars = control_points['scalar'].values
    offsets = control_points['offset'].values

    target_coords = np.column_stack([test_lons, test_lats])

    rbf_scalar = RBFInterpolator(coords, scalars, kernel=kernel)
    scalar_pred = rbf_scalar(target_coords)

    rbf_offset = RBFInterpolator(coords, offsets, kernel=kernel)
    offset_pred = rbf_offset(target_coords)

    return scalar_pred, offset_pred


# ============================================================================
# Visualization
# ============================================================================

def plot_interpolation_comparison(datasets, control_points, output_dir):
    """Create comparison maps of different interpolation methods.

    Args:
        datasets: Dict of {method: xr.Dataset} with interpolated grids.
        control_points: DataFrame with control point locations.
        output_dir: Path to save plots.
    """
    print("\nCreating interpolation comparison plots...")

    # Apply thesis plotting style
    style = thesis_plot_style()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_methods = len(datasets)

    # Plot scalar comparisons
    cm = style['cm']
    fig, axes = plt.subplots(2, (n_methods + 1) // 2, figsize=(18*cm, 10*cm))
    axes = axes.flatten()

    # Compute scalar range across all methods
    scalar_min = min([ds['scalar'].min().values for ds in datasets.values()])
    scalar_max = max([ds['scalar'].max().values for ds in datasets.values()])

    # Create diverging colormap centered at 1.0 (identity)
    scalar_norm = mcolors.TwoSlopeNorm(vmin=scalar_min, vcenter=1.0, vmax=scalar_max)

    for idx, (method, ds) in enumerate(datasets.items()):
        ax = axes[idx]

        # Plot interpolated scalar field with diverging colormap
        im = ax.pcolormesh(
            ds['lon'], ds['lat'], ds['scalar'],
            cmap='RdBu_r', norm=scalar_norm, shading='auto'
        )

        # Overlay control points
        ax.scatter(
            control_points['lon'], control_points['lat'],
            c='black', s=2, alpha=0.4, marker='x', linewidths=0.5
        )

        ax.set_title(f'{method.upper()} - Scalar', fontweight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_aspect('equal')
        cbar = plt.colorbar(im, ax=ax, label='Scalar')
        # Add horizontal line at identity value (1.0)
        cbar.ax.axhline(1.0, color='black', linewidth=1.5, linestyle='--', alpha=0.7)

    # Remove empty subplots
    for idx in range(n_methods, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    scalar_path = output_dir / 'interpolation_comparison_scalar.png'
    plt.savefig(scalar_path, dpi=style['dpi'], bbox_inches='tight')
    print(f"  ✓ Saved: {scalar_path}")
    plt.close()

    # Plot offset comparisons
    fig, axes = plt.subplots(2, (n_methods + 1) // 2, figsize=(18*cm, 10*cm))
    axes = axes.flatten()

    # Compute offset range across all methods
    offset_min = min([ds['offset'].min().values for ds in datasets.values()])
    offset_max = max([ds['offset'].max().values for ds in datasets.values()])

    # Create diverging colormap centered at 0.0 (identity)
    offset_norm = mcolors.TwoSlopeNorm(vmin=offset_min, vcenter=0.0, vmax=offset_max)

    for idx, (method, ds) in enumerate(datasets.items()):
        ax = axes[idx]

        # Plot interpolated offset field with diverging colormap
        im = ax.pcolormesh(
            ds['lon'], ds['lat'], ds['offset'],
            cmap='RdBu_r', norm=offset_norm, shading='auto'
        )

        # Overlay control points
        ax.scatter(
            control_points['lon'], control_points['lat'],
            c='black', s=2, alpha=0.4, marker='x', linewidths=0.5
        )

        ax.set_title(f'{method.upper()} - Offset', fontweight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_aspect('equal')
        cbar = plt.colorbar(im, ax=ax, label='Offset (m/s)')
        # Add horizontal line at identity value (0.0)
        cbar.ax.axhline(0.0, color='black', linewidth=1.5, linestyle='--', alpha=0.7)

    # Remove empty subplots
    for idx in range(n_methods, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    offset_path = output_dir / 'interpolation_comparison_offset.png'
    plt.savefig(offset_path, dpi=style['dpi'], bbox_inches='tight')
    print(f"  ✓ Saved: {offset_path}")
    plt.close()


def plot_cv_scores(cv_results, output_dir):
    """Plot cross-validation scores comparison.

    Args:
        cv_results: Dict of {method: results_dict}.
        output_dir: Path to save plot.
    """
    print("\nPlotting CV scores...")

    # Apply thesis plotting style
    style = thesis_plot_style()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = list(cv_results.keys())

    # Extract scores
    scalar_mae = [cv_results[m]['scalar_mae_mean'] for m in methods]
    scalar_mae_std = [cv_results[m]['scalar_mae_std'] for m in methods]
    scalar_rmse = [cv_results[m]['scalar_rmse_mean'] for m in methods]
    scalar_rmse_std = [cv_results[m]['scalar_rmse_std'] for m in methods]

    offset_mae = [cv_results[m]['offset_mae_mean'] for m in methods]
    offset_mae_std = [cv_results[m]['offset_mae_std'] for m in methods]
    offset_rmse = [cv_results[m]['offset_rmse_mean'] for m in methods]
    offset_rmse_std = [cv_results[m]['offset_rmse_std'] for m in methods]

    # Create plot
    cm = style['cm']
    fig, axes = plt.subplots(1, 2, figsize=(16*cm, 6*cm))

    # Scalar
    ax = axes[0]
    x = np.arange(len(methods))
    width = 0.35

    ax.bar(x - width/2, scalar_mae, width, yerr=scalar_mae_std,
           label='MAE', alpha=0.8, capsize=3)
    ax.bar(x + width/2, scalar_rmse, width, yerr=scalar_rmse_std,
           label='RMSE', alpha=0.8, capsize=3)

    ax.set_xlabel('Method')
    ax.set_ylabel('Error')
    ax.set_title('Scalar CV Scores', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in methods])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Offset
    ax = axes[1]

    ax.bar(x - width/2, offset_mae, width, yerr=offset_mae_std,
           label='MAE', alpha=0.8, capsize=3)
    ax.bar(x + width/2, offset_rmse, width, yerr=offset_rmse_std,
           label='RMSE', alpha=0.8, capsize=3)

    ax.set_xlabel('Method')
    ax.set_ylabel('Error (m/s)')
    ax.set_title('Offset CV Scores', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in methods])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    cv_path = output_dir / 'cv_scores_comparison.png'
    plt.savefig(cv_path, dpi=style['dpi'], bbox_inches='tight')
    print(f"  ✓ Saved: {cv_path}")
    plt.close()


# ============================================================================
# Main execution
# ============================================================================

def main():
    """Main execution."""
    print("="*80)
    print("Unified Corrections Grid Comparison")
    print("="*80)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load unified corrections
    df = load_unified_corrections(UNIFIED_CORRECTIONS_CSV)

    # Prepare control points (onshore + offshore)
    onshore, offshore = prepare_control_points(df)

    # For now, combine onshore and offshore for interpolation
    # (In future, could do separate interpolations and merge)
    all_points = pd.concat([onshore, offshore], ignore_index=True)

    print(f"\n{'='*80}")
    print(f"Using {len(all_points)} total control points for interpolation")
    print(f"{'='*80}")

    # Create Europe grid
    grid_ds = create_europe_grid(resolution=GRID_RESOLUTION, extent=EUROPE_EXTENT)

    # Interpolate using all methods
    print(f"\n{'='*80}")
    print("Running interpolations")
    print(f"{'='*80}")

    interpolated_datasets = {}
    for method in METHODS:
        try:
            ds = interpolate_to_grid(all_points, grid_ds, method=method)

            # Apply distance mask to set neutral values (scalar=1, offset=0) for distant regions
            ds_masked = apply_distance_mask(ds, all_points, max_distance_deg=MAX_DISTANCE_DEG)
            interpolated_datasets[method] = ds_masked

            # Save masked version to NetCDF
            nc_path = OUTPUT_DIR / f'europe_corrections_{method}.nc'
            ds_masked.to_netcdf(nc_path)
            print(f"  ✓ Saved: {nc_path}")

            # Also save unmasked version for comparison
            nc_path_unmasked = OUTPUT_DIR / f'europe_corrections_{method}_unmasked.nc'
            ds.to_netcdf(nc_path_unmasked)
            print(f"  ✓ Saved (unmasked): {nc_path_unmasked}")

        except Exception as e:
            print(f"  ✗ Error with {method}: {e}")
            continue

    # Cross-validation
    print(f"\n{'='*80}")
    print("Running cross-validation")
    print(f"{'='*80}")

    cv_results = {}
    for method in METHODS:
        if method == 'nearest':
            # Skip CV for nearest (too slow for 1835 points)
            print(f"\n  Skipping CV for {method.upper()} (not informative)")
            continue

        try:
            results = evaluate_interpolation_cv(
                all_points, method=method, n_splits=CV_FOLDS
            )
            cv_results[method] = results
        except Exception as e:
            print(f"  ✗ Error with {method}: {e}")
            continue

    # Save CV results to CSV
    if cv_results:
        cv_df = pd.DataFrame(cv_results).T
        cv_path = OUTPUT_DIR / 'cv_scores.csv'
        cv_df.to_csv(cv_path)
        print(f"\n  ✓ Saved CV scores: {cv_path}")

    # Visualizations
    print(f"\n{'='*80}")
    print("Creating visualizations")
    print(f"{'='*80}")

    if interpolated_datasets:
        plot_interpolation_comparison(interpolated_datasets, all_points, OUTPUT_DIR)

    if cv_results:
        plot_cv_scores(cv_results, OUTPUT_DIR)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    print(f"\nControl points: {len(all_points)}")
    print(f"  - Onshore: {len(onshore)}")
    print(f"  - Offshore: {len(offshore)}")

    print(f"\nDistance masking applied:")
    print(f"  - Max distance: {MAX_DISTANCE_DEG}° (~{int(MAX_DISTANCE_DEG * 111)}km at equator)")
    print(f"  - Neutral values (scalar=1, offset=0) applied beyond this distance")

    print(f"\nInterpolation methods completed: {len(interpolated_datasets)}")
    for method in interpolated_datasets:
        print(f"  - {method.upper()}")

    if cv_results:
        print(f"\nCross-validation results ({CV_FOLDS} folds):")
        for method, results in cv_results.items():
            print(f"\n  {method.upper()}:")
            print(f"    Scalar  - MAE: {results['scalar_mae_mean']:.4f} ± {results['scalar_mae_std']:.4f}")
            print(f"            - RMSE: {results['scalar_rmse_mean']:.4f} ± {results['scalar_rmse_std']:.4f}")
            print(f"    Offset  - MAE: {results['offset_mae_mean']:.4f} ± {results['offset_mae_std']:.4f}")
            print(f"            - RMSE: {results['offset_rmse_mean']:.4f} ± {results['offset_rmse_std']:.4f}")

    print(f"\n{'='*80}")
    print("✓ COMPLETE")
    print(f"{'='*80}")
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("\nFiles created:")
    print("  - europe_corrections_<method>.nc (NetCDF grids with distance masking)")
    print("  - europe_corrections_<method>_unmasked.nc (without distance masking)")
    print("  - interpolation_comparison_scalar.png")
    print("  - interpolation_comparison_offset.png")
    print("  - cv_scores.csv")
    print("  - cv_scores_comparison.png")
    print("\nNote: Masked grids use neutral values (scalar=1, offset=0) beyond {:.1f}° from control points".format(MAX_DISTANCE_DEG))


if __name__ == "__main__":
    main()
