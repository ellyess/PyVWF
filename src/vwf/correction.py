"""Bias correction utilities for PyVWF."""
import numpy as np
from scipy.optimize import minimize, minimize_scalar

from vwf.wind import interpolate_wind, train_simulate_wind, prepare_offset_arrays, fast_simulate_cf
from vwf.time_utils import parse_time_slice


def calculate_scalar(gen_cf, time_res):
    """Calculate multiplicative (scalar) correction factors.

    Args:
        gen_cf: DataFrame with observed and simulated capacity factors.
        time_res: Time resolution used for aggregation.

    Returns:
        pandas.DataFrame: DataFrame with ``year``, ``time_slice``, ``cluster``,
        ``obs``, ``sim``, and ``scalar`` columns.
    """
    # # Simple mean aggregation (no capacity weighting)
    # # Scalars represent spatial reanalysis bias, not capacity distribution
    # # Capacity weighting should only occur during final aggregation to country level
    # df = gen_cf.groupby([time_res, 'cluster', 'year']).agg({
    #                         "obs": "mean",
    #                         "sim": "mean",
    #                         })
    
    # OLD APPROACH: Capacity-weighted averaging (commented out)
    # This was causing double-weighting issues where scalars were influenced by turbine size
    # rather than just representing the meteorological bias at that location

    def weighted_avg(group_df, whole_df, values, weights):
        """Compute a weighted average for a group."""
        v = whole_df.loc[group_df.index, values]
        w = whole_df.loc[group_df.index, weights]
        # Use min_count=1 so that all-NaN groups return NaN instead of 0.0
        # (pd.Series.sum() returns 0.0 for all-NaN by default with skipna=True)
        return (v * w).sum(min_count=1) / w.sum()
        
    df = gen_cf.groupby([time_res, 'cluster', 'year']).agg({
                            "obs": lambda x: weighted_avg(x, gen_cf, 'obs', 'capacity'),
                            "sim": lambda x: weighted_avg(x, gen_cf, 'sim', 'capacity'),
                            })
        
    df['scalar'] = df['obs'] / df['sim']
    
    # Constrain scalars to prevent extreme corrections
    # Values outside [0.5, 1.5] indicate potential overfitting or data issues
    # df['scalar'] = df['scalar'].clip(lower=0.1, upper=2.0)
    
    df = df.reset_index()
    df.columns = ['time_slice', 'cluster', 'year', 'obs', 'sim', 'scalar']
        
    return df[['year', 'time_slice', 'cluster', 'obs', 'sim', 'scalar']]
    
def _find_offset_iterative(row, offset_arrays,
                           max_iter=100, tolerance=0.002, initial_step=10.0):
    """Fast iterative optimization using cube root step sizing.

    Args:
        row: Row with year, cluster, time_slice, obs, sim, scalar
        offset_arrays: Pre-extracted numpy arrays from prepare_offset_arrays
        max_iter: Maximum iterations (default: 100)
        tolerance: Convergence tolerance
        initial_step: Initial step size (default: 10.0 m/s)

    Returns:
        float: Optimized offset (or np.nan if failed)
    """
    step = np.sign(row.obs - row.sim) * initial_step
    step_prev = step
    offset = 0.0

    for _ in range(max_iter):
        # Check convergence
        if np.abs(step) <= tolerance:
            return offset

        # Simulate with current offset using fast numpy path
        mean_sim_cf = fast_simulate_cf(offset_arrays, row.scalar, offset)

        # Calculate error and step (cube root for power ~ wind^3)
        error = row.obs - mean_sim_cf
        step = np.cbrt(error)

        # Prevent oscillation
        if np.sign(step) != np.sign(step_prev) and np.abs(step) > np.abs(step_prev):
            step = -step_prev / 2
        elif np.sign(step) == np.sign(step_prev) and np.abs(step) > np.abs(step_prev):
            step = step_prev / 2

        # Update
        step_prev = step
        offset += step

    # Failed to converge - fall back to scipy
    return np.nan


def _find_offset_scipy(row, offset_arrays, bounds=(-3, 3)):
    """Robust scipy optimization fallback.

    Args:
        row: Row with correction factors
        offset_arrays: Pre-extracted numpy arrays from prepare_offset_arrays
        bounds: Offset search bounds

    Returns:
        float: Optimized offset (or np.nan if failed)
    """
    def objective(offset):
        """Squared error between observed and simulated CF."""
        sim_cf = fast_simulate_cf(offset_arrays, row.scalar, offset)
        return (sim_cf - row.obs) ** 2

    try:
        result = minimize_scalar(
            objective,
            bounds=bounds,
            method='bounded',
            options={'xatol': 0.001}
        )

        if result.success:
            return result.x
        else:
            return np.nan

    except Exception:
        return np.nan


def find_offset(row, turb_info, reanalysis, powerCurveFile,
                max_iter=100, tolerance=0.002, initial_step=10.0,
                bounds=(-10, 10), use_scipy_fallback=True, verbose=False):
    """Optimize the additive offset correction factor.

    Uses a hybrid approach: fast iterative method with scipy fallback for robustness.

    Args:
        row (pandas.Series): Row with ``year``, ``cluster``, ``time_slice``, ``obs``, ``sim``, ``scalar``.
        turb_info (pandas.DataFrame): Turbine metadata including height and coordinates.
        reanalysis (xarray.Dataset): Wind parameters on a grid.
        powerCurveFile (pandas.DataFrame): Capacity factor vs. wind speed curves.
        max_iter (int): Maximum iterations for fast method (default: 100).
        tolerance (float): Convergence tolerance (default: 0.002).
        initial_step (float): Initial step size for fast method (default: 10.0).
        bounds (tuple): Offset bounds for scipy method (default: (-10, 10)).
        use_scipy_fallback (bool): Use scipy if fast method fails (default: True).
        verbose (bool): Print warnings for failed optimizations (default: False).

    Returns:
        float: Best-fit offset value (or np.nan if all methods fail).
    """
    # Parse time slice to months
    months = parse_time_slice(row['time_slice'])

    # Pre-filter reanalysis data once
    reanalysis_filtered = reanalysis.sel(
        time=np.logical_and(
            reanalysis.time.dt.year == row.year,
            reanalysis.time.dt.month.isin(months)
        )
    )

    # Pre-filter cluster turbines once
    cluster_turbs = turb_info.loc[turb_info['cluster'] == row.cluster].copy()

    # Handle case where cluster filtering returns empty
    if len(cluster_turbs) == 0:
        if verbose:
            print(f"Warning: No turbines found for cluster={row.cluster} (available: {sorted(turb_info['cluster'].unique())})")
        return np.nan

    # Pre-compute interpolated wind speeds once for this row
    unc_ws = interpolate_wind(reanalysis_filtered, cluster_turbs)

    # Pre-extract numpy arrays for fast iteration (avoids xarray overhead per iteration)
    offset_arrays = prepare_offset_arrays(unc_ws, powerCurveFile)

    # Try fast iterative method first
    offset = _find_offset_iterative(
        row, offset_arrays, max_iter, tolerance, initial_step
    )

    # If failed and fallback enabled, use scipy
    if np.isnan(offset) and use_scipy_fallback:
        offset = _find_offset_scipy(
            row, offset_arrays, bounds
        )

    # Optional warning for failed optimizations
    if verbose and np.isnan(offset):
        print(f"Warning: Offset optimization failed for cluster={row.cluster}, "
              f"year={row.year}, time_slice={row['time_slice']}")

    return offset


def find_offsets_country_level(year, time_slice, obs_country_cf, scalars_by_cluster, turb_info, reanalysis, powerCurveFile):
    """Optimize offsets for all clusters in country-level mode.

    For country-level data, all clusters share the same country-wide observation.
    This function optimizes each cluster's offset simultaneously to minimize the
    error when aggregated to country level.

    Args:
        year: Year to process
        time_slice: Time period (e.g., '1/1' for fixed, 'winter' for season)
        obs_country_cf: Observed country-wide capacity factor
        scalars_by_cluster: Dict mapping cluster ID to scalar value
        turb_info: Turbine/grid point metadata with cluster assignments
        reanalysis: xarray Dataset with wind data
        powerCurveFile: Power curve lookup table

    Returns:
        dict: Mapping of cluster ID to optimized offset value
    """
    # Parse time_slice to month list
    months = parse_time_slice(time_slice)

    # Filter reanalysis to time period
    reanalysis_period = reanalysis.sel(
        time=np.logical_and(
            reanalysis.time.dt.year == year,
            reanalysis.time.dt.month.isin(months)
        )
    )

    # Get cluster IDs
    clusters = sorted(turb_info['cluster'].unique())

    # Get capacity weights for aggregation
    capacity_by_cluster = turb_info.groupby('cluster')['capacity'].sum()

    def objective(offsets):
        """Objective function: squared error between simulated and observed country CF."""
        cluster_cfs = []
        cluster_weights = []

        for i, cluster_id in enumerate(clusters):
            scalar = scalars_by_cluster.get(cluster_id, 1.0)
            offset = offsets[i]

            # Get turbines in this cluster
            cluster_turbs = turb_info[turb_info['cluster'] == cluster_id]

            if len(cluster_turbs) == 0:
                continue

            # Simulate with this cluster's corrections
            mean_cf = train_simulate_wind(
                reanalysis_period,
                cluster_turbs,
                powerCurveFile,
                scalar,
                offset
            )

            cluster_cfs.append(mean_cf)
            cluster_weights.append(capacity_by_cluster[cluster_id])

        # Aggregate to country level (capacity-weighted average)
        if len(cluster_cfs) == 0 or sum(cluster_weights) == 0:
            return 1e6  # Large penalty

        country_cf_sim = sum(cf * w for cf, w in zip(cluster_cfs, cluster_weights)) / sum(cluster_weights)

        # Return squared error
        error = (country_cf_sim - obs_country_cf) ** 2
        return error

    # Initial guess: all offsets = 0
    x0 = np.zeros(len(clusters))

    # Bounds: offsets between -10 and +10 m/s seem reasonable
    bounds = [(-10, 10) for _ in clusters]

    # Optimize
    try:
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-6}
        )

        # Return dict mapping cluster to offset
        offsets_dict = {cluster_id: result.x[i] for i, cluster_id in enumerate(clusters)}

        return offsets_dict

    except Exception as e:
        print(f"  Warning: Offset optimization failed for year={year}, time_slice={time_slice}: {e}")
        # Return zero offsets as fallback
        return {cluster_id: 0.0 for cluster_id in clusters}
