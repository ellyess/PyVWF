"""Bias correction utilities for PyVWF."""

import numpy as np
from scipy.optimize import brentq, minimize

from vwf.wind import interpolate_wind, train_simulate_wind, prepare_offset_arrays, fast_simulate_cf
from vwf.time_utils import parse_time_slice
from vwf.metrics import weighted_mean


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

    def weighted_avg(group_df, whole_df, values, weights, required=("obs", "sim")):
        """Compute a weighted average for a group, over the ROWS THAT HAVE A VALUE.

        ``sim`` exists for every turbine but ``obs`` does not. Dividing by the
        group's whole capacity would count plants that reported nothing in the
        denominator while the numerator skipped them, scaling ``obs`` down by
        the reporting fraction and leaving ``sim`` untouched, so the scalar
        came out as ``true_scalar * reporting_fraction``. On the real US fleet
        only 43% of capacity reports monthly, which turned a correct scalar of
        1.09 into 0.47 and made the correction worse than no correction at all,
        in-sample as well as out.

        The presence mask is shared across ``required`` rather than computed per
        column, so ``obs`` and ``sim`` are averaged over the SAME plants. Masking
        each column against only its own presence puts ``obs`` over the reporters
        and ``sim`` over the whole fleet, and the ratio then compares two
        different samples: the plant-to-plant spread in ``sim`` no longer cancels
        and the scalar carries the difference. That is harmless when reporting is
        independent of output, but reporting is not independent of output, and at
        the 43% reporting rate above a moderate correlation biases the scalar by
        about 7%, a strong one by about 14%, always in the same direction.
        """
        idx = group_df.index
        w = whole_df.loc[idx, weights]
        present = w.notna()
        for col in required:
            present &= whole_df.loc[idx, col].notna()
        v = whole_df.loc[idx, values]
        # min_count=1 so an all-missing group returns NaN rather than 0.0
        # (an empty sum is 0.0 by default), and NaN/0 keeps that NaN.
        # The mask is shared across ``required``, so it is applied here rather
        # than left to weighted_mean's own per-value masking.
        return weighted_mean(v.where(present), w.where(present))

    df = gen_cf.groupby([time_res, "cluster", "year"]).agg(
        {
            "obs": lambda x: weighted_avg(x, gen_cf, "obs", "capacity"),
            "sim": lambda x: weighted_avg(x, gen_cf, "sim", "capacity"),
        }
    )

    df["scalar"] = df["obs"] / df["sim"]

    # Constrain scalars to prevent extreme corrections
    # Values outside [0.5, 1.5] indicate potential overfitting or data issues
    # df['scalar'] = df['scalar'].clip(lower=0.1, upper=2.0)

    df = df.reset_index()
    df.columns = ["time_slice", "cluster", "year", "obs", "sim", "scalar"]

    return df[["year", "time_slice", "cluster", "obs", "sim", "scalar"]]


#: Width of each step the bracketed search takes outward from zero, in m/s,
#: while it looks for a change of sign in the residual.
OFFSET_BRACKET_STEP = 0.5

#: The bracketed search's tolerance on the offset itself, in m/s.
OFFSET_XTOL = 1e-6

#: Largest capacity-factor residual the bracketed search may leave. Brent's
#: method converges to within ``OFFSET_XTOL`` of the root, and the capacity
#: factor changes by at most about 0.2 per m/s of offset, so a genuine root
#: leaves a residual near 2e-7. A larger one means the residual changed sign by
#: jumping, not by crossing zero: an off-curve value dropping out of the mean.
BRACKETED_MAX_RESIDUAL = 1e-6


def _find_offset_bracketed(
    row,
    offset_arrays,
    bounds=(-10.0, 10.0),
    bracket_step=OFFSET_BRACKET_STEP,
    xtol=OFFSET_XTOL,
    residual_tolerance=BRACKETED_MAX_RESIDUAL,
):
    """The offset at which the simulated capacity factor equals the observed one.

    Steps outward from zero, in the direction that moves the simulation toward
    the observation, until the residual changes sign; then solves that bracket
    with Brent's method. So it returns the root nearest zero on that side, as
    the iterative search it replaced aimed to. The capacity factor is not
    monotonic in the offset once enough speeds pass the cut-out, which is why
    the bracket is found by stepping rather than taken from the bounds.

    Args:
        row: Row with ``obs``, ``scalar`` and the cluster's identity.
        offset_arrays: Pre-extracted numpy arrays from ``prepare_offset_arrays``.
        bounds: The search is confined to this interval, in m/s.
        bracket_step: Width of each outward step, in m/s.
        xtol: Tolerance on the offset, in m/s.
        residual_tolerance: Largest capacity-factor residual accepted.

    Returns:
        float: The offset, or ``np.nan`` when the search refuses: the residual
        does not change sign before a bound (no root inside the bounds on that
        side), the root lies at a bound, or the residual at the root exceeds
        ``residual_tolerance``. Each refusal is explicit, never a value at the
        edge of the interval.
    """
    lo, hi = float(bounds[0]), float(bounds[1])

    def residual(offset):
        return float(row.obs - fast_simulate_cf(offset_arrays, row.scalar, offset))

    start = 0.0
    f_start = residual(start)
    if not np.isfinite(f_start):
        return np.nan
    if f_start == 0.0:
        return start
    direction = 1.0 if f_start > 0 else -1.0  # observed above simulated: raise the speed
    a, f_a = start, f_start
    while True:
        b = min(max(a + direction * bracket_step, lo), hi)
        f_b = residual(b)
        if not np.isfinite(f_b):
            return np.nan
        if f_b == 0.0:
            root = b
            break
        if np.sign(f_b) != np.sign(f_a):
            root = brentq(residual, min(a, b), max(a, b), xtol=xtol)
            break
        if b in (lo, hi):
            return np.nan  # no change of sign inside the bounds
        a, f_a = b, f_b
    if min(abs(root - lo), abs(root - hi)) <= xtol:
        return np.nan  # a root at the edge of the search is not accepted
    if abs(residual(root)) > residual_tolerance:
        return np.nan
    return float(root)


def find_offset(
    row,
    turb_info,
    reanalysis,
    powerCurveFile,
    bounds=(-10, 10),
    verbose=False,
    seasons=None,
):
    """Optimize the additive offset correction factor.

    Solves for the root with :func:`_find_offset_bracketed`: a bracket found by
    stepping outward from zero, then Brent's method, refusing a result at a
    bound or with a residual above :data:`BRACKETED_MAX_RESIDUAL`.

    Args:
        row (pandas.Series): Row with ``year``, ``cluster``, ``time_slice``, ``obs``, ``sim``, ``scalar``.
        turb_info (pandas.DataFrame): Turbine metadata including height and coordinates.
        reanalysis (xarray.Dataset): Wind parameters on a grid.
        powerCurveFile (pandas.DataFrame): Capacity factor vs. wind speed curves.
        bounds (tuple): The interval the search is confined to, in m/s
            (default: (-10, 10)). A root at either end is refused.
        verbose (bool): Print warnings for failed optimizations (default: False).
        seasons: Optional season-name → month-list mapping, for regions
            whose seasons differ from the Northern-Hemisphere defaults.
            Default None keeps the hardcoded NH season months.

    Returns:
        float: The offset, or np.nan when the search refuses (see
        :func:`_find_offset_bracketed`) or the cluster has no units.
    """
    # Parse time slice to months
    months = parse_time_slice(row["time_slice"], seasons)

    # Pre-filter reanalysis data once
    reanalysis_filtered = reanalysis.sel(
        time=np.logical_and(
            reanalysis.time.dt.year == row.year, reanalysis.time.dt.month.isin(months)
        )
    )

    # Pre-filter cluster turbines once
    cluster_turbs = turb_info.loc[turb_info["cluster"] == row.cluster].copy()

    # Handle case where cluster filtering returns empty
    if len(cluster_turbs) == 0:
        if verbose:
            print(
                f"Warning: No turbines found for cluster={row.cluster} (available: {sorted(turb_info['cluster'].unique())})"
            )
        return np.nan

    # Pre-compute interpolated wind speeds once for this row
    unc_ws = interpolate_wind(reanalysis_filtered, cluster_turbs)

    # Pre-extract numpy arrays for fast iteration (avoids xarray overhead per iteration)
    offset_arrays = prepare_offset_arrays(unc_ws, powerCurveFile)

    offset = _find_offset_bracketed(row, offset_arrays, bounds=bounds)

    # Optional warning for failed optimizations
    if verbose and np.isnan(offset):
        print(
            f"Warning: Offset optimization failed for cluster={row.cluster}, "
            f"year={row.year}, time_slice={row['time_slice']}"
        )

    return offset


def find_offsets_country_level(
    year,
    time_slice,
    obs_country_cf,
    scalars_by_cluster,
    turb_info,
    reanalysis,
    powerCurveFile,
    seasons=None,
):
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
        seasons: Optional season-name → month-list mapping, for regions
            whose seasons differ from the Northern-Hemisphere defaults.
            Default None keeps the hardcoded NH season months.

    Returns:
        dict: Mapping of cluster ID to optimized offset value
    """
    # Parse time_slice to month list
    months = parse_time_slice(time_slice, seasons)

    # Filter reanalysis to time period
    reanalysis_period = reanalysis.sel(
        time=np.logical_and(reanalysis.time.dt.year == year, reanalysis.time.dt.month.isin(months))
    )

    # Get cluster IDs
    clusters = sorted(turb_info["cluster"].unique())

    # Get capacity weights for aggregation
    capacity_by_cluster = turb_info.groupby("cluster")["capacity"].sum()

    def objective(offsets):
        """Objective function: squared error between simulated and observed country CF."""
        cluster_cfs = []
        cluster_weights = []

        for i, cluster_id in enumerate(clusters):
            scalar = scalars_by_cluster.get(cluster_id, 1.0)
            offset = offsets[i]

            # Get turbines in this cluster
            cluster_turbs = turb_info[turb_info["cluster"] == cluster_id]

            if len(cluster_turbs) == 0:
                continue

            # Simulate with this cluster's corrections
            mean_cf = train_simulate_wind(
                reanalysis_period, cluster_turbs, powerCurveFile, scalar, offset
            )

            cluster_cfs.append(mean_cf)
            cluster_weights.append(capacity_by_cluster[cluster_id])

        # Aggregate to country level (capacity-weighted average)
        if len(cluster_cfs) == 0 or sum(cluster_weights) == 0:
            return 1e6  # Large penalty

        country_cf_sim = sum(cf * w for cf, w in zip(cluster_cfs, cluster_weights)) / sum(
            cluster_weights
        )

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
            objective, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 50, "ftol": 1e-6}
        )

        # Return dict mapping cluster to offset
        offsets_dict = {cluster_id: result.x[i] for i, cluster_id in enumerate(clusters)}

        return offsets_dict

    except Exception as e:
        print(
            f"  Warning: Offset optimization failed for year={year}, time_slice={time_slice}: {e}"
        )
        # Return zero offsets as fallback
        return {cluster_id: 0.0 for cluster_id in clusters}
