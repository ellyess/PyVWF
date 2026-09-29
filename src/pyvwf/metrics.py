"""Capacity-weighted means and one error summary for simulated against observed CF.

:func:`weighted_mean` and :func:`weighted_mean_by` are the package's one
weighted-mean primitive. :func:`calculate_error` is the ``"total"`` summary the
gridded correction scores with (``pyvwf.extensions.grid.evaluate``); every
harness metric is computed in :mod:`pyvwf.harness.skill` instead.

Until 2026-09-25 this module also carried the legacy path's grouped reports
(six more ``calculate_error`` modes, a training-data branch of
:func:`prepare_monthly_data`, and ``overall_error``, which read a
``results/capacity-factor/`` layout nothing writes any more). The legacy path
was removed on 2026-09-24; they are last present at ``96a6d3e``.
"""

import numpy as np
import pandas as pd


def weighted_mean(values, weights, *, axis=None):
    """The weighted mean of the entries that have both a value and a weight.

    Every weighted mean in the package goes through this function, so that a
    missing value is treated the same way everywhere. Two rules:

    - **A missing entry leaves both the sum and the weights.** Keeping its
      weight in the denominator would scale the result down by the missing
      share of weight, which is not a mean of anything.
    - **Where nothing has a value the result is NaN, never zero.** An empty
      numpy or pandas sum is 0.0, and a zero is a value: downstream code
      scores it instead of skipping it. That is how four UK stations came to
      be scored at zero output against an observed 0.38 for a whole year
      (:func:`pyvwf.harness.skill.collapse_pseudo_replicates`).

    A non-finite or non-positive total weight also gives NaN, so a group with
    no capacity behind it does not divide by zero.

    Args:
        values: Values to average. Array-like, or any shape broadcastable
            against ``weights``.
        weights: Weights, of the same shape as ``values`` or broadcastable to
            it (capacity, in most of this package).
        axis: Axis to reduce, as in numpy. ``None`` reduces everything and
            returns a float; an int returns an array over the other axes.

    Returns:
        float or numpy.ndarray: The weighted mean, NaN where nothing has a
        value and a weight.
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    # numpy broadcasts w against v where the shapes differ; broadcasting it
    # explicitly first would change how the products associate in the sum, and
    # the results of runs already recorded would move in their last digit.
    present = np.isfinite(v) & np.isfinite(w)
    total = np.where(present, w, 0.0).sum(axis=axis)
    numerator = np.where(present, v * w, 0.0).sum(axis=axis)
    usable = total > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(usable, numerator / np.where(usable, total, 1.0), np.nan)
    return float(out) if axis is None else out


def weighted_mean_by(frame, value_col, weight_col, by):
    """:func:`weighted_mean` per group of ``frame``, as a Series indexed by ``by``.

    A thin wrapper: the rule lives in :func:`weighted_mean` and this only
    groups the rows.
    """
    grouped = frame.groupby(by, sort=True)
    return grouped.apply(
        lambda g: weighted_mean(g[value_col], g[weight_col]),
        include_groups=False,
    )


def weighted_average_vectorized(df, value_col, weight_col):
    """Compute a weighted average of one column by another.

    Delegates to :func:`weighted_mean`, so a row with a missing value or
    weight is left out of both the sum and the weights rather than diluting
    the result.

    Args:
        df: Input DataFrame.
        value_col: Column name with values to average.
        weight_col: Column name with weights.

    Returns:
        Weighted average as a float, NaN where no row has both.
    """
    return weighted_mean(df[value_col], df[weight_col])


def prepare_monthly_data(df_sim, df_obs):
    """Monthly means of two wide CF frames, melted to one row per ID and month.

    Args:
        df_sim: Simulated capacity factor, wide (``time`` plus one column per ID).
        df_obs: Observed capacity factor, in the same layout.

    Returns:
        Tuple of (df_sim_monthly, df_obs_monthly), each with ``year``,
        ``month``, ``ID`` and ``cf``, ready for merging.

    Note:
        Neither input is mutated; both are copied before the time columns are
        derived.
    """
    df_sim = df_sim.copy()
    df_obs = df_obs.copy()

    df_obs["time"] = pd.to_datetime(df_obs["time"])
    df_obs["month"] = df_obs.time.dt.month
    df_obs["year"] = df_obs.time.dt.year
    df_obs_monthly = df_obs.drop(columns=["time"]).set_index("month").reset_index()
    df_obs_monthly = df_obs_monthly.melt(id_vars=["year", "month"], var_name="ID", value_name="cf")

    # OPTIMIZATION: Process simulations once
    df_sim["time"] = pd.to_datetime(df_sim["time"])
    df_sim["month"] = df_sim.time.dt.month
    df_sim["year"] = df_sim.time.dt.year
    df_sim_monthly = df_sim.drop(columns=["time"]).groupby(["year", "month"]).mean().reset_index()
    df_sim_monthly = df_sim_monthly.melt(id_vars=["year", "month"], var_name="ID", value_name="cf")

    # OPTIMIZATION: Convert ID to string once
    df_obs_monthly["ID"] = df_obs_monthly["ID"].astype(str)
    df_sim_monthly["ID"] = df_sim_monthly["ID"].astype(str)

    return df_sim_monthly, df_obs_monthly


#: ``calculate_error`` modes removed with the legacy path, for the error message.
REMOVED_MODES = (
    "monthly-error",
    "regional-error",
    "cluster-error",
    "turbine-error",
    "temporal-focus",
    "spatial-focus",
)


def calculate_error(type, df_sim, df_obs, turb_info):
    """Capacity-weighted RMSE, MAE and MBE of per-ID mean monthly errors.

    Each ID's monthly errors are averaged (the difference, its absolute value
    and its square), then the IDs are combined weighted by capacity. A month
    missing on either side, or an ID without a capacity, is left out.

    Args:
        type: ``"total"``, the one mode kept. Named for compatibility with
            callers written when the module had seven.
        df_sim: Simulated capacity factor, wide (``time`` plus one column per ID).
        df_obs: Observed capacity factor, in the same layout.
        turb_info: Fleet table with ``ID`` and ``capacity``.

    Returns:
        Tuple ``(rmse, mae, mbe)``.

    Raises:
        ValueError: For any other ``type``, naming a removed mode as removed.
    """
    if type in REMOVED_MODES:
        raise ValueError(
            f"calculate_error mode {type!r} was removed with the legacy path on "
            "2026-09-25; the harness scores runs in pyvwf.harness.skill"
        )
    if type != "total":
        raise ValueError(f"Unknown error type: {type}")

    df_sim_monthly, df_obs_monthly = prepare_monthly_data(df_sim, df_obs)

    # Convert turb_info ID once, on a copy: callers pass the same fleet table
    # into repeated evaluations and should not have it mutated underneath them.
    turb_info = turb_info.copy()
    turb_info["ID"] = turb_info["ID"].astype(str)

    merged = pd.merge(
        df_sim_monthly, df_obs_monthly, on=["ID", "month", "year"], suffixes=("_sim", "_obs")
    )
    merged = pd.merge(merged, turb_info[["ID", "capacity"]], on="ID")
    merged = merged.dropna(subset=["cf_sim", "cf_obs", "capacity"]).reset_index(drop=True)

    merged["diff"] = merged["cf_sim"] - merged["cf_obs"]
    merged["abdiff"] = np.abs(merged["diff"])
    merged["sqdiff"] = merged["diff"] ** 2
    merged = merged.groupby("ID").mean()

    rmse = np.sqrt(weighted_average_vectorized(merged, "sqdiff", "capacity"))
    mae = weighted_average_vectorized(merged, "abdiff", "capacity")
    mbe = weighted_average_vectorized(merged, "diff", "capacity")
    return rmse, mae, mbe
