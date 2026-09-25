"""Country-level observations and grid fleets, for ``vwf.data``.

A country-level region has one observed series, national or per bidding zone,
and a fleet of grid points carrying one representative turbine. These helpers
prepare that fleet, turn the observed series into monthly capacity factors,
decide whether a fit sees one national observation or one per cluster, and
form the capacity-weighted cluster means the fit is given. Split from
``vwf.data`` on 2026-09-25, which imports every name back.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vwf.curves import _default_power_curve
from vwf.metrics import weighted_mean


def prepare_country_fleet(turb_info, power_curves, fix_turb=None):
    """Coerce and filter country-level grid points into a simulable fleet.

    Both :func:`train_set` and :func:`val_set` call this, and they must call the
    same thing. They used to differ: training coerced the numeric fields and
    dropped rows missing any of them, evaluation did neither, and both then
    passed the result straight through as ``clus_info``. A single unparseable
    capacity in the grid file therefore made evaluation score a larger fleet
    than training had fitted, with no error anywhere.

    Args:
        turb_info: Grid points, with ``capacity``, ``height``, ``lon``, ``lat``
            and ideally ``model``.
        power_curves: Curve table, used to pick a default model when the grid
            has none.
        fix_turb: Explicit model override; wins over the default.

    Returns:
        A copy with numeric fields coerced and unusable rows dropped.
    """
    turb_info = turb_info.copy()

    if "model" not in turb_info.columns or turb_info["model"].isna().all():
        turb_info["model"] = (
            fix_turb if fix_turb is not None else _default_power_curve(power_curves)
        )

    for column in ("capacity", "height", "lon", "lat"):
        turb_info[column] = pd.to_numeric(turb_info[column], errors="coerce")

    return turb_info.dropna(subset=["capacity", "height", "lon", "lat", "model"]).reset_index(
        drop=True
    )


def country_cf_to_monthly(obs):
    """Collapse a country CF series to monthly, weighting by energy.

    The monthly capacity factor of a fleet is total energy over total possible
    energy, ``Σ(gen · Δt) / Σ(cap · Δt)``, not the mean of the instantaneous
    ratios. The two differ whenever installed capacity moves inside the month,
    which it does in every growing wind system, and the mean-of-ratios version
    over-weights the low-capacity start of the month. Turbine-level
    observations are monthly energy from the start, so this is also what makes
    the two paths comparable.

    Falls back to the mean of ``capacity_factor`` when the generation and
    capacity columns are absent, which is all the caller can do with a series
    that only carries the ratio.

    Args:
        obs: DatetimeIndexed observations with ``capacity_factor`` and
            optionally ``generation_mw`` and ``capacity_mw``.

    Returns:
        DataFrame with ``year``, ``month`` and ``obs``.
    """
    obs = obs.copy()
    if not isinstance(obs.index, pd.DatetimeIndex):
        obs.index = pd.to_datetime(obs.index, utc=True, format="mixed")
    if obs.index.tz is not None:
        obs.index = obs.index.tz_convert("UTC").tz_localize(None)
    obs = obs.sort_index()

    if {"generation_mw", "capacity_mw"}.issubset(obs.columns):
        # Interval length per row, so a file whose resolution changes partway
        # through (ES switches from hourly to quarter-hourly between splits)
        # is still weighted correctly.
        hours = obs.index.to_series().diff().shift(-1).dt.total_seconds() / 3600.0
        hours = hours.ffill().bfill()
        if hours.notna().any():
            energy = pd.to_numeric(obs["generation_mw"], errors="coerce") * hours
            possible = pd.to_numeric(obs["capacity_mw"], errors="coerce") * hours
            # A row missing either side contributes to neither, so a gap does
            # not silently deflate the month.
            usable = energy.notna() & possible.notna()
            grouped = (
                pd.DataFrame({"energy": energy.where(usable), "possible": possible.where(usable)})
                .resample("ME")
                .sum(min_count=1)
            )
            return _month_index_to_columns(grouped["energy"] / grouped["possible"])

    return _month_index_to_columns(obs["capacity_factor"].resample("ME").mean())


#: Largest within-period spread of cluster observations still read as one
#: national number, in capacity factor. Rounding in the capacity-weighted
#: cluster means leaves about 1e-16; distinct zonal observations differ by
#: thousandths or more.
OBS_SAME_TOLERANCE = 1e-9


def country_obs_is_per_cluster(train_bias_df, time_res):
    """True when each cluster carries its own observation in every period.

    This is the condition under which per-cluster offsets are estimable. With
    one national number the N cluster offsets are under-determined by N-1 and
    :func:`vwf.correction.find_offsets_country_level` returns wherever L-BFGS-B
    stopped; with one observation per cluster the fit is exactly determined and
    is the same problem the turbine-level path already solves per cluster.

    It is read off the data rather than declared in config because it is a fact
    about the data: an observation source can only make the fit identifiable by
    supplying distinct constraints, and whether it did is visible here.

    Args:
        train_bias_df: Per (year, slice, cluster) frame with an ``obs`` column.
        time_res: Name of the time-slice column.

    Returns:
        True if ``obs`` varies across clusters within at least one period by
        more than :data:`OBS_SAME_TOLERANCE`.
    """
    if "cluster" not in train_bias_df.columns or "obs" not in train_bias_df.columns:
        return False
    # Each cluster's obs is a capacity-weighted mean of its points' values, so
    # one national number reaches the clusters differing in the last bits
    # (1e-16 on the real grids). Counting distinct floats took those for
    # distinct observations and sent every national multi-cluster fit to the
    # per-cluster solver; the spread has to exceed a tolerance instead.
    obs = train_bias_df.groupby(["year", time_res])["obs"]
    spread = obs.max() - obs.min()
    return bool((spread > OBS_SAME_TOLERANCE).any())


def country_zonal_cf_to_monthly(obs):
    """Monthly energy-weighted CF per cluster, for a zonal observation frame.

    Args:
        obs: DatetimeIndexed observations with ``capacity_factor``, ``cluster``
            and optionally ``generation_mw`` and ``capacity_mw``.

    Returns:
        DataFrame with ``year``, ``month``, ``cluster`` and ``obs``.
    """
    frames = []
    for cluster, group in obs.groupby("cluster"):
        monthly = country_cf_to_monthly(group.drop(columns=["cluster"]))
        monthly["cluster"] = cluster
        frames.append(monthly)
    return pd.concat(frames, ignore_index=True)[["year", "month", "cluster", "obs"]]


def country_zonal_to_national(obs, turb_info):
    """Collapse a zonal CF frame to one national series, weighted by capacity.

    Evaluation scores the capacity-weighted national aggregate, so a zonal run
    has to be reduced to the same quantity or its metrics are not comparable
    with a national run's.

    Args:
        obs: Zonal observations with ``capacity_factor`` and ``cluster``.
        turb_info: Grid points with ``cluster`` and ``capacity``.

    Returns:
        DatetimeIndexed frame with a single ``capacity_factor`` column.
    """
    weights = turb_info.groupby("cluster")["capacity"].sum()
    frame = obs.copy()
    frame["_w"] = frame["cluster"].map(weights).astype(float)
    frame = frame.dropna(subset=["capacity_factor", "_w"])
    frame["_wcf"] = frame["capacity_factor"] * frame["_w"]
    grouped = frame.groupby(level=0)[["_wcf", "_w"]].sum()
    national = (grouped["_wcf"] / grouped["_w"]).rename("capacity_factor")
    return national.to_frame()


def _month_index_to_columns(monthly):
    """Turn a month-end indexed Series into year/month/obs columns."""
    out = monthly.rename("obs").reset_index()
    out.columns = ["time", "obs"]
    out["year"] = out["time"].dt.year.astype(int)
    out["month"] = out["time"].dt.month.astype(int)
    return out[["year", "month", "obs"]]


def assign_country_clusters(turb_info, num_clu):
    """Resolve the cluster column a country-level run should use.

    No clustering step runs on the country-level path, so ``num_clu`` was
    previously ignored outright: a config asking for 5 clusters against a grid
    carrying 4 produced a file named ``factors_<slice>_5.csv`` holding 4 rows,
    and a manifest recording 5. Two cases are now legitimate and everything else
    is an error.

    ``num_clu == 1`` collapses the country to a single cluster. That is the
    identifiable baseline: one national observation against one offset is
    exactly determined, which makes the fit structurally identical to the
    turbine-level path rather than a walk across an under-determined solution
    set. Any per-cluster country method should have to beat it.

    ``num_clu == the grid's own cluster count`` keeps the grid's assignments,
    which for the zonal regions are bidding zones and carry real meaning.

    Args:
        turb_info: Grid points with a ``cluster`` column.
        num_clu: Requested cluster count.

    Returns:
        A copy of ``turb_info`` with the cluster column to use.

    Raises:
        ValueError: If ``cluster`` is missing, or ``num_clu`` is neither 1 nor
            the grid's own cluster count.
    """
    if "cluster" not in turb_info.columns:
        raise ValueError(
            "country-level metadata needs a 'cluster' column; no clustering step runs on this path"
        )

    turb_info = turb_info.copy()
    num_clu = int(num_clu)

    if num_clu == 1:
        turb_info["cluster"] = 0
        return turb_info

    present = int(turb_info["cluster"].dropna().nunique())
    if num_clu != present:
        raise ValueError(
            f"country-level run asked for {num_clu} clusters but the grid "
            f"points define {present}. The country path does not cluster, so "
            "the two must agree. Set cluster_list to [1] for the single-cluster "
            f"national baseline, or to [{present}] to use the grid's own "
            "assignments."
        )
    return turb_info


def _country_cluster_means(gen_cf, time_res):
    """Capacity-weighted ``obs``/``sim`` means per (year, slice, cluster).

    Falls back to an equal-weight mean only when a group's capacities are all
    missing, so a grid point table without capacities keeps working. A group
    whose capacities are present and sum to zero is a cluster with no fleet in
    that period, which year-specific weights produce routinely; it is dropped
    rather than resurrected at equal weight. NaN values are skipped in both the
    numerator and the denominator, the same rule
    :func:`vwf.correction.calculate_scalar` applies at turbine level, so a
    partially reporting group is not scaled down by its own reporting fraction.
    """
    keys = ["year", time_res, "cluster"]

    if "capacity" not in gen_cf.columns:
        return gen_cf.groupby(keys, as_index=False)[["obs", "sim"]].mean()

    weights = pd.to_numeric(gen_cf["capacity"], errors="coerce")

    def _agg(group):
        out = {}
        w = weights.loc[group.index]
        no_capacity_data = not w.notna().any()
        for col in ("obs", "sim"):
            v = group[col]
            present = v.notna() & w.notna() & (w > 0)
            wsum = w[present].sum()
            if wsum > 0:
                out[col] = weighted_mean(v.where(present), w.where(present))
            elif no_capacity_data:
                out[col] = v.mean()
            else:
                out[col] = np.nan
        return pd.Series(out)

    means = gen_cf.groupby(keys, as_index=False).apply(_agg, include_groups=False)
    return means.dropna(subset=["obs", "sim"]).reset_index(drop=True)
