"""Data preprocessing and orchestration for PyVWF.

This module provides the main orchestration functions for preparing training
and validation datasets for wind generation modeling.

Main Functions:
    train_set: Prepare training data (observations + simulations + reanalysis)
    val_set: Prepare validation data for testing
    cluster_train_set: Apply clustering and compute bias corrections

Observation Sources (see vwf.sources):
    Observed generation and site metadata are supplied by pluggable
    ObservationSource adapters. prep_country dispatches to one, resolved from the
    country code and obs_level unless an explicit source is passed.

Supporting Functions:
    prep_country: Load observations and metadata via an ObservationSource
    clean_obs_data: Filter and clean observation data
    add_models: Assign turbine models based on metadata
    interp_nans: Interpolate missing observation values

Examples:
    Turbine-level workflow:
        >>> gen_cf, turb_info, reanalysis, power_curves = train_set(
        ...     country='DK',
        ...     calc_z0=True,
        ...     mode='onshore',
        ...     obs_level='turbine'
        ... )
        >>> bias_df, clus_info = cluster_train_set(gen_cf, 'season', 10, turb_info)

    Country-level workflow:
        >>> # Observations are fetched outside the library and wrapped in a source
        >>> from vwf.sources import InMemoryCountrySource
        >>> source = InMemoryCountrySource(data['grid_points'], data['train_obs'])
        >>> gen_cf, turb_info, reanalysis, power_curves = train_set(
        ...     country='NL',
        ...     calc_z0=True,
        ...     obs_level='country',
        ...     source=source,
        ... )
"""

import math
from typing import cast

import numpy as np
import pandas as pd

import vwf.wind as wind
from vwf.datasets.era5 import prep_era5

# from vwf.datasets.era5 import prep_era5_daily_cached
from vwf.clustering import cluster_turbines
import vwf.correction as correction
from vwf.curves import _default_power_curve, load_power_curves


# Import from new utility modules
from vwf.time_utils import add_time_resolution_columns
from vwf.sources import ObservationSource, resolve
from vwf.sources.base import ObsLevel

# ============================================================================
# INTERNAL HELPERS
# ============================================================================


# ============================================================================
# DATA PREPROCESSING AND ORCHESTRATION
# ============================================================================


def prep_country(
    country,
    year_test=None,
    *,
    obs_level: str = "turbine",
    source: ObservationSource | None = None,
):
    """Load observations and site metadata for a country.

    Dispatches to an :class:`~vwf.sources.base.ObservationSource`. When ``source``
    is omitted it is resolved from ``country`` and ``obs_level``.

    Args:
        country: Country code.
        year_test: Optional test year. If None, the source's default training
            window is used.
        obs_level: ``"turbine"`` or ``"country"``.
        source: Explicit observation source, bypassing registry resolution.

    Returns:
        Tuple of (observations, site metadata). The observation shape follows the
        source's ``obs_level``; see :meth:`ObservationSource.load_observations`.

    Raises:
        NotImplementedError: If ``obs_level="country"`` and no source is supplied
            or registered for the country.
        ValueError: If ``obs_level="turbine"`` and the country has no source.
    """
    country = country.upper()

    if source is None:
        # prep_country is public and takes obs_level as a plain string, so
        # validate here rather than letting an unrecognised value fall through
        # the registry as a confusing "no source for this country".
        if obs_level not in ("turbine", "country"):
            raise ValueError(f"obs_level must be 'turbine' or 'country', got {obs_level!r}")
        source = resolve(country, cast(ObsLevel, obs_level))

    turb_info = source.load_metadata()

    if source.obs_level == "country":
        return source.load_observations(), turb_info

    return source.load_observations(year_test, year_test), turb_info


# ============================================================================
# DATA CLEANING AND UTILITIES
# ============================================================================


def clean_obs_data(df, country, train=False):
    """Clean turbine observations for modeling.

    Args:
        df: Observations DataFrame.
        country: Country code.
        train: If True, apply training-specific filters.

    Returns:
        Cleaned observations DataFrame.
    """
    # cf can't be greater than 100%
    df["cf_max"] = df[df.columns[df.columns.str.startswith("obs")]].max(axis=1)
    df = df.drop(df[df["cf_max"] > 1].index)
    df = df.drop("cf_max", axis=1)

    # case exists for Denmark solely, develop a method to consider the weight of missing data
    # remove any turbines that have cf of 0 at any point
    if (train) & (country == "DK"):
        df["cf_min"] = df[df.columns[df.columns.str.startswith("obs")]].min(axis=1)
        df = df.drop(df[df["cf_min"] <= 0.01].index)
        df = df.drop("cf_min", axis=1)

    # turn 0 into nan to not be considered in groupby functions
    df = df.replace(0, np.nan)

    # turbine should atleast have a cf of atleast 1%
    df["cf_mean"] = df[df.columns[df.columns.str.startswith("obs")]].mean(axis=1)
    df = df.drop(df[df["cf_mean"] <= 0.01].index)
    df = df.drop(["cf_mean"], axis=1)

    return df


# ============================================================================
# MAIN ORCHESTRATION FUNCTIONS
# ============================================================================


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
        True if ``obs`` varies across clusters within at least one period.
    """
    if "cluster" not in train_bias_df.columns or "obs" not in train_bias_df.columns:
        return False
    spread = train_bias_df.groupby(["year", time_res])["obs"].nunique(dropna=True)
    return bool((spread > 1).any())


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


def train_set(
    country,
    calc_z0,
    mode="all",
    year_test=None,
    add_nan=None,
    interp_nan=None,
    fix_turb=None,
    *,
    obs_level: str = "turbine",
    source: ObservationSource | None = None,
    era5_dir=None,
    bbox=None,
    allow_extrapolation=False,
    roughness="stored",
):
    """Prepare training inputs for PyVWF.

    Args:
        country: Country code.
        calc_z0: Whether to calculate surface roughness from wind profiles.
        mode: Turbine subset ("all", "onshore", "offshore").
        year_test: Test year (used when loading observations).
        add_nan: Fraction of data to randomly remove.
        interp_nan: Limit on simultaneous missing data points when interpolating.
        fix_turb: Turbine model name to fix to a single model.
        obs_level: Observation level ("turbine" or "country").
        source: Observation source. Resolved from ``country`` when omitted.
        era5_dir: Optional ERA5 directory forwarded to prep_era5 (validation
            harness). Default None keeps the legacy location.
        bbox: Optional bounding box forwarded to prep_era5. Default None keeps
            the legacy BoundingBoxes lookup.

    Returns:
        Tuple of (gen_cf, turb_info, reanalysis, power_curves).
    """
    obs_data, turb_info = prep_country(country, year_test, obs_level=obs_level, source=source)

    if mode != "all":
        turb_info = turb_info[turb_info["type"] == mode].copy()

    if fix_turb is not None:
        turb_info["model"] = fix_turb

    # prep era5 + curves once
    reanalysis = prep_era5(
        country,
        True,
        calc_z0,
        bbox=bbox,
        era5_dir=era5_dir,
        allow_extrapolation=allow_extrapolation,
        roughness=roughness,
    )
    power_curves = load_power_curves()

    # -------------------------
    # Country-level branch
    # -------------------------
    if obs_level == "country":
        # Country-level observations arrive as a DatetimeIndexed capacity-factor
        # series (ENTSO-E derived, for example) from the observation source.
        if "capacity_factor" not in obs_data.columns:
            raise ValueError("Country-level observations must have a 'capacity_factor' column")

        # Energy-weighted monthly capacity factor, matching the monthly energy
        # basis the turbine-level observations already use.
        zonal = "cluster" in obs_data.columns
        obs_country = (
            country_zonal_cf_to_monthly(obs_data) if zonal else country_cf_to_monthly(obs_data)
        )

        turb_info = prepare_country_fleet(turb_info, power_curves, fix_turb)

        # Simulate per-grid-point CF (not aggregated) to enable cluster-specific corrections
        sim_ws, sim_cf = wind.simulate_wind(reanalysis, turb_info, power_curves)

        # Resample to monthly and reshape to long format with ID
        sim_cf = sim_cf.groupby(pd.Grouper(key="time", freq="ME")).mean().reset_index()
        sim_cf["time"] = pd.to_datetime(sim_cf["time"], errors="coerce")
        sim_cf = sim_cf.dropna(subset=["time"]).reset_index(drop=True)

        # Melt to long format (time x ID)
        sim_long = sim_cf.melt(id_vars=["time"], var_name="ID", value_name="sim")
        sim_long["year"] = sim_long["time"].dt.year.astype(int)
        sim_long["month"] = sim_long["time"].dt.month.astype(int)
        sim_long = sim_long[["year", "month", "ID", "sim"]]

        if zonal:
            # Each grid point takes its own zone's observation, so every cluster
            # ends up with its own constraint and the offset fit is determined.
            sim_long = sim_long.merge(
                turb_info[["ID", "cluster"]].assign(ID=lambda d: d["ID"].astype(str)),
                on="ID",
                how="left",
            )
            gen_cf = sim_long.merge(obs_country, on=["year", "month", "cluster"], how="inner")
            gen_cf = gen_cf.drop(columns=["cluster"])
        else:
            # Merge with country-wide observations (same obs for all grid points)
            gen_cf = sim_long.merge(obs_country, on=["year", "month"], how="inner")
        gen_cf = add_time_resolution_columns(gen_cf)

        return gen_cf.reset_index(drop=True), turb_info, reanalysis, power_curves

    # ---------------------------------
    # Turbine-level branch
    # ---------------------------------
    obs_cf = obs_data
    obs_cf = clean_obs_data(obs_cf, country, True)

    year_star = obs_cf.year.min()
    year_end = obs_cf.year.max()

    obs_cf = obs_cf[
        obs_cf.groupby("ID").ID.transform("count") == ((year_end - year_star) + 1)
    ].reset_index(drop=True)

    obs_cf = obs_cf[
        [
            "ID",
            "year",
            "obs_1",
            "obs_2",
            "obs_3",
            "obs_4",
            "obs_5",
            "obs_6",
            "obs_7",
            "obs_8",
            "obs_9",
            "obs_10",
            "obs_11",
            "obs_12",
        ]
    ]
    obs_cf.columns = ["ID", "year", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]

    obs_cf = obs_cf.loc[obs_cf["ID"].isin(turb_info["ID"])].reset_index(drop=True)
    obs_cf = obs_cf.melt(id_vars=["ID", "year"], var_name="month", value_name="obs")
    obs_cf["month"] = obs_cf["month"].astype(int)
    obs_cf["year"] = obs_cf["year"].astype(int)

    if add_nan is not None:
        obs_cf["obs"] = obs_cf["obs"].sample(frac=(1 - add_nan), random_state=42)

    if interp_nan is not None:
        obs_cf = interp_nans(obs_cf, interp_nan)

    turb_info = turb_info.loc[turb_info["ID"].isin(obs_cf["ID"])].reset_index(drop=True)

    # Subset reanalysis to training years so sim_cf matches obs year range
    reanalysis = reanalysis.sel(time=slice(str(year_star), str(year_end)))

    sim_ws, sim_cf = wind.simulate_wind(reanalysis, turb_info, power_curves)

    sim_cf = sim_cf.groupby(pd.Grouper(key="time", freq="ME")).mean().reset_index()
    sim_cf = sim_cf.melt(id_vars=["time"], var_name="ID", value_name="sim")
    sim_cf = add_times(sim_cf)
    sim_cf = add_time_resolution_columns(sim_cf)
    sim_cf["ID"] = sim_cf["ID"].astype(str)
    obs_cf["ID"] = obs_cf["ID"].astype(str)

    gen_cf = pd.merge(sim_cf, obs_cf, on=["ID", "month", "year"], how="left")
    gen_cf = gen_cf.drop(["time"], axis=1).reset_index(drop=True)

    return gen_cf, turb_info, reanalysis, power_curves


def val_set(
    country,
    calc_z0,
    mode="all",
    year_test=None,
    fix_turb=None,
    *,
    obs_level: str = "turbine",
    source: ObservationSource | None = None,
    era5_dir=None,
    bbox=None,
    allow_extrapolation=False,
    roughness="stored",
):
    """Prepare validation data for a country.

    Args:
        country: Country code.
        calc_z0: Whether to compute surface roughness.
        mode: Cluster mode (``"all"``, ``"onshore"``, ``"offshore"``).
        year_test: Test year.
        fix_turb: Optional turbine model override.
        obs_level: ``"turbine"`` or ``"country"``.
        source: Observation source. Resolved from ``country`` when omitted.
        era5_dir: Optional ERA5 directory forwarded to prep_era5 (validation
            harness). Default None keeps the legacy location.
        bbox: Optional bounding box forwarded to prep_era5. Default None keeps
            the legacy BoundingBoxes lookup.
        allow_extrapolation: Forwarded to prep_era5. Default False refuses
            units outside the loaded ERA5 extent.
        roughness: Forwarded to prep_era5: "stored" (default) or "derived".

    Returns:
        Tuple of observations, turbine metadata, reanalysis, and power curves.
    """
    power_curves = load_power_curves()
    obs, turb_info = val_obs_and_fleet(
        country,
        year_test,
        mode,
        fix_turb,
        obs_level=obs_level,
        source=source,
        power_curves=power_curves,
    )

    # preping era5 for val
    reanalysis = prep_era5(
        country,
        False,
        calc_z0,
        bbox=bbox,
        era5_dir=era5_dir,
        allow_extrapolation=allow_extrapolation,
        roughness=roughness,
    )

    # Filter to test year only
    if year_test is not None:
        reanalysis = reanalysis.sel(time=str(year_test))

    return obs, turb_info, reanalysis, power_curves


def val_obs_and_fleet(
    country,
    year_test,
    mode="all",
    fix_turb=None,
    *,
    obs_level: str = "turbine",
    source: ObservationSource | None = None,
    power_curves: pd.DataFrame | None = None,
):
    """The observations and fleet of :func:`val_set`, without the reanalysis.

    What an evaluation scores against, for analyses that re-score recorded
    simulations and so need no ERA5: the observations in the test year and the
    fleet they cover, prepared exactly as :func:`val_set` prepares them.

    Args:
        country: Country code.
        year_test: Test year.
        mode: Cluster mode (``"all"``, ``"onshore"``, ``"offshore"``).
        fix_turb: Optional turbine model override.
        obs_level: ``"turbine"`` or ``"country"``.
        source: Observation source. Resolved from ``country`` when omitted.
        power_curves: The power curves, loaded when omitted. Used by the
            country-level fleet preparation.

    Returns:
        Tuple of observations and turbine metadata. Turbine-level observations
        are one column per unit and one row per month; country-level ones are
        a ``time`` and ``obs`` column.
    """
    obs_data, turb_info = prep_country(country, year_test, obs_level=obs_level, source=source)

    if mode != "all":
        turb_info = turb_info[turb_info["type"] == mode].copy()

    if fix_turb is not None:
        turb_info["model"] = fix_turb

    if obs_level == "country":
        if power_curves is None:
            power_curves = load_power_curves()
        # Country-level observations arrive as a DatetimeIndexed capacity-factor
        # series from the observation source, at its native resolution.
        obs_country = obs_data.copy()

        if not isinstance(obs_country.index, pd.DatetimeIndex):
            obs_country.index = pd.to_datetime(obs_country.index, utc=True)

        # Convert timezone-aware to naive UTC (remove timezone info)
        if obs_country.index.tz is not None:
            obs_country.index = obs_country.index.tz_convert("UTC").tz_localize(None)

        # Format for validation output
        # Same fleet preparation as train_set: evaluation must score the fleet
        # training was fitted on, not a superset of it.
        turb_info = prepare_country_fleet(turb_info, power_curves, fix_turb)

        if "cluster" in obs_country.columns:
            # A zonal run is scored on the same national aggregate a national
            # run is, or the two are not comparable.
            obs_country = country_zonal_to_national(obs_country, turb_info)

        obs_country["time"] = obs_country.index
        obs_country = obs_country.rename(columns={"capacity_factor": "obs"})
        obs_country = obs_country[["time", "obs"]].sort_values("time")

        return obs_country, turb_info

    # turbine-level path
    obs_cf = obs_data
    obs_cf = clean_obs_data(obs_cf, country, False)

    # formatting for testing
    dates = np.arange(str(year_test) + "-01", str(year_test + 1) + "-01", dtype="datetime64[M]")
    cols = dates.tolist()
    obs_cf = obs_cf.drop("year", axis=1)
    obs_cf.columns = ["ID"] + cols
    obs_cf = obs_cf.loc[obs_cf["ID"].isin(turb_info["ID"])].reset_index(drop=True)
    turb_info = turb_info.loc[turb_info["ID"].isin(obs_cf["ID"])].reset_index(drop=True)
    obs_cf = obs_cf.set_index("ID").transpose().rename_axis("time").reset_index()

    return obs_cf, turb_info


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
                out[col] = (v[present] * w[present]).sum() / wsum
            elif no_capacity_data:
                out[col] = v.mean()
            else:
                out[col] = np.nan
        return pd.Series(out)

    means = gen_cf.groupby(keys, as_index=False).apply(_agg, include_groups=False)
    return means.dropna(subset=["obs", "sim"]).reset_index(drop=True)


def cluster_train_set(
    gen_cf, time_res, num_clu, turb_info, *, obs_level: str = "turbine", min_cluster_size: int = 1
):
    """Aggregate the training pairs to one resolution and fit its corrections.

    One call handles one ``(num_clu, time_res)`` combination: the paired
    observed/simulated capacity factors are averaged within each time slice,
    turbines are clustered spatially, and a scalar (plus, for turbine-level
    data, an offset placeholder refined later) is computed per
    ``(cluster, time slice)``.

    For ``obs_level="country"`` no clustering step runs, so ``num_clu`` either
    collapses the country to one cluster or has to match the assignments the
    grid points already carry (see :func:`assign_country_clusters`). Grid points
    are capacity-weighted within a cluster.

    The weighting matters. The observation is national generation over national
    installed capacity, which is a capacity-weighted mean over the real fleet,
    so the simulated aggregate has to be the same functional or the difference
    between the two spatial averages is absorbed into the correction factors as
    if it were reanalysis bias. This branch used to take an unweighted mean
    while :func:`vwf.correction.find_offsets_country_level` and the harness's
    country skill metric both capacity-weighted, so the scalar and the offset
    were fitted against different definitions of "the country". On a grid whose
    points all carry the same capacity the two agree, which is why the
    inconsistency stayed invisible.

    Args:
        gen_cf: Paired training frame with ``year``, the ``time_res`` column,
            ``obs``, ``sim``, and ``ID``.
        time_res: Temporal resolution key: ``"fixed"``, ``"season"``,
            ``"bimonth"``, or ``"month"``.
        num_clu: Number of spatial clusters to fit (ignored for
            country-level, where assignments come with ``turb_info``).
        turb_info: Fleet or grid-point metadata; must carry ``cluster`` for
            country-level data.
        obs_level: ``"turbine"`` or ``"country"``; selects the branch above.
        min_cluster_size: Forwarded to :func:`vwf.clustering.cluster_turbines`;
            merges clusters with fewer training sites than this into their
            nearest neighbour before fitting. Default 1 keeps the legacy
            partition.

    Returns:
        Tuple of ``(train_bias_df, clus_info)``: the per-(cluster, slice)
        correction table with ``scalar`` and ``offset`` columns, and the
        metadata with cluster assignments used to produce it.
    """
    if obs_level == "country":
        # For country-level: gen_cf has columns [year, time_res, obs, sim, ID]
        # turb_info has cluster assignments for each ID
        turb_info = assign_country_clusters(turb_info, num_clu)

        # Merge cluster info with gen_cf
        merge_cols = ["ID", "cluster"]
        if "capacity" in turb_info.columns:
            merge_cols.append("capacity")
        gen_cf_with_cluster = pd.merge(gen_cf, turb_info[merge_cols], on="ID", how="left")

        # Capacity-weighted mean within each cluster, matching the aggregation
        # used by find_offsets_country_level and by the harness skill metric.
        # Points with no usable capacity fall back to equal weights rather than
        # dropping out, so a grid without capacities behaves as it always did.
        df = _country_cluster_means(gen_cf_with_cluster, time_res)

        # Compute scalar per cluster with constraints to prevent extreme corrections
        df["scalar"] = df["obs"] / df["sim"]
        # df["scalar"] = df["scalar"].clip(lower=0.5, upper=1.5)
        df["offset"] = 0.0

        # Keep same column naming convention
        df = df[["year", time_res, "cluster", "obs", "sim", "scalar", "offset"]]

        clus_info = turb_info.copy()

        return df, clus_info

    # turbine-level existing behavior
    gen_cf = gen_cf.groupby(["year", time_res, "ID"], as_index=False)[["obs", "sim"]].mean()

    clus_info = cluster_turbines(num_clu, turb_info, True, min_cluster_size=min_cluster_size)
    gen_cf = pd.merge(
        gen_cf,
        clus_info[["ID", "cluster", "lon", "lat", "capacity", "height", "model"]],
        on="ID",
        how="left",
    )

    train_bias_df = correction.calculate_scalar(gen_cf, time_res)

    return train_bias_df, clus_info


# ============================================================================
# SUPPORTING UTILITY FUNCTIONS
# ============================================================================


def interp_nans(df, limit):
    """Interpolate NaNs in long-form observations.

    Args:
        df: Long-form observations with ``ID``, ``year``, ``month``, ``obs``.
        limit: Maximum consecutive NaNs to interpolate.

    Returns:
        DataFrame with interpolated observations.
    """
    df = df.sort_values(["ID", "year", "month"]).copy()

    # A per-group series transform rather than a frame-level groupby.apply: the
    # old _interp returned each group INCLUDING the grouping column, which
    # pandas 3 excludes from apply, silently dropping ID from the result.
    df["obs"] = df.groupby("ID")["obs"].transform(
        lambda s: s.interpolate(method="linear", limit=limit, limit_direction="both")
    )
    return df.reset_index(drop=True)


#: Share of the training years a factor's accepted years must exceed. A factor
#: is the mean of per-year fits, and a mean over one year of three is a
#: different estimate from a mean over three, with nothing in the table to tell
#: them apart (a US cluster kept a fixed factor from 2019 alone once its 2020
#: and 2021 roots fell outside the offset search's bounds; issue #28). The
#: maintainer set the rule as a simple majority, strictly more than half: two
#: of three, three of four, three of five.
MIN_ACCEPTED_YEAR_SHARE = 0.5


def min_accepted_years(n_training_years: int) -> int:
    """Fewest accepted years a factor needs: strictly more than the share."""
    return math.floor(n_training_years * MIN_ACCEPTED_YEAR_SHARE) + 1


def format_bc_factors(train_bias_df, time_res):
    """Aggregate the per-year fits into one factor per cluster and time slice.

    Each factor averages its scalar and its offset over one set of years: the
    **accepted years**, those whose offset was fitted and accepted. A year with
    no usable observation (NaN or zero) is not a fit and is not in the set, and
    neither is a year whose offset the search refused. Averaging both
    parameters over the same set keeps the pair consistent: an offset solved
    against one year's scalar is never applied with another year's.

    Two cases have no factor to average, and they are kept apart:

    - **Unfitted:** no training year has a usable observation, so no fit was
      attempted. The cluster gets the identity, scalar 1 and offset 0, with
      ``n_years`` 0: its units keep their uncorrected values.
    - **Refused:** fits were attempted, but the accepted years are not a
      majority of the training years (:func:`min_accepted_years`). Scalar and
      offset are NaN, so its units get no corrected values, and
      :func:`vwf.harness.corrections.fit_quality` counts it as a failed offset.
      Before issue #28 a partial set was averaged silently.

    Args:
        train_bias_df: Per-year fits, with columns in the order ``year``, the
            slice, ``cluster``, ``obs``, ``sim``, ``scalar``, ``offset``.
        time_res: Name of the slice column in the result.

    Returns:
        DataFrame with ``cluster``, ``time_res``, ``scalar``, ``offset`` and
        ``n_years``, the number of accepted years the factor rests on.
    """
    obs = train_bias_df["obs"]
    df = train_bias_df.drop(["obs", "sim"], axis=1)
    df.columns = ["year", time_res, "cluster", "scalar", "offset"]
    # A zero scalar comes from a zero observation; it is no fit either.
    df["scalar"] = df["scalar"].replace(0, np.nan)

    usable_obs = obs.notna().to_numpy() & (obs.fillna(0) > 0).to_numpy()
    accepted = (
        usable_obs
        & np.isfinite(df["scalar"].to_numpy(dtype=float))
        & df["offset"].notna().to_numpy()
    )
    keys = ["cluster", time_res]
    attempted = df[usable_obs].groupby(keys, as_index=False).size().rename(columns={"size": "_n"})
    bc_factors = (
        df[keys]
        .drop_duplicates()
        .sort_values(keys)
        .merge(
            df[accepted]
            .groupby(keys, as_index=False)
            .agg(scalar=("scalar", "mean"), offset=("offset", "mean"), n_years=("year", "nunique")),
            on=keys,
            how="left",
        )
        .merge(attempted, on=keys, how="left")
        .reset_index(drop=True)
    )
    bc_factors["n_years"] = bc_factors["n_years"].fillna(0).astype(int)

    unfitted = bc_factors["_n"].isna()
    refused = ~unfitted & (bc_factors["n_years"] < min_accepted_years(int(df["year"].nunique())))
    bc_factors.loc[refused, ["scalar", "offset"]] = np.nan
    bc_factors.loc[unfitted, "scalar"] = 1.0
    bc_factors.loc[unfitted, "offset"] = 0.0
    return bc_factors.drop(columns="_n")


def add_times(df):
    """Add ``year`` and ``month`` columns from a ``time`` column."""
    df["year"] = pd.DatetimeIndex(df["time"]).year
    df["month"] = pd.DatetimeIndex(df["time"]).month
    df.insert(1, "year", df.pop("year"))
    df.insert(2, "month", df.pop("month"))
    df["month"] = df["month"].astype(int)
    df["year"] = df["year"].astype(int)
    return df
