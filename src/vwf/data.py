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

Data Loaders (re-exported from vwf.loaders):
    load_turbine_metadata: Load turbine metadata for DK, DE, UK
    load_turbine_observations: Load turbine-level generation data

Supporting Functions:
    prep_country: Load observations and metadata via an ObservationSource
    clean_obs_data: Filter and clean observation data
    add_models: Assign turbine models based on metadata
    interp_nans: Interpolate missing observation values
    sim_turbines_to_country_cf: Aggregate turbine simulations to country level

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
from typing import cast

import numpy as np
import pandas as pd
import difflib

import vwf.wind as wind
from vwf.datasets.era5 import prep_era5
# from vwf.datasets.era5 import prep_era5_daily_cached
from vwf.clustering import cluster_turbines
import vwf.correction as correction



# Import from new utility modules
from vwf.config import PyVWFPaths
from vwf.time_utils import add_time_resolution_columns
from vwf.loaders import (  # noqa: F401  (re-exported for backward compatibility)
    load_turbine_metadata,
    load_turbine_observations,
)
from vwf.sources import InMemoryCountrySource, ObservationSource, resolve
from vwf.sources.base import ObsLevel

# Keep for backward compatibility with any scripts using these paths
COUNTRY_DIR = PyVWFPaths.COUNTRY_DATA
TURBINE_DIR = PyVWFPaths.TURBINE_DATA
COUNTRY_LEVEL_DIR = PyVWFPaths.COUNTRY_LEVEL_DATA

# ============================================================================
# INTERNAL HELPERS
# ============================================================================

def _source_for(
    source: ObservationSource | None,
    external_grid_points: pd.DataFrame | None,
    external_obs_data: pd.DataFrame | None,
) -> ObservationSource | None:
    """Pick the observation source for a pipeline call.

    An explicit ``source`` wins. Otherwise a pair of externally supplied frames is
    wrapped in an in-memory country source. ``None`` means "resolve from the
    country code".
    """
    if source is not None:
        return source
    if external_grid_points is not None and external_obs_data is not None:
        return InMemoryCountrySource(external_grid_points, external_obs_data)
    return None


def _default_power_curve(power_curves: pd.DataFrame) -> str:
    """Pick a default turbine model from a power curve table."""
    cols = [c for c in power_curves.columns if c != "data$speed"]
    if not cols:
        raise ValueError("power_curves has no turbine model columns.")
    return cols[0]


# ============================================================================
# DATA LOADING (Re-exported from vwf.loaders for backward compatibility)
# ============================================================================
# The following functions are now imported from vwf.loaders:
# - load_turbine_metadata
# - load_turbine_observations

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
            raise ValueError(
                f"obs_level must be 'turbine' or 'country', got {obs_level!r}"
            )
        source = resolve(country, cast(ObsLevel, obs_level))

    turb_info = source.load_metadata()

    if source.obs_level == "country":
        return source.load_observations(), turb_info

    return source.load_observations(year_test, year_test), turb_info


def sim_turbines_to_country_cf(sim_cf_long: pd.DataFrame, turb_info: pd.DataFrame) -> pd.DataFrame:
    """Aggregate turbine simulations to a country-level series.

    Args:
        sim_cf_long: Long-form simulations with ``year``, ``month``, ``ID``, ``sim``.
        turb_info: Turbine metadata with ``ID`` and ``capacity`` (kW).

    Returns:
        DataFrame with ``year``, ``month``, and capacity-weighted ``sim``.
    """
    df = sim_cf_long.copy()
    df["ID"] = df["ID"].astype(str)
    caps = turb_info[["ID", "capacity"]].copy()
    caps["ID"] = caps["ID"].astype(str)
    df = df.merge(caps, on="ID", how="left")

    df = df.dropna(subset=["capacity", "sim"])
    if df.empty:
        raise ValueError("No valid rows to aggregate in sim_turbines_to_country_cf.")

    sim_country = (
        df.groupby(["year", "month"], as_index=False)[["sim", "capacity"]]
            .apply(lambda g: pd.Series({"sim": (g["sim"] * g["capacity"]).sum() / g["capacity"].sum()}))
            .reset_index(drop=True)
    )
    return sim_country


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


def load_power_curves():
    """Load turbine power curves.

    Reads ``power_curves.csv`` from the configured input root, falling back to
    the open curve library bundled with the package (with a warning). See
    :meth:`vwf.config.PyVWFPaths.reference_file`.

    Returns:
        DataFrame with wind speed in the ``data$speed`` column and one
        capacity-factor column per turbine model, on a 0 to 40 m/s grid.
    """
    return pd.read_csv(PyVWFPaths.reference_file("power_curves.csv"))


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

    return turb_info.dropna(
        subset=["capacity", "height", "lon", "lat", "model"]
    ).reset_index(drop=True)


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
            grouped = pd.DataFrame(
                {"energy": energy.where(usable), "possible": possible.where(usable)}
            ).resample("ME").sum(min_count=1)
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
    external_grid_points: pd.DataFrame | None = None,
    external_obs_data: pd.DataFrame | None = None,
    era5_dir=None,
    bbox=None,
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
        external_grid_points: Optional externally provided grid points (for country-level).
            Superseded by ``source``; retained for backward compatibility.
        external_obs_data: Optional externally provided observations (for country-level).
            Superseded by ``source``; retained for backward compatibility.
        era5_dir: Optional ERA5 directory forwarded to prep_era5 (validation
            harness). Default None keeps the legacy location.
        bbox: Optional bounding box forwarded to prep_era5. Default None keeps
            the legacy BoundingBoxes lookup.

    Returns:
        Tuple of (gen_cf, turb_info, reanalysis, power_curves).
    """
    source = _source_for(source, external_grid_points, external_obs_data)
    obs_data, turb_info = prep_country(country, year_test, obs_level=obs_level, source=source)

    if mode != "all":
        turb_info = turb_info[turb_info["type"] == mode].copy()

    if fix_turb is not None:
        turb_info["model"] = fix_turb

    # prep era5 + curves once
    reanalysis = prep_era5(country, True, calc_z0, bbox=bbox, era5_dir=era5_dir)
    power_curves = load_power_curves()

    # -------------------------
    # Country-level branch
    # -------------------------
    if obs_level == "country":
        # Country-level observations arrive as a DatetimeIndexed capacity-factor
        # series (ENTSO-E derived, for example) from the observation source.
        if 'capacity_factor' not in obs_data.columns:
            raise ValueError("Country-level observations must have a 'capacity_factor' column")

        # Energy-weighted monthly capacity factor, matching the monthly energy
        # basis the turbine-level observations already use.
        zonal = "cluster" in obs_data.columns
        obs_country = (
            country_zonal_cf_to_monthly(obs_data)
            if zonal
            else country_cf_to_monthly(obs_data)
        )

        turb_info = prepare_country_fleet(turb_info, power_curves, fix_turb)

        # Simulate per-grid-point CF (not aggregated) to enable cluster-specific corrections
        sim_ws, sim_cf = wind.simulate_wind(reanalysis, turb_info, power_curves)

        # Resample to monthly and reshape to long format with ID
        sim_cf = sim_cf.groupby(pd.Grouper(key="time", freq="ME")).mean().reset_index()
        sim_cf["time"] = pd.to_datetime(sim_cf["time"], errors="coerce")
        sim_cf = sim_cf.dropna(subset=["time"]).reset_index(drop=True)

        # Melt to long format (time x ID)
        sim_long = sim_cf.melt(
            id_vars=["time"],
            var_name="ID",
            value_name="sim"
        )
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
            gen_cf = sim_long.merge(
                obs_country, on=["year", "month", "cluster"], how="inner"
            )
            gen_cf = gen_cf.drop(columns=["cluster"])
        else:
            # Merge with country-wide observations (same obs for all grid points)
            gen_cf = sim_long.merge(obs_country, on=["year", "month"], how="inner")
        gen_cf = add_time_res(gen_cf)

        return gen_cf.reset_index(drop=True), turb_info, reanalysis, power_curves

    # ---------------------------------
    # Turbine-level branch
    # ---------------------------------
    obs_cf = obs_data
    obs_cf = clean_obs_data(obs_cf, country, True)

    year_star = obs_cf.year.min()
    year_end = obs_cf.year.max()

    obs_cf = obs_cf[obs_cf.groupby("ID").ID.transform("count") == ((year_end - year_star) + 1)].reset_index(drop=True)

    obs_cf = obs_cf[[
        "ID", "year",
        "obs_1","obs_2","obs_3","obs_4","obs_5","obs_6",
        "obs_7","obs_8","obs_9","obs_10","obs_11","obs_12"
    ]]
    obs_cf.columns = ["ID","year","1","2","3","4","5","6","7","8","9","10","11","12"]

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
    sim_cf = add_time_res(sim_cf)
    sim_cf["ID"] = sim_cf["ID"].astype(str)
    obs_cf["ID"] = obs_cf["ID"].astype(str)

    gen_cf = pd.merge(sim_cf, obs_cf, on=["ID", "month", "year"], how="left")
    gen_cf = gen_cf.drop(["time"], axis=1).reset_index(drop=True)

    return gen_cf, turb_info, reanalysis, power_curves


def val_set(country, calc_z0, mode="all", year_test=None, fix_turb=None, *, obs_level: str = "turbine", source: ObservationSource | None = None, external_grid_points: pd.DataFrame | None = None, external_obs_data: pd.DataFrame | None = None, era5_dir=None, bbox=None):
    """Prepare validation data for a country.

    Args:
        country: Country code.
        calc_z0: Whether to compute surface roughness.
        mode: Cluster mode (``"all"``, ``"onshore"``, ``"offshore"``).
        year_test: Test year.
        fix_turb: Optional turbine model override.
        obs_level: ``"turbine"`` or ``"country"``.
        source: Observation source. Resolved from ``country`` when omitted.
        external_grid_points: Optional externally provided grid points (for country-level).
            Superseded by ``source``; retained for backward compatibility.
        external_obs_data: Optional externally provided observations (for country-level).
            Superseded by ``source``; retained for backward compatibility.
        era5_dir: Optional ERA5 directory forwarded to prep_era5 (validation
            harness). Default None keeps the legacy location.
        bbox: Optional bounding box forwarded to prep_era5. Default None keeps
            the legacy BoundingBoxes lookup.

    Returns:
        Tuple of observations, turbine metadata, reanalysis, and power curves.
    """
    source = _source_for(source, external_grid_points, external_obs_data)
    obs_data, turb_info = prep_country(country, year_test, obs_level=obs_level, source=source)

    if mode != "all":
        turb_info = turb_info[turb_info["type"] == mode].copy()

    if fix_turb is not None:
        turb_info["model"] = fix_turb

    # preping era5 for val
    reanalysis = prep_era5(country, False, calc_z0, bbox=bbox, era5_dir=era5_dir)

    # Filter to test year only
    if year_test is not None:
        reanalysis = reanalysis.sel(time=str(year_test))

    power_curves = load_power_curves()

    if obs_level == "country":
        # Country-level observations arrive as a DatetimeIndexed capacity-factor
        # series from the observation source, at its native resolution.
        obs_country = obs_data.copy()

        if not isinstance(obs_country.index, pd.DatetimeIndex):
            obs_country.index = pd.to_datetime(obs_country.index, utc=True)

        # Convert timezone-aware to naive UTC (remove timezone info)
        if obs_country.index.tz is not None:
            obs_country.index = obs_country.index.tz_convert('UTC').tz_localize(None)

        # Format for validation output
        # Same fleet preparation as train_set: evaluation must score the fleet
        # training was fitted on, not a superset of it.
        turb_info = prepare_country_fleet(turb_info, power_curves, fix_turb)

        if "cluster" in obs_country.columns:
            # A zonal run is scored on the same national aggregate a national
            # run is, or the two are not comparable.
            obs_country = country_zonal_to_national(obs_country, turb_info)

        obs_country['year'] = obs_country.index.year
        obs_country['month'] = obs_country.index.month
        obs_country['time'] = obs_country.index
        obs_country = obs_country.rename(columns={'capacity_factor': 'obs'})
        obs_country = obs_country[['time', 'obs']].sort_values('time')

        return obs_country, turb_info, reanalysis, power_curves

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

    return obs_cf, turb_info, reanalysis, power_curves


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
            "country-level metadata needs a 'cluster' column; no clustering "
            "step runs on this path"
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


def cluster_train_set(gen_cf, time_res, num_clu, turb_info, *, obs_level: str = "turbine",
                      min_cluster_size: int = 1):
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
        gen_cf_with_cluster = pd.merge(
            gen_cf,
            turb_info[merge_cols],
            on="ID",
            how="left"
        )

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

    clus_info = cluster_turbines(
        num_clu, turb_info, True, min_cluster_size=min_cluster_size
    )
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


def add_models(df: pd.DataFrame) -> pd.DataFrame:
    """Assign turbine model names based on metadata.

    Args:
        df: Turbine metadata.

    Returns:
        DataFrame with a ``model`` column added.
    """
    models = pd.read_csv(PyVWFPaths.reference_file("models.csv"))
    models["model"] = models["model"].astype("string")
    models["manufacturer"] = models["manufacturer"].astype("string").str.lower().fillna("")
    models = models.sort_values("p_density").reset_index(drop=True)

    df = df.copy()

    # --- Ensure required columns exist ---
    required = ["ID", "capacity", "diameter", "height", "lon", "lat"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"add_models: missing required columns: {missing}")

    # --- Coerce numerics safely ---
    for c in ["capacity", "diameter", "height", "lon", "lat"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows missing core numeric fields
    df = df.dropna(subset=["ID", "capacity", "diameter", "height", "lon", "lat"]).reset_index(drop=True)

    # Remove unrealistic turbines
    df = df.loc[df["height"] >= 1].reset_index(drop=True)

    # --- Manufacturer cleanup (difflib cannot handle pd.NA) ---
    if "manufacturer" in df.columns:
        df["manufacturer"] = df["manufacturer"].astype("string").str.lower().fillna("")
    else:
        df["manufacturer"] = ""

    # Default type
    if "type" not in df.columns:
        df["type"] = "onshore"
    else:
        df["type"] = df["type"].astype("string").fillna("onshore")

    df["ID"] = df["ID"].astype(str)

    # Compute power density (NOTE: your capacity is in kW; convert to W for density)
    df["p_density"] = (df["capacity"] * 1000.0) / (np.pi * (df["diameter"] / 2.0) ** 2)

    # --- Fuzzy match manufacturer against model manufacturers ---
    # Create candidate pairs by manufacturer similarity, then pick closest p_density
    # (This keeps your original logic but makes it robust.)
    cand = models.assign(
        match=models["manufacturer"].apply(lambda x: difflib.get_close_matches(x, df["manufacturer"].tolist(), cutoff=0.3, n=50))
    ).explode("match")

    if cand["match"].isna().all():
        # If no manufacturer matches at all, fall back to nearest p_density later
        df["model"] = pd.NA
    else:
        merged = df.merge(
            cand.drop_duplicates(subset=["manufacturer", "match", "model", "p_density"]),
            left_on="manufacturer",
            right_on="match",
            how="left",
            suffixes=("", "_m"),
        )

        # choose closest p_density among matched manufacturer candidates
        merged["closest"] = (merged["p_density"] - merged["p_density_m"]).abs()
        merged = merged.sort_values(["ID", "closest"])
        merged = merged.drop_duplicates(subset=["ID"], keep="first")

        # accept manufacturer-based match only if close enough
        merged["model"] = merged["model"].where(merged["closest"] < 1, pd.NA)

        df = merged[["ID", "type", "capacity", "diameter", "height", "lon", "lat", "p_density", "model"]].copy()

    # --- Final fallback: nearest p_density across all models ---
    # merge_asof requires sorted keys
    df = df.sort_values("p_density").reset_index(drop=True)
    fallback = pd.merge_asof(
        df[["p_density"]],
        models[["p_density", "model"]],
        on="p_density",
        direction="nearest",
        tolerance=100,
    )["model"]

    df["model"] = df["model"].fillna(fallback)

    # Drop if still no model
    df = df.dropna(subset=["model"]).reset_index(drop=True)

    # Keep types clean
    df["capacity"] = df["capacity"].astype(float)
    df["diameter"] = df["diameter"].astype(float)
    df["height"] = df["height"].astype(float)

    df = df.sort_values("ID").reset_index(drop=True)
    return df


def format_bc_factors(train_bias_df, time_res):
    """Aggregate bias correction factors by cluster and time slice.

    Scalar: varies by (cluster, time_res) - captures temporal/seasonal variation per cluster
    Offset: varies by (cluster, time_res) - captures systematic spatial bias per season/period

    Both are aggregated across years to form a repeatable seasonal pattern.
    """
    train_bias_df = train_bias_df.drop(["obs", "sim"], axis=1)
    train_bias_df["scalar"] = train_bias_df["scalar"].replace(0, np.nan)
    train_bias_df.columns = ["year", time_res, "cluster", "scalar", "offset"]

    # Both scalar and offset: per (cluster, time_res), aggregated across years
    # This captures seasonal spatial patterns that repeat annually
    bc_factors = train_bias_df.groupby(["cluster", time_res], as_index=False).agg({
        "scalar": "mean",
        "offset": "mean"
    })

    # Handle NaN values (zero offset BEFORE setting scalar, so the isna check works)
    bc_factors.loc[bc_factors["scalar"].isna(), "offset"] = 0
    bc_factors.loc[bc_factors["scalar"].isna(), "scalar"] = 1

    return bc_factors


def add_times(df):
    """Add ``year`` and ``month`` columns from a ``time`` column."""
    df["year"] = pd.DatetimeIndex(df["time"]).year
    df["month"] = pd.DatetimeIndex(df["time"]).month
    df.insert(1, "year", df.pop("year"))
    df.insert(2, "month", df.pop("month"))
    df["month"] = df["month"].astype(int)
    df["year"] = df["year"].astype(int)
    return df


# Backward compatibility alias for add_time_res (now in vwf.time_utils)
add_time_res = add_time_resolution_columns
