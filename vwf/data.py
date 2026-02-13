"""Data preprocessing and orchestration for PyVWF.

This module provides the main orchestration functions for preparing training
and validation datasets for wind generation modeling.

Main Functions:
    train_set: Prepare training data (observations + simulations + reanalysis)
    val_set: Prepare validation data for testing
    cluster_train_set: Apply clustering and compute bias corrections

Data Loaders (re-exported from vwf.loaders):
    load_turbine_metadata: Load turbine metadata for DK, DE, UK
    load_turbine_observations: Load turbine-level generation data

Supporting Functions:
    prep_country: Preprocess observations for a country
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
        >>> # Load data externally and use with PyVWF
        >>> model = PyVWF("", "NL", True, True, "all", [5], ["fixed"], obs_level="country")
        >>> gen_cf, turb_info, reanalysis, power_curves = train_set(
        ...     country='NL',
        ...     calc_z0=True,
        ...     obs_level='country',
        ...     external_grid_points=data['grid_points'],
        ...     external_obs_data=data['train_obs']
        ... )
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import difflib
from calendar import monthrange

import vwf.wind as wind
from vwf.datasets.era5 import prep_era5
# from vwf.datasets.era5 import prep_era5_daily_cached
from vwf.clustering import cluster_turbines
import vwf.correction as correction

from vwf.wind import simulate_country_cf

from pathlib import Path

# Import from new utility modules
from vwf.config import PyVWFPaths
from vwf.time_utils import add_time_resolution_columns
from vwf.loaders import (
    load_turbine_metadata,
    load_turbine_observations,
)
from vwf.loaders.country_level_loaders import country_gen_to_cf

# Keep for backward compatibility with any scripts using these paths
COUNTRY_DIR = PyVWFPaths.COUNTRY_DATA
TURBINE_DIR = PyVWFPaths.TURBINE_DATA
COUNTRY_LEVEL_DIR = PyVWFPaths.COUNTRY_LEVEL_DATA

# ============================================================================
# INTERNAL HELPERS
# ============================================================================

def _year_range(train: bool, year_test: int | None, default_train=(2015, 2018)) -> tuple[int, int]:
    """Determine year range for train or test data."""
    if train:
        return int(default_train[0]), int(default_train[1])
    if year_test is None:
        raise ValueError("year_test must be provided when train=False")
    return int(year_test), int(year_test)


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

def prep_country(country, year_test=None, *, obs_level: str = "turbine"):
    """Preprocess observational data for a country.

    Args:
        country: Country code.
        year_test: Optional test year. If None, prepares training data.
        obs_level: ``"turbine"`` or ``"country"``.

    Returns:
        Tuple of observations and turbine metadata, depending on ``obs_level``.
    """
    country = country.upper()
    train = year_test is None

    # Per-country training windows (fallback default)
    default_train = {
        "DK": (2015, 2019),
        "DE": (2015, 2018),
        "UK": (2015, 2018),
    }.get(country, (2015, 2018))

    year_start, year_end = _year_range(train, year_test, default_train=default_train)

    # ---- metadata ----
    turb_raw = load_turbine_metadata(country)

    turb_info = add_models(turb_raw)

    obs_gen = load_turbine_observations(country, year_start, year_end).copy()
    if not {"ID", "year"}.issubset(obs_gen.columns):
        raise ValueError("Turbine observations must contain columns ['ID','year', ...months...]")

    # Standardise month columns to obs_1..obs_12 (if not already)
    month_cols = [c for c in obs_gen.columns if c not in ["ID", "year"]]
    # If columns are "obs_1"... already, keep them; else rename to obs_{col}
    if not any(str(c).startswith("obs_") for c in month_cols):
        obs_gen.columns = [f"obs_{c}" if c not in ["ID", "year"] else c for c in obs_gen.columns]

    # Merge capacity for CF conversion
    obs_gen["ID"] = obs_gen["ID"].astype(str)
    obs_gen["year"] = pd.to_numeric(obs_gen["year"], errors="coerce").astype("Int64")

    obs_gen = obs_gen.merge(turb_info[["ID", "capacity"]], how="left", on="ID")
    obs_gen = obs_gen.dropna(subset=["capacity", "year"]).reset_index(drop=True)

    # Convert monthly output -> CF using hours in month and turbine capacity (kW)
    for m in range(1, 13):
        col = f"obs_{m}"
        if col not in obs_gen.columns:
            continue
        days = obs_gen["year"].astype(int).map(lambda y: monthrange(int(y), int(m))[1]).astype(float)
        obs_gen[col] = pd.to_numeric(obs_gen[col], errors="coerce") / (days * 24.0 * obs_gen["capacity"].astype(float))

    obs_gen = obs_gen.drop(columns=["capacity"])
    return obs_gen, turb_info


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
        df.groupby(["year", "month"], as_index=False)
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
    """Load turbine power curves from the default CSV."""
    file_loc = "input/power_curves.csv"
    df = pd.read_csv(file_loc)
    return df


# ============================================================================
# MAIN ORCHESTRATION FUNCTIONS
# ============================================================================

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
    external_grid_points: pd.DataFrame | None = None,
    external_obs_data: pd.DataFrame | None = None,
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
        external_grid_points: Optional externally provided grid points (for country-level).
        external_obs_data: Optional externally provided observations (for country-level).

    Returns:
        Tuple of (gen_cf, turb_info, reanalysis, power_curves).
    """
    # Use external data if provided (country-level workflow)
    if external_grid_points is not None and external_obs_data is not None:
        turb_info = external_grid_points.copy()
        obs_data = external_obs_data.copy()
    else:
        obs_data, turb_info = prep_country(country, year_test, obs_level=obs_level)

    if mode != "all":
        turb_info = turb_info[turb_info["type"] == mode].copy()

    if fix_turb is not None:
        turb_info["model"] = fix_turb

    # prep era5 + curves once
    reanalysis = prep_era5(country, True, calc_z0)
    power_curves = load_power_curves()

    # -------------------------
    # Country-level branch
    # -------------------------
    if obs_level == "country":
        # Check if obs_data is already in the correct format (external data)
        if external_obs_data is not None:
            # External data is already formatted (from ENTSO-E API)
            # Expected format: index=datetime, columns=['capacity_factor', 'generation_mw', 'capacity_mw']
            if 'capacity_factor' not in obs_data.columns:
                raise ValueError("external_obs_data must have 'capacity_factor' column")

            # Convert to year/month format expected by downstream
            obs_country = obs_data.copy()
            if not isinstance(obs_country.index, pd.DatetimeIndex):
                obs_country.index = pd.to_datetime(obs_country.index, utc=True)

            # Convert timezone-aware to naive UTC (remove timezone info)
            if obs_country.index.tz is not None:
                obs_country.index = obs_country.index.tz_convert('UTC').tz_localize(None)

            # Aggregate to monthly means (matching simulation temporal resolution)
            # Keep only capacity_factor for aggregation
            obs_monthly = obs_country[['capacity_factor']].resample('ME').mean().reset_index()
            obs_monthly.rename(columns={'index': 'time'}, inplace=True)
            obs_monthly['year'] = obs_monthly['time'].dt.year.astype(int)
            obs_monthly['month'] = obs_monthly['time'].dt.month.astype(int)
            obs_country = obs_monthly.rename(columns={'capacity_factor': 'obs'})
            obs_country = obs_country[['year', 'month', 'obs']]
        else:
            # Use legacy loader (converts generation kWh to CF)
            obs_country = country_gen_to_cf(obs_data, turb_info, output_col="output_kwh")

        # -------------------------------------------------
        # Ensure a valid power-curve model exists
        # -------------------------------------------------
        if "model" not in turb_info.columns or turb_info["model"].isna().all():
            default_model = (
                fix_turb
                if fix_turb is not None
                else _default_power_curve(power_curves)
            )
            turb_info = turb_info.copy()
            turb_info["model"] = default_model

        # -------------------------------------------------
        # Ensure numeric fields required by interpolate_wind
        # -------------------------------------------------
        turb_info["capacity"] = pd.to_numeric(turb_info["capacity"], errors="coerce")
        turb_info["height"] = pd.to_numeric(turb_info["height"], errors="coerce")
        turb_info["lon"] = pd.to_numeric(turb_info["lon"], errors="coerce")
        turb_info["lat"] = pd.to_numeric(turb_info["lat"], errors="coerce")

        turb_info = turb_info.dropna(
            subset=["capacity", "height", "lon", "lat", "model"]
        ).reset_index(drop=True)

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

        # Merge with country-wide observations (same obs for all grid points)
        gen_cf = sim_long.merge(obs_country, on=["year", "month"], how="inner")
        gen_cf = add_time_res(gen_cf)

        return gen_cf.reset_index(drop=True), turb_info, reanalysis, power_curves

    # ---------------------------------
    # Turbine-level branch
    # ---------------------------------
    sim_ws, sim_cf = wind.simulate_wind(reanalysis, turb_info, power_curves)

    sim_cf = sim_cf.groupby(pd.Grouper(key="time", freq="ME")).mean().reset_index()
    sim_cf = sim_cf.melt(id_vars=["time"], var_name="ID", value_name="sim")
    sim_cf = add_times(sim_cf)
    sim_cf = add_time_res(sim_cf)
    sim_cf["ID"] = sim_cf["ID"].astype(str)

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

    obs_cf["ID"] = obs_cf["ID"].astype(str)

    gen_cf = pd.merge(sim_cf, obs_cf, on=["ID", "month", "year"], how="left")
    gen_cf = gen_cf.drop(["time"], axis=1).reset_index(drop=True)

    return gen_cf, turb_info, reanalysis, power_curves


def val_set(country, calc_z0, mode="all", year_test=None, fix_turb=None, *, obs_level: str = "turbine", external_grid_points: pd.DataFrame | None = None, external_obs_data: pd.DataFrame | None = None):
    """Prepare validation data for a country.

    Args:
        country: Country code.
        calc_z0: Whether to compute surface roughness.
        mode: Cluster mode (``"all"``, ``"onshore"``, ``"offshore"``).
        year_test: Test year.
        fix_turb: Optional turbine model override.
        obs_level: ``"turbine"`` or ``"country"``.
        external_grid_points: Optional externally provided grid points (for country-level).
        external_obs_data: Optional externally provided observations (for country-level).

    Returns:
        Tuple of observations, turbine metadata, reanalysis, and power curves.
    """
    # Use external data if provided (country-level workflow)
    if external_grid_points is not None and external_obs_data is not None:
        turb_info = external_grid_points.copy()
        obs_data = external_obs_data.copy()
    else:
        obs_data, turb_info = prep_country(country, year_test, obs_level=obs_level)

    if mode != "all":
        turb_info = turb_info[turb_info["type"] == mode].copy()

    if fix_turb is not None:
        turb_info["model"] = fix_turb

    # preping era5 for val
    reanalysis = prep_era5(country, False, calc_z0)

    # Filter to test year only
    if year_test is not None:
        reanalysis = reanalysis.sel(time=str(year_test))

    power_curves = load_power_curves()

    if obs_level == "country":
        # Check if obs_data is already in the correct format (external data)
        if external_obs_data is not None:
            # External data already has capacity_factor
            obs_country = obs_data.copy()

            if not isinstance(obs_country.index, pd.DatetimeIndex):
                obs_country.index = pd.to_datetime(obs_country.index, utc=True)

            # Convert timezone-aware to naive UTC (remove timezone info)
            if obs_country.index.tz is not None:
                obs_country.index = obs_country.index.tz_convert('UTC').tz_localize(None)

            # Format for validation output
            obs_country['year'] = obs_country.index.year
            obs_country['month'] = obs_country.index.month
            obs_country['time'] = obs_country.index
            obs_country = obs_country.rename(columns={'capacity_factor': 'obs'})
            obs_country = obs_country[['time', 'obs']].sort_values('time')
        else:
            # Use legacy loader
            obs_country = country_gen_to_cf(obs_data, turb_info, output_col="output_kwh", capacity_unit="kW")

            # build a time index similar to turbine val output
            obs_country["time"] = pd.to_datetime(dict(year=obs_country["year"], month=obs_country["month"], day=1))
            obs_country = obs_country.sort_values("time")[["time", "obs"]]

        if "model" not in turb_info.columns or turb_info["model"].isna().all():
            default_model = fix_turb if fix_turb is not None else _default_power_curve(power_curves)
            turb_info = turb_info.copy()
            turb_info["model"] = default_model

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


def cluster_train_set(gen_cf, time_res, num_clu, turb_info, *, obs_level: str = "turbine"):
    """Apply temporal resolution and compute correction factors.

    For ``obs_level="country"``, corrections are computed per cluster using
    country-wide observations.
    """
    if obs_level == "country":
        # For country-level: gen_cf has columns [year, time_res, obs, sim, ID]
        # turb_info has cluster assignments for each ID

        # Merge cluster info with gen_cf
        gen_cf_with_cluster = pd.merge(
            gen_cf,
            turb_info[["ID", "cluster"]],
            on="ID",
            how="left"
        )

        # Group by year, time_res, and cluster - simple mean (no capacity weighting)
        # All grid points in a cluster contribute equally to the bias correction
        df = gen_cf_with_cluster.groupby(["year", time_res, "cluster"], as_index=False)[["obs", "sim"]].mean()

        # Compute scalar per cluster with constraints to prevent extreme corrections
        df["scalar"] = df["obs"] / df["sim"]
        df["scalar"] = df["scalar"].clip(lower=0.5, upper=1.5)
        df["offset"] = 0.0

        # Keep same column naming convention
        df = df[["year", time_res, "cluster", "obs", "sim", "scalar", "offset"]]

        clus_info = turb_info.copy()

        return df, clus_info

    # turbine-level existing behavior
    gen_cf = gen_cf.groupby(["year", time_res, "ID"], as_index=False)[["obs", "sim"]].mean()

    clus_info = cluster_turbines(num_clu, turb_info, True)
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

    def _interp(g):
        g = g.copy()
        g["obs"] = g["obs"].interpolate(method="linear", limit=limit, limit_direction="both")
        return g

    return df.groupby("ID", group_keys=False).apply(_interp).reset_index(drop=True)


def add_models(df: pd.DataFrame) -> pd.DataFrame:
    """Assign turbine model names based on metadata.

    Args:
        df: Turbine metadata.

    Returns:
        DataFrame with a ``model`` column added.
    """
    models = pd.read_csv("input/models.csv")
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

    # Handle NaN values
    bc_factors.loc[bc_factors["scalar"].isna(), "scalar"] = 1
    bc_factors.loc[bc_factors["offset"].isna(), "offset"] = 0

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
