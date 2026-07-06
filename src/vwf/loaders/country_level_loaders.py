"""Country-level data loaders for PyVWF.

This module provides functions to load grid points and observations
for country-level (ENTSO-E) workflows.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from calendar import monthrange

from vwf.config import PyVWFPaths


def _hours_in_month(year: int, month: int) -> int:
    """Calculate hours in a given month."""
    return monthrange(int(year), int(month))[1] * 24


def _aggregate_observations_to_monthly(obs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate high-frequency observations to monthly means.

    ENTSO-E data is typically at 15-minute resolution. PyVWF requires monthly
    means to match turbine-level temporal resolution.

    Args:
        obs: DataFrame with datetime index and columns (capacity_factor, generation_mw, capacity_mw).

    Returns:
        DataFrame with monthly-aggregated data, same columns as input.
    """
    # Ensure datetime index
    if not isinstance(obs.index, pd.DatetimeIndex):
        obs.index = pd.to_datetime(obs.index, utc=True)
    
    # Convert timezone-aware to naive UTC if needed
    if obs.index.tz is not None:
        obs.index = obs.index.tz_convert('UTC').tz_localize(None)
    
    # Resample to monthly means (ME = month end frequency)
    obs_monthly = obs.resample('ME').mean()
    
    return obs_monthly


def country_gen_to_cf(
    obs_country_gen: pd.DataFrame,
    turb_info: pd.DataFrame,
    *,
    output_col: str = "output_kwh",
    capacity_unit: str = "kW",  # "kW" or "MW"
) -> pd.DataFrame:
    """Convert country-level monthly generation to capacity factor.

    Args:
        obs_country_gen: DataFrame with columns ``year``, ``month``, and ``output_col``.
        turb_info: Turbine metadata with a ``capacity`` column.
        output_col: Column name for generation output.
        capacity_unit: Capacity units (``"kW"`` or ``"MW"``).

    Returns:
        DataFrame with columns ``year``, ``month``, and ``obs`` (capacity factor).

    Raises:
        ValueError: If required columns are missing or capacity data is invalid.

    Examples:
        >>> obs_gen = pd.DataFrame({'year': [2018], 'month': [1], 'output_kwh': [1000000]})
        >>> turb_info = pd.DataFrame({'capacity': [500, 300]})  # kW
        >>> cf = country_gen_to_cf(obs_gen, turb_info)
        >>> print(cf['obs'].iloc[0])  # Should be generation / (total_capacity * hours)
    """
    if "capacity" not in turb_info.columns:
        raise ValueError("turb_info must contain a 'capacity' column.")

    cap = pd.to_numeric(turb_info["capacity"], errors="coerce")

    if capacity_unit.lower() == "mw":
        cap_kw = float(cap.sum() * 1000.0)
    else:
        cap_kw = float(cap.sum())

    if cap_kw < 10_000:  # entire country < 10 MW? probably wrong units
        raise ValueError("Total capacity looks too small; check units (kW vs MW).")

    if not np.isfinite(cap_kw) or cap_kw <= 0:
        raise ValueError(
            "Total capacity from turb_info is not valid for converting country gen to CF. "
            f"(sum capacity={cap.sum()}, unit={capacity_unit})"
        )

    df = obs_country_gen.copy()
    if not {"year", "month", output_col}.issubset(df.columns):
        raise ValueError(f"country_gen_to_cf expects columns ['year','month','{output_col}'].")

    df["hours"] = df.apply(lambda r: _hours_in_month(r["year"], r["month"]), axis=1)
    df["obs"] = pd.to_numeric(df[output_col], errors="coerce") / (cap_kw * df["hours"].astype(float))
    return df[["year", "month", "obs"]]


def load_year_specific_grid_points(
    country: str,
    years: list[int],
    base_dir: Path = None
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """Load year-specific grid points that reflect capacity changes over time.

    This function loads grid points for each year separately, allowing bias corrections
    to account for the actual installed capacity at each point in time (reflecting
    commissioning and decommissioning of wind farms).

    Args:
        country: Country code (NL, FR, BE, NO, SE, ES, IT, PT, IE).
        years: List of years to load grid points for (e.g., [2015, 2016, 2017]).
        base_dir: Base directory for country-level data. If None, uses PyVWFPaths.COUNTRY_LEVEL_DATA.

    Returns:
        Tuple of (merged_grid_points, grid_points_by_year):
        - merged_grid_points: DataFrame with averaged metadata across all years
        - grid_points_by_year: Dict mapping year -> grid points DataFrame for that year

    Raises:
        FileNotFoundError: If no year-specific grid point files are found.

    Examples:
        >>> grid_merged, grid_by_year = load_year_specific_grid_points('NL', [2015, 2016, 2017])
        >>> print(f"Loaded {len(grid_by_year)} years")
        >>> # Use with PyVWF via load_country_data_with_year_specific()
    """
    if base_dir is None:
        base_dir = PyVWFPaths.COUNTRY_LEVEL_DATA

    country = country.upper()
    grid_points_dir = base_dir / "grid_points" / country.lower()

    all_grid_points = []
    missing_years = []

    for year in sorted(years):
        grid_file = grid_points_dir / f"{country.lower()}_grid_points_{year}.csv"
        
        if not grid_file.exists():
            missing_years.append(year)
            continue

        year_grid = pd.read_csv(grid_file)
        year_grid['_year'] = year
        all_grid_points.append(year_grid)

    if not all_grid_points:
        raise FileNotFoundError(
            f"No year-specific grid point files found in {grid_points_dir}\n"
            f"Expected files like: {country.lower()}_grid_points_YYYY.csv\n"
            f"Generate with: python vwf/datasets/generate_country_level_training_data.py"
        )

    # For missing years, try to use base grid points or nearest available year
    if missing_years:
        base_grid_file = grid_points_dir / f"{country.lower()}_grid_points.csv"
        
        for year in missing_years:
            if base_grid_file.exists():
                fallback_grid = pd.read_csv(base_grid_file)
                fallback_grid['_year'] = year
                all_grid_points.append(fallback_grid)

    # Concatenate all year-specific grid points
    grid_points_all = pd.concat(all_grid_points, ignore_index=True)

    # Create merged version (averaged across years for stability)
    grid_points_merged = grid_points_all.groupby('ID', as_index=False).first().drop(columns=['_year'], errors='ignore')
    
    # Create year-specific dictionary
    grid_points_by_year = {
        year: gp.drop(columns=['_year'], errors='ignore') 
        for year, gp in grid_points_all.groupby('_year')
    }

    return grid_points_merged, grid_points_by_year