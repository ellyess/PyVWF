"""Country-level data loaders for PyVWF.

This module provides functions to load grid points and observations
for country-level (ENTSO-E) workflows.
"""

from __future__ import annotations

from typing import cast

import pandas as pd
from pathlib import Path

from vwf.config import PyVWFPaths


def load_year_specific_grid_points(
    country: str, years: list[int], base_dir: Path | None = None
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
        year_grid["_year"] = year
        all_grid_points.append(year_grid)

    if not all_grid_points:
        raise FileNotFoundError(
            f"No year-specific grid point files found in {grid_points_dir}\n"
            f"Expected files like: {country.lower()}_grid_points_YYYY.csv\n"
            "Generate with: python -m vwf.datasets.generate_country_level_training_data"
        )

    # For missing years, try to use base grid points or nearest available year
    if missing_years:
        base_grid_file = grid_points_dir / f"{country.lower()}_grid_points.csv"

        for year in missing_years:
            if base_grid_file.exists():
                fallback_grid = pd.read_csv(base_grid_file)
                fallback_grid["_year"] = year
                all_grid_points.append(fallback_grid)

    # Concatenate all year-specific grid points
    grid_points_all = pd.concat(all_grid_points, ignore_index=True)

    # Create merged version (averaged across years for stability)
    grid_points_merged = (
        grid_points_all.groupby("ID", as_index=False)
        .first()
        .drop(columns=["_year"], errors="ignore")
    )

    # Create year-specific dictionary. groupby keys are typed as the generic
    # hashable label, so pin them back to int to match the declared return type
    # (`_year` is populated from the loop over `years: list[int]` above).
    grid_points_by_year = {
        int(cast(int, year)): gp.drop(columns=["_year"], errors="ignore")
        for year, gp in grid_points_all.groupby("_year")
    }

    return grid_points_merged, grid_points_by_year
