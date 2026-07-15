"""Time resolution utilities for PyVWF.

This module provides functions for:
- Time slice parsing (winter, spring, 1/6, etc.)
- Adding temporal resolution columns to DataFrames
- Time period definitions and mappings
"""
import pandas as pd


# Time slice to month mapping (used by both turbine-level and country-level)
TIME_SLICE_MAPPING = {
    'spring': [3, 4, 5],
    'summer': [6, 7, 8],
    'autumn': [9, 10, 11],
    'winter': [1, 2, 12],
    '1/6': [1, 2],
    '2/6': [3, 4],
    '3/6': [5, 6],
    '4/6': [7, 8],
    '5/6': [9, 10],
    '6/6': [11, 12],
    '1/1': list(range(1, 13)),
}


def parse_time_slice(time_slice, seasons=None):
    """Convert time_slice string to list of months.

    Args:
        time_slice: String like 'winter', '1/6', '1/1', or numeric month.
        seasons: Optional mapping of season name to month list (e.g. a
            region's explicit season definitions from the validation
            harness). When given, season names resolve through it INSTEAD of
            the hardcoded Northern-Hemisphere mapping; all other slices
            ('1/6', '1/1', numeric months) are hemisphere-neutral and resolve
            as before. Default None preserves the legacy behaviour exactly.

    Returns:
        list: List of month integers (1-12).

    Examples:
        >>> parse_time_slice('winter')
        [1, 2, 12]
        >>> parse_time_slice('winter', seasons={'winter': [6, 7, 8]})
        [6, 7, 8]
        >>> parse_time_slice('1/6')
        [1, 2]
        >>> parse_time_slice('3')
        [3]
    """
    if seasons is not None and time_slice in seasons:
        return list(seasons[time_slice])
    if time_slice in TIME_SLICE_MAPPING:
        return TIME_SLICE_MAPPING[time_slice]
    return [int(time_slice)]


def add_time_resolution_columns(df: pd.DataFrame, seasons=None) -> pd.DataFrame:
    """Add temporal resolution columns (season, bimonth, fixed) to a DataFrame.

    This function adds three columns based on the month:
    - bimonth: Two-month periods (1/6, 2/6, ..., 6/6)
    - season: Meteorological seasons (winter, spring, summer, autumn)
    - fixed: Always '1/1' (representing full year)

    Args:
        df: DataFrame with a ``month`` column.
        seasons: Optional mapping of season name to month list. When given,
            the ``season`` column is assigned from it instead of the
            hardcoded Northern-Hemisphere mapping below (``bimonth`` and
            ``fixed`` are hemisphere-neutral and unaffected). Default None
            preserves the legacy behaviour exactly.

    Returns:
        DataFrame with ``bimonth``, ``season``, and ``fixed`` columns added.

    Note:
        This function modifies the DataFrame in place and also returns it.

    Examples:
        >>> df = pd.DataFrame({'month': [1, 4, 7, 10]})
        >>> df = add_time_resolution_columns(df)
        >>> df[['month', 'bimonth', 'season']]
           month bimonth  season
        0      1     1/6  winter
        1      4     2/6  spring
        2      7     4/6  summer
        3     10     5/6  autumn
    """
    # Monthly assignments for bimonth and season
    month_to_resolution = {
        1: ("1/6", "winter"),
        2: ("1/6", "winter"),
        3: ("2/6", "spring"),
        4: ("2/6", "spring"),
        5: ("3/6", "spring"),
        6: ("3/6", "summer"),
        7: ("4/6", "summer"),
        8: ("4/6", "summer"),
        9: ("5/6", "autumn"),
        10: ("5/6", "autumn"),
        11: ("6/6", "autumn"),
        12: ("6/6", "winter"),
    }

    # Apply assignments
    for month, (bimonth, season) in month_to_resolution.items():
        df.loc[df["month"] == month, ["bimonth", "season"]] = [bimonth, season]

    # Explicit season definitions (validation harness) override the NH map.
    if seasons is not None:
        month_to_season = {
            month: name for name, months in seasons.items() for month in months
        }
        df["season"] = df["month"].map(month_to_season)

    # Fixed resolution for all months
    df["fixed"] = "1/1"

    return df
