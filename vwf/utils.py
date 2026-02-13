"""Common utility functions for PyVWF.

This module provides shared utilities for:
- Numeric data coercion and validation
- Coordinate handling
- Data standardization
"""
import pandas as pd
import numpy as np


def ensure_numeric(df: pd.DataFrame, cols: list[str], errors: str = 'coerce') -> pd.DataFrame:
    """Coerce specified columns to numeric type.

    Args:
        df: Input DataFrame.
        cols: List of column names to convert.
        errors: How to handle conversion errors. Options:
            - 'coerce': Invalid values become NaN (default)
            - 'raise': Raise exception on invalid values
            - 'ignore': Leave invalid values as-is

    Returns:
        DataFrame with specified columns converted to numeric.

    Note:
        This function modifies the DataFrame in place and also returns it.

    Examples:
        >>> df = pd.DataFrame({'lat': ['52.1', '51.8'], 'lon': ['4.3', '4.9']})
        >>> df = ensure_numeric(df, ['lat', 'lon'])
        >>> df.dtypes
        lat    float64
        lon    float64
        dtype: object
    """
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors=errors)
    return df


def ensure_finite_coords(df: pd.DataFrame, lon_col: str = 'lon', lat_col: str = 'lat') -> pd.DataFrame:
    """Ensure coordinates are numeric and finite (not NaN or inf).

    This function:
    1. Coerces lon/lat columns to numeric
    2. Removes rows with NaN or infinite coordinates

    Args:
        df: Input DataFrame with coordinate columns.
        lon_col: Name of longitude column (default: 'lon').
        lat_col: Name of latitude column (default: 'lat').

    Returns:
        DataFrame with only finite coordinate rows.

    Examples:
        >>> df = pd.DataFrame({
        ...     'lon': [4.5, 'invalid', 5.2],
        ...     'lat': [52.1, 51.8, np.nan]
        ... })
        >>> df = ensure_finite_coords(df)
        >>> len(df)
        1
    """
    # Coerce to numeric
    df = ensure_numeric(df, [lon_col, lat_col])

    # Remove non-finite values
    mask = df[lon_col].notna() & df[lat_col].notna()
    mask &= np.isfinite(df[lon_col]) & np.isfinite(df[lat_col])

    n_before = len(df)
    df = df[mask].copy()
    n_after = len(df)

    if n_after < n_before:
        print(f"  Removed {n_before - n_after} rows with invalid coordinates")

    return df


def standardize_turbine_columns(df: pd.DataFrame, minimal: bool = False) -> pd.DataFrame:
    """Standardize turbine metadata column names and types.

    This ensures consistent naming across different data sources.

    Args:
        df: Input DataFrame with turbine metadata.
        minimal: If True, only standardize essential columns (lat, lon, capacity).

    Returns:
        DataFrame with standardized column names and types.

    Column mappings:
        - lat, latitude, Latitude → lat
        - lon, longitude, Longitude, long → lon
        - capacity, Capacity, cap → capacity
        - height, Height, hub_height → height
        - ID, id, turbine_id → ID
    """
    # Column name mappings
    rename_map = {}

    # Latitude
    for old in ['latitude', 'Latitude', 'Lat']:
        if old in df.columns and 'lat' not in df.columns:
            rename_map[old] = 'lat'

    # Longitude
    for old in ['longitude', 'Longitude', 'Long', 'long']:
        if old in df.columns and 'lon' not in df.columns:
            rename_map[old] = 'lon'

    # Capacity
    for old in ['Capacity', 'cap', 'Cap']:
        if old in df.columns and 'capacity' not in df.columns:
            rename_map[old] = 'capacity'

    if not minimal:
        # Height
        for old in ['Height', 'hub_height', 'Hub_height']:
            if old in df.columns and 'height' not in df.columns:
                rename_map[old] = 'height'

        # ID
        for old in ['id', 'turbine_id', 'Turbine_ID']:
            if old in df.columns and 'ID' not in df.columns:
                rename_map[old] = 'ID'

    # Apply renames
    if rename_map:
        df = df.rename(columns=rename_map)

    # Ensure numeric types for key columns
    numeric_cols = ['lat', 'lon', 'capacity']
    if not minimal:
        numeric_cols.extend(['height'])

    df = ensure_numeric(df, [c for c in numeric_cols if c in df.columns])

    return df


def capacity_weighted_average(df: pd.DataFrame, value_col: str, capacity_col: str = 'capacity') -> float:
    """Calculate capacity-weighted average of a column.

    Args:
        df: Input DataFrame.
        value_col: Column name containing values to average.
        capacity_col: Column name containing capacity weights (default: 'capacity').

    Returns:
        Capacity-weighted average.

    Raises:
        ValueError: If required columns are missing or capacity sum is zero.

    Examples:
        >>> df = pd.DataFrame({
        ...     'cf': [0.3, 0.35, 0.4],
        ...     'capacity': [2.0, 3.0, 1.0]
        ... })
        >>> capacity_weighted_average(df, 'cf')
        0.35
    """
    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not found in DataFrame")
    if capacity_col not in df.columns:
        raise ValueError(f"Column '{capacity_col}' not found in DataFrame")

    total_capacity = df[capacity_col].sum()
    if total_capacity == 0:
        raise ValueError("Total capacity is zero, cannot compute weighted average")

    return (df[value_col] * df[capacity_col]).sum() / total_capacity


def fill_missing_with_default(df: pd.DataFrame, column: str, default_value, fillna: bool = True) -> pd.DataFrame:
    """Fill missing values in a column, optionally creating the column if it doesn't exist.

    Args:
        df: Input DataFrame.
        column: Column name to fill.
        default_value: Value to use for missing/new entries.
        fillna: If True, also fill existing NaN values (default: True).

    Returns:
        DataFrame with filled column.

    Examples:
        >>> df = pd.DataFrame({'lat': [52.1, 52.3]})
        >>> df = fill_missing_with_default(df, 'height', 100.0)
        >>> df['height'].tolist()
        [100.0, 100.0]
    """
    if column not in df.columns:
        df[column] = default_value
    elif fillna:
        df[column] = df[column].fillna(default_value)

    return df
