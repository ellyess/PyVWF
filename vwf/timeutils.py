"""
timeutils module.

Small helpers for adding year/month and derived seasonal/bimonth labels.
Kept separate to avoid circular imports between data.py and wind.py.

Conventions
-----------
- Expects a column named 'time' convertible by pandas to datetime.
- Adds integer 'year' and 'month' plus categorical columns:
    - 'bimonth' in {'1/6', ..., '6/6'}
    - 'season' in {'winter','spring','summer','autumn'}
    - 'fixed' = '1/1'
"""
from __future__ import annotations
import pandas as pd


def add_times(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    """
    Add integer year and month columns derived from a datetime column.

    Parameters
    ----------
    df : pandas.DataFrame
    time_col : str
        Name of datetime column (default "time")

    Returns
    -------
    pandas.DataFrame
        Same df with 'year' and 'month' inserted after time_col.
    """
    t = pd.to_datetime(df[time_col])
    df["year"] = t.dt.year.astype(int)
    df["month"] = t.dt.month.astype(int)

    # keep your original column positioning behaviour
    if time_col in df.columns:
        # insert year and month after time_col
        time_idx = df.columns.get_loc(time_col)
        # move columns by popping and reinserting
        year = df.pop("year")
        month = df.pop("month")
        df.insert(time_idx + 1, "year", year)
        df.insert(time_idx + 2, "month", month)
    return df


def add_time_res(df: pd.DataFrame, month_col: str = "month") -> pd.DataFrame:
    """
    Add 'bimonth', 'season', and 'fixed' columns based on month.

    Parameters
    ----------
    df : pandas.DataFrame
    month_col : str
        Name of integer month column (default "month")

    Returns
    -------
    pandas.DataFrame
    """
    m = df[month_col]

    df.loc[m.isin([1, 2]),  ["bimonth", "season"]] = ["1/6", "winter"]
    df.loc[m.isin([3, 4]),  ["bimonth", "season"]] = ["2/6", "spring"]
    df.loc[m.isin([5, 6]),  ["bimonth", "season"]] = ["3/6", "spring"]  # keep your original mapping
    df.loc[m.isin([7, 8]),  ["bimonth", "season"]] = ["4/6", "summer"]
    df.loc[m.isin([9, 10]), ["bimonth", "season"]] = ["5/6", "autumn"]
    df.loc[m.isin([11, 12]),["bimonth", "season"]] = ["6/6", "winter"]  # keep your original mapping

    df["fixed"] = "1/1"
    return df
