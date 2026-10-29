"""Shared steps of the builders that rebuild country-level observations.

The builders in ``scripts/region_tools/build_*`` replace ENTSO-E's series for
Belgium, France, Norway, Portugal and Sweden with each country's own
generation or fleet (``docs/runbooks/entsoe.md`` gives the order). Each one
splits an hourly national series into the training and test files, writes them
without losing the file it replaces, and, where the national source gives the
fleet only at the end of each year or month, interpolates it to every hour.
These are those steps, written once.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def split_utc_years(series: pd.DataFrame, first: int, last: int, tz: str) -> pd.DataFrame:
    """The UTC calendar years ``first`` to ``last``, stamped in local time.

    The years are cut in UTC, so a split holds whole UTC years, and the stamps
    are then converted to ``tz``, the country's own clock.

    Args:
        series: An hourly frame with a timezone-aware index.
        first: The first UTC year kept.
        last: The last UTC year kept.
        tz: The IANA timezone the kept rows are stamped in.

    Returns:
        The kept rows, with an unnamed index in ``tz``.
    """
    years = pd.DatetimeIndex(series.index).year
    out = series[(years >= first) & (years <= last)].copy()
    out.index = pd.DatetimeIndex(out.index).tz_convert(tz)
    out.index.name = None
    return out


def write_keeping_backup(
    frame: pd.DataFrame, path: Path, *, replace: bool, backup_suffix: str
) -> None:
    """Write ``frame`` to ``path``, keeping the file it first replaces.

    The first replacement renames the old file to ``path`` with
    ``backup_suffix``; later ones overwrite ``path`` and leave that backup as
    it is, so the backup always holds the series as it was before any
    rebuild.

    Args:
        frame: The rows to write.
        path: The CSV file to write.
        replace: Whether an existing ``path`` may be replaced.
        backup_suffix: The suffix of the backup, such as ``.entsoe.bak.csv``.

    Raises:
        FileExistsError: If ``path`` exists and ``replace`` is false.
    """
    if path.exists():
        if not replace:
            raise FileExistsError(f"{path} exists; pass --replace to keep it as a backup")
        backup = path.with_suffix(backup_suffix)
        if not backup.exists():
            path.rename(backup)
            print(f"  kept the old file as {backup.name}")
    frame.to_csv(path)


def interpolated_capacity(index: pd.DatetimeIndex, fleet: pd.Series) -> pd.Series:
    """The fleet at each hour, linear from one period's end to the next.

    A fleet counted at the end of each year (France, Sweden) or month
    (Portugal) is taken to hold at the first hour of the next period, and the
    capacity in between is interpolated linearly. A capacity held flat through
    the period would overstate its capacity factor by about half the period's
    growth.

    Args:
        index: UTC hours.
        fleet: MW installed at the end of each period, indexed by the year as
            an integer or by a monthly ``pandas.Period``.

    Returns:
        The capacity at each hour of ``index``; NaN where a neighbouring
        period's fleet is missing.
    """
    by_month = isinstance(fleet.index, pd.PeriodIndex)
    period = index.tz_convert(None).to_period("M" if by_month else "Y")
    start = period.to_timestamp().tz_localize("UTC")
    end = period.shift(1).to_timestamp().tz_localize("UTC")
    fraction = (index - start) / (end - start)
    key = period if by_month else period.year
    before = fleet.reindex(key - 1).to_numpy()
    after = fleet.reindex(key).to_numpy()
    return pd.Series(before + (after - before) * fraction, index=index)


def recapacitate(frame: pd.DataFrame, fleet: pd.Series) -> pd.DataFrame:
    """The same generation over the interpolated fleet, stamped as before.

    Args:
        frame: An observation file as read, with ``generation_mw`` and its
            original timestamps as the index.
        fleet: The fleet, as :func:`interpolated_capacity` takes it.

    Returns:
        A copy with ``capacity_mw`` and ``capacity_factor`` recomputed.
    """
    out = frame.copy()
    utc = pd.DatetimeIndex(pd.to_datetime(out.index, utc=True))
    out["capacity_mw"] = interpolated_capacity(utc, fleet).to_numpy()
    out["capacity_factor"] = out["generation_mw"] / out["capacity_mw"]
    return out
