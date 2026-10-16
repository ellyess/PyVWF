"""The monthly capacity-factor steps every turbine-level adapter shares.

Eight adapters turned their source's series into the same wide contract,
``ID``, ``year``, ``obs_1``..``obs_12``, each with its own copy of the same
steps, and the copies had begun to diverge: one had no guard for an empty
frame (#85). The steps live here once:

- :func:`coverage_mean`: a sample-weighted monthly mean, NaN where a month has
  less than ``min_coverage`` of its expected samples;
- :func:`mask_before_commissioning`: NaN for every month that starts before a
  unit's commissioning date;
- :func:`wide_monthly`: the pivot to the wide contract;
- :func:`strip_commissioning_prefix`: NaN for a plant's leading run of
  near-zero months, the fleet-entry artefact of a static nameplate;
- :func:`naive_local_to_utc`: a source's naive civil time to naive UTC, for a
  zone with no daylight saving.

It lives in ``pyvwf.datasets`` because both the processing modules here and
the adapters in ``pyvwf.sources`` use it, and the import layers let
``sources`` import ``datasets`` but not the reverse.
"""

from __future__ import annotations

from typing import Any

from datetime import tzinfo

import pandas as pd

from pyvwf.time_utils import month_days

#: Months with less than this fraction of their expected samples are NaN.
DEFAULT_MIN_COVERAGE = 0.9

#: The monthly columns of the wide observation contract.
OBS_COLUMNS = [f"obs_{m}" for m in range(1, 13)]


def empty_monthly() -> pd.DataFrame:
    """The wide contract with no rows."""
    return pd.DataFrame(columns=["ID", "year"] + OBS_COLUMNS)


def wide_monthly(frame: pd.DataFrame, value: str = "cf", aggfunc: Any = None) -> pd.DataFrame:
    """Pivot a long (``ID``, ``year``, ``month``) frame to the wide contract.

    Args:
        frame: Long frame with ``ID``, ``year``, ``month`` and ``value``.
        value: The column to spread across ``obs_1``..``obs_12``.
        aggfunc: None to require one row per (``ID``, ``year``, ``month``), as
            ``DataFrame.pivot`` does; otherwise passed to ``pivot_table``,
            which also drops a row whose every month is missing. Each adapter
            keeps the choice it made before this helper existed.

    Returns:
        Wide frame ``ID``, ``year``, ``obs_1``..``obs_12``, missing months NaN.
    """
    if aggfunc is None:
        wide = frame.pivot(index=["ID", "year"], columns="month", values=value)
    else:
        wide = frame.pivot_table(
            index=["ID", "year"], columns="month", values=value, aggfunc=aggfunc
        )
    wide = wide.reindex(columns=range(1, 13)).reset_index()
    wide.columns = ["ID", "year"] + OBS_COLUMNS
    return wide


def coverage_mean(
    frame: pd.DataFrame,
    *,
    samples_per_day: float,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    value: str = "cf",
) -> pd.DataFrame:
    """Monthly mean of ``value`` per unit, NaN where too few samples were seen.

    Args:
        frame: Long frame with ``ID``, ``year``, ``month`` and ``value``, one
            row per sample (an hour, a half-hour, a SCADA interval).
        samples_per_day: Samples a complete day has (24 hourly, 48 half-hourly).
        min_coverage: Fraction of a month's expected samples required; months
            below it are NaN rather than biased by a gap.
        value: The column averaged.

    Returns:
        Long frame ``ID``, ``year``, ``month``, ``value``, ``n`` (samples seen).
    """
    agg = (
        frame.groupby(["ID", "year", "month"])
        .agg(**{value: (value, "mean"), "n": (value, "count")})
        .reset_index()
    )
    expected = month_days(agg["year"], agg["month"]) * float(samples_per_day)
    agg.loc[agg["n"] / expected < min_coverage, value] = float("nan")
    return agg


def mask_before_commissioning(
    monthly: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    coerce: bool = False,
    value: str = "cf",
) -> None:
    """Set ``value`` to NaN, in place, for every month that starts before commissioning.

    The first fully post-commissioning month is the first valid one: a month
    that contains the commissioning date is masked too.

    Args:
        monthly: Long frame with ``ID``, ``year``, ``month`` and ``value``.
        metadata: Unit metadata with ``ID`` and, if present,
            ``commissioning_date``; without that column nothing is masked.
        coerce: Turn an unparseable date into a missing one (the EIA adapter's
            choice) rather than raise (AEMO's).
        value: The column masked.
    """
    if "commissioning_date" not in metadata.columns:
        return
    dates = metadata.set_index("ID")["commissioning_date"]
    if coerce:
        commissioning = pd.to_datetime(dates, errors="coerce").dropna()
    else:
        commissioning = pd.to_datetime(dates.dropna())
    month_start = pd.to_datetime(
        monthly["year"].astype(str) + "-" + monthly["month"].astype(str) + "-01"
    )
    commissioned = monthly["ID"].map(commissioning)
    monthly.loc[commissioned.notna() & (month_start < commissioned), value] = float("nan")


def strip_commissioning_prefix(wide: pd.DataFrame, *, threshold: float = 0.02) -> pd.DataFrame:
    """NaN a plant's leading run of months whose CF is below ``threshold``.

    A plant that came online mid-window reports near-zero energy before it
    runs, and against a static nameplate that reads as a CF near zero: a
    fleet-entry artefact, not resource signal. This strips the leading run of
    sub-threshold months per plant, up to its first operational month. A later
    low month is kept: after commissioning it is a real outage, curtailment or
    low-wind month. It does not fix a partial build, where a farm at half its
    turbines reads at half CF against the full nameplate.

    Args:
        wide: Wide frame ``ID``, ``year``, ``obs_1``..``obs_12``.
        threshold: CF below which a leading month is treated as pre-commissioning.

    Returns:
        A copy with the leading sub-threshold months NaN.
    """
    out = wide.sort_values(["ID", "year"]).copy()
    for _, block in out.groupby("ID"):
        cells = [
            (row.name, f"obs_{m}", row[f"obs_{m}"])
            for _, row in block.iterrows()
            for m in range(1, 13)
        ]
        for idx, col, val in cells:
            if pd.notna(val) and val >= threshold:
                break
            if pd.notna(val):
                out.at[idx, col] = float("nan")
    return out


def naive_local_to_utc(timestamps: pd.Series, tz: tzinfo, what: str) -> pd.Series:
    """Naive civil timestamps in ``tz`` to naive UTC.

    For a zone with a fixed offset (Chile's standard time, Brasília, AEMO's
    market time). New Zealand's trading periods, which cross daylight saving,
    keep their own conversion in ``pyvwf.datasets.emi_nz``.

    Args:
        timestamps: Naive timestamps, or strings ``pd.to_datetime`` parses.
        tz: The zone they are civil time in.
        what: How the error names the input, e.g. ``"CEN timestamps must be
            naive Chilean standard time"``.

    Raises:
        ValueError: If the timestamps are already timezone-aware.
    """
    ts = pd.to_datetime(timestamps)
    if getattr(ts.dt, "tz", None) is not None:
        raise ValueError(f"{what}; got timezone-aware input")
    return ts.dt.tz_localize(tz).dt.tz_convert("UTC").dt.tz_localize(None)
