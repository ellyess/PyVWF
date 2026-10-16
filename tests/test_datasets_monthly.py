"""The shared monthly capacity-factor steps (pyvwf.datasets.monthly, #85).

Eight adapters now build their wide observation frames through these. The
real-data pins check that no adapter's output moved; these check each step on
a frame small enough to compute by hand.
"""

from __future__ import annotations

from datetime import timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from pyvwf.datasets import monthly as mo
from pyvwf.sources.base import METADATA_COLUMNS, require_columns


def hourly(unit, year, month, hours, cf=0.3):
    return pd.DataFrame({"ID": unit, "year": year, "month": month, "cf": [cf] * hours})


def test_coverage_mean_keeps_a_full_month_and_blanks_a_thin_one():
    # January 2021 has 744 hours; 0.9 coverage needs 670.
    full = hourly("a", 2021, 1, 744, cf=0.4)
    thin = hourly("b", 2021, 1, 600, cf=0.4)
    agg = mo.coverage_mean(pd.concat([full, thin]), samples_per_day=24)
    got = agg.set_index("ID")
    assert got.loc["a", "cf"] == pytest.approx(0.4)
    assert np.isnan(got.loc["b", "cf"])
    assert got["n"].tolist() == [744, 600]


def test_wide_monthly_spreads_twelve_months_and_its_two_pivots_differ_only_as_documented():
    long = pd.DataFrame(
        {"ID": ["a", "a", "b"], "year": [2021] * 3, "month": [1, 3, 2], "cf": [0.1, 0.3, np.nan]}
    )
    wide = mo.wide_monthly(long)
    assert list(wide.columns) == ["ID", "year"] + mo.OBS_COLUMNS
    assert wide.set_index("ID").loc["a", ["obs_1", "obs_2", "obs_3"]].tolist()[::2] == [0.1, 0.3]
    assert "b" in set(wide["ID"])
    # pivot_table drops a unit whose every month is missing; pivot keeps it.
    assert "b" not in set(mo.wide_monthly(long, aggfunc="mean")["ID"])


def test_a_month_starting_before_commissioning_is_masked():
    monthly = pd.DataFrame(
        {"ID": ["a"] * 3, "year": [2021] * 3, "month": [1, 2, 3], "cf": [0.1, 0.2, 0.3]}
    )
    meta = pd.DataFrame({"ID": ["a"], "commissioning_date": ["2021-02-15"]})
    mo.mask_before_commissioning(monthly, meta)
    # February contains the date, so it is masked too; March is the first valid month.
    assert monthly["cf"].isna().tolist() == [True, True, False]


def test_an_unparseable_commissioning_date_raises_unless_coerced():
    monthly = pd.DataFrame({"ID": ["a"], "year": [2021], "month": [1], "cf": [0.1]})
    meta = pd.DataFrame({"ID": ["a"], "commissioning_date": ["not a date"]})
    with pytest.raises((ValueError, TypeError)):
        mo.mask_before_commissioning(monthly.copy(), meta)
    mo.mask_before_commissioning(monthly, meta, coerce=True)
    assert monthly["cf"].tolist() == [0.1]


def test_the_leading_near_zero_run_is_stripped_and_later_low_months_kept():
    wide = pd.DataFrame({"ID": ["a"], "year": [2021], **{f"obs_{m}": [0.3] for m in range(1, 13)}})
    wide.loc[0, ["obs_1", "obs_2", "obs_6"]] = [0.0, 0.01, 0.0]
    out = mo.strip_commissioning_prefix(wide)
    assert out[["obs_1", "obs_2"]].isna().all(axis=None)
    assert out.loc[0, "obs_6"] == 0.0


def test_naive_local_time_converts_and_aware_time_is_refused():
    ts = pd.Series(pd.to_datetime(["2021-01-01 00:00"]))
    got = mo.naive_local_to_utc(ts, timezone(timedelta(hours=-3)), "X must be naive")
    assert got.iloc[0] == pd.Timestamp("2021-01-01 03:00")
    with pytest.raises(ValueError, match="X must be naive; got timezone-aware"):
        mo.naive_local_to_utc(ts.dt.tz_localize("UTC"), timezone.utc, "X must be naive")


def test_metadata_missing_a_column_is_refused_by_name():
    meta = pd.DataFrame({c: [1] for c in METADATA_COLUMNS - {"model"}})
    with pytest.raises(ValueError, match=r"meta.csv is missing required columns \['model'\]"):
        require_columns(meta, METADATA_COLUMNS, "meta.csv")
    require_columns(meta.assign(model="m"), METADATA_COLUMNS, "meta.csv")
