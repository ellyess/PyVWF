"""Plausibility gates for country-level observed capacity factors.

These exist because the affine correction cannot fail loudly on bad
observations. A national series that is uniformly a factor too small still
produces a scalar that fits it, so the model looks healthy in sample and is
wrong out of it. The gate is the only place the error is visible.

The real motivating cases are recorded in
docs/findings/method-country-level.md: NL never exceeds CF 0.57 over four
years of quarter-hourly data, and IE rails against the fetcher's 1.5 clip.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vwf.loaders.country_obs_checks import (
    CLIP_CEILING,
    MIN_PEAK_CF,
    check_country_cf,
)


def series(values, freq="h", tz="UTC", periods=None, **extra):
    """A DatetimeIndexed CF frame; `values` is broadcast if scalar."""
    n = periods if periods is not None else (1 if np.isscalar(values) else len(values))
    idx = pd.date_range("2015-01-01", periods=n, freq=freq, tz=tz)
    return pd.DataFrame({"capacity_factor": values, **extra}, index=idx)


def healthy(n=2000, seed=0):
    """A series that passes every gate: national-looking mean and a real peak."""
    rng = np.random.default_rng(seed)
    cf = np.clip(rng.beta(1.4, 4.0, n), 0, 1)
    cf[0] = 0.93  # a windy hour, as every real fleet has
    return series(cf, periods=n)


# ---------------------------------------------------------------------------
# The gate passes real-looking data
# ---------------------------------------------------------------------------

def test_healthy_series_passes():
    report = check_country_cf(healthy(), "XX train")
    assert report.ok
    assert report.issues == []
    assert 0.1 < report.mean_cf < 0.4
    assert report.step_hours == pytest.approx(1.0)


def test_no_warning_when_healthy(recwarn):
    check_country_cf(healthy(), "XX train")
    assert [w for w in recwarn if issubclass(w.category, UserWarning)] == []


# ---------------------------------------------------------------------------
# The gate catches each failure mode
# ---------------------------------------------------------------------------

def test_understated_generation_is_caught_by_the_peak_test():
    """NL's signature. The mean can sit inside a plausible band while every
    value is scaled down, so the peak is what distinguishes it."""
    cf = healthy()["capacity_factor"].to_numpy() * 0.45
    report = check_country_cf(series(cf), "XX", warn=False)
    assert not report.ok
    assert any("never reaches" in issue for issue in report.issues)


def test_capacity_factor_above_one_is_caught():
    cf = healthy()["capacity_factor"].to_numpy().copy()
    cf[5] = 1.29
    report = check_country_cf(series(cf), "XX", warn=False)
    assert any("exceeds 1" in issue for issue in report.issues)


def test_clip_saturation_is_reported_separately_from_exceeding_one():
    """Rows on the fetcher's ceiling are not merely high; their true value was
    discarded, so they cannot be treated as observations at all."""
    cf = healthy()["capacity_factor"].to_numpy().copy()
    cf[:3] = CLIP_CEILING
    report = check_country_cf(series(cf), "XX", warn=False)
    assert any("clip ceiling" in issue for issue in report.issues)
    assert report.frac_clipped == pytest.approx(3 / len(cf))


def test_mean_outside_the_national_band_is_caught():
    report = check_country_cf(series(0.72, periods=500), "XX", warn=False)
    assert any("plausible national band" in issue for issue in report.issues)


def test_frozen_capacity_register_is_caught():
    """IE's signature: capacity pinned at one value for years, which inflates
    every capacity factor derived from it."""
    frame = healthy(n=40000)
    frame["capacity_mw"] = 1919.0
    report = check_country_cf(frame, "XX", warn=False)
    assert any("register did not update" in issue for issue in report.issues)


def test_moving_capacity_register_is_not_flagged():
    frame = healthy(n=40000)
    frame["capacity_mw"] = np.linspace(1900.0, 4300.0, len(frame))
    report = check_country_cf(frame, "XX", warn=False)
    assert not any("register" in issue for issue in report.issues)


def test_coarse_but_moving_register_is_not_flagged():
    """PT's register is two numbers over seven years, but the fleet genuinely
    only grew 15%. Coarseness alone is not evidence of a stalled register."""
    frame = healthy(n=40000)
    half = len(frame) // 2
    frame["capacity_mw"] = np.r_[np.full(half, 4486.0), np.full(len(frame) - half, 5181.0)]
    report = check_country_cf(frame, "XX", warn=False)
    assert not any("register" in issue for issue in report.issues)


def test_coarse_and_static_register_is_flagged():
    """IE's register: two numbers 0.6% apart over seven years, while Ireland's
    real wind fleet nearly doubled. Every derived CF is inflated."""
    frame = healthy(n=40000)
    half = len(frame) // 2
    frame["capacity_mw"] = np.r_[np.full(half, 1907.0), np.full(len(frame) - half, 1919.0)]
    report = check_country_cf(frame, "XX", warn=False)
    assert any("not tracking the fleet" in issue for issue in report.issues)


# ---------------------------------------------------------------------------
# Coverage drift
# ---------------------------------------------------------------------------

def drifting(annual_cf, seed=0):
    """One year per entry, at the given annual mean capacity factor."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=len(annual_cf) * 8760, freq="h", tz="UTC")
    cf = np.concatenate(
        [np.clip(rng.beta(1.4, 4.0, 8760) * (m / 0.259), 0, 1) for m in annual_cf]
    )
    cf[0] = 0.93
    return pd.DataFrame({"capacity_factor": cf}, index=idx)


def test_coverage_drift_is_caught():
    """NL's decisive signature: the annual mean CF rises 4.3x while capacity
    rises 2.9x, so the generation series covers a growing share of the fleet.
    No constant rescaling repairs that, which is why the peak test alone is not
    enough to say what NL needs."""
    report = check_country_cf(drifting([0.043, 0.063, 0.101, 0.171, 0.188]), "XX", warn=False)
    assert any("more than weather" in issue for issue in report.issues)


def test_ordinary_interannual_variation_is_not_drift():
    report = check_country_cf(drifting([0.24, 0.21, 0.26, 0.23, 0.25]), "XX", warn=False)
    assert not any("more than weather" in issue for issue in report.issues)


def test_drift_test_needs_several_years():
    """Two years of data cannot distinguish drift from a windy year."""
    report = check_country_cf(drifting([0.12, 0.30]), "XX", warn=False)
    assert not any("more than weather" in issue for issue in report.issues)


def test_mostly_missing_series_is_caught():
    cf = healthy()["capacity_factor"].to_numpy().copy()
    cf[::2] = np.nan
    report = check_country_cf(series(cf), "XX", warn=False)
    assert any("missing" in issue for issue in report.issues)


# ---------------------------------------------------------------------------
# Resolution handling
# ---------------------------------------------------------------------------

def test_peak_test_is_skipped_on_a_monthly_series():
    """A monthly mean legitimately never approaches the fleet peak. Applying
    the sub-daily threshold there would fail every correctly aggregated file."""
    monthly = series(np.full(60, 0.25), freq="MS")
    report = check_country_cf(monthly, "XX", warn=False)
    assert not any("never reaches" in issue for issue in report.issues)


def test_mixed_offset_index_still_gets_the_resolution_checks():
    """read_csv(parse_dates=True) hands back an object Index whenever the
    timestamps cross a DST boundary, which every ENTSO-E export does. Without
    coercion the resolution-dependent gates silently skip the worst files."""
    idx = pd.date_range("2015-01-01", periods=300, freq="15min", tz="UTC")
    frame = pd.DataFrame(
        {"capacity_factor": 0.2}, index=pd.Index(idx.astype(str), dtype=object)
    )
    report = check_country_cf(frame, "XX", warn=False)
    assert report.step_hours == pytest.approx(0.25)
    assert any("never reaches" in issue for issue in report.issues)


# ---------------------------------------------------------------------------
# Calling conventions
# ---------------------------------------------------------------------------

def test_strict_raises_instead_of_warning():
    with pytest.raises(ValueError, match="never reaches"):
        check_country_cf(series(0.1, periods=500), "XX", strict=True)


def test_warns_by_default_and_still_returns_the_report():
    with pytest.warns(UserWarning, match="XX train"):
        report = check_country_cf(series(0.1, periods=500), "XX train")
    assert not report.ok


def test_warn_false_stays_silent(recwarn):
    report = check_country_cf(series(0.1, periods=500), "XX", warn=False)
    assert not report.ok
    assert [w for w in recwarn if issubclass(w.category, UserWarning)] == []


def test_missing_column_always_raises():
    with pytest.raises(ValueError, match="capacity_factor"):
        check_country_cf(pd.DataFrame({"generation_mw": [1.0]}), "XX")


def test_as_row_is_tabulatable():
    rows = [check_country_cf(healthy(), "A", warn=False).as_row()]
    rows.append(check_country_cf(series(0.1, periods=500), "B", warn=False).as_row())
    table = pd.DataFrame(rows)
    assert list(table["label"]) == ["A", "B"]
    assert list(table["ok"]) == [True, False]


def test_threshold_constants_are_physically_ordered():
    assert 0.0 < MIN_PEAK_CF < 1.0 < CLIP_CEILING
