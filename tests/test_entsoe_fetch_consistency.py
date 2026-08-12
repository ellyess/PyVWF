"""The ENTSO-E capacity factor's numerator and denominator must cover one fleet.

NL's stored series does not. Its capacity factor climbs 4.3x between 2015 and
2021 while installed capacity climbs 2.9x, the signature of a generation series
covering a growing share of the fleet the capacity counts. Downstream nothing
can see it: the affine correction absorbs the error into the scalar and reports
a clean in-sample fit. The only place to catch it is where the ratio is formed.

See docs/findings/country_level_method_review.md.
"""
from __future__ import annotations

import pandas as pd
import pytest

from vwf.datasets.fetch_entsoe_capacity_factors import (
    ENTSOEWindDataFetcher,
    _wind_kinds,
)


def test_wind_kinds_from_plain_columns():
    assert _wind_kinds(["Wind Onshore", "Wind Offshore"]) == {"onshore", "offshore"}


def test_wind_kinds_from_multiindex_columns():
    """entsoe-py returns (type, aggregation) tuples for some queries."""
    columns = [("Wind Onshore", "Actual Aggregated")]
    assert _wind_kinds(columns) == {"onshore"}


def test_wind_kinds_falls_back_for_an_undifferentiated_column():
    assert _wind_kinds(["Wind"]) == {"wind"}


def fetcher_with(gen_kinds, cap_kinds):
    """A fetcher whose two queries report the given coverage."""
    index = pd.date_range("2015-01-01", periods=4, freq="h", tz="UTC")
    gen = pd.DataFrame({"generation_mw": [100.0] * 4}, index=index)
    cap = pd.DataFrame({"capacity_mw": [1000.0] * 4}, index=index)
    if gen_kinds is not None:
        gen.attrs["wind_columns"] = set(gen_kinds)
    if cap_kinds is not None:
        cap.attrs["wind_columns"] = set(cap_kinds)

    obj = object.__new__(ENTSOEWindDataFetcher)  # no API key needed
    obj.fetch_generation = lambda *a, **k: gen
    obj.fetch_installed_capacity = lambda *a, **k: cap
    return obj


def calculate(fetcher):
    return fetcher.calculate_capacity_factor(
        "NL", pd.Timestamp("2015-01-01", tz="UTC"), pd.Timestamp("2015-01-02", tz="UTC")
    )


def test_matching_coverage_produces_a_capacity_factor():
    result = calculate(fetcher_with({"onshore", "offshore"}, {"onshore", "offshore"}))
    assert result["capacity_factor"].iloc[0] == pytest.approx(0.1)


def test_generation_narrower_than_capacity_is_refused():
    """The NL failure mode: offshore-only generation over a whole-fleet
    denominator understates every capacity factor, by a share that moves."""
    with pytest.raises(ValueError, match="generation covers"):
        calculate(fetcher_with({"offshore"}, {"onshore", "offshore"}))


def test_capacity_narrower_than_generation_is_refused():
    with pytest.raises(ValueError, match="installed"):
        calculate(fetcher_with({"onshore", "offshore"}, {"onshore"}))


def test_the_error_says_how_to_fix_it():
    with pytest.raises(ValueError) as excinfo:
        calculate(fetcher_with({"offshore"}, {"onshore", "offshore"}))
    assert "psr_type" in str(excinfo.value)


def test_unlabelled_queries_are_not_blocked():
    """A single-psr_type fetch records no coverage, and must still work rather
    than failing closed on missing metadata."""
    assert calculate(fetcher_with(None, None))["capacity_factor"].iloc[0] == pytest.approx(0.1)
