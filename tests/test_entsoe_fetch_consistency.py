"""The ENTSO-E capacity factor's numerator and denominator must cover one fleet.

NL's stored series does not. Its capacity factor climbs 4.3x between 2015 and
2021 while installed capacity climbs 2.9x, the signature of a generation series
covering a growing share of the fleet the capacity counts. Downstream nothing
can see it: the affine correction absorbs the error into the scalar and reports
a clean in-sample fit. The only place to catch it is where the ratio is formed.

See docs/results.md#the-country-level-estimator.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pyvwf.datasets.fetch_entsoe_capacity_factors import (
    NORWAY_ZONES,
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


def test_missing_capacity_is_refused_not_estimated():
    """An empty register used to become gen.max() / 0.9, a denominator made
    from the numerator, which is how every Swedish zonal series came to peak at
    exactly 0.900."""
    fetcher = fetcher_with(None, None)
    fetcher.fetch_installed_capacity = lambda *a, **k: pd.DataFrame()
    with pytest.raises(ValueError, match="no installed capacity"):
        calculate(fetcher)


def test_unlabelled_queries_are_not_blocked():
    """A single-psr_type fetch records no coverage, and must still work rather
    than failing closed on missing metadata."""
    assert calculate(fetcher_with(None, None))["capacity_factor"].iloc[0] == pytest.approx(0.1)


def belgian_offshore_query():
    """A generation result shaped as entsoe-py returned Belgium's in 2021 and 2023.

    Offshore output and consumption as tuple columns, and the year's last chunk,
    which had no consumption, as a plain column on its own hours.
    """
    index = pd.date_range("2023-12-28 22:00", periods=4, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            ("Wind Offshore", "Actual Consumption"): [3.0, None, None, None],
            ("Wind Offshore", "Actual Aggregated"): [100.0, 200.0, None, None],
            "Wind Offshore": [None, None, 300.0, 400.0],
        },
        index=index,
    )


def fetcher_returning(frame):
    obj = object.__new__(ENTSOEWindDataFetcher)  # no API key needed
    obj.client = type("Client", (), {"query_generation": lambda self, **k: frame})()
    return obj


@pytest.mark.parametrize("country, zones", [("BE", 1), ("NO", len(NORWAY_ZONES))])
@pytest.mark.parametrize("psr_type", ["all", "offshore"])
def test_consumption_is_not_counted_as_output(psr_type, country, zones):
    """Belgium's offshore wind reports consumption, which was summed into
    output; the last chunk's plain column is output and must stay. Norway sums
    its zones, each answered here with the same frame."""
    gen = fetcher_returning(belgian_offshore_query()).fetch_generation(
        country,
        pd.Timestamp("2023-12-28", tz="UTC"),
        pd.Timestamp("2024-01-01", tz="UTC"),
        psr_type=psr_type,
    )
    assert gen["generation_mw"].tolist() == [zones * v for v in (100.0, 200.0, 300.0, 400.0)]


def mixed_hours_query():
    """Onshore and offshore output across five hours: offshore first reports in
    the second hour, onshore is missing in the third, and nothing reports in
    the fourth."""
    index = pd.date_range("2023-07-20 05:00", periods=5, freq="h", tz="UTC")
    nan = float("nan")
    return pd.DataFrame(
        {
            ("Wind Onshore", "Actual Aggregated"): [90.0, 100.0, nan, nan, 120.0],
            ("Wind Offshore", "Actual Aggregated"): [nan, 50.0, 3.0, nan, 60.0],
        },
        index=index,
    )


def test_an_hour_missing_one_wind_type_is_dropped_not_reported_as_the_fleet():
    """France's 2023-07-20 07:00 UTC had offshore output and no onshore value,
    and was kept as 3 MW against the whole fleet's capacity. An hour with no
    value at all became zero output. Both are dropped; the hour before offshore
    first reports is kept, as offshore output not yet in the series."""
    gen = fetcher_returning(mixed_hours_query()).fetch_generation(
        "FR",
        pd.Timestamp("2023-07-20", tz="UTC"),
        pd.Timestamp("2023-07-21", tz="UTC"),
        psr_type="all",
    )
    assert [t.hour for t in gen.index] == [5, 6, 9]
    assert gen["generation_mw"].tolist() == [90.0, 150.0, 180.0]


def test_an_hour_with_no_output_is_dropped_not_zero():
    index = pd.date_range("2023-07-20 05:00", periods=3, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            ("Wind Offshore", "Actual Aggregated"): [50.0, float("nan"), 60.0],
            "Wind Offshore": [float("nan")] * 3,
        },
        index=index,
    )
    gen = fetcher_returning(frame).fetch_generation(
        "FR",
        pd.Timestamp("2023-07-20", tz="UTC"),
        pd.Timestamp("2023-07-21", tz="UTC"),
        psr_type="offshore",
    )
    assert gen["generation_mw"].tolist() == [50.0, 60.0]
