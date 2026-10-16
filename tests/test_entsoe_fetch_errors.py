"""An ENTSO-E query that fails must fail, not come back empty.

The four fetch methods caught every exception and returned an empty frame, and
every caller reads an empty frame as "ENTSO-E has no data here": a country is
skipped, and a Norwegian zone is left out of the national sum. So a network or
authentication error became a missing country, or a national total quietly
short of one zone's fleet. Only ``NoMatchingDataError``, ENTSO-E's own answer
for a period it holds nothing for, still means empty.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pyvwf.datasets.fetch_entsoe_capacity_factors import (
    NORWAY_ZONES,
    ENTSOEWindDataFetcher,
    NoMatchingDataError,
)

START = pd.Timestamp("2020-01-01", tz="UTC")
END = pd.Timestamp("2020-01-02", tz="UTC")

# Each fetch method, through the single-area path (FR) and the Norwegian
# per-zone path (NO).
METHODS = [
    (method, country)
    for method in ("fetch_generation", "fetch_installed_capacity")
    for country in ("FR", "NO")
]


def wind_frame(value: float) -> pd.DataFrame:
    index = pd.date_range(START, periods=4, freq="h")
    return pd.DataFrame({"Wind Onshore": [value] * 4}, index=index)


class FakeClient:
    """Answers each query per area: a frame, or an exception to raise."""

    def __init__(self, answers: dict[str, object]):
        self.answers = answers

    def _answer(self, country_code, **_):
        answer = self.answers[country_code]
        if isinstance(answer, Exception):
            raise answer
        return answer

    query_generation = _answer
    query_installed_generation_capacity = _answer


def fetcher(answers: dict[str, object]) -> ENTSOEWindDataFetcher:
    obj = object.__new__(ENTSOEWindDataFetcher)  # no API key needed
    obj.client = FakeClient(answers)
    return obj


def areas(country: str) -> list[str]:
    return list(NORWAY_ZONES) if country == "NO" else [country]


@pytest.mark.parametrize("method, country", METHODS)
def test_a_failed_query_raises(method, country):
    """Every area fails with a network error, and the error reaches the caller."""
    answers = {a: ConnectionError("ENTSO-E unreachable") for a in areas(country)}
    with pytest.raises(ConnectionError):
        getattr(fetcher(answers), method)(country, START, END)


@pytest.mark.parametrize("method, country", METHODS)
def test_no_matching_data_is_still_empty(method, country):
    """ENTSO-E's own "nothing for this period" keeps its meaning."""
    answers = {a: NoMatchingDataError() for a in areas(country)}
    assert getattr(fetcher(answers), method)(country, START, END).empty


@pytest.mark.parametrize("method", ["fetch_generation", "fetch_installed_capacity"])
def test_one_failed_norwegian_zone_fails_the_national_fetch(method):
    """The partial sum is the dangerous case: four zones' fleet reported as
    Norway's, with nothing downstream able to tell."""
    answers: dict[str, object] = {zone: wind_frame(100.0) for zone in NORWAY_ZONES}
    answers["NO_2"] = TimeoutError("read timed out")
    with pytest.raises(TimeoutError):
        getattr(fetcher(answers), method)("NO", START, END)


@pytest.mark.parametrize("method", ["fetch_generation", "fetch_installed_capacity"])
def test_a_norwegian_zone_with_no_data_is_skipped(method):
    """NO_5 has no wind, which ENTSO-E reports as no data; the other four sum."""
    answers: dict[str, object] = {zone: wind_frame(100.0) for zone in NORWAY_ZONES}
    answers["NO_5"] = NoMatchingDataError()
    result = getattr(fetcher(answers), method)("NO", START, END)
    assert result.iloc[:, 0].tolist() == [400.0] * 4


class SplitClient:
    """Separate answers for generation and for installed capacity, per zone."""

    def __init__(self, generation: dict[str, object], capacity: dict[str, object]):
        self.generation, self.capacity = generation, capacity

    @staticmethod
    def _answer(answers, country_code):
        answer = answers[country_code]
        if isinstance(answer, Exception):
            raise answer
        return answer

    def query_generation(self, country_code, **_):
        return self._answer(self.generation, country_code)

    def query_installed_generation_capacity(self, country_code, **_):
        return self._answer(self.capacity, country_code)


def split_fetcher(generation, capacity) -> ENTSOEWindDataFetcher:
    obj = object.__new__(ENTSOEWindDataFetcher)
    obj.client = SplitClient(generation, capacity)
    return obj


@pytest.mark.parametrize("psr_type", ["all", "onshore"])
def test_norway_refuses_zones_present_on_one_side_only(psr_type):
    """Known positive: capacity reports NO_3, generation has no data for it (#79)."""
    generation: dict[str, object] = {zone: wind_frame(50.0) for zone in NORWAY_ZONES}
    capacity: dict[str, object] = {zone: wind_frame(100.0) for zone in NORWAY_ZONES}
    generation["NO_5"] = capacity["NO_5"] = NoMatchingDataError()
    generation["NO_3"] = NoMatchingDataError()
    with pytest.raises(ValueError, match=r"capacity only: \['NO_3'\]"):
        split_fetcher(generation, capacity).calculate_capacity_factor(
            "NO", START, END, psr_type=psr_type
        )


@pytest.mark.parametrize("psr_type", ["all", "onshore"])
def test_norway_with_the_same_zones_on_both_sides_passes(psr_type):
    """Known negative: NO_5 missing from both sides is the normal case."""
    generation: dict[str, object] = {zone: wind_frame(50.0) for zone in NORWAY_ZONES}
    capacity: dict[str, object] = {zone: wind_frame(100.0) for zone in NORWAY_ZONES}
    generation["NO_5"] = capacity["NO_5"] = NoMatchingDataError()
    cf = split_fetcher(generation, capacity).calculate_capacity_factor(
        "NO", START, END, psr_type=psr_type
    )
    assert cf["capacity_factor"].tolist() == [0.5] * 4
