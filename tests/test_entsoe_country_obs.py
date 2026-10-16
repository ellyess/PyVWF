"""The country-level observation writer (pyvwf.datasets.entsoe_country_obs), offline.

``generate_country_level_training_data`` writes every country-level training
and test file through these functions, and Norway's and Sweden's national
series are sums over their bidding zones. No test imported the module (#86),
and #84 will merge its Norwegian and Swedish copies, so these cases pin what
each writes: file names, the zone sums, and a zone with no data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyvwf.datasets import entsoe_country_obs as obs
from pyvwf.datasets.country_grid import NORWAY_ZONES, SWEDEN_ZONES


class FakeFetcher:
    """One hourly series per area: generation and capacity from the area's index."""

    def __init__(self, empty=()):
        self.empty = set(empty)
        self.calls = []

    def calculate_capacity_factor(self, country, start, end, psr_type="all"):
        self.calls.append(country)
        if country in self.empty:
            return pd.DataFrame()
        index = pd.date_range(start, periods=6, freq="h")
        k = 1 + sum(ord(c) for c in country) % 7
        frame = pd.DataFrame(
            {"generation_mw": np.full(6, 10.0 * k), "capacity_mw": np.full(6, 40.0 * k)},
            index=index,
        )
        frame["capacity_factor"] = frame["generation_mw"] / frame["capacity_mw"]
        return frame


def test_a_country_writes_its_train_and_test_files_by_the_names_the_source_reads(tmp_path):
    result = obs.fetch_country_observations(FakeFetcher(), "FR", [2015, 2016], 2017, tmp_path)
    d = tmp_path / "observations" / "fr"
    assert (d / "fr_train_2015_2016.csv").is_file()
    assert (d / "fr_test_2017.csv").is_file()
    assert set(result) == {"train", "test"}


def test_a_country_with_no_data_writes_nothing_and_returns_none(tmp_path):
    assert obs.fetch_country_observations(FakeFetcher({"FR"}), "FR", [2015], 2016, tmp_path) is None
    assert not (tmp_path / "observations").exists()


@pytest.mark.parametrize(
    "code, zones, fetch",
    [
        ("no", NORWAY_ZONES, obs.fetch_norway_zone_observations),
        ("se", SWEDEN_ZONES, obs.fetch_sweden_zone_observations),
    ],
)
def test_the_national_series_is_the_zone_sum_and_skips_a_zone_with_no_data(
    tmp_path, code, zones, fetch
):
    ids = list(zones)
    fetcher = FakeFetcher(empty={ids[-1]})
    result = fetch(fetcher, [2015, 2016], 2017, tmp_path)

    d = tmp_path / "observations" / code
    for zone in ids[:-1]:
        assert (
            tmp_path / "observations" / zone.lower() / f"{zone.lower()}_train_2015_2016.csv"
        ).is_file()
    assert not (tmp_path / "observations" / ids[-1].lower()).exists()
    agg = pd.read_csv(d / f"{code}_train_2015_2016_aggregated.csv", index_col=0)
    assert (d / f"{code}_test_2017_aggregated.csv").is_file()

    present = [
        fetcher.calculate_capacity_factor(z, pd.Timestamp("2015-01-01", tz="UTC"), None)
        for z in ids[:-1]
    ]
    gen = sum(f["generation_mw"].iloc[0] for f in present)
    cap = sum(f["capacity_mw"].iloc[0] for f in present)
    assert agg["generation_mw"].iloc[0] == pytest.approx(gen)
    assert agg["capacity_mw"].iloc[0] == pytest.approx(cap)
    assert agg["capacity_factor"].iloc[0] == pytest.approx(gen / cap)
    assert set(result["zones"]["train"]) == set(ids[:-1])


def test_norway_and_sweden_are_routed_to_their_zones_and_others_fetched_whole(tmp_path):
    fetcher = FakeFetcher()
    results = obs.fetch_observations(fetcher, ["FR", "NO", "SE"], [2015], 2016, tmp_path)
    assert set(results) == {"FR", "NO", "SE"}
    assert "NO" not in fetcher.calls and "SE" not in fetcher.calls
    assert set(NORWAY_ZONES) | set(SWEDEN_ZONES) <= set(fetcher.calls)
