"""Italy's observations keep their generation and take Eurostat's fleet.

``scripts/region_tools/build_it_capacity_from_eurostat.py`` replaces ENTSO-E's
Italian register, which runs about two years behind the fleet. These cases
check it on synthetic files and a synthetic Eurostat response.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from support import ROOT, load_script


def _module():
    return load_script(
        ROOT / "scripts/region_tools/build_it_capacity_from_eurostat.py", "build_it_cap"
    )


def observations(start: str, periods: int) -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq="h", tz="UTC").tz_convert("Europe/Rome")
    return pd.DataFrame(
        {"generation_mw": 500.0, "capacity_mw": 900.0, "capacity_factor": 500.0 / 900.0},
        index=index,
    )


def eurostat_response(path, fleet: dict[str, float]) -> None:
    """A JSON-stat body shaped like Eurostat's, with the years out of order in
    ``value`` so a parser that assumed order would misplace them."""
    years = sorted(fleet)
    body = {
        "dimension": {"time": {"category": {"index": {y: i for i, y in enumerate(years)}}}},
        "value": {str(i): fleet[y] for i, y in reversed(list(enumerate(years)))},
    }
    path.write_text(json.dumps(body))


def test_the_fleet_is_read_by_year(tmp_path):
    cache = tmp_path / "nrg_inf_epcrw_IT.json"
    eurostat_response(cache, {"2021": 11253.733, "2022": 11820.508, "2023": 12307.3})
    fleet = _module().fetch_fleet(cache)
    assert fleet.to_dict() == {2021: 11253.733, 2022: 11820.508, 2023: 12307.3}


def test_a_second_run_reads_the_backup_not_its_own_output(tmp_path, monkeypatch):
    """The generation must come from the file as fetched, which a new fetch
    puts in the backup. The backup's generation is changed after the first
    run, so a second run that read its own output would keep the old one."""
    module = _module()
    obs = tmp_path / "it"
    obs.mkdir()
    observations("2023-01-01 00:00", 3).to_csv(obs / "it_test_2023.csv")
    cache = tmp_path / "nrg_inf_epcrw_IT.json"
    eurostat_response(cache, {"2022": 1000.0, "2023": 2000.0})
    argv = ["build", "--obs", str(obs), "--cache", str(cache), "--replace"]
    monkeypatch.setattr("sys.argv", argv)
    module.main()
    out = pd.read_csv(obs / "it_test_2023.csv", index_col=0)
    # The first UTC hour of 2023 sits on the end-2022 fleet.
    assert out["capacity_factor"].iloc[0] == pytest.approx(0.5)
    backup = obs / "it_test_2023.entsoe-capacity.bak.csv"
    fetched = pd.read_csv(backup, index_col=0)
    fetched["generation_mw"] = 600.0
    fetched.to_csv(backup)
    module.main()
    out = pd.read_csv(obs / "it_test_2023.csv", index_col=0)
    assert out["capacity_factor"].iloc[0] == pytest.approx(0.6)
