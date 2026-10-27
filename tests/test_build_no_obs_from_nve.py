"""Norway's observations rebuilt from NVE count each farm only when its output and capacity agree.

``scripts/region_tools/build_no_obs_from_nve.py`` replaces ENTSO-E files whose
generation fell short of the official totals in 2015-21 and whose capacity was
ENTSO-E's register. These cases check the reading and the farm-month rule on
synthetic data.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from support import ROOT, load_script


def _module():
    return load_script(ROOT / "scripts/region_tools/build_no_obs_from_nve.py", "build_no_obs")


def register(tmp_path, groups):
    """A register file with one farm (ID 1) and the given turbine groups."""
    turbines = [
        {"AntallTurbiner": n, "TurbinStorrelse_kW": kw, "DatoIdriftsatt": on, "DatoUtavdrift": off}
        for n, kw, on, off in groups
    ]
    path = tmp_path / "register.json"
    path.write_text(json.dumps([{"VindkraftAnleggId": 1, "Turbiner": turbines}]))
    return path


def test_metered_stamps_are_utc_plus_one_hour_beginnings(tmp_path):
    pytest.importorskip("openpyxl")
    sheet = pd.DataFrame(
        [["kraftverknavn", "Farm"], ["kraftverknr", 7], ["timestamp", None]]
        + [[pd.Timestamp("2020-01-01 00:00"), 5.0], [pd.Timestamp("2020-01-01 01:00"), None]]
    )
    path = tmp_path / "metered.xlsx"
    sheet.to_excel(path, header=False, index=False)
    out = _module().read_metered(path)
    assert out.columns.tolist() == [7]
    assert out.index[0] == pd.Timestamp("2019-12-31 23:00", tz="UTC")
    assert out.iloc[0, 0] == 5.0 and pd.isna(out.iloc[1, 0])


def test_a_group_counts_from_commissioning_until_decommissioning(tmp_path):
    path = register(tmp_path, [(2, 1000, "2020-01-02", None), (1, 500, "2020-01-01", "2020-01-03")])
    index = pd.date_range("2020-01-01", periods=4, freq="1D", tz="UTC")
    fleet = _module().register_fleet(path, index)
    assert fleet[1].tolist() == [0.5, 2.5, 2.0, 2.0]


def hours(month: str) -> pd.DatetimeIndex:
    return pd.date_range(month, pd.Period(month).end_time, freq="1h", tz="UTC")


@pytest.mark.parametrize(
    ("case", "counts"),
    [("settled", True), ("gap", False), ("capacity changes", False), ("above capacity", False)],
)
def test_a_farm_counts_in_a_month_only_when_output_and_capacity_agree(case, counts):
    """A farm-month counts when it is metered every hour, its registered
    capacity is fixed and above zero, and its output never exceeds 1.05 times
    it: the register dates turbines after they start producing."""
    index = hours("2020-03")
    metered = pd.DataFrame({1: 5.0, 2: 3.0}, index=index)
    fleet = pd.DataFrame({1: 10.0, 2: 10.0}, index=index)
    if case == "gap":
        metered.iloc[100, 0] = None
    elif case == "capacity changes":
        fleet.iloc[300:, 0] = 20.0
    elif case == "above capacity":
        metered.iloc[200, 0] = 10.6
    out = _module().national(metered, fleet)
    assert out["generation_mw"].iloc[0] == (8.0 if counts else 3.0)
    assert out["capacity_mw"].iloc[0] == (20.0 if counts else 10.0)


def test_split_takes_utc_years_and_stamps_oslo_time():
    module = _module()
    index = pd.date_range("2018-12-31 22:00", periods=3, freq="1h", tz="UTC")
    series = pd.DataFrame({"generation_mw": 1.0, "capacity_mw": 2.0}, index=index)
    out = module.split(series, 2018, 2018)
    assert [t.isoformat() for t in out.index] == [
        "2018-12-31T23:00:00+01:00",
        "2019-01-01T00:00:00+01:00",
    ]


def test_write_refuses_to_overwrite_without_replace(tmp_path):
    module = _module()
    path = tmp_path / "no_test_2019_aggregated.csv"
    path.write_text("old\n")
    frame = pd.DataFrame({"capacity_factor": [0.5]})
    with pytest.raises(FileExistsError):
        module.write(frame, path, replace=False)
    module.write(frame, path, replace=True)
    assert (tmp_path / "no_test_2019_aggregated.entsoe.bak.csv").read_text() == "old\n"
