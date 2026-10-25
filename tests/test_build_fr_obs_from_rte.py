"""France's observations rebuilt from RTE cover the same fleet in every hour.

``scripts/region_tools/build_fr_obs_from_rte.py`` replaces ENTSO-E files whose
2023 test year left out offshore generation and whose 2018-19 capacity fell
behind the fleet. These cases check the aggregation on synthetic rows shaped
like ODRE's export of RTE's national series.
"""

from __future__ import annotations

import pandas as pd
import pytest
from support import ROOT, load_script


def _module():
    return load_script(ROOT / "scripts/region_tools/build_fr_obs_from_rte.py", "build_fr_obs")


def odre_rows(start: str, periods: int, output: float = 1000.0) -> pd.DataFrame:
    """Half-hourly rows stamped in UTC, as ODRE exports them."""
    stamps = pd.date_range(start, periods=periods, freq="30min", tz="UTC")
    return pd.DataFrame(
        {"date_heure": stamps.strftime("%Y-%m-%dT%H:%M:%S+00:00"), "eolien": output}
    )


FLEET = pd.Series({2022: 1000.0, 2023: 2000.0})


def test_half_hours_are_averaged_into_utc_hours():
    out = _module().hourly_output(odre_rows("2023-01-01 00:00", 4))
    assert [t.isoformat() for t in out.index] == [
        "2023-01-01T00:00:00+00:00",
        "2023-01-01T01:00:00+00:00",
    ]
    assert out.tolist() == [1000.0, 1000.0]


def test_an_hour_missing_a_half_hour_is_dropped_not_halved():
    raw = odre_rows("2023-01-01 00:00", 4).drop(index=3)
    out = _module().hourly_output(raw)
    assert [t.isoformat() for t in out.index] == ["2023-01-01T00:00:00+00:00"]


def test_the_clock_change_hours_odre_cannot_place_are_dropped():
    """ODRE inherits RTE's 48 half-hours on both change days. In March it gives
    01:00 and 01:30 UTC twice; in October it leaves 00:00 and 00:30 UTC empty
    and puts the once-written 02:00 Paris row at 01:00 UTC, which could equally
    be 00:00. The hours it cannot place go empty and every other hour stays."""
    module = _module()
    march_rows = odre_rows("2023-03-25 23:00", 46)
    repeated = march_rows[march_rows["date_heure"].str.startswith("2023-03-26T01:")]
    march = module.hourly_output(pd.concat([march_rows, repeated]))
    assert len(march) == 22
    assert pd.Timestamp("2023-03-26 01:00", tz="UTC") not in march.index
    october_rows = odre_rows("2023-10-28 22:00", 50)
    october_rows = october_rows[~october_rows["date_heure"].str.startswith("2023-10-29T00:")]
    october = module.hourly_output(october_rows)
    assert len(october) == 23
    assert pd.Timestamp("2023-10-29 00:00", tz="UTC") not in october.index
    assert pd.Timestamp("2023-10-29 01:00", tz="UTC") not in october.index


def test_capacity_runs_from_one_31_december_to_the_next():
    module = _module()
    index = pd.DatetimeIndex(["2023-01-01 00:00", "2023-07-02 12:00"], tz="UTC")
    capacity = module.interpolated_capacity(index, FLEET)
    assert capacity.iloc[0] == pytest.approx(1000.0)
    assert capacity.iloc[1] == pytest.approx(1500.0)


def test_capacity_factor_is_output_over_interpolated_capacity():
    out = _module().national(odre_rows("2023-01-01 00:00", 2), FLEET)
    assert out["capacity_factor"].iloc[0] == pytest.approx(1.0)


def test_split_takes_utc_years_and_stamps_paris_time():
    module = _module()
    fleet = pd.Series({2022: 1000.0, 2023: 2000.0, 2024: 3000.0})
    series = module.national(odre_rows("2023-12-31 22:00", 6), fleet)
    out = module.split(series, 2023, 2023)
    # The last UTC hour of 2023 is 1 January in Paris.
    assert [t.isoformat() for t in out.index] == [
        "2023-12-31T23:00:00+01:00",
        "2024-01-01T00:00:00+01:00",
    ]


def test_write_refuses_to_overwrite_without_replace(tmp_path):
    module = _module()
    path = tmp_path / "fr_test_2023.csv"
    path.write_text("old\n")
    frame = pd.DataFrame({"capacity_factor": [0.5]})
    with pytest.raises(FileExistsError):
        module.write(frame, path, replace=False)
    module.write(frame, path, replace=True)
    assert (tmp_path / "fr_test_2023.entsoe.bak.csv").read_text() == "old\n"
    assert "capacity_factor" in path.read_text()
