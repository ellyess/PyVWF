"""The steps the country-observation builders share.

Each builder under ``scripts/region_tools/build_*`` carried its own copy of
these until they moved to ``pyvwf.datasets.country_obs_rebuild``; the
builders' own tests keep the parts that differ by country.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyvwf.datasets.country_obs_rebuild import (
    interpolated_capacity,
    recapacitate,
    split_utc_years,
    write_keeping_backup,
)

YEARLY = pd.Series({2022: 1000.0, 2023: 2000.0})
MONTHLY = pd.Series([100.0, 200.0], index=pd.PeriodIndex(["2023-01", "2023-02"], freq="M"))


def test_years_are_cut_in_utc_and_stamped_in_local_time():
    index = pd.date_range("2022-12-31 22:00", periods=4, freq="1h", tz="UTC")
    series = pd.DataFrame({"generation_mw": 1.0}, index=index)
    out = split_utc_years(series, 2023, 2023, "Europe/Paris")
    # 23:00 UTC on 31 December is already 2023 in Paris, and stays out.
    assert [t.isoformat() for t in out.index] == [
        "2023-01-01T01:00:00+01:00",
        "2023-01-01T02:00:00+01:00",
    ]
    assert out.index.name is None


def test_write_refuses_to_replace_without_the_flag(tmp_path):
    path = tmp_path / "xx_test_2023.csv"
    path.write_text("old\n")
    with pytest.raises(FileExistsError):
        write_keeping_backup(
            pd.DataFrame({"a": [1]}), path, replace=False, backup_suffix=".x.bak.csv"
        )
    assert path.read_text() == "old\n"


def test_write_refuses_even_when_a_backup_exists(tmp_path):
    # Portugal's and Sweden's builders used to overwrite silently once a backup
    # existed, although their docstrings said nothing is overwritten without
    # --replace.
    path = tmp_path / "xx_test_2023.csv"
    path.write_text("rebuilt\n")
    (tmp_path / "xx_test_2023.x.bak.csv").write_text("old\n")
    with pytest.raises(FileExistsError):
        write_keeping_backup(
            pd.DataFrame({"a": [1]}), path, replace=False, backup_suffix=".x.bak.csv"
        )
    assert path.read_text() == "rebuilt\n"


def test_write_keeps_the_file_it_first_replaced(tmp_path):
    path = tmp_path / "xx_test_2023.csv"
    path.write_text("old\n")
    for value in (1, 2):
        write_keeping_backup(
            pd.DataFrame({"a": [value]}), path, replace=True, backup_suffix=".x.bak.csv"
        )
    assert (tmp_path / "xx_test_2023.x.bak.csv").read_text() == "old\n"
    assert pd.read_csv(path, index_col=0)["a"].tolist() == [2]


def test_a_new_file_needs_no_flag(tmp_path):
    path = tmp_path / "xx_test_2023.csv"
    write_keeping_backup(pd.DataFrame({"a": [1]}), path, replace=False, backup_suffix=".x.bak.csv")
    assert path.exists()
    assert not (tmp_path / "xx_test_2023.x.bak.csv").exists()


@pytest.mark.parametrize(
    ("fleet", "hours", "expected"),
    [
        (YEARLY, ["2023-01-01 00:00", "2023-07-02 12:00"], [1000.0, 1500.0]),
        (MONTHLY, ["2023-02-01 00:00", "2023-02-15 00:00"], [100.0, 150.0]),
    ],
    ids=["year-end", "month-end"],
)
def test_capacity_runs_from_one_period_end_to_the_next(fleet, hours, expected):
    capacity = interpolated_capacity(pd.DatetimeIndex(hours, tz="UTC"), fleet)
    assert capacity.to_numpy() == pytest.approx(expected)


def test_capacity_is_missing_where_a_neighbouring_period_is():
    capacity = interpolated_capacity(pd.DatetimeIndex(["2022-06-01 00:00"], tz="UTC"), YEARLY)
    assert np.isnan(capacity.iloc[0])


def test_recapacitating_keeps_generation_and_stamps():
    index = pd.date_range("2023-01-01 00:00", periods=3, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {"generation_mw": 500.0, "capacity_mw": 8354.0, "capacity_factor": 0.06},
        index=index.tz_convert("Europe/Stockholm").astype(str),
    )
    out = recapacitate(frame, YEARLY)
    assert out.index.equals(frame.index)
    assert out["generation_mw"].tolist() == frame["generation_mw"].tolist()
    assert out["capacity_factor"].iloc[0] == pytest.approx(0.5)
