"""Portugal's observations keep their generation over REN's month-end wind fleet.

``scripts/region_tools/build_pt_capacity_from_ren.py`` replaces a capacity that
was frozen over 2015-19 and onshore-only from 2020 while the generation
included offshore. These cases check the interpolation and the rewrite.
"""

from __future__ import annotations

import pandas as pd
import pytest
from support import ROOT, load_script


def _module():
    return load_script(ROOT / "scripts/region_tools/build_pt_capacity_from_ren.py", "build_pt_cap")


FLEET = pd.Series([100.0, 200.0], index=pd.PeriodIndex(["2023-01", "2023-02"], freq="M"))


def test_capacity_runs_from_one_month_end_to_the_next():
    module = _module()
    index = pd.DatetimeIndex(["2023-02-01 00:00", "2023-02-15 00:00"], tz="UTC")
    capacity = module.interpolated_capacity(index, FLEET)
    assert capacity.iloc[0] == pytest.approx(100.0)
    assert capacity.iloc[1] == pytest.approx(150.0)


def test_the_cached_table_is_read_without_a_download(tmp_path):
    cache = tmp_path / "ren.csv"
    cache.write_text("month,wind_mw\n2023-01,100\n2023-02,200\n")
    fleet = _module().fetch_fleet(cache, 2024, 2024)
    assert fleet.to_dict() == {pd.Period("2023-01", "M"): 100, pd.Period("2023-02", "M"): 200}


def test_generation_is_kept_and_divided_by_the_fleet():
    frame = pd.DataFrame(
        {"generation_mw": [75.0], "capacity_mw": [999.0], "capacity_factor": [0.1]},
        index=["2023-02-15 01:00:00+01:00"],
    )
    out = _module().recapacitate(frame, FLEET)
    assert out.index.tolist() == frame.index.tolist()
    assert out["generation_mw"].iloc[0] == 75.0
    assert out["capacity_mw"].iloc[0] == pytest.approx(150.0)
    assert out["capacity_factor"].iloc[0] == pytest.approx(0.5)


def test_write_keeps_the_old_file_once(tmp_path):
    module = _module()
    path = tmp_path / "pt_test_2023.csv"
    path.write_text("old\n")
    frame = pd.DataFrame({"capacity_factor": [0.5]})
    with pytest.raises(FileExistsError):
        module.write(frame, path, replace=False)
    module.write(frame, path, replace=True)
    module.write(frame, path, replace=True)
    assert (tmp_path / "pt_test_2023.entsoe-capacity.bak.csv").read_text() == "old\n"
