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
