"""Belgium's observations rebuilt from Elia cover the same fleet in every hour.

``scripts/region_tools/build_be_obs_from_elia.py`` replaces ENTSO-E files that
held onshore generation in some years and onshore plus offshore in others. These
cases check the aggregation on a synthetic Elia export.
"""

from __future__ import annotations

import pandas as pd
from support import ROOT, load_script


def _module():
    return load_script(ROOT / "scripts/region_tools/build_be_obs_from_elia.py", "build_be_obs")


def elia_rows(start: str, periods: int, onshore=(100.0, 200.0), offshore=(300.0, 600.0)):
    """A 15-minute export with constant output and capacity per segment."""
    times = pd.date_range(start, periods=periods, freq="15min", tz="UTC")
    rows = []
    for kind, (measured, capacity) in (("Onshore", onshore), ("Offshore", offshore)):
        rows += [
            {"datetime": t, "offshoreonshore": kind, "measured": measured, "capacity": capacity}
            for t in times
        ]
    return pd.DataFrame(rows)


def test_both_segments_are_summed_before_the_ratio():
    out = _module().hourly_national(elia_rows("2023-01-01", 8))
    assert len(out) == 2
    assert out["generation_mw"].tolist() == [400.0, 400.0]
    assert out["capacity_mw"].tolist() == [800.0, 800.0]
    assert out["capacity_factor"].tolist() == [0.5, 0.5]


def test_an_hour_missing_one_segment_is_dropped_not_shrunk():
    raw = elia_rows("2023-01-01", 8)
    raw = raw[~((raw.offshoreonshore == "Offshore") & (raw.datetime >= "2023-01-01 01:00"))]
    out = _module().hourly_national(raw)
    # The second hour has only onshore rows. Keeping it would report onshore
    # generation as the national fleet, the defect this script exists to fix.
    assert len(out) == 1


def test_a_step_with_no_capacity_is_a_gap_not_an_infinite_factor():
    raw = elia_rows("2023-01-01", 8)
    gap = raw.datetime == pd.Timestamp("2023-01-01 00:15", tz="UTC")
    raw.loc[gap, "capacity"] = 0.0
    out = _module().hourly_national(raw)
    assert out["capacity_factor"].tolist() == [0.5, 0.5]
    raw.loc[raw.datetime < pd.Timestamp("2023-01-01 01:00", tz="UTC"), "capacity"] = 0.0
    assert len(_module().hourly_national(raw)) == 1


def test_split_takes_utc_years_and_stamps_brussels_time():
    module = _module()
    raw = elia_rows("2022-12-31 22:00", 4 * 4)
    series = module.hourly_national(raw)
    out = module.split_utc_years(series, 2023, 2023, module.TZ)
    assert [t.isoformat() for t in out.index] == [
        "2023-01-01T01:00:00+01:00",
        "2023-01-01T02:00:00+01:00",
    ]
