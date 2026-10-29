"""Sweden's observations keep their generation and take Energimyndigheten's fleet.

``scripts/region_tools/build_se_capacity_from_energimyndigheten.py`` replaces a
capacity-factor denominator that an old fetcher derived from generation (flat
at 8,354 MW in 2015-19) and one that counted onshore wind at the start of each
year. These cases check it on synthetic files.
"""

from __future__ import annotations

import pandas as pd
import pytest
from support import ROOT, load_script


def _module():
    return load_script(
        ROOT / "scripts/region_tools/build_se_capacity_from_energimyndigheten.py", "build_se_cap"
    )


FLEET = pd.Series({2022: 1000.0, 2023: 2000.0})


def observations(start: str, periods: int) -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq="h", tz="UTC").tz_convert("Europe/Stockholm")
    return pd.DataFrame(
        {"generation_mw": 500.0, "capacity_mw": 8354.0, "capacity_factor": 500.0 / 8354.0},
        index=index,
    )


def test_generation_and_stamps_are_kept_and_the_ratio_follows():
    frame = observations("2023-01-01 00:00", 3)
    out = _module().recapacitate(frame, FLEET)
    assert out.index.equals(frame.index)
    assert out["generation_mw"].tolist() == frame["generation_mw"].tolist()
    assert out["capacity_factor"].iloc[0] == pytest.approx(0.5)


def test_a_second_run_reads_the_backup_not_its_own_output(tmp_path, monkeypatch):
    """The generation must come from the file as fetched, so running twice
    gives the same files rather than compounding."""
    module = _module()
    obs = tmp_path / "se"
    obs.mkdir()
    observations("2023-01-01 00:00", 3).to_csv(obs / "se_test_2023_aggregated.csv")
    cache = tmp_path / "EN0105_1.csv"
    cache.write_text(
        '"År","Installerad effekt, MW"\n"2022",1000.00\n"2023",2000.00\n', encoding="latin-1"
    )
    argv = ["build", "--obs", str(obs), "--cache", str(cache), "--replace"]
    monkeypatch.setattr("sys.argv", argv)
    module.main()
    first = (obs / "se_test_2023_aggregated.csv").read_text()
    module.main()
    assert (obs / "se_test_2023_aggregated.csv").read_text() == first
    assert (obs / "se_test_2023_aggregated.entsoe-capacity.bak.csv").exists()
