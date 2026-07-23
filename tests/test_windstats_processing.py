"""WindStats (ES/SE/FI) transforms: metadata, output->CF, twp->coords match.

All fixtures are synthetic: the real WindStats data is CONFIDENTIAL and never
enters the repo. These pin the reshaping and the coordinate-join guards.
"""
import numpy as np
import pandas as pd
import pytest

from vwf.config import PyVWFPaths
from vwf.datasets.windstats import (
    build_windstats_metadata,
    match_twp_to_coords,
    windstats_metadata,
    windstats_monthly_cf,
)
from vwf.sources.windstats import WindStatsSource, _base_country


def test_es_metadata_column_mapping():
    md = pd.DataFrame({
        "Unnamed: 0": ["A FARM 1", "A FARM 2"], "Manufacturer": ["Gamesa", "Made"],
        "kW": [660.0, 850.0], "Rotor (m)": [47.0, 52.0], "Tower (m)": [45.0, 55.0],
    })
    out = windstats_metadata(md, "ES")
    assert list(out["ID"]) == ["A FARM 1", "A FARM 2"]
    assert out["link"].tolist() == ["A FARM 1", "A FARM 2"]   # ES links on id
    assert out["capacity"].iloc[0] == 660.0 and out["type"].iloc[0] == "onshore"


def test_se_links_on_location_and_reads_onoffshore():
    md = pd.DataFrame({
        "Unnamed: 0": [1, 2], "Location": ["Aapua", "Lillgrund"],
        "Manufacturer": ["Enercon", "Siemens"], "On/offshore": ["onshore", "offshore"],
        "Rotor (m)": [48.0, 93.0], "Tower (m)": [50.0, 68.5], "kW": [800.0, 2300.0],
    })
    out = windstats_metadata(md, "SE")
    assert out["link"].tolist() == ["Aapua", "Lillgrund"]      # SE links on Location
    assert out["type"].tolist() == ["onshore", "offshore"]


def test_monthly_cf_from_output():
    # 660 kW turbine at CF 0.3 in a 30-day month: kWh = 660*0.3*30*24
    data = pd.DataFrame({"ID": ["A"], "Year": [1999], "Month": [4],
                         "Output": [660 * 0.3 * 30 * 24]})
    cap = pd.DataFrame({"ID": ["A"], "capacity": [660.0]})
    wide = windstats_monthly_cf(data, cap, 1999, 1999)
    assert wide.iloc[0]["obs_4"] == pytest.approx(0.3, abs=1e-9)


def test_monthly_cf_drops_no_capacity():
    data = pd.DataFrame({"ID": ["A"], "Year": [1999], "Month": [1], "Output": [1000.0]})
    cap = pd.DataFrame({"ID": ["B"], "capacity": [660.0]})
    assert len(windstats_monthly_cf(data, cap, 1999, 1999)) == 0


def test_twp_to_coords_fuzzy_matches_stripping_suffix():
    geo = pd.DataFrame({"ws": ["FARM X 1", "FARM Y 1"],
                        "twp": ["Aapua", "Nonexistent Place"]})
    gwpt = pd.DataFrame({"Project Name": ["Aapua wind farm"],
                         "Longitude": [23.4], "Latitude": [66.5]})
    m = match_twp_to_coords(geo, gwpt)
    assert m["ws"].tolist() == ["FARM X 1"]                    # only Aapua matched
    assert m["lon"].iloc[0] == pytest.approx(23.4)


def test_build_metadata_fails_loudly_and_excludes():
    smd = pd.DataFrame({
        "ID": ["a", "b"], "link": ["FX", "FY"], "capacity": [660.0, 850.0],
        "diameter": [47.0, 52.0], "height": [45.0, 0.0], "model": ["Gamesa", "Made"],
        "type": ["onshore", "onshore"],
    })
    coords = pd.DataFrame({"ws": ["FX"], "lon": [-3.0], "lat": [43.0]})
    with pytest.raises(ValueError, match="no coordinate"):
        build_windstats_metadata(smd, coords)
    md = build_windstats_metadata(smd, coords, exclude=("FY",), height_default=80.0)
    assert md["ID"].tolist() == ["a"]
    assert md["coord_source"].iloc[0] == "gwpt-open"
    assert md["model"].iloc[0] == "2019COE_Market_Average_2.6MW_121"  # uniform default
    assert md["windstats_manufacturer"].iloc[0] == "Gamesa"           # provenance kept


def test_region_code_maps_to_base_country():
    assert _base_country("ES-WS") == "ES"
    assert _base_country("SE-WS") == "SE"


def test_source_resolves_and_loads(monkeypatch, tmp_path):
    from vwf.sources import get_source
    es = tmp_path / "ES"
    es.mkdir()
    pd.DataFrame({"ID": ["a"], "lon": [-3.0], "lat": [43.0], "height": [45.0],
                  "capacity": [660.0], "model": ["M"], "type": ["onshore"]}
                 ).to_csv(es / "es_md.csv", index=False)
    pd.DataFrame({"ID": ["a", "a"], "year": [1998, 2000],
                  **{f"obs_{m}": [0.3, 0.35] for m in range(1, 13)}}
                 ).to_csv(es / "es_obs.csv", index=False)
    monkeypatch.setattr(PyVWFPaths, "TURBINE_DATA", tmp_path)
    src = get_source("windstats", "ES-WS")
    assert isinstance(src, WindStatsSource)
    assert src.load_metadata().iloc[0]["ID"] == "a"
    assert src.load_observations(1998, 1999)["year"].tolist() == [1998]


def test_missing_files_raise_with_runbook_pointer(monkeypatch, tmp_path):
    (tmp_path / "ES").mkdir()
    monkeypatch.setattr(PyVWFPaths, "TURBINE_DATA", tmp_path)
    src = WindStatsSource("ES-WS")
    with pytest.raises(FileNotFoundError, match="runbooks/ES"):
        src.load_metadata()
