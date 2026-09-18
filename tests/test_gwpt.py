"""The GWPT filters and plant-name keys (vwf.datasets.gwpt), on synthetic rows.

The real workbook is local only, so its pins (tests/test_pin_gwpt.py) skip in
CI. These cases cover the same functions where CI can run them, including the
one real difference between the two filters: whether the country string is
stripped before it is compared.
"""
import pandas as pd
import pytest

from vwf.datasets import gwpt


def tracker() -> pd.DataFrame:
    return pd.DataFrame({
        "Country/Area": ["Ireland", "Ireland ", "Ireland", "Ireland", "Ireland", "France"],
        "Status": ["operating", "Operating", "operating", "retired", "operating", "operating"],
        "Project Name": ["A", "B", "C", "D", "E", "F"],
        "GEM phase ID": ["G1", "G2", "G3", "G4", "G5", "G6"],
        "Latitude": [53.0, 53.1, None, 53.3, 53.4, 46.0],
        "Longitude": [-8.0, -8.1, -8.2, -8.3, -8.4, 2.0],
        "Capacity (MW)": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "Start year": [2010, 2012, 2012, 2005, None, 2015],
        "Retired year": [None, None, None, 2016, None, None],
    })


def test_fleet_for_compares_the_country_as_written_and_needs_coordinates():
    fleet = gwpt.fleet_for(tracker(), "IE", None)
    # B's trailing space excludes it; C has no latitude; D is retired.
    assert fleet["mw"].tolist() == [10.0, 50.0]
    assert list(fleet.columns) == ["lat", "lon", "mw"]


def test_fleet_for_as_of_a_year_keeps_projects_without_a_start_year():
    assert gwpt.fleet_for(tracker(), "IE", 2011)["mw"].tolist() == [10.0, 50.0]
    assert gwpt.fleet_for(tracker(), "IE", 2009)["mw"].tolist() == [50.0]


def test_fleet_for_drops_curated_exclusions(capsys):
    assert gwpt.fleet_for(tracker(), "IE", None, {"G1"})["mw"].tolist() == [50.0]
    assert "excluding 1 curated GWPT record(s): A" in capsys.readouterr().out


def test_fleet_for_refuses_an_unmapped_region():
    with pytest.raises(KeyError, match="XX"):
        gwpt.fleet_for(tracker(), "XX", None)


def test_operating_projects_strips_the_country_and_keeps_every_operating_row():
    names = gwpt.operating_projects(tracker(), "Ireland")["Project Name"].tolist()
    assert names == ["A", "B", "C", "E"]


def test_plant_key_drops_words_and_optionally_stage_numerals():
    name = "Parque Eólico Los Meandros III"
    assert gwpt.plant_key(name, gwpt.DROP_CL) == "MEANDROS III"
    assert gwpt.plant_key(name, gwpt.DROP_AR, drop_roman=True) == "MEANDROS"


def test_projects_with_keys_columns():
    frame = gwpt.projects_with_keys(tracker(), "Ireland", lambda s: s.lower())
    assert list(frame.columns) == ["Project Name", "norm", "cap", "Latitude", "Longitude"]
    assert frame["norm"].tolist() == ["a", "b", "c", "e"]


def test_load_exclusions_of_a_missing_file_is_empty(tmp_path):
    assert gwpt.load_exclusions(tmp_path / "absent.csv") == set()
