"""The GWPT filters and plant-name keys (pyvwf.datasets.gwpt), on synthetic rows.

The real workbook is local only, so its pins (tests/test_pin_gwpt.py) skip in
CI. These cases cover the same functions where CI can run them, including the
one real difference between the two filters: whether the country string is
stripped before it is compared.
"""

import pandas as pd
import pytest

from pyvwf.datasets import gwpt


def tracker() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Country/Area": ["Ireland", "Ireland ", "Ireland", "Ireland", "Ireland", "France"],
            "Status": ["operating", "Operating", "operating", "retired", "operating", "operating"],
            "Project Name": ["A", "B", "C", "D", "E", "F"],
            "GEM phase ID": ["G1", "G2", "G3", "G4", "G5", "G6"],
            "Latitude": [53.0, 53.1, None, 53.3, 53.4, 46.0],
            "Longitude": [-8.0, -8.1, -8.2, -8.3, -8.4, 2.0],
            "Capacity (MW)": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "Start year": [2010, 2012, 2012, 2005, None, 2015],
            "Retired year": [None, None, None, 2016, None, None],
        }
    )


def test_fleet_for_compares_the_country_as_written_and_needs_coordinates():
    fleet = gwpt.fleet_for(tracker(), "IE", None)
    # B's trailing space excludes it; C has no latitude; D is retired.
    assert fleet["mw"].tolist() == [10.0, 50.0]
    assert list(fleet.columns) == ["lat", "lon", "mw"]


def test_fleet_for_as_of_a_year_keeps_projects_without_a_start_year():
    # E has no start year and is in every year; D (retired, 2005 to 2016) is
    # in both, as it stood then.
    assert gwpt.fleet_for(tracker(), "IE", 2011)["mw"].tolist() == [10.0, 40.0, 50.0]
    assert gwpt.fleet_for(tracker(), "IE", 2009)["mw"].tolist() == [40.0, 50.0]


def test_fleet_for_as_of_a_year_places_a_retired_project_by_its_years():
    """Known positive and negative: in the fleet before retirement, not from it."""
    assert 40.0 in gwpt.fleet_for(tracker(), "IE", 2015)["mw"].tolist()
    assert 40.0 not in gwpt.fleet_for(tracker(), "IE", 2016)["mw"].tolist()
    assert 40.0 not in gwpt.fleet_for(tracker(), "IE", 2004)["mw"].tolist()


def test_fleet_for_leaves_out_a_retired_project_it_cannot_place():
    frame = tracker()
    frame.loc[frame["Project Name"] == "D", "Retired year"] = None
    assert 40.0 not in gwpt.fleet_for(frame, "IE", 2011)["mw"].tolist()


def test_fleet_for_backfills_a_missing_start_year_and_never_replaces_one():
    # E is undated, so in 2011 without the backfill; with it, from 2019 only.
    dated = {"G5": 2019, "G1": 2030}
    assert gwpt.fleet_for(tracker(), "IE", 2011, start_years=dated)["mw"].tolist() == [10.0, 40.0]
    assert 50.0 in gwpt.fleet_for(tracker(), "IE", 2019, start_years=dated)["mw"].tolist()
    # A's tracker year, 2010, stands against the table's 2030.
    assert 10.0 in gwpt.fleet_for(tracker(), "IE", 2011, start_years=dated)["mw"].tolist()


def test_the_current_fleet_ignores_retired_projects_and_the_backfill():
    assert gwpt.fleet_for(tracker(), "IE", None, start_years={"G5": 2019})["mw"].tolist() == [
        10.0,
        50.0,
    ]


def test_load_start_years(tmp_path):
    assert gwpt.load_start_years(tmp_path / "absent.csv") == {}
    path = tmp_path / "years.csv"
    pd.DataFrame({"gem_phase_id": ["G5"], "start_year": [2019], "rule": ["name"]}).to_csv(
        path, index=False
    )
    assert gwpt.load_start_years(path) == {"G5": 2019}


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
