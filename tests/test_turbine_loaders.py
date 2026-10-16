"""The DK, DE and UK turbine loaders (pyvwf.loaders.turbine_loaders), on synthetic files.

The ``european-turbine`` adapter reads every DK, DE and UK fleet and
observation through these functions, but no test imported them (#86). Each
case writes the on-disk layout the loader reads into a temporary input root.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pyvwf.config import PyVWFPaths
from pyvwf.loaders import turbine_loaders as tl


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(PyVWFPaths, "TURBINE_DATA", tmp_path)
    for code in ("DK", "DE", "UK"):
        (tmp_path / code).mkdir()
    return tmp_path


def test_dk_metadata_maps_location_type_and_cleans_the_fleet(root):
    pd.DataFrame(
        {
            "ID": [1, 2, 3, 4],
            "manufacturer": ["Vestas Wind Systems A/S", "Siemens Gamesa", "Vestas", "Nordex"],
            "capacity": [2000, 3600, 660, 2500],
            "diameter": [80, 120, 47, None],
            "height": [80, 90, 0.5, 100],
            "lon": [8.1, 7.0, 9.0, 10.0],
            "lat": [56.0, 55.5, 56.5, 57.0],
            "location_type": ["Land", "Hav", "Land", "Land"],
        }
    ).to_csv(root / "DK/dk_md.csv", index=False)
    md = tl.load_turbine_metadata("dk")
    # 3 has a 0.5 m hub height and 4 no rotor diameter: both are dropped.
    assert md["ID"].tolist() == ["1", "2"]
    assert md["type"].tolist() == ["onshore", "offshore"]
    assert md["manufacturer"].tolist() == ["Vestas", "Siemens"]


def test_dk_observations_pivot_to_months_within_the_years_asked(root):
    rows = [
        {"ID": 1, "year": y, "month": m, "generation_kwh": 100.0 * m}
        for y in (2014, 2015, 2016)
        for m in range(1, 13)
    ]
    pd.DataFrame(rows).to_csv(root / "DK/dk_obs_2002_2020.csv", index=False)
    obs = tl.load_turbine_observations("DK", 2015, 2016)
    assert sorted(obs["year"]) == [2015, 2016]
    assert [str(m) for m in range(1, 13)] == [c for c in obs.columns if c not in ("ID", "year")]
    assert obs.loc[obs["year"] == 2015, "3"].item() == 300.0
    assert obs["ID"].tolist() == ["1", "1"]


def test_de_metadata_takes_coordinates_from_the_postcode_in_the_id(root):
    pd.DataFrame(
        {
            "V1": ["2569112345", "8000099999"],
            "Manufacturer": ["Enercon", "Vestas"],
            "kW": [2000, 3000],
            "Rotor..m.": [82, 112],
            "Tower..m.": [98, 119],
        }
    ).to_csv(root / "DE/DE_md.csv", index=False)
    pd.DataFrame({"postcode": [25691], "lon": [9.1], "lat": [54.1]}).to_csv(
        root / "DE/geolocate.germany.csv", index=False
    )
    md = tl.load_turbine_metadata("DE")
    # The second unit's postcode, 80000, has no coordinate: dropped.
    assert md["ID"].tolist() == ["2569112345"]
    assert md[["lon", "lat"]].iloc[0].tolist() == [9.1, 54.1]
    assert md["type"].tolist() == ["onshore"]


def test_de_observations_drop_downtime_and_pivot(root):
    pd.DataFrame(
        {
            "ID": ["a", "a", "a"],
            "Year": [2014, 2015, 2015],
            "Month": [1, 1, 2],
            "Output": [5.0, 10.0, 20.0],
            "Downtime": [0, 1, 2],
        }
    ).to_csv(root / "DE/DE_data.csv", index=False)
    obs = tl.load_turbine_observations("DE", 2015, 2015)
    assert obs["year"].tolist() == [2015]
    assert obs[[1, 2]].iloc[0].tolist() == [10.0, 20.0]
    assert "Downtime" not in obs.columns


def test_uk_metadata_and_observations_pass_through_with_the_standard_checks(root):
    pd.DataFrame(
        {
            "ID": ["s1", "s2"],
            "capacity": [1000, 2000],
            "diameter": [60, 90],
            "height": [70, None],
            "lon": [-3.0, -2.0],
            "lat": [55.0, 56.0],
        }
    ).to_csv(root / "UK/uk_md.csv", index=False)
    md = tl.load_turbine_metadata("UK")
    assert md["ID"].tolist() == ["s1"]  # s2 has no hub height
    assert md["type"].tolist() == ["onshore"]
    pd.DataFrame({"ID": ["s1", "s1"], "year": [2018, 2019], "1": [1.0, 2.0]}).to_csv(
        root / "UK/ukobs.csv", index=False
    )
    assert tl.load_turbine_observations("UK", 2019, 2019)["year"].tolist() == [2019]


def test_an_unsupported_region_and_a_fleet_with_no_hub_heights_are_refused(root):
    with pytest.raises(ValueError, match="Unsupported"):
        tl.load_turbine_metadata("FR")
    with pytest.raises(ValueError, match="No turbine-level observation loader"):
        tl.load_turbine_observations("FR", 2015, 2016)
    with pytest.raises(ValueError, match="must contain 'ID'"):
        tl._standardise_turb_info_minimal(pd.DataFrame({"height": [80.0]}))
    with pytest.raises(ValueError, match="valid hub height"):
        tl._standardise_turb_info_minimal(
            pd.DataFrame({"ID": [1], "height": [0.0], "capacity": [1.0], "lon": [0], "lat": [0]})
        )
