"""The start-year backfill's matching rules, on synthetic registers.

``scripts/region_tools/backfill_gwpt_start_years.py`` dates tracker records
from NVE (Norway) and ODRE (France). The registers are local, so these cases
check each rule where CI can run it: that it dates a record it should, and
refuses one it should not.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/region_tools/backfill_gwpt_start_years.py"
    spec = importlib.util.spec_from_file_location("backfill_gwpt_start_years", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def turbine(date, n, kw, out=None):
    return {
        "DatoIdriftsatt": date,
        "DatoUtavdrift": out,
        "AntallTurbiner": n,
        "TurbinStorrelse_kW": kw,
    }


def test_norm_folds_nordic_and_french_letters():
    m = _module()
    assert m.norm("Tysvær") == m.norm("Tysvaer wind farm".replace(" wind farm", ""))
    assert m.norm("Svåheia") == "SVAHEIA"
    assert m.norm("Sainte-Valière") == "STE VALIERE"


def test_a_repowered_plant_takes_its_new_turbines_year():
    """The Havoygavlen shape: one old turbine kept, the old phase retired, new turbines."""
    m = _module()
    turbines = [
        turbine("2011-01-01", 1, 3000),
        turbine("2002-10-15", 15, 2500, out="2021-06-18"),
        turbine("2021-11-11", 9, 4200),
    ]
    assert m.nve_start_year(turbines) == 2021
    assert m.nve_start_year([turbine("2002-10-15", 15, 2500, out="2021-06-18")]) is None


def tracker(**overrides) -> pd.DataFrame:
    row = {
        "GEM phase ID": "G1",
        "Project Name": "Tysvaer wind farm",
        "Phase Name": "--",
        "Capacity (MW)": 47.0,
        "Latitude": 59.30,
        "Longitude": 5.56,
        "City": None,
        "Country/Area": "Norway",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_norway_matches_by_name_and_refuses_a_capacity_mismatch():
    m = _module()
    nve = pd.DataFrame(
        {
            "VindkraftAnleggId": [1089],
            "Navn": ["Tysvær"],
            "Turbiner": [[turbine("2021-10-21", 11, 4300)]],
        }
    )
    assert m.date_norway(tracker(), nve)["year"].tolist() == [2021]
    assert m.date_norway(tracker(), nve)["ids"].tolist() == ["nve:1089"]
    # The same plant at twice the capacity is a different project.
    assert m.date_norway(tracker(**{"Capacity (MW)": 94.0}), nve)["year"].isna().all()
    assert m.date_norway(tracker(**{"Project Name": "Other wind farm"}), nve)["year"].isna().all()


def odre(rows) -> pd.DataFrame:
    frame = pd.DataFrame(
        rows, columns=["commune", "codeinseecommune", "mw", "year", "lat", "lon", "nominstallation"]
    )
    frame["commune_n"] = frame["commune"].map(_module().norm)
    frame["name_n"] = frame["nominstallation"].map(_module().norm)
    frame["rid"] = [f"r{i}" for i in range(len(frame))]
    return frame


def fr_tracker(mw=17.6):
    return tracker(
        **{
            "Project Name": "Lunaires wind farm",
            "Capacity (MW)": mw,
            "Latitude": 49.0,
            "Longitude": 2.0,
            "Country/Area": "France",
        }
    )


def test_france_sums_one_communes_records_and_needs_one_year():
    m = _module()
    one_year = odre(
        [
            ("Lunaires", "10001", 8.8, 2020, 49.0, 2.01, "Confidentiel"),
            ("Lunaires", "10001", 8.8, 2020, 49.0, 2.01, "Confidentiel"),
        ]
    )
    got = m.date_france(fr_tracker(), one_year)
    assert got["year"].tolist() == [2020]
    assert "commune_sum" in got["rule"].iloc[0]
    two_years = one_year.assign(year=[2019, 2023])
    assert m.date_france(fr_tracker(), two_years)["year"].isna().all()


def test_france_refuses_rules_that_disagree_by_more_than_a_year():
    m = _module()
    # The name rule finds the 2008 record at the commune; the one record at
    # 5% capacity within 5 km is a different, 2019 farm.
    reg = odre(
        [
            ("Lunaires", "10001", 30.0, 2008, np.nan, np.nan, "Confidentiel"),
            ("Autre", "10002", 17.6, 2019, 49.0, 2.02, "Confidentiel"),
        ]
    )
    reg.loc[0, "mw"] = 17.6
    got = m.date_france(fr_tracker(), reg)
    assert got["name"].tolist() == [2008] and got["geo"].tolist() == [2019]
    assert got["year"].isna().all()
