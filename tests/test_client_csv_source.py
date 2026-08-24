"""Tests for the client CSV observation adapter (the delivery seam).

These exercise the column-mapping and CF-conversion contract with tiny in-repo
fixtures; they do not touch ERA5, so they run fast and offline.
"""
from __future__ import annotations

import pandas as pd
import pytest

from vwf.sources import available_sources
from vwf.sources.client_csv import ClientCsvTurbineSource


def _write_fleet(tmp_path):
    meta = pd.DataFrame({
        "site_id": ["A", "B"],
        "longitude": [9.0, 9.5],
        "latitude": [56.0, 56.2],
        "rated_mw": [3.0, 2.0],          # MW
        "hub_m": [100.0, 90.0],
        "rotor_m": [112.0, 90.0],
    })
    gen = pd.DataFrame([
        {"ID": "A", "year": 2020, "month": 1, "generation_mwh": 0.30 * 31 * 24 * 3.0},
        {"ID": "B", "year": 2020, "month": 1, "generation_mwh": 0.20 * 31 * 24 * 2.0},
    ])
    mp = tmp_path / "meta.csv"
    gp = tmp_path / "gen.csv"
    meta.to_csv(mp, index=False)
    gen.to_csv(gp, index=False)
    return mp, gp


COLUMN_MAP = {
    "ID": "site_id", "lon": "longitude", "lat": "latitude",
    "capacity": "rated_mw", "height": "hub_m", "diameter": "rotor_m",
}


def _source(tmp_path, **kw):
    mp, gp = _write_fleet(tmp_path)
    return ClientCsvTurbineSource(
        mp, gp, country="DK", column_map=COLUMN_MAP,
        capacity_unit="mw", generation_column="generation_mwh", **kw,
    )


def test_registered():
    assert "client-csv-turbine" in available_sources()


def test_metadata_standardised_and_curve_matched(tmp_path):
    md = _source(tmp_path).load_metadata()
    for col in ("ID", "lon", "lat", "capacity", "height", "model", "type"):
        assert col in md.columns
    assert (md["capacity"] == [3000.0, 2000.0]).all()  # MW -> kW
    assert md["model"].notna().all()                    # curve matched by spec power
    assert (md["type"] == "onshore").all()              # default


def test_energy_to_capacity_factor(tmp_path):
    obs = _source(tmp_path).load_observations(2020, 2020)
    assert {"ID", "year", "obs_1"}.issubset(obs.columns)
    by_id = obs.set_index("ID")["obs_1"]
    assert by_id["A"] == pytest.approx(0.30, abs=1e-6)
    assert by_id["B"] == pytest.approx(0.20, abs=1e-6)


def test_generation_already_cf(tmp_path):
    mp, gp = _write_fleet(tmp_path)
    pd.DataFrame([
        {"ID": "A", "year": 2020, "month": 1, "cf": 0.42},
    ]).to_csv(gp, index=False)
    src = ClientCsvTurbineSource(
        mp, gp, country="DK", column_map=COLUMN_MAP, capacity_unit="mw",
        generation_column="cf", generation_is_cf=True,
    )
    obs = src.load_observations()
    assert obs.set_index("ID")["obs_1"]["A"] == pytest.approx(0.42)


def test_missing_required_metadata_column_raises(tmp_path):
    mp, gp = _write_fleet(tmp_path)
    pd.read_csv(mp).drop(columns=["hub_m"]).to_csv(mp, index=False)  # drop hub height
    src = ClientCsvTurbineSource(
        mp, gp, country="DK", column_map=COLUMN_MAP, capacity_unit="mw",
        generation_column="generation_mwh",
    )
    with pytest.raises(ValueError, match="height"):
        src.load_metadata()


def test_year_filtering(tmp_path):
    mp, gp = _write_fleet(tmp_path)
    pd.DataFrame([
        {"ID": "A", "year": 2019, "month": 1, "generation_mwh": 100.0},
        {"ID": "A", "year": 2020, "month": 1, "generation_mwh": 200.0},
    ]).to_csv(gp, index=False)
    src = ClientCsvTurbineSource(
        mp, gp, country="DK", column_map=COLUMN_MAP, capacity_unit="mw",
        generation_column="generation_mwh",
    )
    obs = src.load_observations(2020, 2020)
    assert (obs["year"] == 2020).all()
