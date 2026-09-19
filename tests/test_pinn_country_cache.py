"""Country-level caches for the physics-informed model.

A country-level region has one national series per month and a fleet of grid
points. The cache has to turn the observations into the same monthly target
the harness fits to, put grid capacities in the turbine tier's unit, and give
every year of the split its own capacities, refusing any grid that is not that
year's fleet. These tests pin those four things. Nothing here imports torch, so
this file runs in CI.
"""

import numpy as np
import pandas as pd
import pytest

from test_harness_driver import make_spec
from vwf.config import PyVWFPaths
from vwf.pinn.cache import (
    MW_TO_KW,
    NATIONAL_ID,
    RegionCache,
    _country_observations,
    load_cache,
    save_cache,
)

IDS = ["grid_0001", "grid_0002", "grid_0003"]


def _grid(path, capacities_mw, *, ids=IDS, shuffle=False):
    grid = pd.DataFrame(
        {
            "lat": [50.0, 50.5, 51.0],
            "lon": [4.0, 4.5, 5.0],
            "weight": capacities_mw,
            "ID": ids,
            "height": 100.0,
            "model": "Vestas.V90.3000",
            "capacity": capacities_mw,
            "type": "onshore",
            "cluster": [0, 0, 1],
        }
    )
    if shuffle:
        grid = grid.iloc[::-1]
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(path, index=False)


def _observations(path, start, end, capacity_mw):
    """Hourly national observations whose capacity steps up mid-month."""
    index = pd.date_range(start, end, freq="h", tz="UTC")
    capacity = np.where(index.day < 16, capacity_mw, 2 * capacity_mw)
    generation = 0.25 * capacity + np.where(index.hour < 12, 10.0, 0.0)
    frame = pd.DataFrame(
        {
            "generation_mw": generation,
            "capacity_mw": capacity,
            "capacity_factor": generation / capacity,
        },
        index=index,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path)
    return frame


@pytest.fixture
def country_root(tmp_path, monkeypatch):
    base = tmp_path / "observations" / "country"
    monkeypatch.setattr(PyVWFPaths, "COUNTRY_LEVEL_DATA", base)
    grids = base / "grid_points" / "zz"
    _grid(grids / "zz_grid_points_2015.csv", [1.0, 0.0, 2.0])
    _grid(grids / "zz_grid_points_2016.csv", [1.5, 0.5, 2.0], shuffle=True)
    _grid(grids / "zz_grid_points_2023.csv", [3.0, 1.0, 4.0])
    obs = base / "observations" / "zz"
    train = _observations(obs / "zz_train_2015_2016.csv", "2015-01-01", "2017-01-31 23:00", 100.0)
    _observations(obs / "zz_test_2023.csv", "2023-01-01", "2023-12-31 23:00", 300.0)
    spec = make_spec(
        code="ZZ",
        source="entsoe-country",
        obs_level="country",
        obs_unit="country",
        train_years=(2015, 2016),
        test_years=(2023,),
    )
    return spec, base, train


def test_the_target_is_energy_weighted_and_limited_to_the_split(country_root):
    spec, _, train = country_root
    grid, obs, years, capacity, record = _country_observations(spec, "train")
    assert set(obs["ID"]) == {NATIONAL_ID}
    # The observation file runs into January 2017, outside the window.
    assert sorted(obs["year"].unique()) == [2015, 2016]
    assert len(obs) == 24
    january = train.loc["2015-01"]
    energy = january["generation_mw"].sum() / january["capacity_mw"].sum()
    mean_of_ratios = january["capacity_factor"].mean()
    value = float(obs.query("year == 2015 and month == 1")["obs"].iloc[0])
    assert value == pytest.approx(energy, rel=1e-9)
    assert value != pytest.approx(mean_of_ratios, rel=1e-6)
    assert record["observation_months"] == 24


def test_capacities_are_in_kw_and_aligned_by_id_in_every_year(country_root):
    spec, _, _ = country_root
    grid, _, years, capacity, record = _country_observations(spec, "train")
    assert list(years) == [2015, 2016]
    # The split's fleet file is the train-end year, 2016.
    assert grid.set_index("ID").loc[IDS, "capacity"].tolist() == [1500.0, 500.0, 2000.0]
    # 2016's file is stored in reverse order; the rows still follow the fleet's IDs.
    order = grid["ID"].tolist()
    expected = {2015: dict(zip(IDS, [1.0, 0.0, 2.0])), 2016: dict(zip(IDS, [1.5, 0.5, 2.0]))}
    for k, year in enumerate(years):
        assert capacity[k].tolist() == [expected[int(year)][i] * MW_TO_KW for i in order]
    assert record["capacity_mw_by_year"] == {"2015": 3.0, "2016": 4.0}
    assert record["points_with_capacity_by_year"] == {"2015": 2, "2016": 3}


def test_the_test_split_uses_its_own_year(country_root):
    spec, _, _ = country_root
    grid, obs, years, capacity, record = _country_observations(spec, "test")
    assert list(years) == [2023]
    assert capacity.sum() == pytest.approx(8.0 * MW_TO_KW)
    assert sorted(obs["month"]) == list(range(1, 13))
    assert record["fleet_year"] == 2023


def test_a_static_grid_is_refused_for_a_missing_year(country_root):
    """The static fallback is a uniform lattice for SE and NO, not a fleet."""
    spec, base, _ = country_root
    grids = base / "grid_points" / "zz"
    (grids / "zz_grid_points_2015.csv").unlink()
    _grid(grids / "zz_grid_points.csv", [9.0, 9.0, 9.0])
    with pytest.raises(FileNotFoundError, match="static grid"):
        _country_observations(spec, "train")


def test_a_year_with_different_points_is_refused(country_root):
    spec, base, _ = country_root
    _grid(
        base / "grid_points" / "zz" / "zz_grid_points_2015.csv",
        [1.0, 1.0, 1.0],
        ids=["grid_0001", "grid_0002", "grid_9999"],
    )
    with pytest.raises(ValueError, match="different grid points"):
        _country_observations(spec, "train")


def test_a_country_cache_round_trips_and_an_old_cache_loads_as_turbine(tmp_path):
    days = pd.date_range("2015-01-01", periods=31, freq="D")
    w = np.full((31, 2), 7.0, dtype="float32")
    base = dict(
        dates=days,
        meta=pd.DataFrame({"ID": ["a", "b"]}),
        obs=pd.DataFrame({"ID": [NATIONAL_ID], "year": [2015], "month": [1], "obs": [0.3]}),
        w_mean=w,
        w_std=w,
        z0=w,
        shear=w,
        curve_speeds=np.arange(3.0),
        curve_cf=np.zeros((1, 3)),
        curve_names=["M"],
        turbine_curve=np.zeros(2, dtype="int64"),
    )
    cache = RegionCache(
        code="ZZ",
        split="train",
        level="country",
        capacity_years=np.array([2015]),
        capacity_by_year=np.array([[1000.0, 2000.0]]),
        fleet_record={"level": "country"},
        **base,
    )
    save_cache(cache, tmp_path)
    back = load_cache("ZZ", "train", tmp_path)
    assert back.level == "country"
    assert back.capacity_years.tolist() == [2015]
    assert back.capacity_by_year.tolist() == [[1000.0, 2000.0]]
    assert back.fleet_record == {"level": "country"}

    # A cache written before the country tier has none of those arrays.
    save_cache(RegionCache(code="YY", split="train", **base), tmp_path)
    z = dict(np.load(tmp_path / "YY_train" / "fields.npz", allow_pickle=True))
    for key in ("level", "capacity_years", "capacity_by_year"):
        z.pop(key)
    np.savez_compressed(tmp_path / "YY_train" / "fields.npz", **z)
    (tmp_path / "YY_train" / "fleet_record.json").unlink()
    old = load_cache("YY", "train", tmp_path)
    assert old.level == "turbine"
    assert old.capacity_by_year.size == 0
    assert old.fleet_record == {}
