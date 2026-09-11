"""Units outside the loaded ERA5 extent are refused, unless a run opts in.

``xarray`` interpolation with ``fill_value=None`` extrapolates linearly past
the grid. The European ERA5 download stopped at 42N, and the Spanish, Italian
and Portuguese country grids were simulated from winds extrapolated up to five
degrees beyond it, with nothing recorded. These tests pin the refusal, the
opt-in and its record, and the load-time check that the data covers the
requested bbox.
"""
import json
import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import test_pipeline as tp
from test_harness_driver import make_spec, synthetic_dk  # noqa: F401  (fixture)
from vwf.datasets.era5 import prep_era5
from vwf.harness.driver import run_evaluate, run_train
from vwf.harness.regions import load_region
from vwf.sources import InMemoryCountrySource
from vwf.wind import (
    EXTRAPOLATION_ATTR,
    ExtrapolationError,
    interpolate_wind,
    loaded_extent_coverage,
)


def _grid(allow=None):
    times = pd.date_range("2016-01-01", periods=3, freq="D")
    lat, lon = np.array([55.0, 55.5, 56.0]), np.array([8.0, 8.75, 9.5])
    ds = xr.Dataset(
        {
            "wnd100m": (("time", "lat", "lon"), np.full((3, 3, 3), 8.0)),
            "roughness": (("time", "lat", "lon"), np.full((3, 3, 3), 0.05)),
        },
        coords={"time": times, "lat": lat, "lon": lon},
    )
    if allow is not None:
        ds.attrs[EXTRAPOLATION_ATTR] = allow
    return ds


def _units(lons, lats, capacity=(1.0, 3.0)):
    return pd.DataFrame({
        "ID": [f"u{i}" for i in range(len(lons))], "lon": lons, "lat": lats,
        "height": 100.0, "capacity": list(capacity), "model": "m",
    })


def test_coverage_counts_units_outside_the_loaded_extent():
    cov = loaded_extent_coverage(_grid(), _units([8.5, 10.0], [55.5, 55.25]))
    assert cov["loaded_extent"] == [8.0, 9.5, 55.0, 56.0]
    assert cov["units_outside_loaded_extent"] == 1
    assert cov["capacity_share_outside_loaded_extent"] == pytest.approx(0.75)
    assert cov["max_degrees_outside_loaded_extent"] == pytest.approx(0.5)
    assert cov["ids_outside"] == ["u1"]


def test_a_unit_on_the_edge_is_inside():
    cov = loaded_extent_coverage(_grid(), _units([8.0, 9.5], [55.0, 56.0]))
    assert cov["units_outside_loaded_extent"] == 0


def test_interpolation_refuses_a_unit_outside_by_default():
    with pytest.raises(ExtrapolationError) as err:
        interpolate_wind(_grid(), _units([8.5, 10.0], [55.5, 55.25]))
    message = str(err.value)
    assert "outside the loaded ERA5 extent" in message
    assert "does not verify the data in those cells" in message
    assert "75.0% of capacity" in message


def test_the_permission_travels_with_the_dataset():
    units = _units([8.5, 10.0], [55.5, 55.25])
    ws = interpolate_wind(_grid(allow=True), units)
    assert ws.sizes["turbine"] == 2
    # An explicit argument overrides the dataset's permission, both ways.
    interpolate_wind(_grid(allow=False), units, allow_extrapolation=True)
    with pytest.raises(ExtrapolationError):
        interpolate_wind(_grid(allow=True), units, allow_extrapolation=False)


def test_a_fleet_inside_the_extent_is_unchanged():
    units = _units([8.5, 9.0], [55.5, 55.75])
    a = interpolate_wind(_grid(), units)
    b = interpolate_wind(_grid(allow=True), units)
    np.testing.assert_array_equal(a.values, b.values)


def test_prep_era5_warns_when_the_data_stops_short_of_the_bbox(tmp_path):
    tp._write_era5(tmp_path / "era5")
    with pytest.warns(UserWarning, match="stops short of the requested bbox"):
        ds = prep_era5("ZZ", False, True, bbox=(7.0, 9.5, 55.0, 56.0), era5_dir=tmp_path / "era5")
    assert ds.attrs[EXTRAPOLATION_ATTR] is False
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        warnings.filterwarnings("ignore", message=".*BUNDLED.*")
        ds = prep_era5("ZZ", False, True, bbox=(8.0, 9.5, 55.0, 56.0),
                       era5_dir=tmp_path / "era5", allow_extrapolation=True)
    assert ds.attrs[EXTRAPOLATION_ATTR] is True


def test_region_config_parses_the_opt_in(tmp_path):
    base = open("configs/regions/scorecard/fr_country.toml").read()
    on = base.replace("[era5]\n", "[era5]\nallow_extrapolation = true\n", 1)
    bad = base.replace("[era5]\n", "[era5]\nallow_extrapolation = \"yes\"\n", 1)
    (tmp_path / "on.toml").write_text(on)
    (tmp_path / "bad.toml").write_text(bad)
    assert load_region("configs/regions/scorecard/fr_country.toml").allow_extrapolation is False
    assert load_region(tmp_path / "on.toml").allow_extrapolation is True
    with pytest.raises(ValueError, match="allow_extrapolation"):
        load_region(tmp_path / "bad.toml")


def _country_run(synthetic_dk, spec):  # noqa: F811
    grid = pd.DataFrame({
        "ID": ["g0", "g1", "g2", "g3"],
        # g3 lies half a degree east of the synthetic grid's 9.5E edge.
        "lon": [8.1, 8.3, 9.2, 10.0], "lat": [55.2, 55.4, 55.6, 55.8],
        "height": [100.0] * 4, "capacity": [2000.0, 2000.0, 4000.0, 4000.0],
        "model": ["2019COE_Market_Average_2.6MW_121"] * 4,
        "cluster": [0, 0, 1, 1], "type": ["onshore"] * 4,
    })
    train_idx = pd.date_range("2015-01-01", "2015-12-31 23:00", freq="h", tz="UTC")
    test_idx = pd.date_range("2016-01-01", "2016-12-31 23:00", freq="h", tz="UTC")
    out = synthetic_dk["root"] / "extent"
    train_dir = run_train(spec, out, source=InMemoryCountrySource(
        grid, pd.DataFrame({"capacity_factor": 0.2}, index=train_idx)), run_name="t")
    eval_dir = run_evaluate(spec, train_dir, out, source=InMemoryCountrySource(
        grid, pd.DataFrame({"capacity_factor": 0.2}, index=test_idx)), run_name="e")
    return train_dir, eval_dir


def test_a_harness_run_outside_the_extent_is_refused(synthetic_dk):  # noqa: F811
    spec = make_spec(source="in-memory-country", obs_level="country", obs_unit="country")
    with pytest.raises(ExtrapolationError):
        _country_run(synthetic_dk, spec)


def test_an_opted_in_run_records_the_extrapolated_share(synthetic_dk):  # noqa: F811
    spec = make_spec(source="in-memory-country", obs_level="country", obs_unit="country",
                     allow_extrapolation=True)
    with pytest.warns(UserWarning, match="simulated from extrapolated winds"):
        train_dir, eval_dir = _country_run(synthetic_dk, spec)
    for run in (train_dir, eval_dir):
        record = json.loads((run / "run_manifest.json").read_text())["era5_extent"]
        assert record["units_outside_loaded_extent"] == 1
        assert record["capacity_share_outside_loaded_extent"] == pytest.approx(4000 / 12000)
        assert record["allow_extrapolation"] is True
        assert "does not verify" in record["meaning"]
    metrics = pd.read_csv(eval_dir / "metrics.csv")
    assert np.allclose(metrics["extrapolated_capacity_share"], 4000 / 12000)


def test_a_run_inside_the_extent_records_zero(synthetic_dk):  # noqa: F811
    spec = make_spec()
    out = synthetic_dk["root"] / "inside"
    train_dir = run_train(spec, out, mode="onshore", run_name="t")
    eval_dir = run_evaluate(spec, train_dir, out, mode="onshore", run_name="e")
    record = json.loads((eval_dir / "run_manifest.json").read_text())["era5_extent"]
    assert record["units_outside_loaded_extent"] == 0
    assert record["allow_extrapolation"] is False
    assert (pd.read_csv(eval_dir / "metrics.csv")["extrapolated_capacity_share"] == 0).all()
