"""Tests for wind interpolation, height extrapolation, and power conversion."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from vwf.wind import (
    interpolate_wind,
    simulate_wind,
    train_simulate_wind,
    aggregate_turbines_to_grid,
    prepare_offset_arrays,
    fast_simulate_cf,
)


def test_interpolate_wind_shape(reanalysis, turbines):
    ws = interpolate_wind(reanalysis, turbines)
    assert ws.dims == ("time", "turbine")
    assert ws.sizes["time"] == reanalysis.sizes["time"]
    assert ws.sizes["turbine"] == len(turbines)
    assert "model" in ws.coords
    assert "capacity" in ws.coords


def test_height_log_law_identity_at_100m(reanalysis, turbines):
    """At hub height 100 m the log-law factor is 1, so interpolated speed
    equals the spatially interpolated 100 m field."""
    ws = interpolate_wind(reanalysis, turbines)
    # Independently interpolate the raw 100 m field to the same points.
    lat = xr.DataArray(turbines["lat"].values, dims="turbine")
    lon = xr.DataArray(turbines["lon"].values, dims="turbine")
    expected = reanalysis["wnd100m"].interp(lon=lon, lat=lat)
    np.testing.assert_allclose(ws.values, expected.values, rtol=1e-6)


def test_height_log_law_scaling(make_reanalysis):
    """Below 100 m, wind speed scales by ln(h/z0)/ln(100/z0) < 1."""
    z0 = 0.03
    ds = make_reanalysis(n_hours=6, z0=z0, mean_speed=10.0, seed=1)
    turb = pd.DataFrame({
        "ID": ["a", "b"],
        "lat": [55.5, 55.5], "lon": [8.5, 8.5],
        "height": [100.0, 80.0],
        "model": ["GE.1.5sle", "GE.1.5sle"],
        "capacity": [1.0, 1.0],
    })
    ws = interpolate_wind(ds, turb)
    factor = np.log(80.0 / z0) / np.log(100.0 / z0)
    ratio = (ws.isel(turbine=1) / ws.isel(turbine=0)).values
    np.testing.assert_allclose(ratio, factor, rtol=1e-6)
    assert factor < 1.0


def test_simulate_wind_cf_bounds(reanalysis, turbines, power_curve):
    _, cf = simulate_wind(reanalysis, turbines, power_curve)
    vals = cf[["t1", "t2"]].to_numpy()
    assert np.all(vals >= -0.05)
    assert np.all(vals <= 1.05)


def test_power_curve_extremes(make_reanalysis, power_curve):
    """Zero wind -> ~0 CF; rated wind -> ~1 CF."""
    calm = make_reanalysis(n_hours=4, mean_speed=0.0, seed=2)
    calm["wnd100m"].values[:] = 0.0
    windy = make_reanalysis(n_hours=4, mean_speed=18.0, seed=3)
    windy["wnd100m"].values[:] = 18.0  # within rated plateau (13-25 m/s)
    turb = pd.DataFrame({
        "ID": ["a"], "lat": [55.5], "lon": [8.5], "height": [100.0],
        "model": ["GE.1.5sle"], "capacity": [1.0],
    })
    cf_calm = train_simulate_wind(calm, turb, power_curve, 1.0, 0.0)
    cf_windy = train_simulate_wind(windy, turb, power_curve, 1.0, 0.0)
    assert cf_calm == pytest.approx(0.0, abs=1e-6)
    assert cf_windy == pytest.approx(1.0, abs=1e-3)


def test_offset_increases_cf_monotonically(reanalysis, turbines, power_curve):
    ws = interpolate_wind(reanalysis, turbines)
    arrays = prepare_offset_arrays(ws, power_curve)
    cfs = [fast_simulate_cf(arrays, 1.0, off) for off in (-3.0, -1.0, 0.0, 1.0, 3.0)]
    assert all(b >= a for a, b in zip(cfs, cfs[1:]))


def test_fast_simulate_cf_matches_train_simulate(reanalysis, turbines, power_curve):
    """The numpy fast path must agree with the xarray reference path."""
    ws = interpolate_wind(reanalysis, turbines)
    arrays = prepare_offset_arrays(ws, power_curve)
    fast = fast_simulate_cf(arrays, 0.9, 0.5)
    ref = train_simulate_wind(reanalysis, turbines, power_curve, 0.9, 0.5)
    assert fast == pytest.approx(ref, rel=1e-6)


def test_aggregate_turbines_to_grid_conserves_capacity(reanalysis):
    turb = pd.DataFrame({
        "ID": [f"t{i}" for i in range(6)],
        "lat": [55.05, 55.06, 55.5, 55.51, 56.0, 56.0],
        "lon": [8.05, 8.06, 8.5, 8.51, 9.0, 9.0],
        "height": [100.0] * 6,
        "model": ["GE.1.5sle"] * 6,
        "capacity": [100.0, 200.0, 50.0, 50.0, 10.0, 10.0],
    })
    agg = aggregate_turbines_to_grid(turb, reanalysis)
    assert agg["capacity"].sum() == pytest.approx(turb["capacity"].sum())
    assert len(agg) <= len(turb)


def test_aggregate_turbines_raises_on_empty(reanalysis):
    turb = pd.DataFrame({
        "ID": ["x"], "lat": [np.nan], "lon": [np.nan],
        "height": [np.nan], "model": ["GE.1.5sle"], "capacity": [np.nan],
    })
    with pytest.raises(ValueError):
        aggregate_turbines_to_grid(turb, reanalysis)
