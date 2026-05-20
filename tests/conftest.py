"""Shared pytest fixtures for PyVWF.

All fixtures build small *synthetic* datasets so the test suite runs without
ERA5 reanalysis files, the ENTSO-E API, or any large input data. The synthetic
reanalysis follows the same schema the production loaders produce: an
``xarray.Dataset`` with ``time``/``lat``/``lon`` coordinates and ``wnd100m``
(100 m wind speed) plus ``roughness`` (surface roughness z0) data variables.
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr


@pytest.fixture
def grid():
    """A small regular lat/lon grid covering Denmark-ish coordinates."""
    return {
        "lats": np.array([55.0, 55.5, 56.0]),
        "lons": np.array([8.0, 8.5, 9.0]),
    }


def _make_reanalysis(times, lats, lons, mean_speed=8.0, z0=0.03, seed=0):
    rng = np.random.default_rng(seed)
    shape = (len(times), len(lats), len(lons))
    wnd = mean_speed + rng.normal(0.0, 1.0, size=shape)
    wnd = np.clip(wnd, 0.0, None)
    rough = np.full(shape, z0)
    return xr.Dataset(
        {
            "wnd100m": (("time", "lat", "lon"), wnd),
            "roughness": (("time", "lat", "lon"), rough),
        },
        coords={"time": times, "lat": lats, "lon": lons},
    )


@pytest.fixture
def reanalysis(grid):
    """Two days of hourly synthetic reanalysis on a 3x3 grid."""
    times = pd.date_range("2020-01-01", periods=48, freq="h")
    return _make_reanalysis(times, grid["lats"], grid["lons"], seed=0)


@pytest.fixture
def make_reanalysis(grid):
    """Factory so individual tests can request custom-length reanalysis."""
    def _factory(n_hours=48, start="2020-01-01", mean_speed=8.0, z0=0.03, seed=0):
        times = pd.date_range(start, periods=n_hours, freq="h")
        return _make_reanalysis(
            times, grid["lats"], grid["lons"],
            mean_speed=mean_speed, z0=z0, seed=seed,
        )
    return _factory


@pytest.fixture
def turbines():
    """Two identical turbines at hub height 100 m inside the grid."""
    return pd.DataFrame(
        {
            "ID": ["t1", "t2"],
            "lat": [55.2, 55.8],
            "lon": [8.2, 8.8],
            "height": [100.0, 100.0],
            "model": ["GE.1.5sle", "GE.1.5sle"],
            "capacity": [1500.0, 1500.0],
        }
    )


@pytest.fixture
def power_curve():
    """A simple synthetic, monotonic power curve as a single-model table.

    Columns match the production format: ``data$speed`` plus one model column
    holding capacity factor (0-1) versus wind speed.
    """
    speed = np.arange(0.0, 40.01, 0.5)
    cut_in, rated, cut_out = 3.0, 13.0, 25.0
    cf = np.zeros_like(speed)
    ramp = (speed >= cut_in) & (speed < rated)
    cf[ramp] = ((speed[ramp] - cut_in) / (rated - cut_in)) ** 3
    cf[(speed >= rated) & (speed <= cut_out)] = 1.0
    cf[speed > cut_out] = 0.0
    return pd.DataFrame({"data$speed": speed, "GE.1.5sle": cf})
