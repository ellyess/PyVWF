"""The physics-informed path's ERA5 reduction: roughness treatment and extent.

``pyvwf.pinn`` reduces hourly ERA5 to daily statistics itself, rather than through
``prep_era5``, because it keeps the within-day spread. That makes it a second
implementation of two decisions the harness records in every manifest: which
temporal treatment of the roughness a run applies, and where the fleet lies
against the loaded extent. These tests pin that the second implementation
follows the region's setting the way ``prep_era5`` does, agrees with it
numerically, and records both decisions in the harness's own format.

Nothing here imports torch, so this file runs in CI, unlike
``test_pinn_physics.py``.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pyvwf.era5 import prep_era5
from pyvwf.pinn.cache import era5_record_for
from pyvwf.pinn.era5_stats import daily_stats_at_points

LAT = np.array([55.0, 55.25, 55.5])
LON = np.array([8.0, 8.25, 8.5])


def _hourly_file(path, *, with_z0=True, with_10m=True, seed=0):
    """Two days of hourly winds on a 3 x 3 grid, with 100 m faster than 10 m."""
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    times = pd.date_range("2015-01-01", periods=48, freq="h")
    shape = (len(times), len(LAT), len(LON))
    u100 = rng.uniform(6.0, 12.0, shape)
    data = {
        "u100": (("time", "lat", "lon"), u100),
        "v100": (("time", "lat", "lon"), np.zeros(shape)),
    }
    if with_10m:
        # A 10 m speed between 60% and 85% of the 100 m one, so the log-profile
        # inversion is defined everywhere and varies from hour to hour.
        data["u10"] = (("time", "lat", "lon"), u100 * rng.uniform(0.6, 0.85, shape))
        data["v10"] = (("time", "lat", "lon"), np.zeros(shape))
    if with_z0:
        data["z0"] = (("lat", "lon"), np.full((len(LAT), len(LON)), 0.25))
    xr.Dataset(data, coords={"time": times, "lat": LAT, "lon": LON}).to_netcdf(
        path / "era5_2015_01.nc"
    )
    return path


def _reduce(era5_dir, roughness, lon=(8.25,), lat=(55.25,)):
    return daily_stats_at_points(
        era5_dir, None, np.array(lon), np.array(lat), [2015], roughness=roughness
    )


def test_stored_uses_the_field_the_file_carries(tmp_path):
    *_, z0, _, record = _reduce(_hourly_file(tmp_path / "e"), "stored")
    assert record["roughness"] == {"requested": "stored", "applied": "stored"}
    assert np.allclose(z0, 0.25)


def test_derived_ignores_the_stored_field(tmp_path):
    *_, z0, _, record = _reduce(_hourly_file(tmp_path / "e"), "derived")
    assert record["roughness"] == {"requested": "derived", "applied": "derived"}
    assert not np.allclose(z0, 0.25)


def test_a_file_with_no_stored_field_is_derived_and_says_so(tmp_path):
    """The case the US and Brazilian hourly files are in: a stored field is
    requested by default and none exists, so the record must show both."""
    *_, record = _reduce(_hourly_file(tmp_path / "e", with_z0=False), "stored")
    assert record["roughness"] == {"requested": "stored", "applied": "derived"}


def test_derived_without_the_10m_winds_says_what_is_missing(tmp_path):
    era5 = _hourly_file(tmp_path / "e", with_10m=False)
    with pytest.raises(ValueError, match="10 m wind components"):
        _reduce(era5, "derived")


def test_an_unknown_treatment_is_refused(tmp_path):
    with pytest.raises(ValueError, match="roughness must be one of"):
        _reduce(_hourly_file(tmp_path / "e"), "annual-mean")


def test_the_derived_roughness_matches_prep_era5(tmp_path):
    """The two implementations must agree, or a comparison between a
    physics-informed run and a scorecard row measures the implementations.

    Compared at a grid node, so interpolation plays no part, and on one file,
    so the backfill has the same span in both.
    """
    era5 = _hourly_file(tmp_path / "e")
    dates, _, _, z0, _, _ = _reduce(era5, "derived")
    ds = prep_era5("ZZ", False, True, era5_dir=era5, roughness="derived")
    assert ds.attrs["pyvwf_roughness_treatment"] == "derived"
    reference = ds["roughness"].sel(lon=8.25, lat=55.25).sel(time=dates).values
    assert np.allclose(z0[:, 0], reference, rtol=1e-6, atol=1e-7)


def test_the_loaded_extent_and_a_unit_beyond_it_are_recorded(tmp_path):
    """A unit outside the grid gets no wind at all, and the record says how
    much of the fleet that is, in the keys a harness manifest uses."""
    era5 = _hourly_file(tmp_path / "e", with_z0=False)
    _, w, _, _, _, reduction = _reduce(era5, "derived", lon=(8.25, 9.4), lat=(55.25, 55.25))
    assert reduction["loaded_extent"] == [8.0, 8.5, 55.0, 55.5]
    assert np.isfinite(w[:, 0]).all()
    assert np.isnan(w[:, 1]).all()

    fleet = pd.DataFrame(
        {
            "ID": ["in", "out"],
            "lon": [8.25, 9.4],
            "lat": [55.25, 55.25],
            "capacity": [3000.0, 1000.0],
        }
    )
    spec = SimpleNamespace(bbox=(8.0, 8.5, 55.0, 55.5), allow_extrapolation=True)
    record = era5_record_for(reduction, fleet, spec)
    extent = record["era5_extent"]
    assert extent["units_outside_loaded_extent"] == 1
    assert extent["capacity_share_outside_loaded_extent"] == pytest.approx(0.25)
    assert extent["ids_outside"] == ["out"]
    assert extent["allow_extrapolation"] is True
    assert "never extrapolated" in extent["meaning"]
    assert record["era5_roughness"] == {"requested": "derived", "applied": "derived"}
