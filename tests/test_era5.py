"""ERA5 longitude normalisation on load."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from vwf.config import PyVWFPaths
from vwf.datasets.era5 import _normalise_longitudes, _slice_bbox, prep_era5


def _marked_dataset(lons):
    """A dataset whose values encode their original longitude, so reordering
    bugs show up as value/coordinate mismatches, not just coordinate sorting."""
    lats = np.array([50.0, 52.0])
    values = np.tile(np.asarray(lons, dtype=float), (len(lats), 1))
    return xr.Dataset(
        {"marker": (("lat", "lon"), values)},
        coords={"lat": lats, "lon": np.asarray(lons, dtype=float)},
    )


def test_normalise_wraps_0_360_and_keeps_data_aligned():
    ds = _normalise_longitudes(_marked_dataset([0.0, 90.0, 180.0, 270.0, 350.0]))
    # Half-open [-180, 180): 180 E maps to -180, as in ERA5's 0..359.75 grids.
    assert list(ds.lon.values) == [-180.0, -90.0, -10.0, 0.0, 90.0]
    # 270 E must have become 90 W *with its data*: the marker still reads 270.
    assert float(ds["marker"].sel(lon=-90.0).isel(lat=0)) == 270.0
    assert float(ds["marker"].sel(lon=-10.0).isel(lat=0)) == 350.0
    assert float(ds["marker"].sel(lon=-180.0).isel(lat=0)) == 180.0


def test_normalise_is_noop_for_minus180_180():
    original = _marked_dataset([-10.0, 0.0, 10.0])
    ds = _normalise_longitudes(original)
    # Frame-equality pin AND object identity: data already in [-180, 180]
    # passes through untouched, not even a copy.
    xr.testing.assert_identical(ds, original)
    assert ds is original


def test_slice_bbox_after_normalisation_finds_negative_lons():
    """The silent-failure mode this exists to kill: a 0..360 dataset sliced
    with a western-Europe bbox returns EMPTY without normalisation."""
    ds_0360 = _marked_dataset([350.0, 352.0, 354.0, 356.0, 358.0])  # -10..-2 E
    bbox = (-11.0, -5.0, 49.0, 53.0)

    raw = _slice_bbox(ds_0360, bbox)
    assert raw.sizes["lon"] == 0  # the failure mode, demonstrated

    sliced = _slice_bbox(_normalise_longitudes(ds_0360), bbox)
    assert list(sliced.lon.values) == [-10.0, -8.0, -6.0]


def test_prep_era5_normalises_0_360_on_load(tmp_path, monkeypatch):
    """prep_era5 reads a 0..360 file and returns [-180, 180] coordinates."""
    times = pd.date_range("2019-01-01", periods=48, freq="h")
    lats = np.array([50.0, 52.0])
    lons_0360 = np.array([350.0, 352.0, 354.0])  # i.e. -10, -8, -6 E
    shape = (len(times), len(lats), len(lons_0360))
    wind_field = np.full(shape, 7.0)

    era5_dir = tmp_path / "era5" / "ZZ"
    era5_dir.mkdir(parents=True)
    xr.Dataset(
        {
            "u100": (("time", "lat", "lon"), wind_field),
            "v100": (("time", "lat", "lon"), wind_field),
        },
        coords={"time": times, "lat": lats, "lon": lons_0360},
    ).to_netcdf(era5_dir / "era5_synthetic_ZZ.nc")
    monkeypatch.setattr(PyVWFPaths, "ERA5_DATA", era5_dir)

    ds = prep_era5(
        "ZZ",  # no BoundingBoxes entry: the bbox comes from the caller
        calc_z0=False,
        bbox=(-11.0, -5.0, 49.0, 53.0),
    )

    assert ds.sizes["lon"] == 3
    assert float(ds.lon.min()) == -10.0 and float(ds.lon.max()) == -6.0
    assert float(ds.lon.max()) <= 180.0
    # sqrt(7^2 + 7^2) ~ 9.9 m/s, daily-resampled
    assert ds["wnd100m"].to_numpy() == pytest.approx(np.hypot(7.0, 7.0))
    assert ds.sizes["time"] == 2  # 48 hourly steps -> 2 daily means
