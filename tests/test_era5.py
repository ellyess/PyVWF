"""ERA5 loading: longitude normalisation, config-driven paths, precomputed fields."""

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pyvwf.config import PyVWFPaths
from pyvwf.era5 import _normalise_longitudes, _slice_bbox, prep_era5


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


def test_prep_era5_respects_precomputed_wnd100m(tmp_path):
    """Pre-combined files carry wnd100m computed from HOURLY speeds; prep
    must NOT overwrite it from (daily-mean) components: mean-of-speed and
    speed-of-mean differ, and the fixture distinguishes them."""
    times = pd.date_range("2019-01-01", periods=2, freq="D")
    lats, lons = np.array([50.0, 52.0]), np.array([5.0, 7.0])
    shape = (2, 2, 2)
    era5_dir = tmp_path / "combined"
    era5_dir.mkdir()
    # Planted: wnd100m = 10 everywhere, while components imply hypot(3,4)=5.
    xr.Dataset(
        {
            "wnd100m": (("time", "lat", "lon"), np.full(shape, 10.0)),
            "roughness": (("time", "lat", "lon"), np.full(shape, 0.05)),
            "u100": (("time", "lat", "lon"), np.full(shape, 3.0)),
            "v100": (("time", "lat", "lon"), np.full(shape, 4.0)),
        },
        coords={"time": times, "lat": lats, "lon": lons},
    ).to_netcdf(era5_dir / "combined.nc")

    # roughness="stored" because the fixture is shaped like a daily pre-combined
    # file: it carries a roughness and no 10 m winds, so the default "derived"
    # has nothing to invert and refuses.
    ds = prep_era5(
        "ZZ", calc_z0=True, bbox=(4.0, 8.0, 49.0, 53.0), era5_dir=era5_dir, roughness="stored"
    )
    assert ds["wnd100m"].to_numpy() == pytest.approx(10.0)  # kept, not recomputed to 5


def test_prep_era5_reads_from_era5_dir_and_normalises(tmp_path):
    """prep_era5 with era5_dir= reads a 0..360 file and returns [-180, 180]."""
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

    ds = prep_era5(
        "ZZ",  # no BoundingBoxes entry: bbox comes from the caller, like the harness
        calc_z0=False,
        bbox=(-11.0, -5.0, 49.0, 53.0),
        era5_dir=era5_dir,
    )

    assert ds.sizes["lon"] == 3
    assert float(ds.lon.min()) == -10.0 and float(ds.lon.max()) == -6.0
    assert float(ds.lon.max()) <= 180.0
    # sqrt(7^2 + 7^2) ~ 9.9 m/s, daily-resampled
    assert ds["wnd100m"].to_numpy() == pytest.approx(np.hypot(7.0, 7.0))
    assert ds.sizes["time"] == 2  # 48 hourly steps -> 2 daily means


def test_log_roughness_inverts_the_log_profile_and_masks_what_it_cannot():
    from pyvwf.era5 import Z0_BOUNDS, log_roughness_from_shear

    def profile(z0, ustar=1.0):
        return ustar * np.log(10 / z0), ustar * np.log(100 / z0)

    # t0 a known profile; t1 no shear, so missing and back-filled from t2;
    # t2 another profile; t3 so smooth that z0 falls below the lower bound.
    w10, w100 = zip(profile(0.03), (5.0, 5.0), profile(0.2), profile(1e-9))
    da = lambda v: xr.DataArray(np.array(v, dtype=float), dims="time")  # noqa: E731
    z0 = np.exp(log_roughness_from_shear(da(w10), da(w100)).values)
    np.testing.assert_allclose(z0[[0, 2]], [0.03, 0.2], rtol=1e-12)
    assert z0[1] == z0[2]
    assert z0[3] == pytest.approx(Z0_BOUNDS[0], rel=1e-12)


# --- years: load only a window, identical to the whole record cut to it ---

WINDOW_BBOX = (7.9, 8.4, 54.9, 55.4)


def _hourly_archive(era5_dir, end, flat=()):
    """Monthly hourly files on a 2x2 grid from November 2019 up to ``end``.

    One file per month, as ERA5 is stored. The winds are the same draws
    whatever ``end`` is, so two archives differ only in where they stop.
    ``flat`` holds ``(lat_index, lon_index, first_hour, last_hour)`` spans where
    the 100 m wind equals the 10 m wind, so the profile cannot be inverted and
    the roughness is missing until the back-fill reaches a valid hour.
    """
    era5_dir.mkdir(parents=True)
    times = pd.date_range("2019-11-01", "2020-03-31T23", freq="h")
    rng = np.random.default_rng(0)
    u10 = rng.uniform(3.0, 8.0, (len(times), 2, 2))
    u100 = u10 * rng.uniform(1.1, 1.5, u10.shape)
    for i, j, first, last in flat:
        span = (times >= pd.Timestamp(first)) & (times <= pd.Timestamp(last))
        u100[span, i, j] = u10[span, i, j]
    ds = xr.Dataset(
        {
            "u10": (("time", "latitude", "longitude"), u10),
            "v10": (("time", "latitude", "longitude"), np.zeros_like(u10)),
            "u100": (("time", "latitude", "longitude"), u100),
            "v100": (("time", "latitude", "longitude"), np.zeros_like(u10)),
        },
        coords={"time": times, "latitude": [55.25, 55.0], "longitude": [8.0, 8.25]},
    ).sel(time=slice(None, end))
    for month, part in ds.groupby("time.month"):
        part.to_netcdf(era5_dir / f"era5_zz_{part.time.dt.year.values[0]}_{month:02d}.nc")


def _prep(era5_dir, years=None):
    return prep_era5("ZZ", bbox=WINDOW_BBOX, era5_dir=era5_dir, years=years)


def _assert_same(windowed, whole):
    """Identical values, dtypes and coordinates; the window attribute aside."""
    windowed = windowed.copy()
    del windowed.attrs["pyvwf_era5_window"]
    xr.testing.assert_identical(windowed, whole)
    for name in windowed.variables:
        assert windowed[name].dtype == whole[name].dtype


def test_a_window_reads_the_january_after_it_and_matches_the_whole_record(tmp_path):
    # The last 2019 hours at one cell are filled from 2 January 2020, inside
    # the January the window reads past its end.
    flat = [(0, 0, "2019-12-31T20", "2020-01-02T05")]
    _hourly_archive(tmp_path / "era5", "2020-03-31T23", flat)
    whole = _prep(tmp_path / "era5")
    windowed = _prep(tmp_path / "era5", years=(2019, 2019))

    assert "pyvwf_era5_window" not in whole.attrs
    assert windowed.attrs["pyvwf_era5_window"] == "2019-2019"  # no whole-record reload
    assert str(windowed.time.values[-1])[:10] == "2019-12-31"
    _assert_same(windowed, whole.sel(time=slice("2019-01-01", "2019-12-31")))


def test_without_the_january_the_windows_last_day_differs(tmp_path):
    """Known positive for the test above: the margin is what makes them equal.

    The same hours with the archive stopping at the window's end leave the
    last day unfilled at that cell, so its roughness is not the whole record's.
    No reload is attempted, since there is nothing past the window to read.
    """
    flat = [(0, 0, "2019-12-31T20", "2020-01-02T05")]
    _hourly_archive(tmp_path / "with", "2020-03-31T23", flat)
    _hourly_archive(tmp_path / "without", "2019-12-31T23", flat)
    whole = _prep(tmp_path / "with").sel(time="2019-12-31")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        cut = _prep(tmp_path / "without", years=(2019, 2019)).sel(time="2019-12-31")

    assert cut["roughness"].values[0, 0] != whole["roughness"].values[0, 0]
    np.testing.assert_array_equal(cut["roughness"].values[1:], whole["roughness"].values[1:])


def test_a_gap_past_the_january_loads_the_whole_record(tmp_path):
    # Unfilled from the last day of 2019 to 10 February 2020, past the January.
    flat = [(1, 1, "2019-12-31T10", "2020-02-10T00")]
    _hourly_archive(tmp_path / "era5", "2020-03-31T23", flat)
    whole = _prep(tmp_path / "era5")
    with pytest.warns(UserWarning, match="loading the whole record"):
        windowed = _prep(tmp_path / "era5", years=(2019, 2019))

    assert windowed.attrs["pyvwf_era5_window"] == "2019-2019, whole record loaded"
    _assert_same(windowed, whole.sel(time=slice("2019-01-01", "2019-12-31")))
