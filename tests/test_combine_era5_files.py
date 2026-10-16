"""The ERA5 combiner (pyvwf.datasets.combine_era5_files), on synthetic files.

It writes the yearly European files whose annual-mean roughness field the
annual-mean treatment reads (``docs/CONTEXT.md``), yet no test imported it and
coverage omitted it (#86). The winds below follow a log profile with a known
roughness length, so the recovered field can be checked against it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pyvwf.datasets import combine_era5_files as ce

Z0 = 0.05


def _write_year(era5_dir, year, *, time_name="valid_time"):
    """u/v at 10 m and 100 m for a log profile u(z) = A ln(z / Z0)."""
    time = pd.date_range(f"{year}-01-01", periods=48, freq="h")
    lat, lon = np.array([55.0, 55.25]), np.array([8.0, 8.25, 8.5])
    rng = np.random.default_rng(year)
    a = rng.uniform(0.5, 1.5, size=(len(time), len(lat), len(lon)))

    def speed(z):
        return a * np.log(z / Z0)

    coords = {time_name: time, "latitude": lat, "longitude": lon}
    dims = (time_name, "latitude", "longitude")
    s10, s100 = speed(10.0), speed(100.0)
    # All the wind from the west-south-west: u = 0.8 s, v = 0.6 s.
    xr.Dataset({"u10": (dims, 0.8 * s10), "v10": (dims, 0.6 * s10)}, coords=coords).to_netcdf(
        era5_dir / f"era5_u10_v10_{year}_months01-12_EU.nc"
    )
    xr.Dataset({"u100": (dims, 0.8 * s100), "v100": (dims, 0.6 * s100)}, coords=coords).to_netcdf(
        era5_dir / f"era5_u100_v100_{year}_months01-12_EU.nc"
    )


def test_available_years_are_read_from_the_file_names(tmp_path):
    for year in (2021, 2019):
        _write_year(tmp_path, year)
    assert ce.find_available_years(tmp_path) == [2019, 2021]


def test_the_combined_file_carries_the_four_winds_and_an_annual_mean_roughness(tmp_path):
    _write_year(tmp_path, 2019)
    out = ce.combine_era5_year(tmp_path, 2019, add_roughness=True, output_dir=tmp_path / "out")
    ds = xr.open_dataset(out)
    assert {"u10", "v10", "u100", "v100", "z0"} <= set(ds.data_vars)
    assert "time" in ds.dims and "valid_time" not in ds.dims
    # One static field per year, not a time series.
    assert ds["z0"].dims == ("latitude", "longitude")
    np.testing.assert_allclose(ds["z0"].values, Z0, rtol=1e-3)
    ds.close()


def test_a_constant_roughness_is_written_as_asked(tmp_path):
    _write_year(tmp_path, 2020, time_name="time")
    out = ce.combine_era5_year(
        tmp_path, 2020, add_roughness=True, roughness_source="constant", z0_constant=0.1
    )
    ds = xr.open_dataset(out)
    assert out.parent == tmp_path
    np.testing.assert_allclose(ds["z0"].values, 0.1)
    ds.close()


def test_without_roughness_no_z0_is_written(tmp_path):
    _write_year(tmp_path, 2020)
    ds = xr.open_dataset(ce.combine_era5_year(tmp_path, 2020))
    assert "z0" not in ds.data_vars
    ds.close()


def test_a_missing_year_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="10m wind file"):
        ce.load_era5_winds(tmp_path, 2030)
