"""Daily ERA5 wind statistics at turbine locations, including sub-daily spread.

``prep_era5`` averages the hourly reanalysis to daily means, and every published
PyVWF result is built on that daily field. Averaging first and converting to
power second is not the same as converting first and averaging second, because
the power curve is not linear: a day whose wind sits near the steep part of the
curve is misestimated by an amount that depends on how much the wind varied
WITHIN that day, and that error is systematic, not random.

The affine correction has no way to see this, so it absorbs the aggregation
error into the same two parameters that are supposed to carry the reanalysis's
wind-speed bias. Here the daily standard deviation is retained alongside the
daily mean, so the forward model can integrate the power curve over the
within-day distribution instead of evaluating it at a point.
"""
from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import xarray as xr

from vwf.datasets.era5 import (
    unify_time_coordinate,
    _normalise_longitudes,
    _slice_bbox,
)

# Matches prep_era5's guards, so the roughness computed here for regions whose
# files carry no z0 agrees with the one the incumbent pipeline derives.
_Z0_MIN, _Z0_MAX = 1e-6, 2.0


def _open_normalised(path: Path, bbox) -> xr.Dataset:
    ds = xr.open_dataset(path)
    ds = unify_time_coordinate(ds)
    for old, new in (("longitude", "lon"), ("latitude", "lat")):
        if old in ds.coords:
            ds = ds.rename({old: new})
    ds = _normalise_longitudes(ds)
    if bbox is not None:
        ds = _slice_bbox(ds, bbox)
    return ds.load()


def _hourly_fields(ds: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Hourly 100 m wind, the roughness the incumbent uses, and the shear exponent.

    Three distinct things, because the archive is not uniform. The pre-combined
    European files ship a single time-AVERAGED ``z0`` field (2-D, no time axis),
    derived once from the 10-100 m shear; the raw American files ship none, so
    ``prep_era5`` inverts the log profile hour by hour and then daily-averages.
    Europe therefore runs on a climatological roughness and the Americas on a
    varying one -- a difference in the character of an input that any
    cross-region method has to live with, so both are reproduced faithfully here
    rather than silently harmonised.

    The shear exponent is computed the same way everywhere and is the physically
    informative quantity: ``w(h) = w100 * (h/100)**shear`` is the power-law
    profile, and the exponent responds to atmospheric stability as well as to
    surface roughness, which a static roughness cannot.
    """
    # cast: numpy's stubs type np.sqrt on a DataArray as ndarray, while at
    # runtime xarray returns a DataArray. Using ** 0.5 instead would type
    # cleanly but is not guaranteed to round identically to sqrt, and this
    # field is pinned bit-for-bit against prep_era5.
    wnd100 = cast(xr.DataArray, ds["wnd100m"] if "wnd100m" in ds.data_vars
                  else np.sqrt(ds["u100"] ** 2 + ds["v100"] ** 2))

    if {"u10", "v10"} <= set(ds.data_vars):
        wnd10 = cast(xr.DataArray, np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)).clip(min=1e-3)
        w100c = wnd100.clip(min=1e-3)
        shear = np.log(w100c / wnd10) / np.log(10.0)
        # Physically admissible band: -0.1 (very unstable, near-uniform or
        # slightly reversed profile) to 0.6 (strongly stable nocturnal shear).
        shear = shear.clip(min=-0.1, max=0.6)
    else:
        shear = xr.zeros_like(wnd100) + np.nan

    if "roughness" in ds.data_vars or "z0" in ds.data_vars:
        z0 = ds["roughness"] if "roughness" in ds.data_vars else ds["z0"]
        if "time" not in z0.dims:
            # Static climatological field: broadcast so the daily reduction is
            # a no-op that still returns a (time, lat, lon) array.
            z0 = z0.broadcast_like(wnd100)
    else:
        # The same inversion of the neutral log profile prep_era5 falls back to.
        wnd10 = cast(xr.DataArray, np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)).clip(min=1e-4)
        w100 = wnd100.clip(min=1e-4)
        num = w100 * np.log(10.0) - wnd10 * np.log(100.0)
        denom = (w100 - wnd10)
        denom = denom.where(np.abs(denom) > 1e-4)
        z0_log = (num / denom).where(lambda a: a < 0)
        z0_log = z0_log.bfill("time").clip(min=np.log(1e-6), max=np.log(_Z0_MAX))
        z0 = np.exp(z0_log)

    return wnd100, z0.clip(min=_Z0_MIN, max=_Z0_MAX), cast(xr.DataArray, shear)


def daily_stats_at_points(
    hourly_dir: str | Path,
    bbox,
    lon: np.ndarray,
    lat: np.ndarray,
    years: range | list[int],
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Daily mean wind, within-day wind spread, and daily mean roughness.

    Each ERA5 file is reduced to daily statistics and interpolated onto the
    requested points before the next is opened, so peak memory is one file, not
    the whole archive.

    Args:
        hourly_dir: Directory of hourly ERA5 NetCDF files.
        bbox: ``(lon_min, lon_max, lat_min, lat_max)`` or None.
        lon: Point longitudes.
        lat: Point latitudes.
        years: Years to include; files whose data falls outside are skipped.

    Returns:
        ``(dates, w_mean, w_std, z0_mean, shear)``. The four arrays are
        ``(n_days, n_points)`` float32: the daily mean 100 m wind speed, the
        standard deviation of the hourly wind speed WITHIN each day, the daily
        mean roughness as the incumbent pipeline would see it, and the daily
        mean power-law shear exponent between 10 m and 100 m.

    Raises:
        FileNotFoundError: If the directory holds no NetCDF files.
    """
    hourly_dir = Path(hourly_dir)
    files = sorted(hourly_dir.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"no NetCDF files under {hourly_dir}")

    wanted = {int(y) for y in years}
    plon = xr.DataArray(np.asarray(lon, dtype=float), dims="point")
    plat = xr.DataArray(np.asarray(lat, dtype=float), dims="point")

    chunks: list[
        tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    for path in files:
        with xr.open_dataset(path) as probe:
            probe = unify_time_coordinate(probe)
            file_years = set(pd.DatetimeIndex(probe["time"].values).year.unique())
        if not (file_years & wanted):
            continue

        ds = _open_normalised(path, bbox)
        wnd100, z0, shear = _hourly_fields(ds)
        daily = xr.Dataset(
            {
                "w_mean": wnd100.resample(time="1D").mean(),
                "w_std": wnd100.resample(time="1D").std(),
                "z0_mean": z0.resample(time="1D").mean(),
                "shear": shear.resample(time="1D").mean(),
            }
        )
        daily = daily.sel(time=daily.time.dt.year.isin(sorted(wanted)))
        if daily.sizes.get("time", 0) == 0:
            ds.close()
            continue
        # Round coordinates exactly as prep_era5 does, so the interpolation
        # lands on the same grid the incumbent pipeline uses.
        daily = daily.assign_coords(
            lon=np.round(daily.lon.astype(float), 5),
            lat=np.round(daily.lat.astype(float), 5),
        )
        at = daily.interp(lon=plon, lat=plat)
        chunks.append((
            pd.DatetimeIndex(at.time.values),
            at["w_mean"].values.astype("float32"),
            at["w_std"].values.astype("float32"),
            at["z0_mean"].values.astype("float32"),
            at["shear"].values.astype("float32"),
        ))
        ds.close()
        del daily, at

    if not chunks:
        raise FileNotFoundError(
            f"no ERA5 files under {hourly_dir} covered years {sorted(wanted)}"
        )

    dates = pd.DatetimeIndex(np.concatenate([c[0].values for c in chunks]))
    order = np.argsort(dates.values)
    dates = dates[order]
    stacked = [np.concatenate([c[i] for c in chunks])[order] for i in (1, 2, 3, 4)]

    keep = ~dates.duplicated()
    out_w, out_sd, out_z0, out_shear = (a[keep] for a in stacked)
    return dates[keep], out_w, out_sd, out_z0, out_shear
