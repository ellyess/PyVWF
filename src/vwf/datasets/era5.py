"""ERA5 reanalysis import and preprocessing utilities."""
from pathlib import Path

import xarray as xr
import numpy as np

from vwf.config import BoundingBoxes, PyVWFPaths


def unify_time_coordinate(ds):
    """Ensure the dataset uses a single ``time`` coordinate.

    Args:
        ds: Input dataset that may contain ``valid_time`` or ``time``.

    Returns:
        Dataset with a normalized ``time`` coordinate.
    """
    # CASE 1: both exist
    if "valid_time" in ds.coords and "time" in ds.coords:
        # check if values identical
        if ds["valid_time"].equals(ds["time"]):
            ds = ds.drop_vars("valid_time")
        else:
            # force rename valid_time → time, overwriting
            ds = ds.drop_vars("time")                     # remove existing time first
            ds = ds.rename({"valid_time": "time"})        # now rename safely

    # CASE 2: only valid_time exists 
    elif "valid_time" in ds.coords and "time" not in ds.coords:
        # print("Only valid_time exists → renaming to time")
        ds = ds.rename({"valid_time": "time"})

    # CASE 3: only time exists → nothing to do
    else:
        pass
    
    # Fix dimensions if needed
    if "valid_time" in ds.dims:
        ds = ds.rename_dims({"valid_time": "time"})
    return ds


def _normalise_longitudes(ds: xr.Dataset) -> xr.Dataset:
    """Bring a 0..360 longitude coordinate onto the [-180, 180] convention.

    The whole pipeline (bounding boxes, turbine metadata) speaks [-180, 180].
    ERA5 files arrive in either convention depending on the download route; a
    0..360 file sliced with a [-180, 180] bbox returns an empty or wrong
    subset SILENTLY, so the convention is normalised here, unconditionally,
    before any slicing. No-op for data already in [-180, 180].
    """
    if "lon" in ds.coords and float(ds.lon.max()) > 180.0:
        ds = ds.assign_coords(lon=((ds.lon + 180.0) % 360.0) - 180.0).sortby("lon")
    return ds


def _slice_bbox(ds: xr.Dataset, bbox: tuple[float, float, float, float]) -> xr.Dataset:
    """Slice a dataset to a lon/lat bounding box.

    Args:
        ds: Input dataset with ``lon`` and ``lat`` coordinates, longitudes in
            [-180, 180] (see ``_normalise_longitudes``).
        bbox: Tuple of ``(lon_min, lon_max, lat_min, lat_max)``.

    Returns:
        Dataset spatially subset to the bounding box.
    """
    lon_min, lon_max, lat_min, lat_max = bbox

    lat_desc = bool(ds.lat[0] > ds.lat[-1])
    lat_slice = slice(lat_max, lat_min) if lat_desc else slice(lat_min, lat_max)

    return ds.sel(lon=slice(lon_min, lon_max), lat=lat_slice)

def prep_era5(country, train=False, calc_z0=True, bbox=None, era5_dir=None):
    """Preprocess ERA5 reanalysis data.

    Args:
        country: Country code used to select data paths and defaults.
        train: If True, use training-period files where applicable.
        calc_z0: If True, compute surface roughness length from 10m/100m winds.
        bbox: Optional ``(lon_min, lon_max, lat_min, lat_max)`` tuple. If None,
            uses ``BoundingBoxes.get(country)`` when available.
        era5_dir: Optional directory holding the ERA5 ``*.nc`` files (the
            validation harness passes the region config's path). Default None
            keeps the legacy ``PyVWFPaths.ERA5_DATA`` location.

    Returns:
        xarray.Dataset: Preprocessed ERA5 dataset.
    """
    print(f"prepping ERA5 data for {country}, train={train}, calc_z0={calc_z0}")

    data_dir = Path(era5_dir) if era5_dir is not None else PyVWFPaths.ERA5_DATA
    path = str(data_dir / "*.nc")
    ds = xr.open_mfdataset(path, combine="by_coords", parallel=False)
    ds = unify_time_coordinate(ds)

    # Standardize coordinate names EARLY (so bbox slicing works)
    for old, new in [("longitude", "lon"), ("latitude", "lat")]:
        if old in ds.coords:
            ds = ds.rename({old: new})

    # 0..360 downloads sliced with [-180, 180] boxes fail silently: normalise.
    ds = _normalise_longitudes(ds)

    # Apply bbox slice early (big memory/time win before .load())
    if bbox is None:
        if BoundingBoxes.has_bbox(country):
            bbox = BoundingBoxes.get(country)
    if bbox is not None:
        ds = _slice_bbox(ds, bbox)

    ds = ds.load()  # now load only the sliced subset

    # Wind speed at 100m. Pre-combined files may already carry wnd100m
    # (computed from HOURLY components before any daily averaging — mean
    # speed, not speed of mean components); recomputing from daily-mean u/v
    # would be wrong, and raw files carry components, so compute only when
    # absent.
    if "wnd100m" not in ds.data_vars:
        ds["wnd100m"] = np.sqrt(ds["u100"] ** 2 + ds["v100"] ** 2)

    if calc_z0:
        ds = ds.drop_vars("fsr", errors="ignore")

        # Check if roughness already exists (from preprocessing)
        if 'z0' in ds.data_vars or 'roughness' in ds.data_vars:
            # Use existing pre-calculated roughness
            if 'z0' in ds.data_vars:
                ds = ds.rename({'z0': 'roughness'})
                print("Using pre-calculated roughness (z0) from combined ERA5 files")
            else:
                print("Using pre-calculated roughness from combined ERA5 files")
            
            # Drop unnecessary wind component variables
            ds = ds.drop_vars(
                ["u100", "v100", "u10", "v10", "number", "expver"],
                errors="ignore",
            )
        else:
            # Calculate roughness from wind shear (fallback if not preprocessed)
            print("Calculating surface roughness from 10m/100m wind shear...")
            
            wnd10m = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)
            
            wnd10m  = wnd10m.clip(min=1e-4)
            ds["wnd100m"] = ds["wnd100m"].clip(min=1e-4)

            num = ds["wnd100m"] * np.log(10) - wnd10m * np.log(100)
            denom = ds["wnd100m"] - wnd10m

            # mask near-zero shear (this is what avoids divide-by-zero)
            denom = denom.where(np.abs(denom) > 1e-4)

            z0_log = (num / denom)

            # physically: log(z0) < 0  →  z0 < 1 m
            z0_log = z0_log.where(z0_log < 0)
            z0_log = z0_log.bfill("time")

            # avoid insane roughness lengths
            z0_log = z0_log.clip(min=np.log(1e-6), max=np.log(2.0))

            ds["roughness"] = np.exp(z0_log)
            ds["roughness"] = ds["roughness"].clip(min=1e-6)  # prevents log(0) later

            ds = ds.drop_vars(
                ["u100", "v100", "u10", "v10", "number", "expver", "wnd10m"],
                errors="ignore",
            )
            print("Calculated surface roughness length")
    else:
        # ds = ds.rename({"fsr": "roughness"})
        ds = ds.drop_vars(["u100", "v100", "u10", "v10", "number", "expver"], errors="ignore")

    # Daily resampling
    ds = ds.resample(time="1D").mean()

    # Rounding coordinates
    ds = ds.assign_coords(
        lon=np.round(ds.lon.astype(float), 5),
        lat=np.round(ds.lat.astype(float), 5),
    )

    print("ERA5 for " + country + " ready")
    return ds
