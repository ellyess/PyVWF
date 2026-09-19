"""ERA5 reanalysis import and preprocessing utilities."""

from pathlib import Path

import xarray as xr
import warnings

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
            ds = ds.drop_vars("time")  # remove existing time first
            ds = ds.rename({"valid_time": "time"})  # now rename safely

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


def _extent_shortfall(ds: xr.Dataset, bbox: tuple[float, float, float, float]) -> dict[str, float]:
    """How far, per side, the loaded grid stops short of the requested bbox.

    A shortfall of up to one grid step is normal (the bbox need not fall on
    grid lines), so only sides short by more than one step are returned.
    """
    lon = np.asarray(ds["lon"].values, dtype=float)
    lat = np.asarray(ds["lat"].values, dtype=float)
    if lon.size == 0 or lat.size == 0:
        return {"west": float("inf")}
    step_lon = float(np.min(np.abs(np.diff(np.sort(lon))))) if lon.size > 1 else 0.25
    step_lat = float(np.min(np.abs(np.diff(np.sort(lat))))) if lat.size > 1 else 0.25
    lon_min, lon_max, lat_min, lat_max = bbox
    short = {
        "west": lon.min() - lon_min,
        "east": lon_max - lon.max(),
        "south": lat.min() - lat_min,
        "north": lat_max - lat.max(),
    }
    steps = {"west": step_lon, "east": step_lon, "south": step_lat, "north": step_lat}
    return {side: float(d) for side, d in short.items() if d > steps[side] + 1e-9}


ROUGHNESS_TREATMENTS = ("stored", "derived")

#: The roughness lengths the inversion may return, in metres.
Z0_BOUNDS = (1e-6, 2.0)


def log_roughness_from_shear(wind10: xr.DataArray, wind100: xr.DataArray) -> xr.DataArray:
    """The log of the roughness length z0, from the 10 m and 100 m wind speeds.

    Inverts the log wind profile between the two heights:
    ``ln z0 = (w100 ln 10 - w10 ln 100) / (w100 - w10)``. The one definition
    all three roughness routes use (``CONTEXT.md``, roughness route): the
    derivation at load in :func:`prep_era5`, the annual-mean field of
    ``combine_era5_files.py`` and the daily files of
    ``scripts/era5/combine.py``.

    Where the shear is near zero (``|w100 - w10| <= 1e-4``) or the result would
    give z0 of 1 m or more, the value is missing, then back-filled along time.
    The result is clipped to the logs of :data:`Z0_BOUNDS`. The caller clips
    the speeds first and takes the exponential, because the routes differ
    there.

    Args:
        wind10: 10 m wind speed, m/s, already clipped away from zero.
        wind100: 100 m wind speed, m/s, already clipped away from zero.
    """
    num = wind100 * np.log(10) - wind10 * np.log(100)
    denom = wind100 - wind10
    # mask near-zero shear (this is what avoids divide-by-zero)
    denom = denom.where(np.abs(denom) > 1e-4)
    z0_log = num / denom
    # physically: log(z0) < 0  ->  z0 < 1 m
    z0_log = z0_log.where(z0_log < 0)
    z0_log = z0_log.bfill("time")
    # avoid insane roughness lengths
    return z0_log.clip(min=np.log(Z0_BOUNDS[0]), max=np.log(Z0_BOUNDS[1]))


def prep_era5(
    country,
    train=False,
    calc_z0=True,
    bbox=None,
    era5_dir=None,
    resample_daily=True,
    allow_extrapolation=False,
    roughness="stored",
):
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
        resample_daily: If True (default), average to daily means, which is what
            every published PyVWF result is built on. Set False to keep the
            file's native resolution, which for the raw hourly downloads is
            hourly. NOTE this is not a free switch: the power curve is convex, so
            the daily mean of simulated power is not the power of the daily mean
            wind, and a run at native resolution is a materially different model,
            not a finer view of the same one. Default True so existing results
            and the golden regression path are unchanged.
        allow_extrapolation: Attached to the returned dataset, where
            ``vwf.wind.interpolate_wind`` reads it: whether units outside the
            loaded extent may have their winds extrapolated. Default False, so
            such units are refused. A region opts in with
            ``[era5] allow_extrapolation = true``.
        roughness: Which temporal treatment of the roughness to apply.
            ``"stored"`` (default) uses a roughness field the files already
            carry, which for the European files is one annual mean per year;
            ``"derived"`` ignores any stored field and inverts the log profile
            per timestep from the 10 m and 100 m winds. Files with no stored
            field are derived either way. Which treatment is better is under
            test (``docs/findings/method-roughness-treatment-prereg.md``); the
            default reproduces what every existing run did. The treatment
            actually applied is attached to the returned dataset as
            ``pyvwf_roughness_treatment``.

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
        # The cheap check: a download that stops short of the box would leave
        # units beyond the data. The ES, IT and PT country boxes reach south of
        # the European files' 42N edge, and nothing said so at load time.
        shortfall = _extent_shortfall(ds, bbox)
        if shortfall:
            sides = ", ".join(f"{side} by {d:.2f} degrees" for side, d in shortfall.items())
            warnings.warn(
                f"ERA5 for {country} stops short of the requested bbox {tuple(bbox)}: "
                f"{sides} (files in {data_dir}). Units beyond the data are refused "
                "unless the region allows extrapolation.",
                stacklevel=2,
            )

    ds = ds.load()  # now load only the sliced subset

    # Wind speed at 100m. Pre-combined files may already carry wnd100m
    # (computed from HOURLY components before any daily averaging: mean
    # speed, not speed of mean components); recomputing from daily-mean u/v
    # would be wrong, and raw files carry components, so compute only when
    # absent.
    if "wnd100m" not in ds.data_vars:
        ds["wnd100m"] = np.sqrt(ds["u100"] ** 2 + ds["v100"] ** 2)

    if roughness not in ROUGHNESS_TREATMENTS:
        raise ValueError(f"roughness must be one of {ROUGHNESS_TREATMENTS}, got {roughness!r}")

    applied = None
    if calc_z0:
        ds = ds.drop_vars("fsr", errors="ignore")

        if roughness == "derived":
            # Asked for the per-timestep derivation, so a stored field is not
            # used even when the files carry one.
            missing = [v for v in ("u10", "v10") if v not in ds.data_vars]
            if missing:
                raise ValueError(
                    f"roughness='derived' needs the 10 m wind components, and the ERA5 "
                    f"files for {country} lack {missing}. The stored field is the only "
                    "roughness those files can supply."
                )
            ds = ds.drop_vars(["z0", "roughness"], errors="ignore")

        # Check if roughness already exists (from preprocessing)
        if "z0" in ds.data_vars or "roughness" in ds.data_vars:
            # Use existing pre-calculated roughness
            if "z0" in ds.data_vars:
                ds = ds.rename({"z0": "roughness"})
                print("Using pre-calculated roughness (z0) from combined ERA5 files")
            else:
                print("Using pre-calculated roughness from combined ERA5 files")

            applied = "stored"
            # Drop unnecessary wind component variables
            ds = ds.drop_vars(
                ["u100", "v100", "u10", "v10", "number", "expver"],
                errors="ignore",
            )
        else:
            applied = "derived"
            # Calculate roughness from wind shear (fallback if not preprocessed)
            print("Calculating surface roughness from 10m/100m wind shear...")

            wnd10m = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)

            wnd10m = wnd10m.clip(min=1e-4)
            ds["wnd100m"] = ds["wnd100m"].clip(min=1e-4)

            z0_log = log_roughness_from_shear(wnd10m, ds["wnd100m"])

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

    # Daily resampling (the published resolution; see resample_daily).
    if resample_daily:
        ds = ds.resample(time="1D").mean()

    # Rounding coordinates
    ds = ds.assign_coords(
        lon=np.round(ds.lon.astype(float), 5),
        lat=np.round(ds.lat.astype(float), 5),
    )

    from vwf.wind import EXTRAPOLATION_ATTR

    ds.attrs[EXTRAPOLATION_ATTR] = bool(allow_extrapolation)
    # What was applied, not what was asked for: a run that asks for the stored
    # field and gets the derivation, because the files carry none, must say so.
    ds.attrs["pyvwf_roughness_treatment"] = applied if applied else "reanalysis-field"
    print("ERA5 for " + country + " ready")
    return ds
