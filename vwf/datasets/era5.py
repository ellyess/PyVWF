"""
era5 module.

Summary
-------
Importing and processing ERA5 reanalysis data.

Data conventions
----------------
Expected dimensions follow xarray conventions (e.g., time × lat × lon) unless stated otherwise.
Time coordinates are assumed to be UTC unless explicitly converted by the caller.

Units
-----
Wind speed: [m s^-1]; Hub height: [m]; Power: [MW]; Energy: [MWh]; Capacity factor: [-] (unless stated otherwise).

Assumptions
-----------
- ERA5/reanalysis fields are treated as representative at the chosen spatial/temporal resolution.
- Wake effects, curtailment, availability losses are not modelled unless explicitly implemented in this module.

References
----------
Add dataset and methodological references relevant to this module.
"""
import xarray as xr
import numpy as np


def unify_time_coordinate(ds):
    """
    Ensure the dataset uses ONLY a 'time' dimension/coordinate.
    Handles cases where both 'valid_time' and 'time' exist.
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

BBOX = {
    "UK": (-11, 3, 49, 61),
    "NL": (2.5, 7.5, 50.5, 53.8),
    "BE": (2.2, 6.6, 49.3, 51.8),
    "DE": (5.0, 16.0, 47.0, 55.2),
    "DK": (7.5, 13.5, 54.0, 58.2),
    "NO": (4.0, 31.5, 57.5, 71.5),
    "FR": (-6.0, 10.5, 41.0, 51.5),
}

def _slice_bbox(ds: xr.Dataset, bbox: tuple[float, float, float, float]) -> xr.Dataset:
    """
    bbox = (lon_min, lon_max, lat_min, lat_max)
    Handles ERA5 lat being descending or ascending.
    """
    lon_min, lon_max, lat_min, lat_max = bbox

    # Ensure lon is in [-180, 180] if your data is 0..360 (optional; only if needed)
    # if ds.lon.max() > 180:
    #     ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180).sortby("lon")

    lat_desc = bool(ds.lat[0] > ds.lat[-1])
    lat_slice = slice(lat_max, lat_min) if lat_desc else slice(lat_min, lat_max)

    return ds.sel(lon=slice(lon_min, lon_max), lat=lat_slice)

def prep_era5(country, train=False, calc_z0=True, bbox=None):
    """
    Memory-light, fast version of ERA5 preprocessing.

    bbox: optional (lon_min, lon_max, lat_min, lat_max). If None, uses BBOX[country] when available.
    """
    print(f"prepping ERA5 data for {country}, train={train}, calc_z0={calc_z0}")

    # Load ERA5 files
    path = f"input/era5/EU/*.nc"
    ds = xr.open_mfdataset(path, combine="by_coords", parallel=False)
    ds = unify_time_coordinate(ds)

    # Standardize coordinate names EARLY (so bbox slicing works)
    for old, new in [("longitude", "lon"), ("latitude", "lat")]:
        if old in ds.coords:
            ds = ds.rename({old: new})

    # Apply bbox slice early (big memory/time win before .load())
    if bbox is None:
        bbox = BBOX.get(country)
    if bbox is not None:
        ds = _slice_bbox(ds, bbox)

    ds = ds.load()  # now load only the sliced subset

    # Wind speed at 100m
    ds["wnd100m"] = np.sqrt(ds["u100"] ** 2 + ds["v100"] ** 2)

    if calc_z0:
        ds = ds.drop_vars("fsr", errors="ignore")

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

# from __future__ import annotations

# from pathlib import Path
# import shutil

# import numpy as np
# import xarray as xr
# import zarr


# def unify_time_coordinate(ds: xr.Dataset) -> xr.Dataset:
#     """
#     Ensure the dataset uses ONLY a 'time' dimension/coordinate.
#     Handles cases where both 'valid_time' and 'time' exist.
#     """
#     if "valid_time" in ds.coords and "time" in ds.coords:
#         if ds["valid_time"].equals(ds["time"]):
#             ds = ds.drop_vars("valid_time")
#         else:
#             ds = ds.drop_vars("time")
#             ds = ds.rename({"valid_time": "time"})
#     elif "valid_time" in ds.coords and "time" not in ds.coords:
#         ds = ds.rename({"valid_time": "time"})

#     if "valid_time" in ds.dims:
#         ds = ds.rename_dims({"valid_time": "time"})
#     return ds


# def _preprocess_keep_only(ds: xr.Dataset) -> xr.Dataset:
#     """
#     Preprocess each file before concatenation:
#     - unify time coord
#     - rename coords to lon/lat
#     - drop nuisance vars
#     - keep only required vars
#     """
#     ds = unify_time_coordinate(ds)

#     rename = {}
#     if "longitude" in ds.coords:
#         rename["longitude"] = "lon"
#     if "latitude" in ds.coords:
#         rename["latitude"] = "lat"
#     if rename:
#         ds = ds.rename(rename)

#     ds = ds.drop_vars(["expver", "number"], errors="ignore")

#     keep = [v for v in ["u10", "v10", "u100", "v100", "fsr"] if v in ds.data_vars]
#     return ds[keep]


# def prep_era5_daily_cached(
#     country: str,
#     *,
#     train: bool = False,
#     calc_z0: bool = True,
#     in_root: str = "input/era5",
#     cache_root: str = "input/cache/era5_daily",
#     bbox: tuple[float, float, float, float] | None = None,  # (lon_min, lon_max, lat_min, lat_max)
#     overwrite: bool = False,
# ) -> xr.Dataset:
#     """
#     Build a DAILY dataset (wnd100m + roughness) from hourly ERA5 u/v and cache to Zarr.

#     Key properties:
#     - daily aggregation happens BEFORE roughness (fast)
#     - avoids bfill/ffill on hourly
#     - writes Zarr v2 with consolidated metadata
#     - ensures coords (time/lat/lon) are chunked (prevents Zarr "chunks=None" error)

#     Notes:
#     - This function currently reads from: in_root/*.nc (one Europe cutout)
#       If you have per-country folders, replace `path` below with:
#         Path(in_root) / country / split / "*.nc"
#     """
#     split = "train" if train else "test"
#     cache_dir = Path(cache_root) / country / split
#     cache_dir.mkdir(parents=True, exist_ok=True)

#     tag = "z0" if calc_z0 else "fsr"
#     bbox_tag = ""
#     if bbox is not None:
#         bbox_tag = f"_lon{bbox[0]}_{bbox[1]}_lat{bbox[2]}_{bbox[3]}".replace(".", "p")
#     zarr_path = cache_dir / f"era5_daily_{tag}{bbox_tag}.zarr"

#     # Fast path: open existing cache (use consolidated=False for resilience)
#     if zarr_path.exists() and not overwrite:
#         return xr.open_zarr(zarr_path, consolidated=False)

#     if zarr_path.exists() and overwrite:
#         shutil.rmtree(zarr_path)

#     # If you truly have per-country/train-test files, switch to:
#     # path = Path(in_root) / country / split / "*.nc"
#     path = Path(in_root) / "*.nc"

#     # Avoid -1 chunks here; keep explicit sizes
#     ds = xr.open_mfdataset(
#         str(path),
#         combine="by_coords",
#         preprocess=_preprocess_keep_only,
#         parallel=False,  # safer; prevents kernel death on many setups
#         chunks={"time": 8760, "lat": 64, "lon": 64},  # yearly time chunks; modest spatial
#     )

#     # Optional spatial subset BEFORE any compute/resample
#     if bbox is not None:
#         lon_min, lon_max, lat_min, lat_max = bbox
#         lat_slice = slice(lat_max, lat_min) if ds.lat[0] > ds.lat[-1] else slice(lat_min, lat_max)
#         ds = ds.sel(lon=slice(lon_min, lon_max), lat=lat_slice)

#     # --- DAILY FIRST: resample u/v to daily means ---
#     u100_d = ds["u100"].resample(time="1D").mean()
#     v100_d = ds["v100"].resample(time="1D").mean()
#     u10_d = ds["u10"].resample(time="1D").mean()
#     v10_d = ds["v10"].resample(time="1D").mean()

#     # Daily wind speeds
#     wnd100_d = np.hypot(u100_d, v100_d).astype("float32")
#     wnd10_d = np.hypot(u10_d, v10_d).astype("float32")

#     # Roughness: either computed from (10m,100m) or from fsr
#     if calc_z0:
#         denom = (wnd100_d - wnd10_d).where(np.abs(wnd100_d - wnd10_d) > 1e-6)
#         num = wnd100_d * np.log(10.0) - wnd10_d * np.log(100.0)
#         z0_log = (num / denom)

#         z0 = np.exp(z0_log.where(z0_log < 0)).astype("float32")  # z0<1m filter
#         z0 = z0.clip(min=1e-5, max=2.0)  # keep stable range
#         z0 = z0.fillna(np.float32(0.03))  # conservative fallback
#         roughness = z0
#     else:
#         roughness = ds["fsr"].resample(time="1D").mean().astype("float32")

#     # Output dataset
#     ds_out = xr.Dataset(
#         {"wnd100m": wnd100_d, "roughness": roughness},
#         coords={"time": wnd100_d.time, "lat": wnd100_d.lat, "lon": wnd100_d.lon},
#     )

#     # Coordinate rounding (optional)
#     ds_out = ds_out.assign_coords(
#         lon=np.round(ds_out.lon.astype("float64"), 5),
#         lat=np.round(ds_out.lat.astype("float64"), 5),
#     )

#     # --- CHUNKING: must chunk coords too (fixes your error) ---
#     tchunk = 366
#     ychunk = int(ds_out.sizes["lat"])
#     xchunk = int(ds_out.sizes["lon"])

#     ds_out = ds_out.chunk({"time": tchunk, "lat": ychunk, "lon": xchunk})

#     # Force coordinate variables to be chunked (they were None)
#     ds_out["time"] = ds_out["time"].chunk({"time": min(tchunk, int(ds_out.sizes["time"]))})
#     ds_out["lat"] = ds_out["lat"].chunk({"lat": ychunk})
#     ds_out["lon"] = ds_out["lon"].chunk({"lon": xchunk})

#     # Ensure consistent chunking across vars
#     ds_out = ds_out.unify_chunks()

#     # Clean any half-written store (safety)
#     if zarr_path.exists():
#         shutil.rmtree(zarr_path)

#     # --- WRITE: Zarr v2 is most stable with xarray ---
#     ds_out.to_zarr(
#         zarr_path,
#         mode="w",
#         consolidated=False,
#         zarr_version=2,
#     )
#     zarr.consolidate_metadata(str(zarr_path))

#     return xr.open_zarr(zarr_path, consolidated=True)