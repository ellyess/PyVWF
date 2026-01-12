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
import os
from pathlib import Path

import xarray as xr
import numpy as np

# Resolve repo root robustly (assumes: <repo_root>/vwf/datasets/era5.py and <repo_root>/input/...)
VWF_ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = Path(os.environ.get("PYVWF_INPUT_ROOT", str(VWF_ROOT / "input"))).resolve()

def _era5_glob(country: str, train: bool) -> str:
    sub = "train" if train else "test"
    p = INPUT_ROOT / "era5" / country / sub
    if not p.exists():
        raise FileNotFoundError(
            f"ERA5 directory not found: {p}\n"
            f"Expected files under: {INPUT_ROOT / 'era5' / country / sub}\n"
            f"Set PYVWF_INPUT_ROOT to override the input root."
        )
    return str(p / "*.nc")

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
        print("Only valid_time exists → renaming to time")
        ds = ds.rename({"valid_time": "time"})

    # CASE 3: only time exists → nothing to do
    else:
        pass
    
    # Fix dimensions if needed
    if "valid_time" in ds.dims:
        ds = ds.rename_dims({"valid_time": "time"})
    return ds

def prep_era5(country, train=False, calc_z0=True):
    """
    Memory-light, fast version of ERA5 preprocessing.
    Keeps speed similar to prep_era5_original, but avoids name conflicts
    and avoids unnecessary .compute().
    
        Args:
            country (str): country code, e.g. "DE", "UK"
            train (boolean): if for training or not
            calc_z0 (boolean): whether to calculate surface roughness length from 10m and 100m winds
        Returns:
            ds (xarray.DataSet): dataset with wind variables on a grid
    """
    print(f"prepping ERA5 data for {country}, train={train}, calc_z0={calc_z0}")

    # Load ERA5 files (in memory, fast)
    path = _era5_glob(country, train)
    ds = xr.open_mfdataset(path, combine="by_coords", parallel=False)
    ds = unify_time_coordinate(ds)

    ds = ds.load()     # <<< loads into memory => FAST operations afterwards

    # Wind speed at 100m
    ds["wnd100m"] = np.sqrt(ds["u100"]**2 + ds["v100"]**2)

    # If calculating surface roughness, z0 from 10m and 100m winds
    if calc_z0:
        ds = ds.drop_vars("fsr", errors="ignore")

        wnd10m = np.sqrt(ds["u10"]**2 + ds["v10"]**2)

        num   = ds["wnd100m"] * np.log(10)  - wnd10m * np.log(100)
        denom = ds["wnd100m"]               - wnd10m

        z0_log = (num / denom).where(lambda x: x < 0)
        z0_log = z0_log.bfill("time")

        ds["roughness"] = np.exp(z0_log)

        # Clean variables
        ds = ds.drop_vars(["u100","v100","u10","v10","number","expver","wnd10m"],
                            errors="ignore")

        print("Calculated surface roughness length")

    else:
        ds = ds.rename({"fsr": "roughness"})
        ds = ds.drop_vars(["u100","v100","u10","v10","number","expver"],
                            errors="ignore")

    # Daily resampling
    ds = ds.resample(time="1D").mean()

    # Standardize coordinate names
    for old, new in [("longitude","lon"),("latitude","lat")]:
        if old in ds.coords:
            ds = ds.rename({old:new})

    # Rounding coordinates
    ds = ds.assign_coords(
        lon=np.round(ds.lon.astype(float), 5),
        lat=np.round(ds.lat.astype(float), 5),
    )
    
    print("ERA5 for "+country+" ready")
    return ds