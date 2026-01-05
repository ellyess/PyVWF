import xarray as xr
import numpy as np

# def prep_era5(country, train=False, calc_z0=True): # added 10m values to calculate own FSR
#     """
#     Reading and processing a saved ERA5 file with 100m wind speeds and fsr.

#         Args:
#             train (boolean): if for training or not
            
#         Returns:
#             ds (xarray.DataSet): dataset with wind variables on a grid
#     """
#     print("prepping ERA5 data for "+country+", train="+str(train)+", calc_z0="+str(calc_z0))
#     # load the corresponding raw ERA5 file
#     if train == True:
#         ds = xr.open_mfdataset('input/era5/'+country+'/train/*.nc')
#     else:
#         ds = xr.open_mfdataset('input/era5/'+country+'/test/*.nc')
#     ds = ds.compute() # this allows it to not be dask chunks
    
#     print("loaded ERA5 data")
#     # converting wind speed components into wind speed
#     ds["wnd100m"] = np.sqrt(ds["u100"] ** 2 + ds["v100"] ** 2).assign_attrs(
#         units=ds["u100"].attrs["units"], long_name="100 metre wind speed"
#     )
    
#     if calc_z0 == True:
#         print("calculating surface roughness length")
#         ds["wnd10m"] = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2).assign_attrs(
#         units=ds["u10"].attrs["units"], long_name="10 metre wind speed"
#         )
#         ds['roughness'] = ((ds["wnd100m"]*np.log(10))-(ds["wnd10m"]*np.log(100))) / (ds["wnd100m"]-ds["wnd10m"])
#         ds['roughness'] = ds['roughness'].where((ds['roughness'] < 0))
#         ds['roughness'] = ds['roughness'].bfill('valid_time')
#         ds['roughness'] = np.exp(ds['roughness']).assign_attrs(
#             units=ds["fsr"].attrs["units"], long_name="calculated surface roughness"
#         )
#         ds = ds.drop_vars(["u100", "v100", "u10", "v10", "number", "expver"])
#         ds = ds.rename({"valid_time": "time"})
#         print("calculating surface roughness length - done")
        
#     else:
#         ds = ds.drop_vars(["u100", "v100","number","expver"])
#         ds = ds.rename({"valid_time": "time", "fsr": "roughness"})
    
#     # turn hourly data into daily for speed of existing code
#     ds = ds.resample(time='1D').mean()
#     print("resampled to daily data")
#     try:
#         ds = ds.rename({"longitude": "lon", "latitude": "lat",})
#     except:
#         pass
        
#     # keeping values at fixed float length
#     ds = ds.assign_coords(
#         lon=np.round(ds.lon.astype(float), 5), lat=np.round(ds.lat.astype(float), 5)
#     )
    
#     print(ds)
#     return ds


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
    path = f"input/era5/{country}/{'train' if train else 'test'}/*.nc"
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