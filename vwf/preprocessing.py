import xarray as xr
import numpy as np
import pandas as pd

def prep_era5(year_star, year_end):
    """
    Reading a saved ERA5 file with 100m wind speeds and fsr.
    changing names and converting wind speed components into wind speed.
    """
    # Load the corresponding raw ERA5 file
    ncFile = 'data/reanalysis/era5/'+str(year_star)+'-'+str(year_end)+'_raw.nc'
    ds = xr.open_dataset(ncFile)
    
    ds["wnd100m"] = np.sqrt(ds["u100"] ** 2 + ds["v100"] ** 2).assign_attrs(
        units=ds["u100"].attrs["units"], long_name="100 metre wind speed"
    )
    
    ds = ds.drop_vars(["u100", "v100"])
    ds = ds.rename({"fsr": "roughness"})
    
    # turn hourly data into daily for speed of existing code
    ds = ds.resample(time='1D').mean()
    return _rename_and_clean_coords(ds, False)
    

#############################################################################################################
#############################################################################################################
##### THE CODE BELOW IS ALL TAKEN FROM ATLITE PLEASE CHANGE AND ADJUST IN FUTURE BEFORE PUBLISHING  #########
#############################################################################################################
#############################################################################################################

def maybe_swap_spatial_dims(ds, namex="x", namey="y"):
    """Swap order of spatial dimensions according to atlite concention."""
    swaps = {}
    lx, rx = ds.indexes[namex][[0, -1]]
    ly, uy = ds.indexes[namey][[0, -1]]

    if lx > rx:
        swaps[namex] = slice(None, None, -1)
    if uy < ly:
        swaps[namey] = slice(None, None, -1)

    return ds.isel(**swaps) if swaps else ds

def _rename_and_clean_coords(ds, add_lon_lat=True):
    """Rename 'longitude' and 'latitude' columns to 'x' and 'y' and fix roundings.

    Optionally (add_lon_lat, default:True) preserves latitude and longitude
    columns as 'lat' and 'lon'.
    """
    try:
        ds = ds.rename({"lon": "longitude", "lat": "latitude"})
    except:
        pass
        
    ds = ds.rename({"longitude": "x", "latitude": "y"})
    # round coords since cds coords are float32 which would lead to mismatches
    ds = ds.assign_coords(
        x=np.round(ds.x.astype(float), 5), y=np.round(ds.y.astype(float), 5)
    )
    # ds = maybe_swap_spatial_dims(ds)
    if add_lon_lat:
        ds = ds.assign_coords(lon=ds.coords["x"], lat=ds.coords["y"])
    return ds