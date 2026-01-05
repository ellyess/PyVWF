"""
merra2 module.

Summary
-------
Importing and processing MERRA-2 reanalysis data.

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

def prep_merra2(country):
    """
    Reading Iain's preprepped MERRA 2 Az file and selecting desired location.
    This is purely for research purposes and should be ignored.
    """
    year_star = 2020
    year_end = 2020
    
    area = [7., 16, 54.3, 58]    
    ncFile = 'input/merra2/DailyAZ/'+str(2020)+'-'+str(2020)+'_dailyAz_ian.nc'
    ds = xr.open_dataset(ncFile)
    ds = ds.sel(lat=slice(area[2], area[3]), lon=slice(area[0], area[1]))
    updated_times = np.asarray(pd.date_range(start=str(year_star)+'/01/01', end=str(year_end+1)+'/01/01', freq='1H'))[:-1]
    ds["time"] = ("time", updated_times)
    
    # turn hourly data into daily for speed of existing code
    ds = ds.resample(time='1D').mean()
    
    try:
        ds = ds.rename({"longitude": "lon", "latitude": "lat"})
    except:
        pass
        
    # keeping values at fixed float length
    ds = ds.assign_coords(
        lon=np.round(ds.lon.astype(float), 5), lat=np.round(ds.lat.astype(float), 5)
    )
    return ds