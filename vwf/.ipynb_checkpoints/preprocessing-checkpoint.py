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
    
    
def prep_metadata_2020():
    """
    Load in turbine metadata and observed generation data.
    The simulation is done to turbines present in both 2020 and the chosen year
    Because if the turbine is no longer present in 2020 it means it has retired
    Load meta info for all turbines in 2020.
    
    In future please automate this and make it friendly for not just 2020
    """

    turb_meta = pd.read_excel('data/turbine_info/Metadata_2020.xlsx', index_col=None)
    # Data pre-processing
    turb_meta['ID'] = turb_meta['ID'].astype(str)
    turb_meta['CF_mean'] = turb_meta.iloc[:,33:45].mean(axis=1)
    turb_meta = turb_meta.loc[turb_meta['height'] > 1]
    turb_meta = turb_meta.loc[turb_meta['CF_mean'] >= 0.01]
    # Select columns for observed CF
    turb_meta = turb_meta.iloc[:, np.r_[1:2, 4:9, 33:45]]
    turb_meta = turb_meta.reset_index(drop=True)
    turb_meta.columns = ['ID','capacity','longitude','latitude','height','turb_match','obs_1','obs_2','obs_3','obs_4','obs_5','obs_6','obs_7','obs_8','obs_9','obs_10','obs_11','obs_12']
    
    # # Saving observation data of 2020
    # obs_cf = turb_meta.filter(regex='ID|obs_').set_index('ID').T.reset_index(drop=True)
    # updated_times = np.asarray(pd.date_range(start='2020/01/01', end='2021/01/01', freq='1M'))
    # obs_cf.insert(0, 'time', updated_times)
    # obs_cf.to_csv('data/results/denmark_obs_cf.csv', index = None) 
    
    return turb_meta
    
def prep_obs_and_turb_info(turb_meta, year_star, year_end):
    """
    Loading the observation data for the training years, and cleaning it.
    The turbine info for training is also prepared.
    """
    # Load observation data and slice the observed CF for chosen year
    appended_data = []
    for i in range(year_star, year_end+1):
        data = pd.read_excel('data/observation/Denmark_'+str(i)+'.xlsx')
        data = data.iloc[3:,np.r_[0:1, 3:15]] # the slicing done here is file dependent please consider this when other files are used
        data.columns = ['ID','1','2','3','4','5','6','7','8','9','10','11','12']
        data.ID = data['ID'].astype(str)
        data = data.reset_index(drop=True)
        data['year'] = i

        appended_data.append(data[:-1])

    obs_gen = pd.concat(appended_data).reset_index(drop=True)
    obs_gen.columns = [f'obs_{i}' if i not in ['ID', 'year'] else f'{i}' for i in obs_gen.columns]

    turb_info = turb_meta.loc[turb_meta['ID'].isin(obs_gen.ID)].reset_index(drop=True)
    
    return obs_gen, turb_info
    
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