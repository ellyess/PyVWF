def prep_era5(country, train=False, calc_z0=True): # added 10m values to calculate own FSR
    """
    Reading and processing a saved ERA5 file with 100m wind speeds and fsr.

        Args:
            train (boolean): if for training or not
            
        Returns:
            ds (xarray.DataSet): dataset with wind variables on a grid
    """
    # load the corresponding raw ERA5 file
    if train == True:
        ds = xr.open_mfdataset('input/era5/'+country+'/train/*.nc')
    else:
        ds = xr.open_mfdataset('input/era5/'+country+'/test/*.nc')
    ds = ds.compute() # this allows it to not be dask chunks
    
    # converting wind speed components into wind speed
    ds["wnd100m"] = np.sqrt(ds["u100"] ** 2 + ds["v100"] ** 2).assign_attrs(
        units=ds["u100"].attrs["units"], long_name="100 metre wind speed"
    )
    
    if calc_z0 == True:
        ds["wnd10m"] = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2).assign_attrs(
        units=ds["u10"].attrs["units"], long_name="10 metre wind speed"
        )
        ds['roughness'] = ((ds["wnd100m"]*np.log(10))-(ds["wnd10m"]*np.log(100))) / (ds["wnd100m"]-ds["wnd10m"])
        ds['roughness'] = ds['roughness'].where((ds['roughness'] < 0))
        ds['roughness'] = ds['roughness'].bfill('valid_time')
        ds['roughness'] = np.exp(ds['roughness']).assign_attrs(
            units=ds["fsr"].attrs["units"], long_name="calculated surface roughness"
        )
    # ds['roughness'] = np.exp(((ds["wnd100m"]*np.log(10))-(ds["wnd10m"]*np.log(100))) / (ds["wnd100m"]-ds["wnd10m"])).assign_attrs(
    #     units=ds["fsr"].attrs["units"], long_name="calculated surface roughness"
    # )
    
    ds = ds.drop_vars(["u100", "v100","u10", "v10","number","expver"])
    ds = ds.rename({"valid_time": "time"})
    # ds = ds.rename({"fsr": "roughness"})
    
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
    
# def prep_era5(country, train=False):
#     """
#     Reading and processing a saved ERA5 file with 100m wind speeds and fsr.

#         Args:
#             train (boolean): if for training or not
            
#         Returns:
#             ds (xarray.DataSet): dataset with wind variables on a grid
#     """
#     # load the corresponding raw ERA5 file
#     if train == True:
#         ds = xr.open_mfdataset('input/era5/'+country+'/train/*.nc')
#     else:
#         ds = xr.open_mfdataset('input/era5/'+country+'/test/*.nc')
#     ds = ds.compute() # this allows it to not be dask chunks
    
#     # converting wind speed components into wind speed
#     ds["wnd100m"] = np.sqrt(ds["u100"] ** 2 + ds["v100"] ** 2).assign_attrs(
#         units=ds["u100"].attrs["units"], long_name="100 metre wind speed"
#     )
    
#     ds = ds.drop_vars(["u100", "v100"])
#     ds = ds.rename({"fsr": "roughness"})
    
#     # turn hourly data into daily for speed of existing code
#     ds = ds.resample(time='1D').mean()
    
#     try:
#         ds = ds.rename({"longitude": "lon", "latitude": "lat"})
#     except:
#         pass
        
#     # keeping values at fixed float length
#     ds = ds.assign_coords(
#         lon=np.round(ds.lon.astype(float), 5), lat=np.round(ds.lat.astype(float), 5)
#     )
#     return ds