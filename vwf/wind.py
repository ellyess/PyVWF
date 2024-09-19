import xarray as xr
import numpy as np
import pandas as pd
from scipy import interpolate

import vwf.data as data

def interpolate_wind(reanalysis, turb_info):    
    """
    Simulate wind speeds at turbine locations.

    Args:
        reanalysis (xarray.Dataset): reanalysis wind data variables assigned to latitude, longitude and time.
        turb_info (pandas.DataFrame): input turbine metadata.

    Returns:
        xarray.DataArray: time-series of simulated wind speeds for every turbine in turb_info
    """
    # heights are assigned to allow speed to be calculated for every height at every gridpoint
    reanalysis = reanalysis.assign_coords(
        height=('height', turb_info['height'].unique()))
    
    # this is for research purposes and is added for the use of merra 2, can be removed for the code below
    try:
        exception_flag = True
        ws = reanalysis.A * np.log(reanalysis.height / reanalysis.z)
        exception_flag = False

    except:
        pass
        
    finally:
        if exception_flag:
            ws = reanalysis.wnd100m * (np.log(reanalysis.height/ reanalysis.roughness) / np.log(100 / reanalysis.roughness))
        
    # # calculating wind speed from reanalysis dataset variables
    # ws = reanalysis.wnd100m * (np.log(reanalysis.height/ reanalysis.roughness) / np.log(100 / reanalysis.roughness))
    
    # creating coordinates to spatially interpolate to
    lat =  xr.DataArray(turb_info['lat'], dims='turbine', coords={'turbine':turb_info['ID']})
    lon =  xr.DataArray(turb_info['lon'], dims='turbine', coords={'turbine':turb_info['ID']})
    height =  xr.DataArray(turb_info['height'], dims='turbine', coords={'turbine':turb_info['ID']})

    # spatial interpolation to turbine positions
    sim_ws = ws.interp(
            lon=lon, lat=lat, height=height,
            kwargs={"fill_value": None}
            )
    
    # assign models and capacity for future functions
    sim_ws = sim_ws.assign_coords({
        'model':('turbine', turb_info['model']),
        'capacity':('turbine', turb_info['capacity']),
    })
    return sim_ws
    
def simulate_wind(reanalysis, turb_info, powerCurveFile, *args):
    """
    Simulate wind speed and capacity factor, optionally can be corrected.

    Args:
        reanalysis (xarray.Dataset): wind parameters on a grid.
        turb_info (pandas.DataFrame): turbine metadata including height and coordinates.
        powerCurveFile (pandas.DataFrame): capacity factor at increasing wind speeds for different models.
        bc_factors (pandas.DataFrame, optional): correction factor for every cluster and time period in time resolution.
        time_res (str, optional): time resolution to be set.

    Returns:
        sim_ws (pandas.DataFrame): time-series of simulated wind speeds at every turbine in turb_info.
        sim_cf (pandas.DataFrame): time-series of simulated capacity factors of every turbine in turb_info.
    """
    sim_ws = interpolate_wind(reanalysis, turb_info)

    if len(args) >= 1: 
        bc_factors = args[0]
        time_res = args[1]
        sim_ws = correct_wind_speed(sim_ws, time_res, bc_factors, turb_info)

    def speed_to_power(data):
        x = powerCurveFile['data$speed']          
        y = powerCurveFile[data.model[0].data]
        f = interpolate.Akima1DInterpolator(x, y)
        return f(data)
    sim_cf = sim_ws.groupby('model').map(speed_to_power)
    
    return sim_ws.to_pandas().reset_index(), sim_cf.to_pandas().reset_index()

def correct_wind_speed(ds, time_res, bc_factors, turb_info):
    """
    Correct simulated windspeeds with assigned bias correction factors

        Args:
            ds (xarray.Dataset): wind speeds and coordinates
            time_res (str): temporal resolution of the bias correction factors
            bc_factors (pandas.DataFrame): the derived bias correction factors
            turb_info (pandas.DataFrame): turbine metadata including height and coordinates

        Returns:
            np.array: corrected wind speeds for every turbine
    """
    ds = ds.assign_coords({'cluster':('turbine', turb_info['cluster'])})
    df = ds.to_dataframe('unc_ws').reset_index()
    df['year'] = pd.DatetimeIndex(df['time']).year
    df['month'] = pd.DatetimeIndex(df['time']).month
    df = data.add_time_res(df)
    df = df.merge(bc_factors, on=['cluster',time_res],how='left').set_index(['time','turbine'])

    ds = df[['scalar','offset','unc_ws']].to_xarray()
    ds = ds.assign(cor_ws= (ds["unc_ws"] * ds["scalar"]) + ds["offset"])
    ds = ds.assign_coords({'model':('turbine', turb_info['model'])})
    return ds.cor_ws   

def train_simulate_wind(reanalysis, turb_info, powerCurveFile, scalar=1, offset=0):
    """
    Simulate average capacity factor of desired resolution for training.

        Args:
            reanalysis (xarray.Dataset): wind parameters on a grid
            turb_info (pandas.DataFrame): turbine metadata including height and coordinates
            powerCurveFile (pandas.DataFrame): capacity factor at increasing wind speeds for different models
            scalar (float): multiplicative correction factor
            offset (float): additive correction factor

        Returns:
            float: weighted average of simulated CF
    """ 
    unc_ws = interpolate_wind(reanalysis, turb_info)
    cor_ws = (unc_ws * scalar) + offset
    
    def speed_to_power(data):
        x = powerCurveFile['data$speed']
        y = powerCurveFile[data.model[0].data]
        f = interpolate.Akima1DInterpolator(x, y)
        return f(data)
    cor_cf = cor_ws.groupby('model').map(speed_to_power)
    avg_cf = cor_cf.weighted(cor_cf['capacity']).mean()
    return avg_cf.data


    
