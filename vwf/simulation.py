import xarray as xr
import numpy as np
import pandas as pd
from scipy import interpolate

from vwf.extras import (
    add_times,
    add_time_res
)

def simulate_wind_speed(reanalysis, turb_info):
    """
    Simulate wind speeds at turbine locations.

        Args:
            reanalysis (xarray.Dataset): wind parameters on a grid
            turb_info (pandas.DataFrame): turbine metadata including height and coordinates

        Returns:
            sim_ws (xarray.DataArray): time-series of simulated wind speeds at every turbine in turb_info
    """
    # heights are assigned to allow speed to be calculated for every height at every gridpoint
    reanalysis = reanalysis.assign_coords(
        height=('height', turb_info['height'].unique()))
    
    # calculating wind speed from reanalysis dataset variables,
    ws = reanalysis.wnd100m * (np.log(reanalysis.height/ reanalysis.roughness) / np.log(100 / reanalysis.roughness))
    
    # creating coordinates to spatially interpolate to
    lat =  xr.DataArray(turb_info['lat'], dims='turbine', coords={'turbine':turb_info['ID']})
    lon =  xr.DataArray(turb_info['lon'], dims='turbine', coords={'turbine':turb_info['ID']})
    height =  xr.DataArray(turb_info['height'], dims='turbine', coords={'turbine':turb_info['ID']})

    # spatial interpolation to turbine positions
    sim_ws = ws.interp(
            lon=lon, lat=lat, height=height,
            kwargs={"fill_value": None}
            )
    
    # can add models like this
    sim_ws = sim_ws.assign_coords({'model':('turbine', turb_info['model'])})
    
    return sim_ws

def speed_to_power(column, powerCurveFile, turb_info):
    """
    Convert wind speed into capacity factor.

        Args:
            column (pandas.Series): column name is the turbine ID and values are wind speeds
            powerCurveFile (pandas.DataFrame): capacity factor at increasing wind speeds for different models
            turb_info (pandas.DataFrame): turbine metadata including height and coordinates

        Returns:
            f(column) (pandas.Series): the original input with the wind speeds as capacity factor
    """
    x = powerCurveFile['data$speed']
    turb_name = turb_info.loc[turb_info['ID'] == column.name, 'model']           
    y = powerCurveFile[turb_name].to_numpy().flatten()
    f = interpolate.Akima1DInterpolator(x, y)
    return f(column)

def simulate_wind(reanalysis, turb_info, powerCurveFile, *args): 
    """
    Simulate wind speed and capacity factor, optionally can be corrected.

        Args:
            reanalysis (xarray.Dataset): wind parameters on a grid
            turb_info (pandas.DataFrame): turbine metadata including height and coordinates
            powerCurveFile (pandas.DataFrame): capacity factor at increasing wind speeds for different models
            Optional:
                args (pandas.DataFrame): correction factor for every cluster and time period in time resolution

        Returns:
            sim_ws (pandas.DataFrame): time-series of simulated wind speeds at every turbine in turb_info
            sim_cf (pandas.DataFrame): time-series of simulated capacity factors of every turbine in turb_info
    """
    sim_ws = simulate_wind_speed(reanalysis, turb_info)
    sim_ws = sim_ws.to_pandas()

    if len(args) >= 1: 
        bc_factors = args[0]
        time_res = args[1]
        
        # adding in turbine ID for merging
        sim_ws = sim_ws.reset_index()
        sim_ws = sim_ws.melt(
            id_vars=["time"], 
            var_name="ID", 
            value_name="ws"
        )

        sim_ws = add_times(sim_ws)
        sim_ws = add_time_res(sim_ws)
        
        sim_ws['month'] = sim_ws['month'].astype(str)
        bc_factors[time_res] = bc_factors[time_res].astype(str)

        # merging to assign correction factors, then applying them to correct wind speed
        # sim_ws = sim_ws.set_index('ID').join(turb_info.set_index('ID'), how='left').reset_index()
        # sim_ws = sim_ws.set_index(['cluster',time_res]).join(bc_factors.set_index(['cluster',time_res]), how='left')
        sim_ws = pd.merge(sim_ws, turb_info[['ID','cluster']], on=['ID'], how='left')
        sim_ws = pd.merge(sim_ws, bc_factors, on=['cluster',time_res], how='left')
        sim_ws['ws'] = (sim_ws.ws * sim_ws.scalar) + sim_ws.offset 
        sim_ws = sim_ws.pivot(index=['time'], columns='ID', values='ws')

    sim_cf = sim_ws.apply(speed_to_power, args=(powerCurveFile, turb_info), axis=0)
    return sim_ws.reset_index(), sim_cf.reset_index()

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
    # calculating wind speed from reanalysis data
    sim_ws = simulate_wind_speed(reanalysis, turb_info)
    unc_ws = sim_ws.to_pandas()
    cor_ws = (unc_ws * scalar) + offset
    
    # print('unc: ', np.mean(unc_ws), 'cor: ', np.mean(cor_ws))
    
    # converting to power
    cor_cf = cor_ws.apply(speed_to_power, args=(powerCurveFile, turb_info), axis=0)
    
    # adding in turbine ID for merging
    cor_cf = cor_cf.reset_index()
    cor_cf = cor_cf.melt( 
        id_vars=["time"], 
        var_name="ID", 
        value_name="cf"
    )
    cor_cf = pd.merge(cor_cf, turb_info[['ID','capacity']], on=['ID'], how='left')
    return np.average(cor_cf.cf, weights=cor_cf.capacity)