import xarray as xr
import numpy as np
import pandas as pd
from scipy import interpolate

from vwf.extras import (
    add_times,
    add_time_res
)

def simulate_wind_speed(reanalysis, turb_info):
    reanalysis = reanalysis.assign_coords(
        height=('height', turb_info['height'].unique()))
    
    # calculating wind speed from reanalysis dataset variables
    ws = reanalysis.wnd100m * (np.log(reanalysis.height/ reanalysis.roughness) / np.log(100 / reanalysis.roughness))
    
    # creating coordinates to spatially interpolate to
    lat =  xr.DataArray(turb_info['lat'], dims='turbine', coords={'turbine':turb_info['ID']})
    lon =  xr.DataArray(turb_info['lon'], dims='turbine', coords={'turbine':turb_info['ID']})
    height =  xr.DataArray(turb_info['height'], dims='turbine', coords={'turbine':turb_info['ID']})

    # spatial interpolating to turbine positions
    sim_ws = ws.interp(
            lon=lon, lat=lat, height=height,
            kwargs={"fill_value": None})
    
    return sim_ws

def speed_to_power(column, powerCurveFile, turb_info):
    x = powerCurveFile['data$speed']
    turb_name = turb_info.loc[turb_info['ID'] == column.name, 'model']           
    y = powerCurveFile[turb_name].to_numpy().flatten()
    f = interpolate.Akima1DInterpolator(x, y)
    return f(column)

def simulate_wind(reanalysis, turb_info, powerCurveFile, *args): 
    # calculating wind speed from reanalysis data
    # start_time = time.time()
    sim_ws = simulate_wind_speed(reanalysis, turb_info)
    sim_ws = sim_ws.to_pandas()

    if len(args) >= 1: 
        bc_factors = args[0]
        time_res = args[1]
        
        sim_ws = sim_ws.reset_index()
        sim_ws = sim_ws.melt(id_vars=["time"], # adding in turbine ID for merging
            var_name="ID", 
            value_name="ws")


        sim_ws = add_times(sim_ws)
        sim_ws = add_time_res(sim_ws)
        
        sim_ws['month'] = sim_ws['month'].astype(str)
        bc_factors[time_res] = bc_factors[time_res].astype(str)
        # turb_info['ID'] = turb_info['ID'].astype(str)
        # bc_factors['ID'] = bc_factors['ID'].astype(str)
        sim_ws = pd.merge(sim_ws, turb_info[['ID','cluster']], on=['ID'], how='left')
        sim_ws = pd.merge(sim_ws, bc_factors, on=['cluster',time_res], how='left')
        sim_ws['ws'] = (sim_ws.ws * sim_ws.scalar) + sim_ws.offset # equation 2
        sim_ws = sim_ws.pivot(index=['time'], columns='ID', values='ws')

    sim_cf = sim_ws.apply(speed_to_power, args=(powerCurveFile, turb_info), axis=0)

    return sim_ws.reset_index(), sim_cf.reset_index()

def train_simulate_wind(reanalysis, turb_info, powerCurveFile, scalar=1, offset=0): 
    # calculating wind speed from reanalysis data
    sim_ws = simulate_wind_speed(reanalysis, turb_info)
    unc_ws = sim_ws.to_pandas()
    cor_ws = (unc_ws * scalar) + offset
    # converting to power
    cor_cf = cor_ws.apply(speed_to_power, args=(powerCurveFile, turb_info), axis=0)
    return np.mean(cor_cf)