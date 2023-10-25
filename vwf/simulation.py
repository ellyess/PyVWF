import xarray as xr
import numpy as np
import pandas as pd
from scipy import interpolate

from vwf.extras import add_times

def extrapolate_wind_speed(reanal_data, turb_info):
 
    # calculating wind speed from reanalysis dataset variables
    ws = reanal_data.wnd100m * (
        np.log(reanal_data.height/ reanal_data.roughness) / np.log(100 / reanal_data.roughness))
    ws = ws.where(ws > 0 , 0)
    ws = ws.where(ws < 40 , 40)
    
    # creating coordinates to spatially interpolate to
    lat =  xr.DataArray(turb_info['lat'], dims='turbine', coords={'turbine':turb_info['ID']})
    lon =  xr.DataArray(turb_info['lon'], dims='turbine', coords={'turbine':turb_info['ID']})
    height =  xr.DataArray(turb_info['height'], dims='turbine', coords={'turbine':turb_info['ID']})

    # spatial interpolating to turbine positions
    raw_ws = ws.interp(
            x=lon, y=lat, height=height,
            kwargs={"fill_value": None})
    
    return raw_ws


def speed_to_power(sim_ws, turb_info, powerCurveFile, train=False): 
    x = powerCurveFile['data$speed']
    
    if train == True:
        turb_name = turb_info.loc[turb_info['ID'] == sim_ws.turbine.data, 'model']
        y = powerCurveFile[turb_name].to_numpy().flatten()
        f = interpolate.Akima1DInterpolator(x, y)
        return f(sim_ws.data)

    else:
        # identifying the model assigned to this turbine ID to access the power curve
        # and covert the speed into power
        sim_cf = sim_ws.copy()
        for i in range(2, len(sim_cf.columns)+1):          
            speed_single = sim_cf.iloc[:,i-1]
            turb_name = turb_info.loc[turb_info['ID'] == speed_single.name, 'model']           
            y = powerCurveFile[turb_name].to_numpy().flatten()
            f = interpolate.Akima1DInterpolator(x, y)
            sim_cf.iloc[:,i-1] = f(speed_single)
        return sim_cf


def simulate_wind(reanal_data, turb_info, powerCurveFile, time_res='month',train=False, bias_correct=False, *args):
    # grouping reanalysis data to speed up calculation
    all_heights = np.sort(turb_info['height'].unique())
    reanal_data = reanal_data.assign_coords(
        height=('height', all_heights))
    # calculating wind speed from reanalysis data
    raw_ws = extrapolate_wind_speed(reanal_data, turb_info)
    
    if train == True:
        scalar, offset = args
        # raw_ws = (raw_ws + offset) * scalar # equation 1 
        raw_ws = (raw_ws * scalar) + offset # equation 2
        raw_ws = raw_ws.where(raw_ws > 0 , 0)
        raw_ws = raw_ws.where(raw_ws < 40 , 40)
        raw_cf = speed_to_power(raw_ws, turb_info, powerCurveFile, train)
        return np.mean(raw_cf)

    else:
        unc_ws = raw_ws.to_pandas().reset_index()

        if bias_correct == False:
            unc_cf = speed_to_power(unc_ws, turb_info[['ID','model']], powerCurveFile)
            return unc_ws, unc_cf
            
        else:
            bc_factors = args[0] # reading in bias correction factors
            unc_ws = unc_ws.melt(id_vars=["time"], # adding in turbine ID for merging
                var_name="ID", 
                value_name="ws")
            unc_ws = add_times(unc_ws)
            
            # matching the correct resolution bias correction factors
            unc_ws = pd.merge(unc_ws, turb_info[['ID', 'cluster']], on='ID', how='left')  
            if time_res == 'year':
                time_factors = bc_factors.groupby(['cluster'], as_index=False).agg({'scalar': 'mean', 'offset': 'mean'})
                unc_ws = pd.merge(unc_ws, time_factors[['cluster','scalar', 'offset']],  how='left', on=['cluster'])
            
            else:        
                unc_ws = pd.merge(unc_ws, bc_factors[['cluster', 'month','two_month','season']],  how='left', on=['cluster', 'month'])
                time_factors = bc_factors.groupby(['cluster',time_res], as_index=False).agg({'scalar': 'mean', 'offset': 'mean'})
                unc_ws = pd.merge(unc_ws, time_factors[['cluster', time_res, 'scalar', 'offset']],  how='left', on=['cluster', time_res])
    
            # applying the bias correction factors to wind speed
            # cor_ws['speed'] = (unc_ws.ws + unc_ws.offset) * gen_speed.scalar # equation 1 
            unc_ws['ws'] = (unc_ws.ws * unc_ws.scalar) + unc_ws.offset # equation 2
            cor_ws = unc_ws.pivot(index=['time'], columns='ID', values='ws').reset_index()
            cor_cf = speed_to_power(cor_ws,turb_info[['ID','model']], powerCurveFile)
            return cor_ws, cor_cf