import xarray as xr
import numpy as np
import pandas as pd
from scipy import interpolate

from vwf.extras import add_times

def extrapolate_wind_speed(ds, turbine_info):
 
    # calculating wind speed from era5 variables
    w = ds.wnd100m * (
        np.log(ds.height/ ds.roughness) / np.log(100 / ds.roughness))

    w = w.where(w > 0 , 0)
    w = w.where(w < 40 , 40)
    
    # creating coordinates to spatially interpolate to
    lat =  xr.DataArray(turbine_info['latitude'], dims='turbine', coords={'turbine':turbine_info['ID']})
    lon =  xr.DataArray(turbine_info['longitude'], dims='turbine', coords={'turbine':turbine_info['ID']})
    height =  xr.DataArray(turbine_info['height'], dims='turbine', coords={'turbine':turbine_info['ID']})

    # spatial interpolating to turbine positions
    wnd_spd = w.interp(
            x=lon, y=lat, height=height,
            kwargs={"fill_value": None})
    
    return wnd_spd


def speed_to_power(speed_frame, turbine_info, powerCurveFile, train=False): 
    x = powerCurveFile['data$speed']
    
    if train == True:
        turb_name = turbine_info.loc[turbine_info['ID'] == speed_frame.turbine.data, 'turb_match']
        y = powerCurveFile[turb_name].to_numpy().flatten()
        f2 = interpolate.Akima1DInterpolator(x, y)
        
        return f2(speed_frame.data)

    else:
        power_frame = speed_frame.copy()
        for i in range(2, len(power_frame.columns)+1):          
            speed_single = power_frame.iloc[:,i-1]
            turb_name = turbine_info.loc[turbine_info['ID'] == speed_single.name, 'turb_match']           
            y = powerCurveFile[turb_name].to_numpy().flatten()
            f2 = interpolate.Akima1DInterpolator(x, y)
            power_frame.iloc[:,i-1] = f2(speed_single)

        return power_frame


def simulate_wind(ds, turbine_info, powerCurveFile, time_res='month',train=False, bias_correct=False, *args):
    all_heights = np.sort(turbine_info['height'].unique())
    ds = ds.assign_coords(
        height=('height', all_heights))

    wnd_spd = extrapolate_wind_speed(ds, turbine_info)
    
    if train == True:
        
        scalar, offset = args
        # wnd_spd = (wnd_spd + offset) * scalar # equation 1 
        wnd_spd = (wnd_spd * scalar) + offset # equation 2
        wnd_spd = wnd_spd.where(wnd_spd > 0 , 0)
        wnd_spd = wnd_spd.where(wnd_spd < 40 , 40)
        
        gen_power = speed_to_power(wnd_spd, turbine_info, powerCurveFile, train)
        
        return np.mean(gen_power)

    else:
        
        # reformatting to include turbine ID for easier merging later
        speed_frame = wnd_spd.to_pandas().reset_index()
        gen_speed = speed_frame

        speed_frame = speed_frame.melt(id_vars=["time"], 
                        var_name="ID", 
                        value_name="speed")

        speed_frame = add_times(speed_frame)
         
        if bias_correct == True:
            bias_factors = args[0]
    
            gen_speed = pd.merge(speed_frame, turbine_info[['ID', 'cluster']], on='ID', how='left')  
            
            if time_res == 'year':
                time_factors = bias_factors.groupby(['cluster'], as_index=False).agg({'scalar': 'mean', 'offset': 'mean'})
                gen_speed = pd.merge(gen_speed, time_factors[['cluster','scalar', 'offset']],  how='left', on=['cluster'])
            
            else:        
                gen_speed = pd.merge(gen_speed, bias_factors[['cluster', 'month','two_month','season']],  how='left', on=['cluster', 'month'])
                time_factors = bias_factors.groupby(['cluster',time_res], as_index=False).agg({'scalar': 'mean', 'offset': 'mean'})
                gen_speed = pd.merge(gen_speed, time_factors[['cluster', time_res, 'scalar', 'offset']],  how='left', on=['cluster', time_res])
    
            # gen_speed['speed'] = (gen_speed.speed + gen_speed.offset) * gen_speed.scalar # equation 1 
            gen_speed['speed'] = (gen_speed.speed * gen_speed.scalar) + gen_speed.offset # equation 2
            
            gen_speed = gen_speed.pivot(index=['time'], columns='ID', values='speed').reset_index()
            gen_power = speed_to_power(gen_speed,turbine_info[['ID','turb_match']], powerCurveFile)
            
            return gen_speed, gen_power
    
        else:
            gen_power = speed_to_power(gen_speed,turbine_info[['ID','turb_match']], powerCurveFile)
            
            return gen_speed, gen_power
