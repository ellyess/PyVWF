import xarray as xr
import numpy as np
import pandas as pd

import time
import utm
from calendar import monthrange
import datetime

from vwf.extras import (
    add_times,
    add_time_res
)

from vwf.simulation import simulate_wind

##############################################################################
# MERRA 2 STUFF PLEASE DELETE THIS 
def prep_merra2_method_1(year_star, year_end):
    """
    Reading Iain's preprepped MERRA 2 Az file and selecting desired location.
    In future this will be replaced with my own Az function.
    """
    
    area = [7., 16, 54.3, 58]    
    ncFile = '../../ReanalysisData/merra2/DailyAZ/'+str(2020)+'-'+str(2020)+'_dailyAz_ian.nc'
    ds = xr.open_dataset(ncFile)
    ds = ds.sel(lat=slice(area[2], area[3]), lon=slice(area[0], area[1]))
    updated_times = np.asarray(pd.date_range(start=str(year_star)+'/01/01', end=str(year_end+1)+'/01/01', freq='1H'))[:-1]
    ds["time"] = ("time", updated_times)
    ds = ds.resample(time='1D').mean()
    return _rename_and_clean_coords(ds, False)
    
    
#################################################################

def prep_era5(train=False):
    """
    Reading a saved ERA5 file with 100m wind speeds and fsr.
    changing names and converting wind speed components into wind speed.
    """
    # Load the corresponding raw ERA5 file
    if train == True:
        ds = xr.open_mfdataset('data/reanalysis/train/*.nc')
    else:
        ds = xr.open_mfdataset('data/reanalysis/test/*.nc')
    ds = ds.compute() # this allows it to not be dask chunks
    
    ds["wnd100m"] = np.sqrt(ds["u100"] ** 2 + ds["v100"] ** 2).assign_attrs(
        units=ds["u100"].attrs["units"], long_name="100 metre wind speed"
    )
    
    ds = ds.drop_vars(["u100", "v100"])
    ds = ds.rename({"fsr": "roughness"})
    
    # turn hourly data into daily for speed of existing code
    ds = ds.resample(time='1D').mean()
    try:
        ds = ds.rename({"longitude": "lon", "latitude": "lat"})
    except:
        pass
        
    ds = ds.assign_coords(
        lon=np.round(ds.lon.astype(float), 5), lat=np.round(ds.lat.astype(float), 5)
    )
    return ds

def prep_obs(country, train=False, *args):
    
    if country == "DK":
        """
        For Denmark's data there had to be a lot of manual manipulation of the excel file. 
        I had to manually match the turbines that exist in the power curves file, with Denmarks naming convention then match it to the ID's. 
        anlaeg.xlsx is the raw file and match_turb_dk.xlsx is where the matching is done.
        After this we are required to fill in missing turbine matches and also convert the coordinate system.
        We also produce the observational data which is again manually seperated into yearly sheets from a megasheet for the years we desire.
        As the observational data is power output we converted that to capacity factor with the matched turbines.
        the ID's here are the gsrn ID
        """
        ##############################
        # producing turb_info
        ##############################
        
        # reading in the messy denmark turbine info that we have matched
        df = pd.read_excel('data/wind_data/DK/raw/match_turb_dk.xlsx')
        columns = ['Turbine identifier (GSRN)','Capacity (kW)','X (east) coordinate\nUTM 32 Euref89','Y (north) coordinate\nUTM 32 Euref89','Hub height (m)', 'Date of original connection to grid', 'turb_match']
        df = df[columns]
        rename_col = ['ID','capacity','x_east_32','y_north_32','height', 'date', 'model']
        df.columns = rename_col
        df = df.dropna()

        # matching modelless turbines with closest model via capacity
        metadata = pd.read_csv('data/turbine_info/models.csv')
        metadata = metadata.sort_values('capacity')

        df['model'][df['model'] == 0] = np.nan
        df['capacity'] = df['capacity'].astype(int)
        df = df.sort_values('capacity').reset_index(drop=True)
        df.loc[df['model'].isna(), 'model'] = pd.merge_asof(df, metadata, left_on=["capacity"], right_on=["capacity"], direction="nearest")['model_y']

        # convert coordinate system
        def rule(row):
            lat, lon = utm.to_latlon(row["x_east_32"], row["y_north_32"], 32, 'W')
            return pd.Series({"lat": lat, "lon": lon})

        df = df.merge(df.apply(rule, axis=1), left_index= True, right_index= True)
        df = df[['ID','capacity','lat','lon','height', 'date', 'model']]
        df['ID'] = df['ID'].astype(str)
        print("Number of observed turbines/farms before preprocessing: ", len(df))
        turb_info = df.drop(df[df['height'] < 1].index).reset_index(drop=True)

        ##############################
        # producing obs_cf
        ##############################

        # Load observation data and slice the observed CF for chosen years
        if train == True:
            year_star = 2015 # start year of training period
            year_end = 2019 # end year of training period

            appended_data = []
            for i in range(year_star, year_end+1):
                data = pd.read_excel('data/wind_data/DK/observation/Denmark_'+str(i)+'.xlsx')
                data = data.iloc[3:,np.r_[0:1, 3:15]] # the slicing done here is file dependent please consider this when other files are used
                data.columns = ['ID','1','2','3','4','5','6','7','8','9','10','11','12']
                data['ID'] = data['ID'].astype(str)
                data = data.reset_index(drop=True)
                data['year'] = i
    
                appended_data.append(data[:-1])
    
            obs_gen = pd.concat(appended_data).reset_index(drop=True)
            obs_gen.columns = [f'obs_{i}' if i not in ['ID', 'year'] else f'{i}' for i in obs_gen.columns]

            # converting obs_gen into obs_cf by turning power into capacity factor
            df = pd.merge(obs_gen, turb_info[['ID', 'capacity']],  how='left', on=['ID'])
            df = df.dropna().reset_index(drop=True)

            def daysDuringMonth(yy, m):
                result = []    
                [result.append(monthrange(y, m)[1]) for y in yy]        
                return result
    
            for i in range(1,13):
                df['obs_'+str(i)] = df['obs_'+str(i)]/(((daysDuringMonth(df.year, i))*df['capacity'])*24)

        else:
            year_test = args[0]
            data = pd.read_excel('data/wind_data/DK/observation/Denmark_'+str(year_test)+'.xlsx')
            data = data.iloc[3:,np.r_[0:1, 3:15]] # the slicing done here is file dependent please consider this when other files are used
            data.columns = ['ID','1','2','3','4','5','6','7','8','9','10','11','12']
            data['ID'] = data['ID'].astype(str)
            
            obs_gen = data.reset_index(drop=True)
            obs_gen.columns = [f'obs_{i}' if i not in ['ID'] else f'{i}' for i in obs_gen.columns]

            # converting obs_gen into obs_cf by turning power into capacity factor
            df = pd.merge(obs_gen, turb_info[['ID', 'capacity']],  how='left', on=['ID'])
            df = df.dropna().reset_index(drop=True)
    
            for i in range(1,13):
                df['obs_'+str(i)] = df['obs_'+str(i)]/(((monthrange(year_test, i)[1])*df['capacity'])*24)

        # need to make sure to remove turbines giving a capacity factor greater than 1
        df = df.drop(['capacity'], axis=1).reset_index(drop=True)
        df['cf_max'] = df.iloc[:,1:13].max(axis=1)
        df = df.drop(df[df['cf_max'] > 1].index)
        df['cf_min'] = df.iloc[:,1:13].min(axis=1)
        
        if train == True:
            df = df.drop(df[df['cf_min'] <= 0.01].index)
            
        df['cf_mean'] = df.iloc[:,1:13].mean(axis=1)
        df = df.drop(df[df['cf_mean'] <= 0.01].index)
        obs_cf = df.drop(['cf_mean', 'cf_max' , 'cf_min'], axis=1).reset_index(drop=True)
        

        if train == True:
            obs_cf = obs_cf.loc[obs_cf['ID'].isin(turb_info['ID'])].reset_index(drop=True)
            obs_cf = obs_cf[obs_cf.groupby('ID').ID.transform('count') == ((year_end-year_star)+1)].reset_index(drop=True)
            obs_cf.columns = ['ID','1','2','3','4','5','6','7','8','9','10','11','12','year']
            
            turb_info = turb_info.loc[turb_info['ID'].isin(obs_cf['ID'])].reset_index(drop=True)
            # turb_info.to_csv('data/wind_data/DK/train_turb_info.csv', index = None)
            
            obs_cf = obs_cf.melt(id_vars=["ID", "year"], 
                            var_name="month", 
                            value_name="obs")

            # obs_cf.to_csv('data/wind_data/DK/train_obs_cf.csv', index = None)

        else:
            # some random stuff to make it easier to plot for research
            dates = np.arange(str(year_test)+'-01', str(year_test+1)+'-01', dtype='datetime64[M]')
            cols = dates.tolist()
            obs_cf.columns = ['ID'] + cols
            obs_cf = obs_cf.loc[obs_cf['ID'].isin(turb_info['ID'])]

            turb_info = turb_info.loc[turb_info['ID'].isin(obs_cf['ID'])].reset_index(drop=True)
            obs_cf = obs_cf.set_index('ID').transpose().rename_axis('time').reset_index()
            # turb_info.to_csv('data/wind_data/DK/'+str(year_test)+'turb_info.csv', index = None)
            # obs_cf.to_csv('data/wind_data/DK/'+str(year_test)+'obs_cf.csv', index = None)

        print("Number of valid observed turbines/farms: ", len(turb_info))
        
        return obs_cf, turb_info


def merge_gen_cf(reanalysis, obs_cf, turb_info, powerCurveFile):
    
    sim_ws, sim_cf = simulate_wind(reanalysis, turb_info, powerCurveFile)
    sim_cf = sim_cf.groupby(pd.Grouper(key='time',freq='M')).mean().reset_index()
    sim_cf = sim_cf.melt(id_vars=["time"], 
                    var_name="ID", 
                    value_name="sim")
    sim_cf = add_times(sim_cf)
    sim_cf = add_time_res(sim_cf)


    sim_cf['ID'] = sim_cf['ID'].astype(str)
    sim_cf['month'] = sim_cf['month'].astype(int)
    sim_cf['year'] = sim_cf['year'].astype(int)
    obs_cf['ID'] = obs_cf['ID'].astype(str)
    obs_cf['month'] = obs_cf['month'].astype(int)
    obs_cf['year'] = obs_cf['year'].astype(int)
    
    gen_cf = pd.merge(sim_cf, obs_cf, on=['ID', 'month', 'year'], how='left')
    gen_cf = gen_cf.drop(['time'], axis=1).reset_index(drop=True)
    
    return gen_cf