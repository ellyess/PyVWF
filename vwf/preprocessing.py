import xarray as xr
import numpy as np
import pandas as pd
import difflib

import time
import utm
from calendar import monthrange
import datetime

from vwf.simulation import simulate_wind

from itertools import product
from random import sample

def prep_merra2(country):
    """
    Reading Iain's preprepped MERRA 2 Az file and selecting desired location.
    In future this will be replaced with my own Az function.
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
    
def prep_era5(country, train=False):
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
    
    ds = ds.drop_vars(["u100", "v100"])
    ds = ds.rename({"fsr": "roughness"})
    
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
    
    
def add_models(df):
    """
    Assign model names to input turbines.
    
    Using our collection of models to assign power curves to observational data,
    matching is done via most similar power density, using the manufacturer if
    possible.

        Args:
            df (pandas.DataFrame): dataframe with input turbine metadata
            
        Returns:
            df (pandas.DataFrame): input df with added column called model
    """

    models = pd.read_csv('input/models.csv')
    models['model'] = models['model'].astype(pd.StringDtype())
    models['manufacturer'] = models['manufacturer'].str.lower()

    print("Total observed turbines/farms before conditions: ", len(df))
    
    # removing turbines that are unrealistic
    df = df.drop(df[df['height'] < 1].index).reset_index(drop=True)
    
    df['capacity'] = df['capacity'].astype(float)
    df['p_density'] = (df['capacity']*1000) / (np.pi * (df['diameter']/2)**2)
    df['capacity'] = df['capacity'].astype(int)
    df['ID'] = df['ID'].astype(str)
    
    # merging by manufacturer and power density if available
    if 'manufacturer' in df:
        df['manufacturer'] = df['manufacturer'].astype(pd.StringDtype())
        df['manufacturer'] = df['manufacturer'].str.lower()

        df = df.merge(models
            .assign(match=models['manufacturer'].apply(lambda x: difflib.get_close_matches(x, df['manufacturer'],cutoff=0.3,n=100)))
            .explode('match').drop_duplicates(),
            # .explode('manufacturer'),
            left_on=['manufacturer'], right_on=['match'],
            how="outer"
        )
        df = df.dropna(subset=['ID'])
        
        df = (
            df.assign(
                closest= np.abs(df['p_density_x'] - df['p_density_y'])
            )
            .sort_values("closest")
            .drop_duplicates(subset=["ID"], keep="first")
        )
        # allowing for better power density match if manufaturer model is not close enough
        df['model'].where(df['closest'] < 1, np.nan, inplace=True)
        df = df.drop(['diameter_y', 'p_density_y', 'offshore','manufacturer_y', 'capacity_y','match','closest','manufacturer_x'], axis=1)

    if 'type' in df.columns:
        df.columns = ['ID','capacity','diameter','height','lon','lat','type','p_density','model']
    else:
        df.columns = ['ID','capacity','diameter','height','lon','lat','p_density','model']
        df['type'] = 'onshore'
        
    df = df[['ID','type','capacity','diameter','height','lon','lat','model', 'p_density']]
    
    # matching on closest power density with a given tolerance
    df = df.sort_values('p_density').reset_index(drop=True)
    models = models.sort_values('p_density')
    df.loc[df['model'].isna(), 'model'] = pd.merge_asof(df, models, on='p_density', direction="nearest",tolerance=100)['model_y']

    df = df.dropna(subset=['model'])
    df = df.sort_values('ID').reset_index(drop=True)
    
    return df


def prep_country(country, year_test=None):
    """
    Country specific preprocessing of observational data.
    
    Prepping the relevant countries observational data into a format that can
    be managed in `prep_obs()` each country will be unique here. Produce turb_info 
    then obs_gen.

        Args:
            country (str): country code e.g. Denmark "DK"
            
        Returns:
            obs_gen (pandas.DataFrame): obs_gen which is a time series of the power generated by a turbine
            turb_info (pandas.DataFrame): consists of the turbines metadata; latitude, longitude, capacity, height, rotor diameter and model
    """

    if year_test != None:
        train = False
    else:
        train = True
        
    # for Denmark the metadata of existing turbines exist on anlaeg.xlsx,
    # sourced from Denmark Energy Agency. DK_md.csv is a tidied version.
    if country == "DK":
        # producing turb_info
        dk_md = pd.read_csv('input/country-data/DK/observations/DK_md.csv')
        # selecting relevent columns
        columns = ['Turbine identifier (GSRN)',
                'Manufacture','Capacity (kW)',
                'Rotor diameter (m)','Hub height (m)',
                'X (east) coordinate\nUTM 32 Euref89',
                'Y (north) coordinate\nUTM 32 Euref89', 
                'Type of location']
        dk_md = dk_md[columns]
        dk_md.columns = ['ID','manufacturer','capacity','diameter','height','x_east_32','y_north_32', 'type']

        # onshore or offshore
        dk_md['type'] = dk_md['type'].str.lower()
        dk_md.loc[dk_md['type'] == 'land', 'type'] = 'onshore'
        dk_md.loc[dk_md['type'] == 'hav', 'type'] = 'offshore'
        
        # convert coordinate system
        dk_md['x_east_32'] = pd.to_numeric(dk_md['x_east_32'],errors = 'coerce')
        dk_md['y_north_32'] = pd.to_numeric(dk_md['y_north_32'],errors = 'coerce')
        dk_md = dk_md.dropna(subset=['capacity', 'diameter', 'x_east_32','y_north_32']).reset_index(drop=True)
        
        def rule(row):
            lat, lon = utm.to_latlon(row["x_east_32"], row["y_north_32"], 32, 'W')
            return pd.Series({"lat": lat, "lon": lon})
        
        dk_md = dk_md.merge(dk_md.apply(rule, axis=1), left_index= True, right_index= True)
        
        dk_md = dk_md[['ID','manufacturer','capacity','diameter','height','lon','lat', 'type']]
        # getting first word of manufacturer name
        dk_md['manufacturer'] = dk_md['manufacturer'].str.split(' ').str[0]
        
        turb_info = add_models(dk_md)
    
        # producing obs_gen
        # load observation data and slice the observed power for chosen years
        if train == True:
            year_star = 2015 # start year of training period
            year_end = 2019 # end year of training period
            
        else: # this is for testing I use year_star and end for the sake of keeping code same
            year_star = year_test 
            year_end = year_test 
            
        appended_data = []
        for i in range(year_star, year_end+1):
            data = pd.read_excel('input/country-data/DK/observations/Denmark_'+str(i)+'.xlsx')
            data = data.iloc[3:,np.r_[0:1, 3:15]] # the slicing done here is file dependent please consider this when other files are used
            data.columns = ['ID','1','2','3','4','5','6','7','8','9','10','11','12']
            data['ID'] = data['ID'].astype(str)
            data = data.reset_index(drop=True)
            data['year'] = i
            appended_data.append(data[:-1])

        obs_gen = pd.concat(appended_data).reset_index(drop=True)
        obs_gen = obs_gen.fillna(0)

    # Germany data is sourced from Iain Staffell
    elif country == "DE":
        # producing turb_info
        de_geo = pd.read_csv('input/country-data/DE/observations/geolocate.germany.csv') # contains the postcode and lat lon
        de_md = pd.read_csv('input/country-data/DE/observations/DE_md.csv') # contains turbine metda data
        
        # selecting relevent columns
        de_md = de_md[['V1','Manufacturer','kW','Rotor..m.','Tower..m.']]
        de_md.columns = ['ID','manufacturer','capacity','diameter','height']
        de_md['postcode'] = de_md['ID'].astype(str).str[:5].astype(int)
    
        # geolocating by postcode
        de_md = pd.merge(de_md, de_geo[['postcode','lon','lat']], on='postcode', how='left')
        de_md = de_md.drop(["postcode"], axis=1)
        de_md = de_md.dropna(subset=['capacity', 'diameter', 'lon', 'lat']).reset_index(drop=True)
        
        turb_info = add_models(de_md)
        
        # producing obs_gen
        if train == True:
            year_star = 2011 # start year of training period
            year_end = 2014 # end year of training period
            
        else: # this is for testing I use year_star and end for the sake of keeping code same
            year_star = year_test 
            year_end = year_test 
            
        # load observation data and slice the observed power for chosen years
        de_data = pd.read_csv('input/country-data/DE/observations/DE_data.csv')
        de_data = de_data.loc[(de_data["Year"] >= year_star) & (de_data["Year"] <= year_end)].drop(['Downtime'], axis=1).reset_index(drop=True)   
        de_data.columns = ['ID','year','month','output']
        de_data = de_data.dropna(subset=['ID', 'year', 'month'])
        obs_gen = de_data.pivot(index=['ID','year'], columns='month', values='output').reset_index()
        obs_gen = obs_gen.fillna(0)
    
    # turning power into capacity factor
    obs_gen.columns = [f'obs_{i}' if i not in ['ID', 'year'] else f'{i}' for i in obs_gen.columns]
    obs_gen = obs_gen.merge(turb_info[['ID', 'capacity']], how='left', on=['ID'])
    obs_gen = obs_gen.dropna().reset_index(drop=True)

    def daysDuringMonth(yy, m):
        result = []    
        [result.append(monthrange(y, m)[1]) for y in yy]        
        return result

    for i in range(1,13):
        obs_gen['obs_'+str(i)] = obs_gen['obs_'+str(i)]/(((daysDuringMonth(obs_gen.year, i))*obs_gen['capacity'])*24)
        
    obs_gen = obs_gen.drop(['capacity'], axis=1)

    return obs_gen, turb_info
            
    
def prep_obs(country, year_test=None, remove=None, interp=None, set_turb=None):
    """
    Preprocess the turbines/farms found in the observed data.
    
    Prepares the observational data (obs_cf and turb_info) for the desired country
    used in the model. Converts observed power generation into observed CF and ensures
    that all turbines in the data are acceptable.

        Args:
            country (str): country code e.g. Denmark "DK"
            
        Returns:
            obs_gen (pandas.DataFrame): obs_gen which is a time series of the power generated by a turbine
            turb_info (pandas.DataFrame): consists of the turbines metadata; latitude, longitude, capacity, height, rotor diameter and model
    """
    if year_test != None:
        train = False
    else:
        train = True
        
    df, turb_info = prep_country(country, year_test)

    ## conditions to remove turbines from the data
    # cf can't be greater than 100%
    df['cf_max'] = df[df.columns[df.columns.str.startswith('obs')]].max(axis=1)
    df = df.drop(df[df['cf_max'] > 1].index)
    df = df.drop('cf_max', axis=1)
    
    # case exists for Denmark solely, develop a method to consider the weight of missing data
    # remove any turbines that have cf of 0 at any point
    if (train == True) & (country == 'DK'):
        df['cf_min'] = df[df.columns[df.columns.str.startswith('obs')]].min(axis=1)
        df = df.drop(df[df['cf_min'] <= 0.01].index)
        df = df.drop('cf_min', axis=1)
    
        
    df = df.replace(0, np.nan)
    # turbine should atleast produce greater than 1% power
    df['cf_mean'] = df[df.columns[df.columns.str.startswith('obs')]].mean(axis=1)
    df = df.drop(df[df['cf_mean'] <= 0.01].index)
    df = df.drop(['cf_mean'], axis=1)

    obs_cf = df.copy()

    # further rules for the training data and formatting
    if train == True:
        year_star = obs_cf.year.min()
        year_end = obs_cf.year.max()
        
        obs_cf = obs_cf[obs_cf.groupby('ID').ID.transform('count') == ((year_end-year_star)+1)].reset_index(drop=True) # turbines should exist the entire period
        obs_cf = obs_cf[['ID','year','obs_1','obs_2','obs_3','obs_4','obs_5','obs_6','obs_7','obs_8','obs_9','obs_10','obs_11','obs_12']] # reordering columns
        obs_cf.columns = ['ID','year','1','2','3','4','5','6','7','8','9','10','11','12'] # renaming columns
        
        obs_cf = obs_cf.loc[obs_cf['ID'].isin(turb_info['ID'])].reset_index(drop=True) # keeping data that has a turbine match
        obs_cf = obs_cf.melt(
            id_vars=["ID", "year"], 
            var_name="month", 
            value_name="obs"
        )
        
        if remove != None:
            obs_cf['obs'] = obs_cf['obs'].sample(frac=(1-remove), random_state=42)
        
        obs_cf['month'] = obs_cf['month'].astype(int)
        obs_cf['year'] = obs_cf['year'].astype(int)

        if interp != None:
            obs_cf = interp_nans(obs_cf, interp)
        
        turb_info = turb_info.loc[turb_info['ID'].isin(obs_cf['ID'])].reset_index(drop=True)

        # setting turbine model to fixed turbine just for testing in data
        if set_turb != None:
            turb_info['model'] = set_turb #'GE.1.5se' # 'Vestas.V66.2000'

    
    # formatting for testing
    else:
        dates = np.arange(str(year_test)+'-01', str(year_test+1)+'-01', dtype='datetime64[M]')
        cols = dates.tolist()
        obs_cf = obs_cf.drop('year', axis=1)
        obs_cf.columns = ['ID'] + cols
        
        obs_cf = obs_cf.loc[obs_cf['ID'].isin(turb_info['ID'])].reset_index(drop=True)
        turb_info = turb_info.loc[turb_info['ID'].isin(obs_cf['ID'])].reset_index(drop=True)
        
        if set_turb != None:
            turb_info['model'] = set_turb #'GE.1.5se' # 'Vestas.V66.2000'
        
        obs_cf = obs_cf.set_index('ID').transpose().rename_axis('time').reset_index()
    
    print("Number of valid observed turbines/farms: ", len(turb_info))
    
    return obs_cf, turb_info

def interp_nans(df, limit):
    data = df.sort_values(['ID','year']).groupby(['ID']).apply(
        lambda group: group.interpolate(
            method='linear',
            limit=limit, 
            limit_direction='both',
    )).reset_index(drop=True).sort_values(['year','month'])
    return data
    
def merge_gen_cf(reanalysis, obs_cf, turb_info, powerCurveFile):
    
    sim_ws, sim_cf = simulate_wind(reanalysis, turb_info, powerCurveFile)
    sim_cf = sim_cf.groupby(pd.Grouper(key='time',freq='M')).mean().reset_index()
    sim_cf = sim_cf.melt(id_vars=["time"], 
                    var_name="ID", 
                    value_name="sim")
    sim_cf = add_times(sim_cf)
    sim_cf = add_time_res(sim_cf)

    sim_cf['ID'] = sim_cf['ID'].astype(str)
    obs_cf['ID'] = obs_cf['ID'].astype(str)
    
    gen_cf = pd.merge(sim_cf, obs_cf, on=['ID', 'month', 'year'], how='left')
    gen_cf = gen_cf.drop(['time'], axis=1).reset_index(drop=True)
    
    return gen_cf
    
def add_times(df):
    """
    Add columns to identify year and month.
    """
    df['year'] = pd.DatetimeIndex(df['time']).year
    df['month'] = pd.DatetimeIndex(df['time']).month
    df.insert(1, 'year', df.pop('year'))
    df.insert(2, 'month', df.pop('month'))
    df['month'] = df['month'].astype(int)
    df['year'] = df['year'].astype(int)
    return df
    
def add_time_res(df):
    """
    Add columns to identify time resolutions.
    """
    df.loc[df['month'] == 1, ['bimonth','season']] = ['1/6', 'winter']
    df.loc[df['month'] == 2, ['bimonth','season']] = ['1/6', 'winter']
    df.loc[df['month'] == 3, ['bimonth','season']] = ['2/6', 'spring']
    df.loc[df['month'] == 4, ['bimonth','season']] = ['2/6', 'spring']
    df.loc[df['month'] == 5, ['bimonth','season']] = ['3/6', 'spring']
    df.loc[df['month'] == 6, ['bimonth','season']] = ['3/6', 'summer']
    df.loc[df['month'] == 7, ['bimonth','season']] = ['4/6', 'summer']
    df.loc[df['month'] == 8, ['bimonth','season']] = ['4/6', 'summer']
    df.loc[df['month'] == 9, ['bimonth','season']] = ['5/6', 'autumn']
    df.loc[df['month'] == 10, ['bimonth','season']] = ['5/6', 'autumn']
    df.loc[df['month'] == 11, ['bimonth','season']] = ['6/6', 'autumn']
    df.loc[df['month'] == 12, ['bimonth','season']] = ['6/6', 'winter']
    df['yearly'] = '1/1'
    return df