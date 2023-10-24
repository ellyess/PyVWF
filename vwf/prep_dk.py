# For Denmark's data there had to be a lot of manual manipulation of the excel file. 
# I had to manually match the turbines that exist in the power curves file, with Denmarks naming convention then match it to the ID's. 
# anlaeg.xlsx is the raw file and match_turb_dk.xlsx is where the matching is done.
# After this we are required to fill in missing turbine matches and also convert the coordinate system.
# We also produce the observational data which is again manually seperated into yearly sheets from a megasheet for the years we desire.
# As the observational data is power output we converted that to capacity factor with the matched turbines.
# the ID's here are the gsrn ID

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utm
from calendar import monthrange


################
### producing turb_info
################
# reading in the messy denmark turbine info that we have matched
df = pd.read_excel('../data/wind_data/DK/match_turb_dk.xlsx')
columns = ['Turbine identifier (GSRN)','Capacity (kW)','X (east) coordinate\nUTM 32 Euref89','Y (north) coordinate\nUTM 32 Euref89','Hub height (m)', 'Date of original connection to grid', 'turb_match']
df = df[columns]
rename_col = ['ID','capacity','x_east_32','y_north_32','height', 'date', 'model']
df.columns = rename_col
df = df.dropna()

# matching modelless turbines with closest model via capacity
metadata = pd.read_csv('../data/turbine_info/models.csv')
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
turb_info = df.drop(df[df['height'] < 1].index).reset_index(drop=True)


################
### producing obs_cf
################

def prep_obs_denmark(year_star, year_end):
    """
    Loading the observation data for the training years, and cleaning it.
    """
    # Load observation data and slice the observed CF for chosen year
    appended_data = []
    for i in range(year_star, year_end+1):
        data = pd.read_excel('../data/wind_data/DK/observation/Denmark_'+str(i)+'.xlsx')
        data = data.iloc[3:,np.r_[0:1, 3:15]] # the slicing done here is file dependent please consider this when other files are used
        data.columns = ['ID','1','2','3','4','5','6','7','8','9','10','11','12']
        data['ID'] = data['ID'].astype(str)
        data = data.reset_index(drop=True)
        data['year'] = i

        appended_data.append(data[:-1])

    obs_gen = pd.concat(appended_data).reset_index(drop=True)
    obs_gen.columns = [f'obs_{i}' if i not in ['ID', 'year'] else f'{i}' for i in obs_gen.columns]
    
    return obs_gen

obs_gen = prep_obs_denmark(2015,2020)

# converting obs_gen into obs_cf by turning power into capacity factor
df = pd.merge(obs_gen, turb_info[['ID', 'capacity']],  how='left', on=['ID'])
df = df.dropna().reset_index(drop=True)

def daysDuringMonth(yy, m):
    result = []    
    [result.append(monthrange(y, m)[1]) for y in yy]        
    return result

for i in range(1,13):
    df['obs_'+str(i)] = df['obs_'+str(i)]/(((daysDuringMonth(df.year, i))*df['capacity'])*24)

df['cf_mean'] = df.iloc[:,1:13].mean(axis=1)
df = df.drop(df[df['cf_mean'] <= 0.01].index)
obs_cf = df.drop(['capacity','cf_mean'], axis=1).reset_index(drop=True)


turb_info.to_csv('../data/wind_data/DK/turb_info.csv', index = None)
obs_cf.to_csv('../data/wind_data/DK/obs_cf.csv', index = None)



# creating a turb_info_test for 2020
turb_info_test = turb_info.loc[turb_info['ID'].isin(obs_cf[obs_cf['year'] == 2020]['ID'])].reset_index(drop=True)
turb_info_test.to_csv('../data/wind_data/DK/turb_info_test.csv', index = None)