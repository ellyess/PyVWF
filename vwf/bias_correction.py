import xarray as xr
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from calendar import monthrange

from vwf.simulation import train_simulate_wind

def cluster_turbines(num_clu, turb_info):
    """
    Spatially clustering turbines.

        Args:
            num_clu (int): number of clusters
            turb_info (pandas.DataFrame): turbine metadata including height and coordinates

        Returns:
            turb_info (pandas.DataFrame): turbine metadata with assigned cluster column
    """
    # generating the cluster labels 
    kmeans = KMeans(
        init="random",
        n_clusters = num_clu,
        n_init = 10,
        max_iter = 300,
        random_state = 42
    )
    
    lat = turb_info['lat']
    lon = turb_info['lon']
    df = pd.DataFrame(list(zip(lat, lon)), columns =['lat', 'lon'])
    kmeans.fit(df)
    turb_info['cluster'] = kmeans.labels_
    
    return turb_info
    
def closest_cluster(clus_info, turb_info):
    """
    Assign new turbine/farm to closest cluster found in training.

        Args:
            clus_info (pandas.DataFrame): clustered turbine metadata from training
            turb_info (pandas.DataFrame): turbine metadata for new turbines

        Returns:
            turb_info (pandas.DataFrame): turbine metadata with assigned cluster column
    """
    clus_info['ID'] = clus_info['ID'].astype(str)
    turb_info['ID'] = turb_info['ID'].astype(str)
    
    # finding cluster center
    avg = clus_info.groupby(['cluster'], as_index=False)[['lat','lon']].mean()
    turb_info = pd.DataFrame.merge(clus_info[['ID','cluster']], turb_info, on='ID', how='right')

    # find smallest distance between the new turbine and cluster centers
    for i in range(len(turb_info)):
        if np.isnan(turb_info.cluster[i]) == True:
            indx = np.argmin(np.sqrt((avg.lat.values - turb_info.lat[i])**2 + (avg.lon.values - turb_info.lon[i])**2))
            turb_info.cluster[i] = avg.cluster[indx]

    turb_info = turb_info.reset_index(drop=True)
    return turb_info

def train_data(time_res, gen_cf, clus_info):
    """
    Create data for training.
    """
    gen_data = gen_cf.groupby(['year',time_res,'ID'], as_index=False)[['obs','sim']].mean()

    gen_data = pd.merge(gen_data, clus_info[['ID', 'cluster', 'lon', 'lat', 'capacity', 'height', 'model']], on='ID', how='left')
    bias_data = calculate_scalar(time_res, gen_data)

    return bias_data
    
def calculate_scalar(time_res, gen_data):
    """
    Calculate the scalar, multiplicative correction factor.

        Args:
            time_res (str): time resolution we want to simulate
            gen_data (pandas.DataFrame): dataframe with the observed and simulated cf for training period

        Returns:
            turb_info (pandas.DataFrame): turbine metadata with assigned cluster column
    """
    def weighted_avg(group_df, whole_df, values, weights):
        v = whole_df.loc[group_df.index, values]
        w = whole_df.loc[group_df.index, weights]
        return (v * w).sum() / w.sum()
        
    bias_data = gen_data.groupby([time_res, 'cluster', 'year']).agg({
                            "obs": lambda x: weighted_avg(x, gen_data, 'obs', 'capacity'),
                            "sim": lambda x: weighted_avg(x, gen_data, 'sim', 'capacity'),
                            })
        

    bias_data['scalar'] = (0.6 * (bias_data['obs'] / bias_data['sim'])) + 0.2
    bias_data = bias_data.reset_index()
    bias_data.columns = ['time_slice', 'cluster', 'year', 'obs', 'sim', 'scalar']
        
    return bias_data[['year', 'time_slice', 'cluster', 'obs', 'sim', 'scalar']]
    
def find_offset(row, turb_info, reanalysis, powerCurveFile):
    """
    Optimize the offset, additive correction factor.
    
    This function applies the correction factors to the simulated wind speed
    which is converted into capacity factor, this is done repeatedly till an
    offset is calculated that makes the simulated capacity factor match the
    observed capacity factor.

        Args:
            row (pandas.Series): row has the year, cluster and time slice
            turb_info (pandas.DataFrame): turbine metadata including height and coordinates
            reanalysis (xarray.Dataset): wind parameters on a grid
            powerCurveFile (pandas.DataFrame): capacity factor at increasing wind speeds for different models
            
        Returns:
            offset (float): best offset value
    """
    if row['time_slice'] == 'spring':
        time_slice = [3,4,5]
    elif row['time_slice'] == 'summer':
        time_slice = [6,7,8]
    elif row['time_slice'] == 'autumn':
        time_slice = [9,10,11]
    elif row['time_slice'] == 'winter':
        time_slice = [1,2,12]
    elif row['time_slice'] == '1/6':
        time_slice = [1,2]
    elif row['time_slice'] == '2/6':
        time_slice = [3,4]
    elif row['time_slice'] == '3/6':
        time_slice = [5,6]
    elif row['time_slice'] == '4/6':
        time_slice = [7,8]
    elif row['time_slice'] == '5/6':
        time_slice = [9,10]
    elif row['time_slice'] == '6/6':
        time_slice = [11,12]
    elif row['time_slice'] == '1/1':
        time_slice = [1,2,3,4,5,6,7,8,9,10,11,12]
    else:
        time_slice = int(row['time_slice'])
    
    #initialize step size
    step = np.sign(row.obs - row.sim) * 10
    step_prev = step
    offset = 0
    n = 30 # max 30 iterations
    while np.abs(step) > 0.002: # loop will run until the step size is smaller than our power curve's resolution
        if n == 0:
            offset = np.nan
            break
        n -= 1
        
        # calculate the mean simulated CF using the new offset
        mean_sim_cf = train_simulate_wind(
            reanalysis.sel(
                    time=np.logical_and(
                    reanalysis.time.dt.year == row.year, 
                    reanalysis.time.dt.month.isin(time_slice)
                    )
            ),
            turb_info.loc[turb_info['cluster'] == row.cluster],
            powerCurveFile, 
            row.scalar, 
            offset
        )
        
        # wind power is roughly cube of wind speed, this provides a good step size
        step = np.cbrt(row.obs - mean_sim_cf)
        # print('obs: ', row.obs, 'sim: ', mean_sim_cf)
        # prevent the step size oscilating between + and -
        if np.sign(step) != np.sign(step_prev) and np.abs(step) > np.abs(step_prev):
            step = -step_prev/2

        # ensure the step size reduces through iterations
        elif np.sign(step) == np.sign(step_prev) and np.abs(step) > np.abs(step_prev):
            step = step_prev/2
            
        step_prev = step
        offset += step
        # print(offset)
    return offset

def format_bc_factors(run, time_res, num_clu, country):
    bias_data = pd.read_csv(run+'/training/correction-factors/'+country+'_factors_'+time_res+'_'+str(num_clu)+'.csv')
    bias_data = clean_bias_data(bias_data)
    bias_data.columns = ['year',time_res,'cluster','scalar','offset']

    bc_factors = bias_data.groupby(['cluster', time_res], as_index=False).agg({'scalar': 'mean', 'offset': 'mean'})
    bc_factors.loc[bc_factors['scalar'].isna(), 'offset'] = 0
    bc_factors.loc[bc_factors['scalar'].isna(), 'scalar'] = 1
    return bc_factors

def fill_factor_nan(df, time_res, num_clu, country):
    df2 = pd.read_csv(run+'/training/correction-factors/'+country+'_factors_'+time_res+'_'+str(num_clu)+'.csv')
    df2 = clean_bias_data(df2)
    df2.columns = ['year',time_res,'cluster','scalar','offset']

    fill = df.merge(
        df2, 
        on=['year','cluster',time_res],
        how='left'
    )

    df.loc[df['scalar'].isna(), 'offset'] = fill['offset_y']
    df.loc[df['scalar'].isna(), 'scalar'] = fill['scalar_y']
    return df
    

def clean_bias_data(df):
    df = df.drop(['obs','sim', 'Unnamed: 0'], axis=1)
    df['scalar'] = df['scalar'].replace(0, np.nan)
    return df
    
    
    
    
# # working offset
# def find_offset(row, turb_info, reanalysis, powerCurveFile):

#     # start_time = time.time()
#     if row['time_slice'] == 'spring':
#         time_slice = [3,4,5]
#     elif row['time_slice'] == 'summer':
#         time_slice = [6,7,8]
#     elif row['time_slice'] == 'autumn':
#         time_slice = [9,10,11]
#     elif row['time_slice'] == 'winter':
#         time_slice = [1,2,12]
#     elif row['time_slice'] == '1/6':
#         time_slice = [1,2]
#     elif row['time_slice'] == '2/6':
#         time_slice = [3,4]
#     elif row['time_slice'] == '3/6':
#         time_slice = [5,6]
#     elif row['time_slice'] == '4/6':
#         time_slice = [7,8]
#     elif row['time_slice'] == '5/6':
#         time_slice = [9,10]
#     elif row['time_slice'] == '6/6':
#         time_slice = [11,12]
#     elif row['time_slice'] == 'year':
#         time_slice = [1,2,3,4,5,6,7,8,9,10,11,12]
#     else:
#         time_slice = int(row['time_slice'])
    
#     # end_time = time.time()
#     # elapsed_time = end_time - start_time
#     # print("If statements took: ", elapsed_time)
    
#     # decide our initial search step size
#     stepSize = -0.64
#     if (row.sim > row.obs):
#         stepSize = 0.64
        
#     # start_time = time.time()
#     myOffset = 0.000
#     while np.abs(stepSize) > 0.002: # Stop when step-size is smaller than our power curve's resolution
#         myOffset += stepSize # If we are still far from energytarget, increase stepsize
        
#         # calculate the mean simulated CF using the new offset
#         mean_sim_cf = train_simulate_wind(
#             reanalysis.sel(
#                     time=np.logical_and(
#                     reanalysis.time.dt.year == row.year, 
#                     reanalysis.time.dt.month.isin(time_slice)
#                 )
#             ),
#             turb_info.loc[turb_info['cluster'] == row.cluster],
#             powerCurveFile, 
#             row.scalar, 
#             myOffset
#         )
        
#         # if we have overshot our target, then repeat, searching the other direction
#         # ((guess < target & sign(step) < 0) | (guess > target & sign(step) > 0))

#         if np.sign(mean_sim_cf - row.obs) == np.sign(stepSize):
#             stepSize = -stepSize / 2
#         # If we have reached unreasonable places, stop
#         if myOffset < -20 or myOffset > 20:
#             myOffset = np.nan
#             break


#     # end_time = time.time()
#     # elapsed_time = end_time - start_time
#     # print("While loop took: ", elapsed_time)
#     return myOffset
    
    
######## POST PROCESSING OF BIAS CORRECTION FACTORS

### USING yearly TIME RES
# def format_bc_factors(time_res, num_clu, country):
#     bias_data = pd.read_csv('data/training/correction-factors/'+country+'_factors_'+time_res+'_'+str(num_clu)+'.csv')
#     bias_data = clean_bias_data(bias_data)
#     bias_data.columns = ['year',time_res,'cluster','scalar','offset']
#     # bias_data.loc[bias_data['scalar'].isna(),'offset'] = np.nan
#     # bias_data.loc[bias_data['offset'].isna(),'scalar'] = np.nan

#     if time_res != 'yearly':
#         core = pd.read_csv('data/training/correction-factors/'+country+'_factors_yearly_'+str(num_clu)+'.csv')
#         core_factors = clean_bias_data(core)
    
#         fill = bias_data.merge(core_factors, on=['year','cluster'],how='left')
    
#         bias_data.loc[bias_data['scalar'].isna(), 'offset'] = fill['offset_y']
#         bias_data.loc[bias_data['scalar'].isna(), 'scalar'] = fill['scalar_y']
        
#     bc_factors = bias_data.groupby(['cluster', time_res], as_index=False).agg({'scalar': 'mean', 'offset': 'mean'})
#     bc_factors.loc[bc_factors['scalar'].isna(), 'offset'] = 0
#     bc_factors.loc[bc_factors['scalar'].isna(), 'scalar'] = 1
#     # bc_factors['offset'] = bc_factors['offset'].fillna(0)
    
#     return bc_factors

# # USING SEQUENTIAL TIME RES
# def format_bc_factors(time_res, num_clu, country):
#     bias_data = pd.read_csv('data/training/correction-factors/'+country+'_factors_'+time_res+'_'+str(num_clu)+'.csv')
#     bias_data = clean_bias_data(bias_data)
#     bias_data.columns = ['year',time_res,'cluster','scalar','offset']
#     # bias_data.columns = ['cluster',time_res,'scalar','year','offset']
#     bias_data = bias_time_res(bias_data, time_res)

#     if time_res == 'month':
#         bias_data = fill_factor_nan(bias_data, 'bimonth', num_clu, country)
        
#     if time_res == 'month' or time_res == 'bimonth':
#         bias_data = fill_factor_nan(bias_data, 'season', num_clu, country)
        
#     if time_res != 'yearly':
#         bias_data = fill_factor_nan(bias_data, 'yearly', num_clu, country)

#     bc_factors = bias_data.groupby(['cluster', time_res], as_index=False).agg({'scalar': 'mean', 'offset': 'mean'})
#     bc_factors.loc[bc_factors['scalar'].isna(), 'offset'] = 0
#     bc_factors.loc[bc_factors['scalar'].isna(), 'scalar'] = 1
#     return bc_factors
    
# # with modification for iains method
# def calculate_scalar(time_res, gen_data, scalar_alpha = 0.6, scalar_beta = 0.2):
#     """
#     Calculate the scalar, multiplicative correction factor.

#         Args:
#             time_res (str): time resolution we want to simulate
#             gen_data (pandas.DataFrame): dataframe with the observed and simulated cf for training period

#         Returns:
#             turb_info (pandas.DataFrame): turbine metadata with assigned cluster column
#     """

#     def weighted_avg(group_df, whole_df, values, weights):
#         v = whole_df.loc[group_df.index, values]
#         w = whole_df.loc[group_df.index, weights]
#         return (v * w).sum() / w.sum()
        
#     if time_res == 'year':
#         bias_data = gen_data.groupby(['year', 'cluster',]).agg({
#                                 "obs": lambda x: weighted_avg(x, gen_data, 'obs', 'capacity'),
#                                 "sim": lambda x: weighted_avg(x, gen_data, 'sim', 'capacity'),
#                                 })
#         bias_data['scalar'] = (scalar_alpha * (bias_data['obs'] / bias_data['sim'])) + scalar_beta
#         bias_data['time_slice'] = 'year'
        
#     else: 
#         bias_data = gen_data.groupby([time_res, 'cluster', 'year']).agg({
#                                 "obs": lambda x: weighted_avg(x, gen_data, 'obs', 'capacity'),
#                                 "sim": lambda x: weighted_avg(x, gen_data, 'sim', 'capacity'),
#                                 })
                                
#     bias_data['scalar'] = (scalar_alpha * (bias_data['obs'] / bias_data['sim'])) + scalar_beta
#     bias_data = bias_data.reset_index()
#     bias_data.columns = ['time_slice', 'cluster', 'year', 'obs', 'sim', 'scalar']
        
#     return bias_data[['year', 'time_slice', 'cluster', 'obs', 'sim', 'scalar']]


# def bias_time_res(df, time_res):
#     """
#     Add columns to identify time resolutions.
#     """
#     if time_res == 'month':
#         df.loc[df['month'] == 1, ['bimonth','season']] = ['1/6', 'winter']
#         df.loc[df['month'] == 2, ['bimonth','season']] = ['1/6', 'winter']
#         df.loc[df['month'] == 3, ['bimonth','season']] = ['2/6', 'spring']
#         df.loc[df['month'] == 4, ['bimonth','season']] = ['2/6', 'spring']
#         df.loc[df['month'] == 5, ['bimonth','season']] = ['3/6', 'spring']
#         df.loc[df['month'] == 6, ['bimonth','season']] = ['3/6', 'summer']
#         df.loc[df['month'] == 7, ['bimonth','season']] = ['4/6', 'summer']
#         df.loc[df['month'] == 8, ['bimonth','season']] = ['4/6', 'summer']
#         df.loc[df['month'] == 9, ['bimonth','season']] = ['5/6', 'autumn']
#         df.loc[df['month'] == 10, ['bimonth','season']] = ['5/6', 'autumn']
#         df.loc[df['month'] == 11, ['bimonth','season']] = ['6/6', 'autumn']
#         df.loc[df['month'] == 12, ['bimonth','season']] = ['6/6', 'winter']
        
#     elif time_res == 'bimonth':
#         df.loc[df['bimonth'] == '1/6', 'season'] = 'winter'
#         df.loc[df['bimonth'] == '2/6', 'season'] = 'spring'
#         df.loc[df['bimonth'] == '3/6', 'season'] = 'spring'
#         df.loc[df['bimonth'] == '4/6', 'season'] = 'summer'
#         df.loc[df['bimonth'] == '5/6', 'season'] = 'autumn'
#         df.loc[df['bimonth'] == '6/6', 'season'] = 'autumn'
#         df = pd.concat([df, df.loc[df['bimonth'] == '3/6'].assign(**{'season': 'summer'})])
#         df = pd.concat([df, df.loc[df['bimonth'] == '6/6'].assign(**{'season': 'winter'})])

#     df['yearly'] = '1/1'
#     return df