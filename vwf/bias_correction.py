import xarray as xr
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from calendar import monthrange

from vwf.simulation import train_simulate_wind


def cluster_turbines(num_clu, turb_train, train=False, *args):
    """
    Spatially cluster the turbines using the coordinates.

        Args:
            num_clu (int): number of clusters to split turbines into
            turb_train (pandas.DataFrame): dataframe with the turbines that exist in training
            train (bool): if the clustering is being used in training or on new turbines
            *args (pandas.DataFrame): dataframe with new turbines to cluster based on the training turbines

        Returns:
            turb_info (pandas.DataFrame): turbine metadata with assigned cluster column
    """
    # fitting clusters to training data
    kmeans = KMeans(
            init="random",
            n_clusters = num_clu,
            n_init = 10,
            max_iter = 300,
            random_state = 42
        )
    kmeans.fit(turb_train[['lat','lon']])
        
    if train == True:
        turb_train['cluster'] = kmeans.predict(turb_train[['lat','lon']])
        return turb_train
    else:
        turb_info = args[0]
        turb_info['cluster'] = kmeans.predict(turb_info[['lat','lon']])
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
        
    # bias_data['scalar'] = (0.6 * (bias_data['obs'] / bias_data['sim'])) + 0.2
    bias_data['scalar'] = bias_data['obs'] / bias_data['sim']
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
    
    # loop will run until the step size is smaller than our power curve's resolution
    while np.abs(step) > 0.002: 
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
        # prevent the step size oscilating between + and -
        if np.sign(step) != np.sign(step_prev) and np.abs(step) > np.abs(step_prev):
            step = -step_prev/2

        # ensure the step size reduces through iterations
        elif np.sign(step) == np.sign(step_prev) and np.abs(step) > np.abs(step_prev):
            step = step_prev/2
            
        step_prev = step
        offset += step
    return offset

def format_bc_factors(bias_data, time_res):
    """
    Mean of all the bias corrections factors calculated for every cluster and time period.
    If the scalar is NA then set to unbiascorrected.
    """
    # clean the bias data up
    bias_data = bias_data.drop(['obs','sim'], axis=1)
    bias_data['scalar'] = bias_data['scalar'].replace(0, np.nan)
    bias_data.columns = ['year',time_res,'cluster','scalar','offset']

    # group and calculate the mean
    bc_factors = bias_data.groupby(['cluster', time_res], as_index=False).agg({'scalar': 'mean', 'offset': 'mean'})
    bc_factors.loc[bc_factors['scalar'].isna(), 'offset'] = 0
    bc_factors.loc[bc_factors['scalar'].isna(), 'scalar'] = 1
    return bc_factors

