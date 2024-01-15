import xarray as xr
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from calendar import monthrange

from vwf.simulation import train_simulate_wind

def cluster_turbines(num_clu, turb_info):
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
    Assign turbines not found in training data to closest cluster.
    """
    # making sure ID column dtype is same   
    clus_info['ID'] = clus_info['ID'].astype(str)
    turb_info['ID'] = turb_info['ID'].astype(str)
    
    avg = clus_info.groupby(['cluster'], as_index=False)[['lat','lon']].mean()
    turb_info = pd.DataFrame.merge(clus_info[['ID','cluster']], turb_info, on='ID', how='right')

    for i in range(len(turb_info)):
        if np.isnan(turb_info.cluster[i]) == True:
            # Find the cluster center closest to the new turbine
            # find smallest distance between the new turbine and cluster centers
            indx = np.argmin(np.sqrt((avg.lat.values - turb_info.lat[i])**2 + (avg.lon.values - turb_info.lon[i])**2))
            turb_info.cluster[i] = avg.cluster[indx]

    turb_info = turb_info.reset_index(drop=True)

    return turb_info
    
def train_data(time_res, gen_cf, clus_info):

    if time_res == 'yearly':
        gen_data = gen_cf.groupby(['year','ID'], as_index=False)[['obs','sim']].mean()
        gen_data['yearly'] = 'year'
        
    else:
        gen_data = gen_cf.groupby(['year',time_res,'ID'], as_index=False)[['obs','sim']].mean()

    gen_data = pd.merge(gen_data, clus_info[['ID', 'cluster', 'lon', 'lat', 'capacity', 'height', 'model']], on='ID', how='left')
    bias_data = calculate_scalar(time_res, gen_data)

    return bias_data
    
def calculate_scalar(time_res, gen_data):
    # scalar_alpha = 0.6
    # scalar_beta = 0.2

    # if time_res == 'year':
    #     bias_data = gen_data.groupby(['year','cluster'], as_index=False)[['obs','sim']].mean()
    #     bias_data['scalar'] = (scalar_alpha * (bias_data['obs'] / bias_data['sim'])) + scalar_beta
    #     bias_data['time_slice'] = 'year'
        
    # else:
    

    def weighted_avg(group_df, whole_df, values, weights):
        v = whole_df.loc[group_df.index, values]
        w = whole_df.loc[group_df.index, weights]
        return (v * w).sum() / w.sum()
        
    bias_data = gen_data.groupby([time_res, 'cluster', 'year']).agg({
                            "obs": lambda x: weighted_avg(x, gen_data, 'obs', 'capacity'),
                            "sim": lambda x: weighted_avg(x, gen_data, 'sim', 'capacity'),
                            })
        
        
    # bias_data = gen_data.groupby([time_res, 'cluster', 'year'])[['obs','sim']].mean()
    # bias_data['scalar'] = (scalar_alpha * (bias_data['obs'] / bias_data['sim'])) + scalar_beta
    bias_data['scalar'] = bias_data['obs'] / bias_data['sim']
    bias_data = bias_data.reset_index()
    bias_data.columns = ['time_slice', 'cluster', 'year', 'obs', 'sim', 'scalar']
        
    return bias_data[['year', 'time_slice', 'cluster', 'obs', 'sim', 'scalar']]
    
def find_offset(row, turb_info, reanalysis, powerCurveFile):

    # start_time = time.time()
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
    elif row['time_slice'] == 'year':
        time_slice = [1,2,3,4,5,6,7,8,9,10,11,12]
    else:
        time_slice = int(row['time_slice'])
    
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print("If statements took: ", elapsed_time)
    
    # decide our initial search step size
    stepSize = -0.64
    if (row.sim > row.obs):
        stepSize = 0.64
        
    # start_time = time.time()
    myOffset = 0.000
    while np.abs(stepSize) > 0.002: # Stop when step-size is smaller than our power curve's resolution
        myOffset += stepSize # If we are still far from energytarget, increase stepsize
        
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
            myOffset
        )
        
        # if we have overshot our target, then repeat, searching the other direction
        # ((guess < target & sign(step) < 0) | (guess > target & sign(step) > 0))
        if mean_sim_cf != 0.000:
            if np.sign(mean_sim_cf - row.obs) == np.sign(stepSize):
                stepSize = -stepSize / 2
            # If we have reached unreasonable places, stop
            if myOffset < -20 or myOffset > 20:
                myOffset = np.nan
                break
        elif mean_sim_cf == 0.000:
            myOffset = np.nan
            break

    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print("While loop took: ", elapsed_time)
    return myOffset