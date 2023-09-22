import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate, optimize
import statsmodels.api as sm
import statsmodels.formula.api as smf
import itertools
# For calculating Error Metrics
from sklearn.metrics import mean_squared_error
from math import sqrt
pd.options.mode.chained_assignment = None
from sklearn.cluster import KMeans

from Core_Functions import wind_speed_to_power_output
from Core_Functions import determine_farm_scalar
from Core_Functions import find_farm_offset


# VARIABLES
mode = 'ERA5'
num_clu = 1 # number of clusters
time_res = 'Month' # time resolution choices: 'Year', 'Month', 'Two-month', 'Season'
year_star = 2017
year_end = 2019

# Load observed and simulated monthly CF, and power curve database
powerCurveFileLoc = '../Data/Metadata/Wind Turbine Power Curves.csv'
powerCurveFile = pd.read_csv(powerCurveFileLoc)

# change this in future to read whatever the clusters are done over
gen_clu = pd.read_csv('../Data/'+mode+'/ClusterLabels/cluster_labels_2015_2019_'+str(num_clu)+'.csv', index_col = None)
gen_clu['ID'] = gen_clu['ID'].astype(str)
id_clus = gen_clu[['ID','cluster']]


# concatinating all yearly simulated CF for total training period
appended_gen = []
for i in range(year_star,year_end+1):
    data = pd.read_excel('../Data/'+mode+'/Sim_MonthlyCF/'+str(i)+'_sim.xlsx', index_col=None)
    data['CF_mean'] = data.iloc[:,6:18].mean(axis=1)
    data = data.loc[data['height'] > 1]
    data = data.loc[data['CF_mean'] >= 0.01]
    data['year'] = i
    appended_gen.append(data)

appended_gen = pd.concat(appended_gen)
appended_gen = appended_gen.reset_index(drop=True)
appended_gen['ID'] = appended_gen['ID'].astype(str)


# Load A and z
nc = '../Data/'+mode+'/DailyAZ/'+str(year_star)+'-'+str(year_end)+'_dailyAz.nc'
ds_new = xr.open_dataset(nc)
updated_times = np.asarray(pd.date_range(start=str(year_star)+'/01/01', end=str(year_end+1)+'/01/01', freq='1d'))[:-1]
ds_new['time'] = ("time", updated_times)



# Average the daily A and z by month
year_month_idx = pd.MultiIndex.from_arrays([ds_new['time.year'].values, ds_new['time.month'].values])
ds_new.coords['year_month'] = ('time', year_month_idx)
ds_mean = ds_new.groupby('year_month').mean()


data_all = []
star = 0
end = 12
for k in range(year_star,year_end+1):
    # Slice the monthly A and z for the chosen year
    
    a_months = np.asarray(ds_mean.A[star:end])
    z_months = np.asarray(ds_mean.z[star:end])
    # Find farms in the chosen year that have a cluster label
    data_w_cluster = pd.DataFrame.merge(id_clus, appended_gen.loc[appended_gen['year'] == k], on='ID', how='left')

    # Get data for an 'average farm' for each cluster:
    # Average lat, lon, height, capacity, simulated and observed CF for all farm in a cluster
    # The turbine model with the most ocurrences is recorded
    data_gen = data_w_cluster.groupby('cluster', as_index=False)[['latitude','longitude','height','CF_1', 
                                                        'CF_2', 'CF_3', 'CF_4', 'CF_5',
                                                        'CF_6', 'CF_7', 'CF_8', 'CF_9', 'CF_10', 'CF_11', 'CF_12', 'sim_1',
                                                        'sim_2', 'sim_3', 'sim_4', 'sim_5', 'sim_6', 'sim_7', 'sim_8', 'sim_9',
                                                        'sim_10', 'sim_11', 'sim_12', ]].mean()
    data_gen['turb_match'] = data_w_cluster.groupby('cluster', as_index=False)['turb_match'].agg(lambda x: pd.Series.mode(x)[0])['turb_match']


    # Main bias correction function
    # iterate through the months, then every cluster
    scalar_all = []
    offset_all = []
    for j in range(0,12):
        # print(j)
        A = a_months[j]
        z = z_months[j]
        energyInitial = data_gen.iloc[:,j+16]
        energyTarget = data_gen.iloc[:,j+4]
        prt = energyTarget/energyInitial
        scalar_list = []
        offset_list = []

        # iterate through the farm clusters
        for i in range(len(prt)):
            PR = prt[i]
            # Find scalar depending on PR = energyTarget/energyInitial
            scalar = determine_farm_scalar(PR, 2, iso = None)
            # Find offset using the iterative process
            offset = find_farm_offset(A, z, data_gen, ds_mean, powerCurveFile, i, scalar, energyInitial[i], energyTarget[i])

            scalar_list.append(scalar)
            offset_list.append(offset)

        scalar_all.append(scalar_list)
        offset_all.append(offset_list)
        
        
    # Concatenate results vertically for comparison and save results
    # the result is reordered such that a month follows another vertically
    month_p = []
    lat_l = []
    lon_l = []
    clu_l = []
    sim_l = []
    obs_l = []
    for i in range(1,13):
        mon = [i]*len(data_gen)
        lat = data_gen.iloc[:,1]
        lon = data_gen.iloc[:,2]
        clu = data_gen.iloc[:,0]
        # simulated CF for all months
        sim = data_gen.iloc[:,i+15]
        # observed CF for all months
        obs = data_gen.iloc[:,i+4]
        lat_l.append(lat)
        lon_l.append(lon)
        month_p.append(mon)
        clu_l.append(clu)
        sim_l.append(sim)
        obs_l.append(obs)


    merged1 = list(itertools.chain(*scalar_all))
    merged2 = list(itertools.chain(*offset_all))
    merged3 = list(itertools.chain(*month_p))
    merged4 = list(itertools.chain(*lat_l))
    merged5 = list(itertools.chain(*lon_l))
    merged6 = list(itertools.chain(*clu_l))
    merged7 = list(itertools.chain(*sim_l))
    merged8 = list(itertools.chain(*obs_l))
    df = pd.DataFrame(list(zip(merged1, merged2, merged3, merged4, merged5, merged6, merged7, merged8)),
                columns =['scalar', 'offset','month', 'latitude', 'longitude','cluster','sim','obs'])
    df['prt'] = df['obs']/df['sim']

    print(df.groupby(['month'], as_index=False)[['scalar','offset']].mean())

    df['year'] = k
    data_all.append(df)
    star += 12
    end += 12
    
df_all = pd.concat(data_all)
df_all.to_csv('../Data/'+mode+'/BCFactors/cluster_'+str(num_clu)+'_'+str(year_star)+'_'+str(year_end)+'.csv', index = None)

