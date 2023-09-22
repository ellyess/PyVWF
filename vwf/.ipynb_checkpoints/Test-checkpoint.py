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

from pathlib import Path

# VARIABLES
mode = 'ERA5'
num_clu = 1 # number of clusters
time_res = 'Month' # time resolution choices: 'Year', 'Month', 'Two-month', 'Season'
year_ = 2020 # test year

# Load observed and simulated monthly CF, and power curve database
powerCurveFileLoc = '../Data/Metadata/Wind Turbine Power Curves.csv'
powerCurveFile = pd.read_csv(powerCurveFileLoc)

# change this in future to read whatever the clusters are done over
gen_clu = pd.read_csv('../Data/'+mode+'/ClusterLabels/cluster_labels_2015_2019_'+str(num_clu)+'.csv', index_col = None)
gen_clu['ID'] = gen_clu['ID'].astype(str)
id_clus = gen_clu[['ID','cluster']]

# Load meta data for 2020 turbines 
gen2020 = pd.read_excel('../Data/Metadata/Metadata_2020.xlsx', index_col=None)
# Data pre-processing
gen2020['CF_mean'] = gen2020.iloc[:,33:45].mean(axis=1)
gen2020= gen2020.loc[gen2020['height'] > 1]
gen2020= gen2020.loc[gen2020['CF_mean'] >= 0.01]

# Select columns for observed CF
gen_obs = gen2020.iloc[:, np.r_[1:2, 4:9, 33:45]]
gen_obs['ID'] = gen_obs['ID'].astype(str)
gen_obs = gen_obs.reset_index(drop=True)

bcfactor_files = Path('../Data/'+mode+'/BCFactors/').glob('*cluster*.csv')
df = pd.concat([pd.read_csv(f, index_col=None) for f in bcfactor_files])
df = df.reset_index(drop=True)

# LOAD A and z for 2020
nc2020 = '../Data/'+mode+'/DailyAZ/'+str(year_)+'-'+str(year_)+'_dailyAz.nc'
ds_new = xr.open_dataset(nc2020)
updated_times = np.asarray(pd.date_range(start='2020/01/01', end='2021/01/01', freq='1D'))[:-1]
ds_new["time"] = ("time", updated_times)

# Average the daily A and z by month
year_month_idx = pd.MultiIndex.from_arrays([ds_new['time.year'].values, ds_new['time.month'].values])
ds_new.coords['year_month'] = ('time', year_month_idx)
ds_mean = ds_new.groupby('year_month').mean()

a_months = np.asarray(ds_mean.A)
z_months = np.asarray(ds_mean.z)

    
# Simulate uncorrected CF
final_uncorr = []
# iterate through the months
for j in range(1,3):
    A = a_months[j-1]
    z = z_months[j-1]
    uncorr_list = []
    
    # iterate through the turbines
    for i in range(len(gen_obs)):
        print('at: ',j,' ', i)
        uncorr = wind_speed_to_power_output(A, z, gen_obs, ds_mean, powerCurveFile, i, False)
        uncorr_list.append(uncorr)

    final_uncorr.append(uncorr_list)
    
# ASSIGN FARMS TO CLOSEST CLUSTER
# FOR EACH NEW TURBINE IN 2020, ASSIGN THE TURBINE TO THE CLOSEST CLUSTER
avg = gen_clu.groupby(['cluster'], as_index=False)[['latitude','longitude']].mean()
gen_data_2020 = pd.DataFrame.merge(id_clus, gen_obs, on='ID', how='right')

for i in range(len(gen_data_2020)):
    if np.isnan(gen_data_2020.cluster[i]) == True:
        # Find the cluster center closest to the new turbine
        # - find smallest distance between the new turbine and cluster centers
        indx = np.argmin(np.sqrt((avg.latitude.values - gen_data_2020.latitude[i])**2 + (avg.longitude.values - gen_data_2020.longitude[i])**2))
        gen_data_2020.cluster[i] = avg.cluster[indx]
gen_data_2020 = gen_data_2020
gen_data_2020 = gen_data_2020.reset_index(drop=True)


# AVERAGE BC FACTORS BY THE CHOSEN TIME RESOLUTION
# ADD IN A COLUMN REPRESENTING THE SEASON
sea = []
for i in range(len(df)):
    if (df.month[i] == 3) or (df.month[i] == 4) or (df.month[i] == 5):
        sea.append('spring')
    if (df.month[i] == 8) or (df.month[i] == 6) or (df.month[i] == 7):
        sea.append('summ')
    if (df.month[i] == 11) or (df.month[i] == 9) or (df.month[i] == 10):
        sea.append('autum')
    if (df.month[i] == 12) or (df.month[i] == 1) or (df.month[i] == 2):
        sea.append('wint')
df['season'] = sea

# ADD IN A COLUMN REPRESENTING THE BI-MONTHLY DIVISION
two = []
for i in range(len(df)):
    if (df.month[i] == 1) or (df.month[i] == 2):
        two.append('01')
    if (df.month[i] == 3) or (df.month[i] == 4):
        two.append('02')
    if (df.month[i] == 5) or (df.month[i] == 6):
        two.append('03')
    if (df.month[i] == 7) or (df.month[i] == 8):
        two.append('04')
    if (df.month[i] == 9) or (df.month[i] == 10):
        two.append('05')
    if (df.month[i] == 11) or (df.month[i] == 12):
        two.append('06')
df['two_month'] = two


# AVERAGE SCALAR AND OFFSET DEPENDING ON WHAT TIME RESOLUTION TO USE
if time_res == 'Year':
    df_mean = df.groupby(['cluster'], as_index=False)[['scalar','offset']].mean()
if time_res == 'Month':
    df_mean = df.groupby(['cluster','month'], as_index=False)[['scalar','offset']].mean()
if time_res == 'Season':
    df_mean = df.groupby(['cluster','season'], as_index=False)[['scalar','offset']].mean()
if time_res == 'Two-month':
    df_mean = df.groupby(['cluster','two_month'], as_index=False)[['scalar','offset']].mean()


# CACULATE RESULTS AFTER BIAS CORRECTION
final_all = []
for j in range(1,3):
    print(j)
    A = a_months[j-1]
    z = z_months[j-1]
    final_list = []
    
    # Get a season indicator depending on the month
    if (j == 3) or (j == 4) or (j == 5):
        sea = 'spring'
    if (j == 8) or (j == 6) or (j == 7):
        sea= 'summ'
    if (j == 11) or (j == 9) or (j == 10):
        sea = 'autum'
    if (j == 12) or (j == 1) or (j == 2):
        sea= 'wint'

    # Get a bi-monthly indicator depending on the month
    if (j == 1) or (j == 2):
        two = '01'
    if (j == 3) or (j == 4):
        two = '02'
    if (j == 5) or (j == 6):
        two = '03'
    if (j == 7) or (j == 8):
        two = '04'
    if (j == 9) or (j == 10):
        two = '05'
    if (j == 11) or (j == 12):
        two = '06'

    # Iterate through the turbine clusters
    # Get corresponding scalar and offset depending on time and cluster choice
    for i in range(len(gen_data_2020)):
        cluster = gen_data_2020['cluster'][i]
        print('at: ',j,' ', i)
        if time_res == 'Year':
            scalar = df_mean.scalar[(df_mean['cluster'] == cluster)].values[0] 
            offset = df_mean.offset[(df_mean['cluster'] == cluster)].values[0]

        if time_res == 'Month':
            scalar = df_mean.scalar[(df_mean['month'] == j)&(df_mean['cluster'] == cluster)].values[0]
            offset = df_mean.offset[(df_mean['month'] == j)&(df_mean['cluster'] == cluster)].values[0]

        if time_res == 'Season':
            scalar = df_mean.scalar[(df_mean['season'] == sea)&(df_mean['cluster'] == cluster)].values[0] 
            offset = df_mean.offset[(df_mean['season'] == sea)&(df_mean['cluster'] == cluster)].values[0]

        if time_res == 'Two-month':
            scalar = df_mean.scalar[(df_mean['two_month'] == two)&(df_mean['cluster'] == cluster)].values[0] 
            offset = df_mean.offset[(df_mean['two_month'] == two)&(df_mean['cluster'] == cluster)].values[0]

        final = wind_speed_to_power_output(A, z, gen_data_2020, ds_mean, powerCurveFile, i, 0.597, 2.836, True)
        # final = wind_speed_to_power_output(A, z, gen_data_2020, ds_mean, powerCurveFile, i, scalar, offset, True)
        final_list.append(final)

    final_all.append(final_list)

print(final_all)

# the result is reordered such that a month follows another vertically
month_p = []
lat_l = []
lon_l = []
cluster_l = []
obs_p = []
cap_l = []
height_l = []
turb_l = []
for i in range(1,13):
    obs = gen_data_2020.iloc[:,i+6] # get observed CF
    mon = [i]*len(gen_data_2020)
    lat = gen_data_2020['latitude']
    lon = gen_data_2020['longitude']
    cluster = gen_data_2020['cluster']
    cap_ = gen_data_2020.capacity
    h = gen_data_2020.height
    t = gen_data_2020.turb_match

    obs_p.append(obs)
    lat_l.append(lat)
    lon_l.append(lon)
    month_p.append(mon)
    cluster_l.append(cluster)
    cap_l.append(cap_)
    height_l.append(h)
    turb_l.append(t)

merged5 = list(itertools.chain(*final_uncorr))
merged6 = list(itertools.chain(*final_all))
merged7 = list(itertools.chain(*obs_p))
merged10 = list(itertools.chain(*month_p))
merged11 = list(itertools.chain(*lat_l))
merged12 = list(itertools.chain(*lon_l))
merged13 = list(itertools.chain(*cluster_l))
merged133 = list(itertools.chain(*cap_l))
merged14 = list(itertools.chain(*height_l))
merged15 = list(itertools.chain(*turb_l))


df_all = pd.DataFrame(list(zip(merged5, merged6, merged7, merged10, merged11, merged12, merged13,merged133, merged14, merged15)),
               columns =['uncorr', 'corrected', 'obs', 'month', 'lat', 'lon','cluster','cap', 'height', 'turb_match'])
df_all['abs_diff'] = abs(df_all['obs'] - df_all['corrected'])
df_all['abs_diff_uncor'] = abs(df_all['obs'] - df_all['uncorr'])

# CALCULATE METRICS
y_actual = df_all.obs
y_predicted = df_all.corrected
y_uncorr = df_all.uncorr

# corrected metrics
rms = sqrt(mean_squared_error(y_actual, y_predicted))
abs_diff = abs(y_actual - y_predicted).mean()

# uncorrected metrics 
rms1 = sqrt(mean_squared_error(y_actual, y_uncorr))
abs_diff1 = abs(y_actual - y_uncorr).mean()
df_result = pd.DataFrame(list(zip([rms1], [abs_diff1], [rms], [abs_diff])), 
             columns =['rms_uncorrected', 'abs_diff_uncorrected','rms_corrected', 'abs_diff_corrected'])
             
# SAVE THE RESULTS
df_all.to_csv('../Data/Results/'+mode+'_cluster_results_'+str(num_clu)+'_'+time_res +'.csv', index = None)
df_result.to_csv('../Data/Results/Metrics/'+mode+'_cluster_metrics_'+str(num_clu)+'_'+time_res +'.csv', index = None)