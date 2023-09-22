import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate, optimize
import statsmodels.api as sm
import statsmodels.formula.api as smf
import itertools
from sklearn.metrics import mean_squared_error
from math import sqrt
from calendar import monthrange
pd.options.mode.chained_assignment = None

from Core_Functions import wind_speed_to_power_output

# Power curve files come from Renewables.ninja
powerCurveFileLoc = '../Data/Metadata/Wind Turbine Power Curves.csv'
powerCurveFile = pd.read_csv(powerCurveFileLoc)

mode = 'ERA5'
# Define the year we're doing the simulation on
year_ = 2020
# The start and end index for the chosen year in its corresponding Az file 
# (2014-2016's, 2017-2019's Az file was done on a 3-year basis, so it's necessary to sub-index)
year_star = 2020
year_end = 2020

# Load the corresponding Az file
ncFile = '../Data/'+mode+'/DailyAZ/'+str(year_star)+'-'+str(year_end)+'_dailyAz.nc'
reanal_data = xr.open_dataset(ncFile)
reanal_data['time'] = pd.to_datetime(reanal_data.time)

# Slice the Az variables for the chosen year
star = (year_ - year_star)*365
endd = (year_ - year_star + 1)*365
A_list = reanal_data.A[star:endd]
z_list = reanal_data.z[star:endd]

# Load in turbine metadata and observed generation data.
# The simulation is done to turbines present in both 2020 and the chosen year
# Because if the turbine is no longer present in 2020 it means it has retired
# Load meta info for all turbines in 2020
metadata = pd.read_excel('../Data/Metadata/Metadata_2020.xlsx', index_col=None)
metadata.ID = metadata['ID'].astype(str)
# Load generation data
obs_gen = pd.read_excel('../Data/ObservationData/Denmark_'+str(year_)+'.xlsx')


# SIMULATING WIND POWER
# Slice the observed CF for that year
obs_gen = obs_gen.iloc[3:,np.r_[0:1, 3:15]] # the slicing done here is file dependent please consider this when other files are used
obs_gen.columns = ['ID','1','2','3','4','5','6','7','8','9','10','11','12']
obs_gen.ID = obs_gen['ID'].astype(str)


# Find turbines that are present in both 2020 and the chosen year
obs_meta = metadata.loc[metadata['ID'].isin(obs_gen.ID)].reset_index(drop=True)

# Convert wind speed to wind power, iterate through the months, then every turbine
# THIS TAKES 1.5 HOURS TO RUN FOR EACH YEAR
final_all = []
for j in range(len(A_list)):
    A = A_list[j]
    z = z_list[j]
    final_list = []
    
    # generate CF for all turbines
    for i in range(len(obs_meta)):
        print('at: ',j,' ', i)
        final = wind_speed_to_power_output(A, z, obs_meta, reanal_data, powerCurveFile, i, False)
        final_list.append(final)
    
    final_all.append(final_list)
    

# AVERAGING THE DAILY CF TO MONTHS FOR EVERY TURBINE
# Create index for days in every month
day_list = [0]
k = 0
for i in range(1,13):
    j = monthrange(year_, i)[1]
    k = k+j
    day_list.append(k)
    
all_months = []
for i in range(0,12):
    a = final_all[day_list[i]:day_list[i+1]]
    mon_mean = np.mean(a, axis=0)
    all_months.append(mon_mean)
    
    
# COMBINE SIMULATION RESULTS WITH OBSERVATIONS AND SAVE
# Select the observation results for turbines that exist that year
join_ = pd.DataFrame.merge(obs_meta, obs_gen, on='ID', how='left')


# CALCULATE CAPACITY FACTORS FOR GENERATION DATA
# This is because the original generation data is in kWh 
# CF = monthly kWh/(no. days in the month)*24hours*capacity
for i in range(1,13):
    join_['CF_'+str(i)] = join_[str(i)]/(monthrange(year_, i)[1]*24*join_['capacity'])
    
# Put in the monthly simulation data
sim = pd.DataFrame(all_months).T
names = []
for i in range(1,13):
    names.append('sim_'+str(i))
sim.columns = names
sim['ID'] = obs_meta.ID.astype(str)

# Combine everything together
all_ = pd.DataFrame.merge(join_, sim, on='ID', how='left')

# Select the columns wanted
# This previously caused an issue (FIXED WITH REGEX)
all_ = all_.filter(regex='ID|capacity|longitude|latitude|height|turb_match|CF|sim_')
# drop NAs and reset index
final_= all_.dropna().reset_index(drop=True)
final_.to_excel('../Data/'+mode+'/Sim_MonthlyCF/'+str(year_)+'_sim.xlsx', index=None)

# Save another version of the result: result concatenated vertically, 
# this is for easier comparison and plotting purpose
gendata = final_

month_p = []
lat_l = []
lon_l = []
obs_p = []
sim_p = []
id_l = []
cap_l = []
height_l = []
turb_l = []
for i in range(1,13):
    obs = gendata.iloc[:,i+5]
    sim = gendata.iloc[:,i+17]
    mon = [i]*len(gendata)
    lat = gendata.latitude
    lon = gendata.longitude
    id_ = gendata.ID.astype(str)
    cap_ = gendata.capacity
    h = gendata.height
    t = gendata.turb_match
    
    obs_p.append(obs)
    sim_p.append(sim)
    lat_l.append(lat)
    lon_l.append(lon)
    month_p.append(mon)
    id_l.append(id_)
    cap_l.append(cap_)
    height_l.append(h)
    turb_l.append(t)

merged6 = list(itertools.chain(*id_l))
merged7 = list(itertools.chain(*obs_p))
merged8 = list(itertools.chain(*sim_p))
merged10 = list(itertools.chain(*month_p))
merged11 = list(itertools.chain(*lat_l))
merged12 = list(itertools.chain(*lon_l))
merged13 = list(itertools.chain(*cap_l))
merged14 = list(itertools.chain(*height_l))
merged15 = list(itertools.chain(*turb_l))
df_all = pd.DataFrame(list(zip(merged6, merged7, merged8, merged10, merged11, merged12, merged13, merged14, merged15)),
               columns =['ID', 'obs', mode, 'month', 'lat', 'lon', 'cap', 'height', 'turb_match'])
               
# Save the reordered results
df_all.to_excel('../Data/'+mode+'/Sim_MonthlyCF/'+str(year_)+'_sim_reorder.xlsx', index=None)