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


# VARIABLES
mode = 'MERRA2'
num_clu = 1 # number of clusters
year_star = 2020
year_end = 2020

# Load observed and simulated monthly CF, and power curve database
powerCurveFileLoc = '../Data/Metadata/Wind Turbine Power Curves.csv'
powerCurveFile = pd.read_csv(powerCurveFileLoc)


# First find out the turbines present in all training years
id_list = []
for i in range(year_star,year_end+1):
    data = pd.read_excel('../Data/'+mode+'/Sim_MonthlyCF/'+str(i)+'_sim.xlsx', index_col=None)
    # print(i, len(data.ID))
    data['CF_mean'] = data.iloc[:,6:18].mean(axis=1)
    data = data.loc[data['height'] > 1]
    data = data.loc[data['CF_mean'] >= 0.01]
    print(i, 'updated', len(data.ID))
    id_list.append(data.ID)

ID_all = set.intersection(*map(set,id_list))
gendata = data.loc[data['ID'].isin(ID_all)]
gendata['ID'] = gendata['ID'].astype(str)

# Use K-means clustering, cluster the turbines by different number of clusters, and save the cluster labels.
def cluster_labels(num_clu, gendata):
    # for num_clu in [1, 2, 3, 5, 10, 15, 20, 30, 50, 100, 150, 200, 300, 400]:
    lat = gendata['latitude']
    lon = gendata['longitude']

    df = pd.DataFrame(list(zip(lat, lon)),
                    columns =['lat', 'lon'])

    # create kmeans model/object
    kmeans = KMeans(
        init="random",
        n_clusters = num_clu,
        n_init = 10,
        max_iter = 300,
        random_state = 42
    )
    kmeans.fit(df)
    labels = kmeans.labels_
    
    return labels

gen_clu = gendata.copy()
labels = cluster_labels(num_clu, gen_clu)
gen_clu['cluster'] = labels

gen_clu = gen_clu[['ID','capacity','latitude','longitude','height','turb_match','cluster']]
gen_clu['ID'] = gen_clu['ID'].astype(str)

# the part '2015-2019' means that the farms are present in all these years, and clusters are based on this.
gen_clu.to_csv('../Data/'+mode+'/ClusterLabels/cluster_labels_'+str(year_star)+'_'+str(year_end)+'_'+str(num_clu)+'.csv', index = None)