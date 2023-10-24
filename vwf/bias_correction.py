import xarray as xr
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from calendar import monthrange
import itertools

from vwf.extras import add_times
from vwf.simulation import simulate_wind

# Use K-means clustering, cluster the turbines by different number of clusters, and save the cluster labels.
def cluster_labels(num_clu, sim_cf):
    # for num_clu in [1, 2, 3, 5, 10, 15, 20, 30, 50, 100, 150, 200, 300, 400]:
    lat = sim_cf['lat']
    lon = sim_cf['lon']

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
    
def closest_cluster(clus_info, turb_info):
    """
    Assign turbines not found in training data to closest cluster.
    """
    
    # making sure ID column dtype is same   
    clus_info['ID'] = clus_info['ID'].astype(str)
    turb_info['ID'] = turb_info['ID'].astype(str)
    
    avg = clus_info.groupby(['cluster'], as_index=False)[['lat','lon']].mean()
    turb_info_new = pd.DataFrame.merge(clus_info[['ID','cluster']], turb_info, on='ID', how='right')

    for i in range(len(turb_info_new)):
        if np.isnan(turb_info_new.cluster[i]) == True:
            # Find the cluster center closest to the new turbine
            # - find smallest distance between the new turbine and cluster centers
            indx = np.argmin(np.sqrt((avg.lat.values - turb_info_new.lat[i])**2 + (avg.lon.values - turb_info_new.lon[i])**2))
            turb_info_new.cluster[i] = avg.cluster[indx]

    turb_info_new = turb_info_new.reset_index(drop=True)

    return turb_info_new
    
    
def generate_training_data(unc_cf, obs_cf, turb_info, num_clu):

    unc_cf = unc_cf.groupby(pd.Grouper(key='time',freq='M')).mean().reset_index()
    unc_cf = unc_cf.melt(id_vars=["time"], 
                    var_name="ID", 
                    value_name="cf")
    unc_cf.columns = ['time','ID', 'cf']
    unc_cf = add_times(unc_cf)

    df = pd.merge(unc_cf, turb_info[['ID', 'lon', 'lat', 'capacity', 'height', 'model']], on='ID', how='left')       
    df['cf'].where(df['cf'] > 0 , 0, inplace=True)
    df['cf'].where(df['cf'] < 1 , 1, inplace=True) 
    unc_cf = df.pivot(index=['ID', 'lon', 'lat', 'capacity', 'height', 'year', 'model'], columns='month', values='cf').reset_index()
    unc_cf.columns = [f'sim_{i}' if i not in ['ID', 'year', 'lon', 'lat', 'capacity', 'height', 'model'] else f'{i}' for i in unc_cf.columns]
    
        
    # combining sim_cf and obs_cf
    train_data = pd.merge(unc_cf, obs_cf,  how='left', on=['ID', 'year'])
    train_data['ID'] = train_data['ID'].astype(str)
    
    # generating the labels
    labels = cluster_labels(num_clu, train_data)
    df = train_data.copy()
    df['cluster'] = labels
    df = df.drop_duplicates(subset=['ID'])
    clus_info = df[['ID','lon','lat','cluster']].reset_index(drop=True)
    
    return train_data, clus_info


def training_bias(train_data, reanal_data, clus_info, year_star, year_end, powerCurveFile):
    """
    Traiaing the bias correction factors, for all clusters, every training 
    year on a monthly resolution.
    
    Inputs:
        train_data: Dataframe consisting of the simulated CF,
                       observed CF, turbine metadata and coordinates
                       for every month in the training years.
        reanal_data: Xarray Dataset with relevant wind speed variables
                     for target area.
        clus_data: Dataframe of turbine ID, coordinates and assigned cluster
        year_star: Training start year
        year_end: Training end year
        powerCurveFile: Dataframe with the power curve of various turbines.
                     
    Returns:
        Monthly bias correction factors for each cluster per training year.
    """
    data_all = []
    star = 0
    end = 12
    for k in range(year_star,year_end+1):
    
        # Find farms in the chosen year that have a cluster label
        data_w_cluster = pd.DataFrame.merge(clus_info[['ID','cluster']], train_data.loc[train_data['year'] == k], on='ID', how='left')
    
        # Get data for an 'average farm' for each cluster:
        # Average lat, lon, height, capacity, simulated and observed CF for all farm in a cluster
        # The turbine model with the most ocurrences is recorded
        data_gen = data_w_cluster.groupby('cluster', as_index=False)[['lat','lon','height','obs_1', 
                                                            'obs_2', 'obs_3', 'obs_4', 'obs_5',
                                                            'obs_6', 'obs_7', 'obs_8', 'obs_9', 'obs_10', 'obs_11', 'obs_12', 'sim_1',
                                                            'sim_2', 'sim_3', 'sim_4', 'sim_5', 'sim_6', 'sim_7', 'sim_8', 'sim_9',
                                                            'sim_10', 'sim_11', 'sim_12', ]].mean()
        data_gen['model'] = data_w_cluster.groupby('cluster', as_index=False)['model'].agg(lambda x: pd.Series.mode(x)[0])['model']
        data_gen['ID'] = data_w_cluster.groupby('cluster', as_index=False)['ID'].agg(lambda x: pd.Series.mode(x)[0])['ID']
        
        # Main bias correction function
        # iterate through the months, then every cluster
        scalar_all = []
        offset_all = []
        for j in range(0,12):
            test_id = star
            test_id += j
            scalar_list = []
            offset_list = []
            
            energyInitial = data_gen.iloc[:,j+16]
            energyTarget = data_gen.iloc[:,j+4]
            prt = energyTarget/energyInitial
            # iterate through the farm clusters
            for i in range(len(prt)):
                PR = prt[i]
                # Find scalar depending on PR = energyTarget/energyInitial
                scalar = determine_farm_scalar(PR, 2)
                # Find offset using the iterative process
                offset = find_farm_offset(data_gen.iloc[[i]], reanal_data.sel(time=slice(str(k)+'-'+str(j+1)+'-01', str(k)+'-'+str(j+1)+'-'+str(monthrange(k, j+1)[1]))), powerCurveFile, scalar, energyInitial[i], energyTarget[i])
    
                scalar_list.append(scalar)
                offset_list.append(offset)
    
            scalar_all.append(scalar_list)
            offset_all.append(offset_list)
            test_id += j
            
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
                    columns =['scalar', 'offset','month', 'lat', 'lon','cluster','sim','obs'])
        df['prt'] = df['obs']/df['sim']
    
        # print(df.groupby(['month'], as_index=False)[['scalar','offset']].mean())
    
        df['year'] = k
        data_all.append(df)
        star += 12
        end += 12
    
    results_df = pd.concat(data_all)

    return results_df.reset_index(drop=True)


def format_bcfactors(df):
    """
    Generating the bias correction factors for every month 
    in each cluster.
    
    Inputs:
        df: Dataframe from training_bias() that has monthly
            bias correction factors for every year in training
    
    Returns:
        Monthly bias correction factors with associated time resolution
    """
    # this takes in the results of the training function.
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

    bias_factors = df.groupby(['cluster', 'month'], as_index=False).agg({'scalar': 'mean', 'offset': 'mean', 'season': 'first', 'two_month' : 'first'})
    
    year_start = df.year.min()
    year_end = df.year.max()
    num_clu = df.cluster.max()+1

    return bias_factors


def find_farm_offset(gendata, azfile, powerCurveFile, myScalar, energyInitial, energyTarget):
    """ 
    Iterative process to find the fixed offset beta in 
    bias correction formula w_corr = alpha*w + beta such that 
    the resulting load factor from simulation model equals energyTarget
    
    Inputs:
        A: Derived A value for height interpolation function: w(h) = A * np.log(h / z)
        z: Derived z value for height interpolation function: w(h) = A * np.log(h / z)
        gendata: DataFrame that contains the meta data for wind turbines
        farm_ind: Turbine row number in gendata
        myScalar: Scalar alpha in formula w_corr = alpha*w + beta 
        energyInitial: Uncorrected CF for this turbine
        energyTarget: Observed CF for this turbine

    Returns:
        Offset beta used in bias correction: w_corr = alpha*w + beta
    """
    myOffset = 0
    
    # decide our initial search step size
    stepSize = -0.64
    if (energyTarget > energyInitial):
        stepSize = 0.64
        
    # Stop when step-size is smaller than our power curve's resolution
    while np.abs(stepSize) > 0.002:
        # If we are still far from energytarget, increase stepsize
        myOffset += stepSize
        
        # Calculate the simulated CF using the new offset
        mylf = simulate_wind(azfile, gendata, powerCurveFile, 'month', True,False,myScalar,myOffset)
        # print(mylf)

        # If we have overshot our target, then repeat, searching the other direction
        # ((guess < target & sign(step) < 0) | (guess > target & sign(step) > 0))
        if mylf != 0:
            energyGuess = np.mean(mylf)
            if np.sign(energyGuess - energyTarget) == np.sign(stepSize):
                stepSize = -stepSize / 2
            # If we have reached unreasonable places, stop
            if myOffset < -20 or myOffset > 20:
                break
        elif mylf == 0:
            myOffset = 0
            break
    
    return myOffset

def determine_farm_scalar(PR, match_method=2):
    """ 
    Calculate scalar based on the error factor PR = CF_obs/CF_sim
    
    Inputs:
        PR: A ratio derived from PR = CF_obs/CF_sim
        match_method: 1 or 2, to signify which method to use (see below)

    Returns:
        Scalar alpha used in bias correction: w_corr = alpha*w + beta 
    """
    # This was used as an alternative method in RN
    if match_method == 1:
        scalar = 0.85
    
    # This is the method used in my study
    if match_method == 2:
        scalar_alpha = 0.6
        scalar_beta = 0.2
        scalar = (scalar_alpha * PR) + scalar_beta
    return scalar