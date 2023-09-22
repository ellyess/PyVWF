import xarray as xr
import numpy as np
import pandas as pd
from scipy import interpolate
from sklearn.cluster import KMeans
from calendar import monthrange
import itertools


def simulate_wind_train(ds, turbine_info, powerCurveFile, scalar, offset, i):

        
    w = ds.A * np.log(turbine_info.height[i] / ds.z)
    w = w.where(w > 0 , 0)
    w = w.where(w < 40 , 40)

    # spatial interpolating to turbine positions
    wnd_spd = w.interp(
            lat=turbine_info.latitude, lon=turbine_info.longitude,
            kwargs={"fill_value": None})

    wnd_spd = (wnd_spd * scalar) + offset
    wnd_spd = wnd_spd.where(wnd_spd > 0 , 0)
    wnd_spd = wnd_spd.where(wnd_spd < 40 , 40)
    x = powerCurveFile['data$speed']
    y = powerCurveFile[turbine_info.turb_match]

    f2 = interpolate.Akima1DInterpolator(x, y)

    gen_power = np.mean(f2(wnd_spd))

    return np.mean(gen_power)

def extrapolate_wind_speed(ds, turbine_info, train=False):
 
    # calculating wind speeds at given heights
    if train == True:
        w = ds.A * np.log(turbine_info.height / ds.z)
    else:
        w = ds.A * np.log(ds.height / ds.z)

    w = w.where(w > 0 , 0)
    w = w.where(w < 40 , 40)
    
    # creating coordinates to spatially interpolate to
    lat =  xr.DataArray(turbine_info['latitude'], dims='turbine', coords={'turbine':turbine_info['ID']})
    lon =  xr.DataArray(turbine_info['longitude'], dims='turbine', coords={'turbine':turbine_info['ID']})
    height =  xr.DataArray(turbine_info['height'], dims='turbine', coords={'turbine':turbine_info['ID']})


    # spatial interpolating to turbine positions
    wnd_spd = w.interp(
            lat=lat, lon=lon, height=height,
            kwargs={"fill_value": None})


    # removing unreasonable wind speeds
    # wnd_spd = wnd_spd.where(wnd_spd > 0 , 0)
    # wnd_spd = wnd_spd.where(wnd_spd < 40 , 40)
    
    return wnd_spd



def speed_to_power(speed_frame, turbine_info, powerCurveFile, train=False):
    
    x = powerCurveFile['data$speed']
    
    if train == True:
        turb_name = turbine_info.loc[turbine_info['ID'] == speed_frame.turbine.data, 'turb_match']
        print(turb_name)
        y = powerCurveFile[turb_name].to_numpy().flatten()
        # f2 = interpolate.interp1d(x, y, kind='cubic')
        print(len(y), len(x))
        print(y)
        f2 = interpolate.Akima1DInterpolator(x, y)
        return f2(speed_frame.data)

    else:
        power_frame = speed_frame.copy()
        for i in range(2, len(power_frame.columns)+1):            
            speed_single = power_frame.iloc[:,i-1]
            turb_name = turbine_info.loc[turbine_info['ID'] == speed_single.name, 'turb_match']
            
            y = powerCurveFile[turb_name].to_numpy().flatten()
            # f2 = interpolate.interp1d(x, y, kind='cubic') 
            f2 = interpolate.Akima1DInterpolator(x, y)
            power_frame.iloc[:,i-1] = f2(speed_single)

        return power_frame



def simulate_wind(ds, turbine_info, powerCurveFile, time_res='month',train=False, bias_correct=False, *args):

    
    if train == True:

        wnd_spd = extrapolate_wind_speed(ds, turbine_info, train)
        
        scalar, offset = args
        # wnd_spd = (wnd_spd + offset) * scalar
        wnd_spd = (wnd_spd * scalar) + offset
        wnd_spd = wnd_spd.where(wnd_spd > 0 , 0)
        wnd_spd = wnd_spd.where(wnd_spd < 40 , 40)
        
        gen_power = speed_to_power(wnd_spd, turbine_info, powerCurveFile, train)
        
        return np.mean(gen_power)

    else:
        all_heights = np.sort(turbine_info['height'].unique())
        ds = ds.assign_coords(
            height=('height', all_heights))

        wnd_spd = extrapolate_wind_speed(ds, turbine_info, train)
        
        # reformatting to include turbine ID for easier merging later
        speed_frame = wnd_spd.to_pandas().reset_index()

        gen_speed = speed_frame
        # return speed_frame
        speed_frame = speed_frame.melt(id_vars=["time"], 
                        var_name="ID", 
                        value_name="speed")
        speed_frame.columns = ['date','ID', 'speed']
        speed_frame = add_times(speed_frame)
         
        if bias_correct == True:
            bias_factors = args[0]
    
            gen_speed = pd.merge(speed_frame, turbine_info[['ID', 'cluster']], on='ID', how='left')       
            gen_speed = pd.merge(gen_speed, bias_factors[['cluster', 'month','two_month','season']],  how='left', on=['cluster', 'month'])

            time_factors = bias_factors.groupby(['cluster',time_res], as_index=False).agg({'month' : 'first', 'two_month' : 'first', 'season': 'first', 'scalar': 'mean', 'offset': 'mean'})
            gen_speed = pd.merge(gen_speed, time_factors[['cluster', time_res, 'scalar', 'offset']],  how='left', on=['cluster', time_res])
    
            gen_speed['speed'] = (gen_speed.speed + gen_speed.offset) * gen_speed.scalar
            # gen_speed['speed'] = (gen_speed.speed * gen_speed.scalar) + gen_speed.offset 

            # gen_speed['speed'].where(gen_speed['speed'] > 0 , 0, inplace=True)
            # gen_speed['speed'].where(gen_speed['speed'] < 40 , 40, inplace=True)
            
            gen_speed = gen_speed.pivot(index=['date'], columns='ID', values='speed').reset_index()
            gen_power = speed_to_power(gen_speed,turbine_info[['ID','turb_match']], powerCurveFile, train)   
            
            return gen_speed, gen_power
    
        else:

            gen_power = speed_to_power(gen_speed,turbine_info[['ID','turb_match']], powerCurveFile, train)
        
            return gen_speed, gen_power



def add_times(data):
    data['year'] = pd.DatetimeIndex(data['date']).year
    data['month'] = pd.DatetimeIndex(data['date']).month
    data.insert(1, 'year', data.pop('year'))
    data.insert(2, 'month', data.pop('month'))
    return data

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
    
def generate_training_data(uncorr_cf, obs_gen, turbine_info, num_clu):

    uncorr_cf = uncorr_cf.groupby(pd.Grouper(key='time',freq='M')).mean().reset_index()
    uncorr_cf = uncorr_cf.melt(id_vars=["time"], 
                    var_name="ID", 
                    value_name="cf")
    uncorr_cf.columns = ['date','ID', 'cf']
    uncorr_cf = add_times(uncorr_cf)

    gen_cf = pd.merge(uncorr_cf, turbine_info[['ID', 'longitude', 'latitude', 'capacity', 'height', 'turb_match']], on='ID', how='left')       
    gen_cf['cf'].where(gen_cf['cf'] > 0 , 0, inplace=True)
    gen_cf['cf'].where(gen_cf['cf'] < 1 , 1, inplace=True)
    gen_cf = gen_cf.pivot(index=['ID', 'longitude', 'latitude', 'capacity', 'height', 'year', 'turb_match'], columns='month', values='cf').reset_index()
    gen_cf.columns = [f'sim_{i}' if i not in ['ID', 'year', 'longitude', 'latitude', 'capacity', 'height', 'turb_match'] else f'{i}' for i in gen_cf.columns]
    
    def daysDuringMonth(yy, m):
        result = []    
        [result.append(monthrange(y, m)[1]) for y in yy]        
        return result
        
    all_gen = pd.merge(gen_cf, obs_gen,  how='left', on=['ID', 'year'])
    
    for i in range(1,13):
        all_gen['obs_'+str(i)] = all_gen['obs_'+str(i)]/(((daysDuringMonth(all_gen.year, i))*all_gen['capacity'])*24)


    all_gen['CF_mean'] = all_gen.filter(regex='sim_').mean(axis=1)
    all_gen = all_gen.loc[all_gen['height'] > 1]
    all_gen = all_gen.loc[all_gen['CF_mean'] >= 0.01]
    all_gen['ID'] = all_gen['ID'].astype(str)
    all_gen = all_gen.drop(['CF_mean'], axis=1)
    
    # generating the labels
    labels = cluster_labels(num_clu, all_gen)
    clus_data = all_gen.copy()
    clus_data['cluster'] = labels
    clus_data = clus_data.drop_duplicates(subset=['ID'])
    clus_id = clus_data[['ID','cluster']].reset_index(drop=True)
           
    return all_gen, clus_id, clus_data.reset_index(drop=True)


def training_bias(training_data, reanal_data, clus_id, year_star, year_end, powerCurveFile):
    data_all = []
    star = 0
    end = 12
    for k in range(year_star,year_end+1):
    
    
        # Find farms in the chosen year that have a cluster label
        data_w_cluster = pd.DataFrame.merge(clus_id, training_data.loc[training_data['year'] == k], on='ID', how='left')
    
        # Get data for an 'average farm' for each cluster:
        # Average lat, lon, height, capacity, simulated and observed CF for all farm in a cluster
        # The turbine model with the most ocurrences is recorded
        data_gen = data_w_cluster.groupby('cluster', as_index=False)[['latitude','longitude','height','obs_1', 
                                                            'obs_2', 'obs_3', 'obs_4', 'obs_5',
                                                            'obs_6', 'obs_7', 'obs_8', 'obs_9', 'obs_10', 'obs_11', 'obs_12', 'sim_1',
                                                            'sim_2', 'sim_3', 'sim_4', 'sim_5', 'sim_6', 'sim_7', 'sim_8', 'sim_9',
                                                            'sim_10', 'sim_11', 'sim_12', ]].mean()
        data_gen['turb_match'] = data_w_cluster.groupby('cluster', as_index=False)['turb_match'].agg(lambda x: pd.Series.mode(x)[0])['turb_match']
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
                print(i)
                PR = prt[i]
                # Find scalar depending on PR = energyTarget/energyInitial
                scalar = determine_farm_scalar(PR, 2)
                # Find offset using the iterative process
                offset = find_farm_offset(data_gen, reanal_data.sel(time=slice(str(k)+'-'+str(j+1)+'-01', str(k)+'-'+str(j+1)+'-'+str(monthrange(k, j+1)[1]))), powerCurveFile, scalar, energyInitial[i], energyTarget[i], i)
    
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
                    columns =['scalar', 'offset','month', 'latitude', 'longitude','cluster','sim','obs'])
        df['prt'] = df['obs']/df['sim']
    
        # print(df.groupby(['month'], as_index=False)[['scalar','offset']].mean())
    
        df['year'] = k
        data_all.append(df)
        star += 12
        end += 12
    
    results_df = pd.concat(data_all)

    return results_df.reset_index(drop=True)





def format_bcfactors(df):
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
    
    return bias_factors



    

def find_farm_offset(gendata, azfile, powerCurveFile, myScalar, energyInitial, energyTarget, i):
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
        # mylf = simulate_wind(azfile, gendata, powerCurveFile, 'month', True,False,myScalar,myOffset)
        mylf = simulate_wind_train(azfile, gendata, powerCurveFile, myScalar, myOffset, i)
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












# def extrapolate_wind_speed(ds, turbine_info, train=False):
 
#     # calculating wind speeds at given heights
#     w = ds.A * np.log(ds.height / ds.z)


#     if train == True:
#         # creating coordinates to spatially interpolate to
#         lat =  xr.DataArray(turbine_info['latitude'], dims='turbine', coords={'turbine':turbine_info['turb_match']})
#         lon =  xr.DataArray(turbine_info['longitude'], dims='turbine', coords={'turbine':turbine_info['turb_match']})
#         height =  xr.DataArray(turbine_info['height'], dims='turbine', coords={'turbine':turbine_info['turb_match']})

#     else:
#         # creating coordinates to spatially interpolate to
#         lat =  xr.DataArray(turbine_info['latitude'], dims='turbine', coords={'turbine':turbine_info['ID']})
#         lon =  xr.DataArray(turbine_info['longitude'], dims='turbine', coords={'turbine':turbine_info['ID']})
#         height =  xr.DataArray(turbine_info['height'], dims='turbine', coords={'turbine':turbine_info['ID']})

#     # spatial interpolating to turbine positions
#     wnd_spd = w.interp(
#             lat=lat, lon=lon, height=height,
#             kwargs={"fill_value": None})

#     wnd_spd.attrs.update(
#         {
#             "long name": "extrapolated wind speed to turbines using logarithmic "
#             "method with A and z",
#             "units": "m s**-1",
#         }
#     )

#     # removing unreasonable wind speeds
#     wnd_spd = wnd_spd.where(wnd_spd > 0 , 0)
#     wnd_spd = wnd_spd.where(wnd_spd < 40 , 40)
    
#     return wnd_spd

# def speed_to_power(speed, powerCurveFile, train=False):
    
#     x = powerCurveFile['data$speed']
#     if train == True:
#         y = powerCurveFile[speed.turbine.data].to_numpy().flatten()
#         f2 = interpolate.interp1d(x, y, kind='cubic')
#         return f2(speed.data)

#     else:
#         y = powerCurveFile[speed.turb_match]
#         f2 = interpolate.interp1d(x, y, kind='cubic')
#         return f2(speed.speed)

# def simulate_wind(ds, turbine_info, powerCurveFile, time_res='month',train=False, bias_correct=False, *args):
    
#     all_heights = np.sort(turbine_info['height'].unique())
#     ds = ds.assign_coords(
#         height=('height', all_heights))

#     wnd_spd = extrapolate_wind_speed(ds, turbine_info, train)

    
#     if train == True:
#         scalar, offset = args
#         wnd_spd = (wnd_spd + offset) * scalar
#         wnd_spd = wnd_spd.where(wnd_spd > 0 , 0)
#         wnd_spd = wnd_spd.where(wnd_spd < 40 , 40)
        
#         gen_power = speed_to_power(wnd_spd, powerCurveFile, train)
        
#         return gen_power

#     else:
#         # reformatting to include turbine ID for easier merging later
#         speed_frame = wnd_spd.to_pandas().reset_index()
#         speed_frame = speed_frame.melt(id_vars=["time_level_0", "time_level_1"], 
#                         var_name="ID", 
#                         value_name="speed")
#         speed_frame.columns = ['year','month','ID', 'speed']

 
        
#         if bias_correct == True:
#             bias_factors = args[0]
    
#             gen_speed = pd.DataFrame.merge(speed_frame, turbine_info[['ID','turb_match', 'cluster', 'latitude', 'longitude', 'height', 'capacity']], on='ID', how='left')       
#             gen_speed = pd.merge(gen_speed, bias_factors[['cluster', 'month','two_month','season','year']],  how='left', on=['cluster', 'month', 'year'])
    
#             time_factors = bias_factors.groupby(['cluster',time_res], as_index=False).agg({'month' : 'first', 'two_month' : 'first', 'season': 'first', 'scalar': 'mean', 'offset': 'mean'})
#             gen_speed = pd.merge(gen_speed, time_factors[['cluster', time_res, 'scalar', 'offset']],  how='left', on=['cluster', time_res])
    
#             # main bias correction bit (also the slowest bit of this code, is there a way to speed this up)
#             gen_speed['speed'] = (gen_speed.speed + gen_speed.offset) * gen_speed.scalar

#             gen_speed['speed'].where(gen_speed['speed'] > 0 , 0, inplace=True)
#             gen_speed['speed'].where(gen_speed['speed'] < 40 , 40, inplace=True)
#             gen_speed['cf'] = gen_speed.apply(lambda x: speed_to_power(x, powerCurveFile), axis=1)
            
#             gen_power = gen_speed[['ID', 'year', 'month', 'cf', 'turb_match', 'cluster', 'longitude', 'latitude', 'height', 'capacity']]
#             gen_power = gen_power.pivot(index=['ID','turb_match','year', 'cluster', 'longitude', 'latitude', 'height', 'capacity'], columns='month', values='cf')
#             gen_power.columns.name = None
#             gen_power.columns = [f'sim_{i}' if i not in ['ID','turb_match','year', 'cluster', 'longitude', 'latitude', 'height', 'capacity'] else f'{i}' for i in gen_power.columns]
            
#             return gen_power.reset_index()
    
#         else:
#             gen_speed = pd.DataFrame.merge(speed_frame, turbine_info[['ID','turb_match', 'latitude', 'longitude', 'height', 'capacity']], on='ID', how='left')
            
#             gen_speed['cf'] = gen_speed.apply(lambda x: speed_to_power(x, powerCurveFile), axis=1)
#             gen_power = gen_speed[['ID', 'year', 'month', 'cf', 'turb_match', 'longitude', 'latitude', 'height', 'capacity']]
#             gen_power = gen_power.pivot(index=['ID','turb_match','year', 'longitude', 'latitude', 'height', 'capacity'], columns='month', values='cf')
#             gen_power.columns.name = None
#             gen_power.columns = [f'sim_{i}' if i not in ['ID','turb_match','year', 'longitude', 'latitude', 'height', 'capacity'] else f'{i}' for i in gen_power.columns]
            
#             return gen_power.reset_index()






# def speed_to_power2(speed_frame, powerCurveFile):    
#     x = powerCurveFile['data$speed']
#     y = powerCurveFile[speed_frame.turbine.data].to_numpy().flatten()
#     f2 = interpolate.interp1d(x, y, kind='cubic')

#     return f2(speed_frame.data)

# def simulate_power(ds, turbine_info, powerCurveFile, scalar=1, offset=0):
    
#     all_heights = np.sort(turbine_info['height'].unique())
#     ds = ds.assign_coords(
#         height=('height', all_heights))

    
#     wnd_spd = extrapolate_wind_speed(ds, turbine_info, scalar, offset)
    
#     # converting speed to power
#     gen_power = speed_to_power2(wnd_spd, powerCurveFile)
    
#     return gen_power